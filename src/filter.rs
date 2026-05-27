//! Conformer-ensemble filtering — Rust port of
//! `chemtrayzer.core.coords.ConformerEnsemble.filter`.
//!
//! The Python version optionally enumerates all graph-isomorphic atom
//! permutations via VF2++. The Rust port deliberately omits that path;
//! callers pass the atom mappings they care about explicitly via the
//! `mappings` slice. An empty slice is interpreted as the identity
//! mapping only.
//!
//! ## Algorithm summary
//!
//! 1. Sort the ensemble in ascending energy order (stable).
//! 2. Drop conformers whose energy exceeds the minimum by more than
//!    `energy_window` (Hartree).
//! 3. Iterate the remaining conformers in order; greedily keep the
//!    current one when it differs from every conformer already kept,
//!    according to [`is_different_conformer`]. With `mirror_check` set,
//!    additionally require that the mirror of the candidate differs from
//!    the candidate itself and from every kept conformer.
//! 4. If `cum_boltzmann_threshold < 1`, truncate the kept set so that the
//!    cumulative Boltzmann weight (at `temperature_k`) just exceeds the
//!    threshold.

use ndarray::{Array2, ArrayView2, ShapeBuilder};
use rayon::prelude::*;

use crate::elements::atomic_weights;
use crate::error::ConformerError;
use crate::ensemble::ConformerEnsemble;
use crate::geometry::{AtomType, Geometry};
use crate::inertia::{rotational_constants_hz, RotationalConstants};
use crate::rmsd::{qcp_rmsd_from_stats, QcpRmsdWorkspace};
use crate::{BOLTZMANN_CONSTANT_J_PER_K, HARTREE_ENERGY_J};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Pairwise-difference options shared by [`is_different_conformer`] and
/// [`filter_ensemble`]. Set any threshold to [`f64::INFINITY`] to skip the
/// corresponding check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfDiffOptions {
    /// RMSD threshold [Å]. Pairs with RMSD above this are considered
    /// different conformers.
    pub rmsd_threshold: f64,
    /// Rotational-constant threshold [%]. Pairs whose minimum
    /// element-wise relative difference (in %) of the rotational
    /// constants exceeds this value are considered different conformers.
    pub rot_threshold: f64,
    /// Energy threshold [Hartree]. Pairs with `|ΔE|` above this are
    /// considered different conformers (energies must be supplied).
    pub energy_threshold: f64,
    /// Mass-weight the RMSD (and intermediate Kabsch alignment).
    pub mass_weighted: bool,
    /// Apply a rigid-body (Kabsch / QCP) alignment before computing RMSD.
    pub rigid_transform: bool,
    /// When `true`, the filter additionally requires that the mirror of
    /// each candidate is different from the candidate itself and from
    /// every previously kept conformer. Only honoured when `is_chiral`
    /// is `false` (matching the Python guard).
    pub mirror_check: bool,
}

impl Default for ConfDiffOptions {
    fn default() -> Self {
        Self {
            rmsd_threshold: 0.125,
            rot_threshold: 1.0,
            energy_threshold: 0.00038,
            mass_weighted: false,
            rigid_transform: true,
            mirror_check: false,
        }
    }
}

/// Ensemble-level filter options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfFilterOptions {
    /// Pairwise-difference options.
    pub diff: ConfDiffOptions,
    /// Temperature [K] used for the cumulative-Boltzmann cutoff.
    pub temperature_k: f64,
    /// Cumulative-Boltzmann-weight cutoff in `(0, 1]`. When `< 1`, the
    /// filter truncates the result to the smallest prefix whose
    /// cumulative Boltzmann weight exceeds the threshold.
    pub cum_boltzmann_threshold: f64,
    /// Energy window [Hartree]. Conformers with `E - E_min > energy_window`
    /// are dropped before the pairwise loop runs.
    pub energy_window: f64,
    /// Whether the molecule is chiral. Mirrors the Python `_is_chiral`
    /// flag and gates the `mirror_check` branch: mirror checks only run
    /// when `mirror_check && !is_chiral`.
    pub is_chiral: bool,
}

impl Default for ConfFilterOptions {
    fn default() -> Self {
        Self {
            diff: ConfDiffOptions::default(),
            temperature_k: 1500.0,
            cum_boltzmann_threshold: 1.0,
            energy_window: f64::INFINITY,
            is_chiral: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Permutation validation
// ---------------------------------------------------------------------------

/// Validates a dense permutation: length matches `n_atoms`, every entry is
/// in `0..n_atoms`, and `self_atom_types[i] == other_atom_types[mapping[i]]`.
pub fn validate_permutation(
    mapping: &[usize],
    self_atom_types: &[AtomType],
    other_atom_types: &[AtomType],
) -> Result<(), ConformerError> {
    let n = self_atom_types.len();
    if mapping.len() != n {
        return Err(ConformerError::InvalidPermutationLength {
            permutation_len: mapping.len(),
            expected_len: n,
        });
    }
    if other_atom_types.len() != n {
        return Err(ConformerError::AtomCountMismatch {
            expected: n,
            found: other_atom_types.len(),
        });
    }
    for (i, &j) in mapping.iter().enumerate() {
        if j >= n {
            return Err(ConformerError::InvalidPermutationIndex {
                index: j,
                n_atoms: n,
            });
        }
        if self_atom_types[i] != other_atom_types[j] {
            return Err(ConformerError::PermutationAtomTypeMismatch {
                self_index: i,
                self_type: self_atom_types[i],
                other_index: j,
                other_type: other_atom_types[j],
            });
        }
    }
    Ok(())
}

/// A list of dense atom permutations whose validity has been checked
/// once against a specific atom-type list. Construct via
/// [`ValidatedMappings::new`] (or [`ValidatedMappings::empty`] for the
/// identity-only case); downstream consumers (`filter_ensemble`,
/// `is_different_conformer`, `min_rmsd_over_mappings`) accept `&Self`
/// and skip per-pair revalidation — every entry is guaranteed to be a
/// well-formed permutation of length `n_atoms` with matching atom types.
#[derive(Debug, Clone, Default)]
pub struct ValidatedMappings {
    inner: Vec<Vec<usize>>,
    n_atoms: usize,
}

impl ValidatedMappings {
    /// The empty (identity-only) mapping list.
    pub fn empty() -> Self {
        Self { inner: Vec::new(), n_atoms: 0 }
    }

    /// Validates every mapping in `mappings` against `atom_types`
    /// exactly once and returns an immutable, trusted handle on success.
    /// Returns the first error encountered.
    pub fn new(
        mappings: Vec<Vec<usize>>,
        atom_types: &[AtomType],
    ) -> Result<Self, ConformerError> {
        mappings
            .iter()
            .try_for_each(|m| validate_permutation(m, atom_types, atom_types))?;
        Ok(Self { inner: mappings, n_atoms: atom_types.len() })
    }

    /// View the mappings as a slice.
    pub fn as_slice(&self) -> &[Vec<usize>] {
        &self.inner
    }

    /// Number of mappings stored.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` if no mappings are stored (treated as identity-only by
    /// consumers).
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Atom count the mappings were validated against. `0` for an empty
    /// mapping list.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn iter_or_identity(&self) -> impl Iterator<Item = Option<&[usize]>> + '_ {
        self.inner
            .iter()
            .map(|m| Some(m.as_slice()))
            .chain(self.inner.is_empty().then_some(None))
    }
}

// ---------------------------------------------------------------------------
// Mass-weighted / permutation-aware RMSD
// ---------------------------------------------------------------------------

/// Computes the RMSD between `left` and `right` under a dense atom
/// permutation `mapping` (where `mapping[i] = j` means "left atom `i`
/// corresponds to right atom `j`"). `weights` defaults to uniform 1.0 when
/// `None`. When `rigid_transform` is `true`, performs a (mass-weighted)
/// Kabsch / QCP alignment before computing the residual; otherwise uses
/// the raw coordinates (matching Python `Geometry.rmsd(..., rigid_transform=False)`).
///
/// `mapping == None` is treated as the identity permutation.
///
/// Validates pair atom count, `mapping`, and optional `weights` length on
/// every call. Coordinate shape is assumed to already be `(n_atoms, 3)`.
/// Hot paths inside this module use the [`rmsd_permuted_unchecked`]
/// variant after a one-shot [`ValidatedMappings`] check.
pub fn rmsd_permuted(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
    weights: Option<&[f64]>,
    mapping: Option<&[usize]>,
    rigid_transform: bool,
) -> Result<f64, ConformerError> {
    let n = left.nrows();
    if right.nrows() != n {
        return Err(ConformerError::AtomCountMismatch {
            expected: n,
            found: right.nrows(),
        });
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(ConformerError::AtomCountMismatch {
                expected: n,
                found: w.len(),
            });
        }
    }
    if let Some(m) = mapping {
        if m.len() != n {
            return Err(ConformerError::InvalidPermutationLength {
                permutation_len: m.len(),
                expected_len: n,
            });
        }
        for &j in m {
            if j >= n {
                return Err(ConformerError::InvalidPermutationIndex {
                    index: j,
                    n_atoms: n,
                });
            }
        }
    }
    rmsd_permuted_unchecked(left, right, weights, mapping, rigid_transform)
}

/// Body of [`rmsd_permuted`] that skips per-mapping length/index
/// validation. Safe to call only when pair atom counts, coordinate shape,
/// and optional weight length are already known to be valid — e.g. for
/// coordinates that came from [`ConformerEnsemble`] / [`Geometry`] and a
/// mapping that came from [`ValidatedMappings`].
pub(crate) fn rmsd_permuted_unchecked(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
    weights: Option<&[f64]>,
    mapping: Option<&[usize]>,
    rigid_transform: bool,
) -> Result<f64, ConformerError> {
    let n = left.nrows();
    debug_assert_eq!(right.nrows(), n);
    debug_assert_eq!(left.ncols(), 3);
    debug_assert_eq!(right.ncols(), 3);
    debug_assert!(weights.is_none_or(|w| w.len() == n));

    if n == 0 {
        return Ok(0.0);
    }

    let weight_at = |i: usize| weights.map_or(1.0, |w| w[i]);
    let right_idx = |i: usize| mapping.map_or(i, |m| m[i]);

    // Total weight; for uniform weights this is `n`.
    let mut weight_sum = 0.0;
    for i in 0..n {
        weight_sum += weight_at(i);
    }
    debug_assert!(weight_sum > 0.0);

    if !rigid_transform {
        let mut acc = 0.0;
        for i in 0..n {
            let j = right_idx(i);
            let dx = left[[i, 0]] - right[[j, 0]];
            let dy = left[[i, 1]] - right[[j, 1]];
            let dz = left[[i, 2]] - right[[j, 2]];
            acc += weight_at(i) * (dx * dx + dy * dy + dz * dz);
        }
        return Ok((acc / weight_sum).sqrt());
    }

    // Weighted centroids (over `left[i]` and `right[mapping[i]]`).
    let mut lc = [0.0_f64; 3];
    let mut rc = [0.0_f64; 3];
    for i in 0..n {
        let w = weight_at(i);
        let j = right_idx(i);
        lc[0] += w * left[[i, 0]];
        lc[1] += w * left[[i, 1]];
        lc[2] += w * left[[i, 2]];
        rc[0] += w * right[[j, 0]];
        rc[1] += w * right[[j, 1]];
        rc[2] += w * right[[j, 2]];
    }
    lc.iter_mut().for_each(|v| *v /= weight_sum);
    rc.iter_mut().for_each(|v| *v /= weight_sum);

    // Weighted cross-covariance + Σ w·|x|², Σ w·|y|².
    let mut g_l = 0.0_f64;
    let mut g_r = 0.0_f64;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut sxz = 0.0;
    let mut syx = 0.0;
    let mut syy = 0.0;
    let mut syz = 0.0;
    let mut szx = 0.0;
    let mut szy = 0.0;
    let mut szz = 0.0;

    for i in 0..n {
        let w = weight_at(i);
        let j = right_idx(i);
        let lx = left[[i, 0]] - lc[0];
        let ly = left[[i, 1]] - lc[1];
        let lz = left[[i, 2]] - lc[2];
        let rx = right[[j, 0]] - rc[0];
        let ry = right[[j, 1]] - rc[1];
        let rz = right[[j, 2]] - rc[2];

        g_l += w * (lx * lx + ly * ly + lz * lz);
        g_r += w * (rx * rx + ry * ry + rz * rz);

        let wlx = w * lx;
        let wly = w * ly;
        let wlz = w * lz;
        sxx += wlx * rx;
        sxy += wlx * ry;
        sxz += wlx * rz;
        syx += wly * rx;
        syy += wly * ry;
        syz += wly * rz;
        szx += wlz * rx;
        szy += wlz * ry;
        szz += wlz * rz;
    }

    Ok(qcp_rmsd_from_stats(
        weight_sum, g_l, g_r, sxx, sxy, sxz, syx, syy, syz, szx, szy, szz,
    ))
}

/// Minimum RMSD across an explicit list of pre-validated dense atom
/// permutations.
///
/// An empty [`ValidatedMappings`] is treated as a single identity
/// mapping. Mapping counts are typically small (≤ a few dozen) and this
/// routine is itself invoked from inside the rayon-parallel
/// `check_against_kept` loop, so the per-mapping evaluations are
/// deliberately serial: nesting rayon inside an already-parallel context
/// would only add scheduling overhead without delivering useful
/// parallelism. Because the mappings were validated once at construction
/// time, the inner loop calls the unchecked RMSD variant.
pub fn min_rmsd_over_mappings(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
    weights: Option<&[f64]>,
    mappings: &ValidatedMappings,
    rigid_transform: bool,
) -> Result<f64, ConformerError> {
    mappings.iter_or_identity().try_fold(f64::INFINITY, |best, mapping| {
        Ok(best.min(rmsd_permuted_unchecked(
            left,
            right,
            weights,
            mapping,
            rigid_transform,
        )?))
    })
}

/// Returns `true` when every permitted mapping yields an RMSD strictly
/// above `rmsd_threshold`.
///
/// This is the hot-path predicate used by the filter: the caller only
/// needs the boolean threshold comparison, not the exact minimum RMSD.
/// That lets the mapping scan short-circuit as soon as one mapping lands
/// at or below the threshold instead of evaluating the remaining
/// permutations just to compute an exact minimum that will be discarded.
fn all_mappings_exceed_rmsd_threshold(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
    weights: Option<&[f64]>,
    mappings: &ValidatedMappings,
    rigid_transform: bool,
    rmsd_threshold: f64,
) -> Result<bool, ConformerError> {
    debug_assert!(rmsd_threshold.is_finite());

    for mapping in mappings.iter_or_identity() {
        let rmsd = rmsd_permuted_unchecked(
            left,
            right,
            weights,
            mapping,
            rigid_transform,
        )?;
        if rmsd <= rmsd_threshold {
            return Ok(false);
        }
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// is_different_conformer
// ---------------------------------------------------------------------------

/// Runs the two cheap pairwise difference checks (energy, rotational
/// constants) in their cheap-first order. Returns `Ok(true)` as soon as
/// either check proves the pair differs, `Ok(false)` when neither check
/// can decide (RMSD is required), or an error if a finite energy
/// threshold was requested but energies are missing.
fn cheap_checks_imply_different(
    left_energy: Option<f64>,
    right_energy: Option<f64>,
    left_rot: Option<&RotationalConstants>,
    right_rot: Option<&RotationalConstants>,
    opts: &ConfDiffOptions,
) -> Result<bool, ConformerError> {
    if opts.energy_threshold.is_finite() {
        match (left_energy, right_energy) {
            (Some(a), Some(b)) => {
                if (a - b).abs() > opts.energy_threshold {
                    return Ok(true);
                }
            }
            _ => return Err(ConformerError::MissingEnergyForThreshold),
        }
    }

    if opts.rot_threshold.is_finite() {
        let l = left_rot
            .expect("left rot constants must be supplied when rot_threshold is finite");
        let r = right_rot
            .expect("right rot constants must be supplied when rot_threshold is finite");
        if rotational_constants_diff_percent(l, r) > opts.rot_threshold {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Cheap-first pairwise difference check using precomputed per-conformer
/// quantities. Order: energy ➜ rotational constants ➜ RMSD. RMSD is
/// evaluated last because it is by far the most expensive step (QCP
/// eigensolve, optionally repeated over `mappings`).
///
/// `left_rot` / `right_rot` are required when `opts.rot_threshold` is
/// finite. `weights` is passed straight through to the RMSD step.
#[allow(clippy::too_many_arguments)]
fn is_different_precomputed(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
    left_energy: Option<f64>,
    right_energy: Option<f64>,
    left_rot: Option<&RotationalConstants>,
    right_rot: Option<&RotationalConstants>,
    weights: Option<&[f64]>,
    mappings: &ValidatedMappings,
    opts: &ConfDiffOptions,
) -> Result<bool, ConformerError> {
    if cheap_checks_imply_different(
        left_energy, right_energy, left_rot, right_rot, opts,
    )? {
        return Ok(true);
    }

    if !opts.rmsd_threshold.is_finite() {
        return Ok(false);
    }
    all_mappings_exceed_rmsd_threshold(
        left,
        right,
        weights,
        mappings,
        opts.rigid_transform,
        opts.rmsd_threshold,
    )
}

/// Returns `true` when `left` and `right` differ on at least one of the
/// configured criteria (energy, rotational constants, RMSD). Mirrors the
/// Python `ConformerEnsemble.is_different_conformer` semantics.
///
/// `atom_types` is the shared atom-type slice for both geometries (used to
/// look up masses for the rotational-constants and mass-weighted RMSD
/// checks). `mappings` is the pre-validated permutation list passed
/// through to the RMSD step; pass [`ValidatedMappings::empty`] for the
/// identity-only case.
///
/// For batch use inside [`filter_ensemble`] the masses and rotational
/// constants are precomputed once per conformer; standalone callers pay
/// the recomputation cost on every invocation.
#[allow(clippy::too_many_arguments)]
pub fn is_different_conformer(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
    atom_types: &[AtomType],
    left_energy: Option<f64>,
    right_energy: Option<f64>,
    mappings: &ValidatedMappings,
    opts: &ConfDiffOptions,
) -> Result<bool, ConformerError> {
    let need_masses =
        opts.rot_threshold.is_finite() || opts.mass_weighted;
    let masses: Vec<f64> = if need_masses {
        atomic_weights(atom_types)?
    } else {
        Vec::new()
    };
    let weights = opts.mass_weighted.then_some(masses.as_slice());

    let (left_rot, right_rot) = if opts.rot_threshold.is_finite() {
        (
            Some(rotational_constants_hz(left, &masses)?),
            Some(rotational_constants_hz(right, &masses)?),
        )
    } else {
        (None, None)
    };

    is_different_precomputed(
        left,
        right,
        left_energy,
        right_energy,
        left_rot.as_ref(),
        right_rot.as_ref(),
        weights,
        mappings,
        opts,
    )
}

/// Minimum element-wise `|a - b| / min(a, b) * 100`. For a linear/non-linear
/// pair, falls back to the longer of the two slices by replicating the
/// linear value across its three components (this is what the NumPy
/// broadcasting in the Python implementation effectively does).
fn rotational_constants_diff_percent(
    a: &RotationalConstants,
    b: &RotationalConstants,
) -> f64 {
    let (sa, sb) = match (a, b) {
        (RotationalConstants::Linear(la), RotationalConstants::Linear(lb)) => {
            return percent_diff(*la, *lb);
        }
        (RotationalConstants::Nonlinear(va), RotationalConstants::Nonlinear(vb)) => {
            (va.as_slice(), vb.as_slice())
        }
        (RotationalConstants::Linear(la), RotationalConstants::Nonlinear(vb)) => {
            let arr = [*la; 3];
            return vb
                .iter()
                .zip(arr.iter())
                .map(|(x, y)| percent_diff(*x, *y))
                .fold(f64::INFINITY, f64::min);
        }
        (RotationalConstants::Nonlinear(va), RotationalConstants::Linear(lb)) => {
            let arr = [*lb; 3];
            return va
                .iter()
                .zip(arr.iter())
                .map(|(x, y)| percent_diff(*x, *y))
                .fold(f64::INFINITY, f64::min);
        }
    };
    sa.iter()
        .zip(sb.iter())
        .map(|(x, y)| percent_diff(*x, *y))
        .fold(f64::INFINITY, f64::min)
}

#[inline]
fn percent_diff(a: f64, b: f64) -> f64 {
    let denom = a.min(b);
    if denom == 0.0 {
        f64::INFINITY
    } else {
        (a - b).abs() / denom * 100.0
    }
}

// ---------------------------------------------------------------------------
// filter_ensemble
// ---------------------------------------------------------------------------

/// Returns the filtered ensemble, keeping the lowest-energy representative
/// of each unique conformer. Mirrors Python
/// `ConformerEnsemble.filter(opts, inplace=False)`.
///
/// `mappings` is forwarded to every pairwise RMSD comparison; pass
/// [`ValidatedMappings::empty`] for the identity-only case. Mapping
/// validity was checked once at [`ValidatedMappings::new`] construction
/// time, so the hot loop performs no further per-pair revalidation. The
/// caller is expected to construct `ValidatedMappings` against the same
/// `atom_types` as `ensemble`.
pub fn filter_ensemble(
    ensemble: &ConformerEnsemble,
    opts: &ConfFilterOptions,
    mappings: &ValidatedMappings,
) -> Result<ConformerEnsemble, ConformerError> {
    let atom_types = ensemble.atom_types();
    if !mappings.is_empty() && mappings.n_atoms() != atom_types.len() {
        return Err(ConformerError::InvalidPermutationLength {
            permutation_len: mappings.n_atoms(),
            expected_len: atom_types.len(),
        });
    }

    // 1. Sort ascending by energy (stable).
    let mut working = ensemble.clone();
    working.sort_by_energy(false);

    if working.is_empty() {
        return Ok(working);
    }

    // 2. Energy-window prefilter.
    let e_min = working.energies()[0];
    let energies = working.energies().to_vec();
    let n = energies.len();
    let cutoff = if opts.energy_window.is_finite() {
        energies
            .iter()
            .position(|&e| (e - e_min) > opts.energy_window)
            .unwrap_or(n)
    } else {
        n
    };

    // 3. Precompute per-conformer quantities used by every pair test.
    //    Doing this once turns the inner loop's O(K) recomputations into
    //    O(1) lookups (was: `atomic_weights` + two `rotational_constants_hz`
    //    per pair).
    let need_masses =
        opts.diff.rot_threshold.is_finite() || opts.diff.mass_weighted;
    let masses: Vec<f64> = if need_masses {
        atomic_weights(atom_types)?
    } else {
        Vec::new()
    };
    let weights = opts.diff.mass_weighted.then_some(masses.as_slice());

    let rot_consts: Vec<RotationalConstants> =
        if opts.diff.rot_threshold.is_finite() {
            (0..cutoff)
                .map(|i| {
                    rotational_constants_hz(
                        working.conformer_coords(i).expect("index in range"),
                        &masses,
                    )
                })
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };

    // Fast-path precomputation: when the RMSD step does not need atom
    // permutations or mass weighting and a finite rigid-transform RMSD
    // threshold is in play, we route the inner kept loop through
    // `QcpRmsdWorkspace`: each rayon worker amortises a single
    // `prepare_reference(cand)` over its chunk of kept comparisons and
    // benefits from pulp SIMD + the column-major zero-copy SoA load. The
    // workspace assumes both sides are pre-centered on their own
    // centroid (it does not subtract centroids itself), so we centre
    // every kept-eligible conformer once here in column-major layout.
    let fast_rmsd_path = mappings.is_empty()
        && !opts.diff.mass_weighted
        && opts.diff.rigid_transform
        && opts.diff.rmsd_threshold.is_finite();
    let centered_coords: Vec<Array2<f64>> = if fast_rmsd_path {
        (0..cutoff)
            .map(|i| {
                centered_coords_f(
                    working.conformer_coords(i).expect("index in range"),
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    let centered_slice: Option<&[Array2<f64>]> =
        fast_rmsd_path.then_some(centered_coords.as_slice());

    // 4. Greedy pairwise filter.
    //
    // The outer loop is inherently sequential (each candidate's fate
    // depends on the already-kept set), but the inner "is this candidate
    // different from every kept conformer?" check parallelises cleanly
    // via `try_for_each` over `&kept_indices`, which short-circuits on
    // the first duplicate found and on the first error propagated from
    // any thread.
    //
    // Mirror handling: a reflection leaves both the energy and the
    // moment-of-inertia eigenvalues invariant, so the mirror conformer
    // reuses the candidate's precomputed energy and rotational
    // constants. Mirror-vs-self therefore reduces to a single RMSD
    // comparison (energy and rot checks pass trivially).
    let mirror_check_active = opts.diff.mirror_check && !opts.is_chiral;
    let mut kept_indices: Vec<usize> = Vec::new();

    'outer: for i in 0..cutoff {
        let cand_coords = working.conformer_coords(i).expect("index in range");
        let cand_centered: Option<ArrayView2<'_, f64>> =
            centered_slice.map(|s| s[i].view());
        let cand_energy = energies[i];
        let cand_rot = rot_consts.get(i);
        let duplicates_with_kept =
            |probe_coords: ArrayView2<'_, f64>,
             probe_centered: Option<ArrayView2<'_, f64>>| {
                check_against_kept(
                    probe_coords,
                    probe_centered,
                    cand_energy,
                    cand_rot,
                    &kept_indices,
                    &working,
                    centered_slice,
                    &energies,
                    &rot_consts,
                    weights,
                    mappings,
                    &opts.diff,
                )
            };

        if duplicates_with_kept(cand_coords, cand_centered)? {
            continue 'outer;
        }

        if mirror_check_active {
            // No RMSD threshold ➜ mirror-vs-self can never differ; cheap
            // checks pass trivially because energy + rot constants are
            // mirror-invariant.
            if !opts.diff.rmsd_threshold.is_finite() {
                continue 'outer;
            }

            // Mirror of an already-centered cloud is still centered
            // (x ↦ -x preserves the zero centroid), so on the fast path
            // we mirror the centred buffer directly and skip a second
            // centring pass.
            let mirror_source = cand_centered.unwrap_or(cand_coords);
            let mirror = mirror_coords(mirror_source);
            let mirror_view = mirror.view();
            let mirror_centered = fast_rmsd_path.then_some(mirror_view);

            // Mirror vs self: identical energy and rot constants ➜ only
            // RMSD matters. On the fast path, route through a single
            // workspace call with both sides already centred.
            let mirror_self_rmsd = if fast_rmsd_path {
                let n = atom_types.len();
                if n == 0 {
                    0.0
                } else {
                    let mut ws = QcpRmsdWorkspace::new(n);
                    ws.rmsd(mirror_view, mirror_source)
                        .expect("validated shapes")
                }
            } else {
                rmsd_permuted_unchecked(
                    mirror_view,
                    cand_coords,
                    weights,
                    None,
                    opts.diff.rigid_transform,
                )?
            };
            if mirror_self_rmsd <= opts.diff.rmsd_threshold {
                continue 'outer;
            }

            // Mirror vs every previously kept (reuses cand's rot consts).
            if duplicates_with_kept(mirror_view, mirror_centered)? {
                continue 'outer;
            }
        }

        kept_indices.push(i);
    }

    // Materialise the kept subset.
    let kept_coords: Vec<Array2<f64>> = kept_indices
        .iter()
        .map(|&i| working.coords()[i].clone())
        .collect();
    let kept_energies: Vec<f64> = kept_indices.iter().map(|&i| energies[i]).collect();

    let mut filtered = ConformerEnsemble::new(
        atom_types.to_vec(),
        kept_coords,
        kept_energies,
    )?;

    // 4. Boltzmann-weight cutoff.
    if opts.cum_boltzmann_threshold < 1.0 && !filtered.is_empty() {
        let weights = filtered.boltzmann_weights(opts.temperature_k)?;
        let mut cum = 0.0;
        let mut cutoff_idx = filtered.n_conformers();
        for (i, &w) in weights.iter().enumerate() {
            cum += w;
            if cum > opts.cum_boltzmann_threshold {
                cutoff_idx = i + 1;
                break;
            }
        }
        if cutoff_idx < filtered.n_conformers() {
            let truncated_coords: Vec<Array2<f64>> =
                filtered.coords()[..cutoff_idx].to_vec();
            let truncated_energies: Vec<f64> =
                filtered.energies()[..cutoff_idx].to_vec();
            filtered = ConformerEnsemble::new(
                atom_types.to_vec(),
                truncated_coords,
                truncated_energies,
            )?;
        }
    }

    Ok(filtered)
}

/// Parallel "is `cand` different from every conformer in `kept_indices`?"
/// check. Returns `Ok(true)` as soon as any kept conformer is found that
/// is *not* different from the candidate (i.e. the candidate is a
/// duplicate), `Ok(false)` when the candidate differs from every kept
/// conformer, or propagates the first error from any worker.
///
/// Uses [`ParallelIterator::try_for_each`] with a `Halt` enum as the
/// single early-exit channel: returning `Err(Halt::Duplicate)` stops
/// every worker immediately, and `Err(Halt::Err(e))` propagates the
/// first observed `ConformerError`. `with_min_len` keeps groups of
/// comparisons on the same worker, amortising rayon's per-task overhead
/// for the common case of small to medium kept sets without needing a
/// hand-tuned sequential fallback.
///
/// When `cand_centered` and `centered_coords` are both `Some` the inner
/// RMSD step routes through a per-worker [`QcpRmsdWorkspace`]: each
/// worker allocates its workspace once (in rayon's `init` closure),
/// runs `prepare_reference(cand)` once, and then reuses the prepared
/// reference for every kept comparison in its chunk. This buys SIMD
/// cross-covariance accumulation, the column-major zero-copy SoA load,
/// and amortises the candidate-side centroid/squared-norm computation
/// across the worker's chunk. Both centred slices must be pre-centred
/// on their own centroid; the workspace does not subtract centroids.
#[allow(clippy::too_many_arguments)]
fn check_against_kept(
    cand_coords: ArrayView2<'_, f64>,
    cand_centered: Option<ArrayView2<'_, f64>>,
    cand_energy: f64,
    cand_rot: Option<&RotationalConstants>,
    kept_indices: &[usize],
    working: &ConformerEnsemble,
    centered_coords: Option<&[Array2<f64>]>,
    energies: &[f64],
    rot_consts: &[RotationalConstants],
    weights: Option<&[f64]>,
    mappings: &ValidatedMappings,
    diff_opts: &ConfDiffOptions,
) -> Result<bool, ConformerError> {
    /// Minimum chunk size handed to a rayon worker. Each comparison is
    /// O(n_atoms) for the energy/rot checks plus an O(n_atoms) QCP
    /// eigensolve for the RMSD step — microseconds-scale for typical
    /// molecules. Bundling several comparisons per task amortises the
    /// scheduler overhead.
    const KEPT_CHUNK_MIN: usize = 8;

    enum Halt {
        Duplicate,
        Err(ConformerError),
    }

    // Fast path: per-worker workspace with prepared reference.
    if let (Some(cand_c), Some(centered)) = (cand_centered, centered_coords) {
        debug_assert!(mappings.is_empty());
        debug_assert!(weights.is_none());
        debug_assert!(diff_opts.rigid_transform);
        debug_assert!(diff_opts.rmsd_threshold.is_finite());

        let n_atoms = cand_c.nrows();
        let threshold = diff_opts.rmsd_threshold;
        let result = kept_indices
            .par_iter()
            .with_min_len(KEPT_CHUNK_MIN)
            .try_for_each_init(
                || {
                    let mut ws = QcpRmsdWorkspace::new(n_atoms);
                    if n_atoms > 0 {
                        ws.prepare_reference(cand_c)
                            .expect("validated shape");
                    }
                    ws
                },
                |ws, &k| {
                    match cheap_checks_imply_different(
                        Some(cand_energy),
                        Some(energies[k]),
                        cand_rot,
                        rot_consts.get(k),
                        diff_opts,
                    ) {
                        Ok(true) => return Ok(()),
                        Ok(false) => {}
                        Err(e) => return Err(Halt::Err(e)),
                    }
                    let rmsd = if n_atoms == 0 {
                        0.0
                    } else {
                        match ws.rmsd_prepared(centered[k].view()) {
                            Ok(v) => v,
                            Err(e) => return Err(Halt::Err(e)),
                        }
                    };
                    if rmsd > threshold {
                        Ok(())
                    } else {
                        Err(Halt::Duplicate)
                    }
                },
            );

        return match result {
            Ok(()) => Ok(false),
            Err(Halt::Duplicate) => Ok(true),
            Err(Halt::Err(e)) => Err(e),
        };
    }

    // Generic path: mappings, mass weighting, non-rigid transform, or
    // infinite RMSD threshold. Falls back to `rmsd_permuted_unchecked`
    // via `is_different_precomputed`; the workspace does not support
    // these cases.
    let result = kept_indices
        .par_iter()
        .with_min_len(KEPT_CHUNK_MIN)
        .try_for_each(|&k| {
            let kept_coords =
                working.conformer_coords(k).expect("index in range");
            match is_different_precomputed(
                cand_coords,
                kept_coords,
                Some(cand_energy),
                Some(energies[k]),
                cand_rot,
                rot_consts.get(k),
                weights,
                mappings,
                diff_opts,
            ) {
                Ok(true) => Ok(()),
                Ok(false) => Err(Halt::Duplicate),
                Err(e) => Err(Halt::Err(e)),
            }
        });

    match result {
        Ok(()) => Ok(false),
        Err(Halt::Duplicate) => Ok(true),
        Err(Halt::Err(e)) => Err(e),
    }
}

/// Returns mirrored coordinates (x ↦ -x) in a fresh column-major `(n, 3)`
/// array.
fn mirror_coords(coords: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = coords.nrows();
    let mut buf = Vec::with_capacity(3 * n);
    for i in 0..n {
        buf.push(-coords[[i, 0]]);
    }
    for i in 0..n {
        buf.push(coords[[i, 1]]);
    }
    for i in 0..n {
        buf.push(coords[[i, 2]]);
    }
    Array2::from_shape_vec((n, 3).f(), buf)
        .expect("3*n elements fit a column-major (n,3) array")
}

/// Returns a column-major `(n, 3)` copy of `coords` translated so its
/// unweighted centroid is at the origin. Column-major layout means the
/// flat buffer is exactly `[x | y | z]`, which lets
/// [`QcpRmsdWorkspace::prepare_reference`] / [`QcpRmsdWorkspace::rmsd_prepared`]
/// take their zero-copy SoA fast path.
fn centered_coords_f(coords: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = coords.nrows();
    let mut buf = Vec::with_capacity(3 * n);
    if n == 0 {
        return Array2::from_shape_vec((0, 3).f(), buf)
            .expect("empty (0,3) array");
    }
    let inv_n = 1.0 / n as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for i in 0..n {
        cx += coords[[i, 0]];
        cy += coords[[i, 1]];
        cz += coords[[i, 2]];
    }
    cx *= inv_n;
    cy *= inv_n;
    cz *= inv_n;
    for i in 0..n {
        buf.push(coords[[i, 0]] - cx);
    }
    for i in 0..n {
        buf.push(coords[[i, 1]] - cy);
    }
    for i in 0..n {
        buf.push(coords[[i, 2]] - cz);
    }
    Array2::from_shape_vec((n, 3).f(), buf)
        .expect("3*n elements fit a column-major (n,3) array")
}

/// Convenience wrapper for callers that already have [`Geometry`] objects.
///
/// `mappings` is forwarded to [`is_different_conformer`]; pass
/// [`ValidatedMappings::empty`] for the identity-only case.
pub fn is_different_geometry(
    left: &Geometry,
    right: &Geometry,
    left_energy: Option<f64>,
    right_energy: Option<f64>,
    mappings: &ValidatedMappings,
    opts: &ConfDiffOptions,
) -> Result<bool, ConformerError> {
    if left.atom_types() != right.atom_types() {
        return Err(ConformerError::InconsistentAtomTypes);
    }
    is_different_conformer(
        left.coords(),
        right.coords(),
        left.atom_types(),
        left_energy,
        right_energy,
        mappings,
        opts,
    )
}

// ---------------------------------------------------------------------------
// Boltzmann factor (suppressed warning shim — already implemented on
// ConformerEnsemble; use that for the cutoff. These constants are imported
// purely to document the unit conventions used by the filter.)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const _BOLTZMANN_REF: f64 = BOLTZMANN_CONSTANT_J_PER_K;
#[allow(dead_code)]
const _HARTREE_REF: f64 = HARTREE_ENERGY_J;
