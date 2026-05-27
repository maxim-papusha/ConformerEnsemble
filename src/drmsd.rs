//! Distance-RMSD (dRMSD) and all-pairs Euclidean distances.
//!
//! dRMSD is the rotation- and translation-invariant RMSD over the upper
//! triangular pairwise-distance matrices of two structures with the same
//! atom count `N`:
//!
//! ```text
//! dRMSD(A, B) = sqrt( (1 / M) * sum_{i<j} (d^A_{ij} - d^B_{ij})^2 )
//! ```
//!
//! where `d^X_{ij} = ||x_i - x_j||` and `M = N * (N - 1) / 2`. Unlike the
//! QCP RMSD in [`crate::rmsd`], dRMSD does not require pre-centering and
//! does not depend on any rotational alignment.
//!
//! # Memory layout
//!
//! Mirrors the QCP RMSD workspace: per-conformer input is `(N, 3)`
//! `float64`, ideally column-major (Fortran-ordered). In that case the
//! flat memory is the internal SoA layout `[x | y | z]` and loading skips
//! the AoS→SoA shuffle. On the Python side this maps directly onto an
//! `np.asfortranarray((n_atoms, 3))` view.
//!
//! # Kernel strategy
//!
//! This module is **scalar only**. The inner loop fixes one atom `i`
//! (broadcast `xi`, `yi`, `zi`) and sweeps `j > i` over SoA streams,
//! computing `sqrt((xj-xi)^2 + (yj-yi)^2 + (zj-zi)^2)`. The loops are
//! written so LLVM autovectorizes them cleanly under `-O3`. An explicit
//! `std::simd` + `multiversion` variant lives behind the optional
//! `portable-simd` feature in [`crate::drmsd_simd`].

use ndarray::ArrayView2;

use crate::error::ConformerError;

/// Reusable workspace for high-throughput dRMSD evaluations on coordinate
/// sets with a fixed atom count.
///
/// Construction performs **one** heap allocation of `6 * n_atoms` `f64`s
/// for SoA scratch plus one of `n_atoms * (n_atoms - 1) / 2` for the
/// reference pairwise distances. The hot path (`prepare_reference`,
/// `drmsd_prepared`, `drmsd`) does no further allocation.
#[derive(Debug, Clone)]
pub struct DrmsdWorkspace {
    n_atoms: usize,
    /// `[ref_x | ref_y | ref_z | right_x | right_y | right_z]`,
    /// each block of length `n_atoms`. Single contiguous allocation.
    coords: Vec<f64>,
    /// Upper-triangular pairwise distance vector for the reference,
    /// length `n_atoms * (n_atoms - 1) / 2`. Packed row-major:
    /// `d(0,1) .. d(0,n-1) | d(1,2) .. d(1,n-1) | ... | d(n-2, n-1)`.
    reference_distances: Vec<f64>,
    reference_prepared: bool,
}

impl DrmsdWorkspace {
    /// Creates a workspace for repeated dRMSD computations with `n_atoms`.
    #[inline]
    pub fn new(n_atoms: usize) -> Self {
        Self {
            n_atoms,
            coords: vec![0.0; 6 * n_atoms],
            reference_distances: vec![0.0; n_pairs(n_atoms)],
            reference_prepared: false,
        }
    }

    /// Back-compat constructor. The `_use_simd` flag is now a no-op: this
    /// module is scalar-only. The explicit-SIMD variant lives in
    /// [`crate::drmsd_simd`] behind the `portable-simd` feature.
    #[inline]
    pub fn with_simd(n_atoms: usize, _use_simd: bool) -> Self {
        Self::new(n_atoms)
    }

    /// Number of atoms this workspace is configured for.
    #[inline(always)]
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Number of atom pairs this workspace tracks (`n * (n - 1) / 2`).
    #[inline(always)]
    pub fn n_pairs(&self) -> usize {
        self.reference_distances.len()
    }

    /// Returns the pairwise distance matrix for the prepared reference, in
    /// flat upper-triangular row-major packing.
    #[inline(always)]
    pub fn reference_distances(&self) -> &[f64] {
        &self.reference_distances
    }

    /// Prepares the left-hand coordinates once for repeated comparisons.
    pub fn prepare_reference(
        &mut self,
        left: ArrayView2<'_, f64>,
    ) -> Result<(), ConformerError> {
        validate_coord_shape(left)?;
        validate_workspace_n_atoms(self.n_atoms, left.nrows())?;
        let n = self.n_atoms;
        let (ref_scratch, _) = self.coords.split_at_mut(3 * n);
        load_to_soa(ref_scratch, n, left);
        let (lx, ly, lz) = split3(ref_scratch, n);
        compute_pairwise_distances_dispatch(lx, ly, lz, &mut self.reference_distances);
        self.reference_prepared = true;
        Ok(())
    }

    /// Computes dRMSD of `right` against the prepared reference.
    ///
    /// The right-side pairwise distances are computed on the fly in chunks
    /// and consumed immediately against the cached reference distances, so
    /// no `O(n^2)` right-side distance buffer is materialised.
    pub fn drmsd_prepared(
        &mut self,
        right: ArrayView2<'_, f64>,
    ) -> Result<f64, ConformerError> {
        if !self.reference_prepared {
            return Err(ConformerError::WorkspaceReferenceNotPrepared);
        }
        validate_coord_shape(right)?;
        validate_workspace_n_atoms(self.n_atoms, right.nrows())?;
        let n = self.n_atoms;
        let m = self.reference_distances.len();
        if m == 0 {
            return Ok(0.0);
        }

        let (_, right_scratch) = self.coords.split_at_mut(3 * n);
        load_to_soa(right_scratch, n, right);
        let (rx, ry, rz) = split3(right_scratch, n);

        let ssd = sum_sq_distance_diffs_dispatch(rx, ry, rz, &self.reference_distances);

        Ok((ssd / m as f64).sqrt())
    }

    /// Computes dRMSD in one shot, equivalent to
    /// [`prepare_reference`](Self::prepare_reference) followed by
    /// [`drmsd_prepared`](Self::drmsd_prepared).
    pub fn drmsd(
        &mut self,
        left: ArrayView2<'_, f64>,
        right: ArrayView2<'_, f64>,
    ) -> Result<f64, ConformerError> {
        validate_pair_shape(left, right)?;
        self.prepare_reference(left)?;
        self.drmsd_prepared(right)
    }
}

/// Computes dRMSD between two structures.
///
/// **Allocates** a fresh [`DrmsdWorkspace`] per call. For repeated calls on
/// geometries of a fixed size, prefer [`DrmsdWorkspace::drmsd`] or the
/// prepared-reference workflow.
#[inline]
pub fn drmsd(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
) -> Result<f64, ConformerError> {
    let mut workspace = DrmsdWorkspace::new(left.nrows());
    workspace.drmsd(left, right)
}

/// Returns the flat upper-triangular pairwise Euclidean distances for
/// `coords`, length `n * (n - 1) / 2`, packed row-major:
/// `d(0,1), d(0,2), ..., d(0,n-1), d(1,2), ..., d(n-2, n-1)`.
pub fn pairwise_distances(
    coords: ArrayView2<'_, f64>,
) -> Result<Vec<f64>, ConformerError> {
    validate_coord_shape(coords)?;
    let n = coords.nrows();
    let m = n_pairs(n);
    let mut soa = vec![0.0_f64; 3 * n];
    load_to_soa(&mut soa, n, coords);
    let (x, y, z) = split3(&soa, n);
    let mut out = vec![0.0_f64; m];
    compute_pairwise_distances_dispatch(x, y, z, &mut out);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[inline(always)]
fn validate_coord_shape(coords: ArrayView2<'_, f64>) -> Result<(), ConformerError> {
    let (rows, cols) = coords.dim();
    if cols != 3 {
        return Err(ConformerError::InvalidCoordShape { rows, cols });
    }
    Ok(())
}

#[inline(always)]
fn validate_workspace_n_atoms(
    workspace_n_atoms: usize,
    coords_n_atoms: usize,
) -> Result<(), ConformerError> {
    if workspace_n_atoms != coords_n_atoms {
        return Err(ConformerError::WorkspaceAtomCountMismatch {
            workspace_n_atoms,
            coords_n_atoms,
        });
    }
    Ok(())
}

#[inline(always)]
fn validate_pair_shape(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
) -> Result<(), ConformerError> {
    validate_coord_shape(left)?;
    validate_coord_shape(right)?;
    if left.nrows() != right.nrows() {
        return Err(ConformerError::AtomCountMismatch {
            expected: left.nrows(),
            found: right.nrows(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SoA loaders & helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn n_pairs(n_atoms: usize) -> usize {
    n_atoms.saturating_sub(1) * n_atoms / 2
}

#[inline(always)]
fn split3(soa: &[f64], n: usize) -> (&[f64], &[f64], &[f64]) {
    (&soa[0..n], &soa[n..2 * n], &soa[2 * n..3 * n])
}

/// Loads `coords` into the three SoA streams `[x | y | z]` packed in `soa`.
/// Fast-paths column-major `(n, 3)` input as a contiguous `memcpy`.
#[inline]
fn load_to_soa(soa: &mut [f64], n: usize, coords: ArrayView2<'_, f64>) {
    if let Some(buf) = fortran_soa_slice(coords) {
        soa[..3 * n].copy_from_slice(buf);
        return;
    }
    let (x, rest) = soa.split_at_mut(n);
    let (y, z) = rest.split_at_mut(n);
    for (i, row) in coords.rows().into_iter().enumerate() {
        x[i] = row[0];
        y[i] = row[1];
        z[i] = row[2];
    }
}

/// Returns the flat SoA memory buffer when `view` is a column-major
/// (Fortran-ordered) `(n, 3)` array.
#[inline]
fn fortran_soa_slice<'a>(view: ArrayView2<'a, f64>) -> Option<&'a [f64]> {
    let n = view.nrows() as isize;
    if view.ncols() != 3 || view.strides() != [1, n] {
        return None;
    }
    view.to_slice_memory_order()
}

// ---------------------------------------------------------------------------
// Top-level dispatch — routes to SIMD kernels when the `portable-simd`
// feature is enabled, otherwise to the scalar (autovectorized) kernels.
// ---------------------------------------------------------------------------

#[inline]
fn compute_pairwise_distances_dispatch(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    out: &mut [f64],
) {
    #[cfg(feature = "portable-simd")]
    {
        crate::drmsd_simd::compute_pairwise_distances_simd(x, y, z, out);
    }
    #[cfg(not(feature = "portable-simd"))]
    {
        compute_pairwise_distances_scalar(x, y, z, out);
    }
}

#[inline]
fn sum_sq_distance_diffs_dispatch(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ref_dists: &[f64],
) -> f64 {
    #[cfg(feature = "portable-simd")]
    {
        crate::drmsd_simd::sum_sq_distance_diffs_simd(x, y, z, ref_dists)
    }
    #[cfg(not(feature = "portable-simd"))]
    {
        sum_sq_distance_diffs_scalar(x, y, z, ref_dists)
    }
}

// ---------------------------------------------------------------------------
// Scalar kernels (also exposed for the SIMD module's tail handling)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn compute_pairwise_distances_scalar(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    out: &mut [f64],
) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let mut offset = 0usize;
    for i in 0..(n - 1) {
        let row_len = n - i - 1;
        let xi = x[i];
        let yi = y[i];
        let zi = z[i];
        let xs = &x[i + 1..n];
        let ys = &y[i + 1..n];
        let zs = &z[i + 1..n];
        let dst = &mut out[offset..offset + row_len];
        pairwise_distances_row_scalar(xi, yi, zi, xs, ys, zs, dst);
        offset += row_len;
    }
    debug_assert_eq!(offset, out.len());
}

#[inline]
pub(crate) fn sum_sq_distance_diffs_scalar(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ref_dists: &[f64],
) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut ssd = 0.0_f64;
    let mut offset = 0usize;
    for i in 0..(n - 1) {
        let row_len = n - i - 1;
        let xi = x[i];
        let yi = y[i];
        let zi = z[i];
        let xs = &x[i + 1..n];
        let ys = &y[i + 1..n];
        let zs = &z[i + 1..n];
        let rd = &ref_dists[offset..offset + row_len];
        ssd += sq_dist_diff_row_scalar(xi, yi, zi, xs, ys, zs, rd);
        offset += row_len;
    }
    debug_assert_eq!(offset, ref_dists.len());
    ssd
}

#[inline]
pub(crate) fn pairwise_distances_row_scalar(
    xi: f64,
    yi: f64,
    zi: f64,
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert_eq!(xs.len(), zs.len());
    debug_assert_eq!(xs.len(), out.len());
    for k in 0..xs.len() {
        let dx = xs[k] - xi;
        let dy = ys[k] - yi;
        let dz = zs[k] - zi;
        out[k] = (dx * dx + dy * dy + dz * dz).sqrt();
    }
}

#[inline]
pub(crate) fn sq_dist_diff_row_scalar(
    xi: f64,
    yi: f64,
    zi: f64,
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    ref_dists: &[f64],
) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert_eq!(xs.len(), zs.len());
    debug_assert_eq!(xs.len(), ref_dists.len());
    let mut ssd = 0.0_f64;
    for k in 0..xs.len() {
        let dx = xs[k] - xi;
        let dy = ys[k] - yi;
        let dz = zs[k] - zi;
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        let diff = d - ref_dists[k];
        ssd += diff * diff;
    }
    ssd
}
