//! QCP-based rotational RMSD kernels.
//!
//! Generic over the floating-point precision of the **inner kernels** via
//! the [`QcpFloat`] trait. The accompanying SoA cross-covariance and
//! squared-norm sums are accumulated in the scalar type `T` (`f32` or
//! `f64`); the small fixed-cost polynomial root solve afterwards always
//! runs in `f64` for numerical stability.
//!
//! Two concrete workspaces are exposed:
//!
//! * [`QcpRmsdWorkspace`] — `f64` everywhere (back-compat default).
//! * [`QcpRmsdWorkspaceF32`] — `f32` storage + SoA accumulation, then
//!   `f64` polynomial root finding. Use for high-throughput screening
//!   where ~1e-7 relative precision is sufficient.
//!
//! Input structures are assumed to be pre-centered to the same origin;
//! this kernel does not subtract centroids.
//!
//! # Memory layout
//!
//! Each workspace holds **one** `Vec<T>` of `3 * n_atoms` elements that
//! backs the three reference SoA streams `[ref_x | ref_y | ref_z]` in a
//! single contiguous heap region. The hot path (`prepare_reference`,
//! `rmsd_prepared`, `rmsd`) performs **zero** heap allocations.
//!
//! The right-hand structure is consumed in a single fused pass: when it
//! is column-major (Fortran-ordered) the three SoA streams are read
//! directly from its memory; otherwise the AoS rows are streamed into FP
//! registers alongside the prepared reference SoA in a single pass.
//!
//! References:
//! 1. D. L. Theobald, Acta Crystallographica A 61 (2005), 478–480.
//!    doi:10.1107/S0108767305015266
//! 2. P. Liu, D. K. Agrafiotis, D. L. Theobald, J. Comput. Chem. 31
//!    (2010), 1561–1563. doi:10.1002/jcc.21439

use ndarray::ArrayView2;

use crate::error::ConformerError;

// ---------------------------------------------------------------------------
// QcpFloat trait — scalar abstraction for f32 / f64 inner kernels.
// ---------------------------------------------------------------------------

/// Scalar precision used for the SoA reference storage and the
/// cross-covariance accumulation inside [`QcpRmsdWorkspaceT`].
///
/// The eigenvalue polynomial solver always runs in `f64` regardless of
/// `T`; the trait provides cheap conversions for the boundary.
pub trait QcpFloat:
    Copy + Default + Send + Sync + std::fmt::Debug + 'static
{
    /// Multiplicative identity zero.
    const ZERO: Self;

    /// `self * a + b`, using FMA when the target supports it.
    fn fma(self, a: Self, b: Self) -> Self;

    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;

    /// SoA squared norm `sum(x_i^2 + y_i^2 + z_i^2)`, accumulated in `T`.
    fn squared_norm_soa(x: &[Self], y: &[Self], z: &[Self]) -> Self;

    /// 9-element SoA cross-covariance, accumulated in `T`.
    fn cross_cov_soa(
        lx: &[Self],
        ly: &[Self],
        lz: &[Self],
        rx: &[Self],
        ry: &[Self],
        rz: &[Self],
    ) -> [Self; 9];
}

impl QcpFloat for f64 {
    const ZERO: Self = 0.0;

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        // Use mul_add for true FMA on AVX2+FMA / NEON.
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x
    }

    #[inline]
    fn squared_norm_soa(x: &[Self], y: &[Self], z: &[Self]) -> Self {
        #[cfg(feature = "portable-simd")]
        {
            crate::rmsd_simd::squared_norm_soa_f64(x, y, z)
        }
        #[cfg(not(feature = "portable-simd"))]
        {
            squared_norm_soa_scalar(x, y, z)
        }
    }

    #[inline]
    fn cross_cov_soa(
        lx: &[Self],
        ly: &[Self],
        lz: &[Self],
        rx: &[Self],
        ry: &[Self],
        rz: &[Self],
    ) -> [Self; 9] {
        #[cfg(feature = "portable-simd")]
        {
            crate::rmsd_simd::cross_cov_soa_f64(lx, ly, lz, rx, ry, rz)
        }
        #[cfg(not(feature = "portable-simd"))]
        {
            cross_cov_soa_scalar(lx, ly, lz, rx, ry, rz)
        }
    }
}

impl QcpFloat for f32 {
    const ZERO: Self = 0.0;

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x as f32
    }

    #[inline]
    fn squared_norm_soa(x: &[Self], y: &[Self], z: &[Self]) -> Self {
        #[cfg(feature = "portable-simd")]
        {
            crate::rmsd_simd::squared_norm_soa_f32(x, y, z)
        }
        #[cfg(not(feature = "portable-simd"))]
        {
            squared_norm_soa_scalar(x, y, z)
        }
    }

    #[inline]
    fn cross_cov_soa(
        lx: &[Self],
        ly: &[Self],
        lz: &[Self],
        rx: &[Self],
        ry: &[Self],
        rz: &[Self],
    ) -> [Self; 9] {
        #[cfg(feature = "portable-simd")]
        {
            crate::rmsd_simd::cross_cov_soa_f32(lx, ly, lz, rx, ry, rz)
        }
        #[cfg(not(feature = "portable-simd"))]
        {
            cross_cov_soa_scalar(lx, ly, lz, rx, ry, rz)
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar reference kernels (used by both T=f64 and T=f32 in the absence
// of the `portable-simd` feature, and as tail handlers for the SIMD
// kernels). Generic so they autovectorize once per concrete type.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn squared_norm_soa_scalar<T: QcpFloat>(
    x: &[T],
    y: &[T],
    z: &[T],
) -> T {
    let mut g = T::ZERO;
    let xs = x.iter().copied();
    let ys = y.iter().copied();
    let zs = z.iter().copied();
    for ((xi, yi), zi) in xs.zip(ys).zip(zs) {
        g = xi.fma(xi, g);
        g = yi.fma(yi, g);
        g = zi.fma(zi, g);
    }
    g
}

#[inline]
pub(crate) fn cross_cov_soa_scalar<T: QcpFloat>(
    lx: &[T],
    ly: &[T],
    lz: &[T],
    rx: &[T],
    ry: &[T],
    rz: &[T],
) -> [T; 9] {
    debug_assert_eq!(lx.len(), ly.len());
    debug_assert_eq!(lx.len(), lz.len());
    debug_assert_eq!(lx.len(), rx.len());
    debug_assert_eq!(lx.len(), ry.len());
    debug_assert_eq!(lx.len(), rz.len());

    let mut sxx = T::ZERO;
    let mut sxy = T::ZERO;
    let mut sxz = T::ZERO;
    let mut syx = T::ZERO;
    let mut syy = T::ZERO;
    let mut syz = T::ZERO;
    let mut szx = T::ZERO;
    let mut szy = T::ZERO;
    let mut szz = T::ZERO;

    for (((((ax, ay), az), bx), by), bz) in lx
        .iter()
        .copied()
        .zip(ly.iter().copied())
        .zip(lz.iter().copied())
        .zip(rx.iter().copied())
        .zip(ry.iter().copied())
        .zip(rz.iter().copied())
    {
        sxx = ax.fma(bx, sxx);
        sxy = ax.fma(by, sxy);
        sxz = ax.fma(bz, sxz);
        syx = ay.fma(bx, syx);
        syy = ay.fma(by, syy);
        syz = ay.fma(bz, syz);
        szx = az.fma(bx, szx);
        szy = az.fma(by, szy);
        szz = az.fma(bz, szz);
    }
    [sxx, sxy, sxz, syx, syy, syz, szx, szy, szz]
}

// ---------------------------------------------------------------------------
// Generic workspace
// ---------------------------------------------------------------------------

/// Reusable workspace generic over scalar precision.
///
/// **One** heap allocation of `3 * n_atoms` `T`s holds the reference SoA
/// streams contiguously. After construction, the hot path performs zero
/// further allocations.
#[derive(Debug, Clone)]
pub struct QcpRmsdWorkspaceT<T: QcpFloat> {
    n_atoms: usize,
    storage: Vec<T>,
    reference_g: T,
    reference_prepared: bool,
}

impl<T: QcpFloat> QcpRmsdWorkspaceT<T> {
    /// Creates a workspace for repeated RMSD computations with `n_atoms`.
    #[inline]
    pub fn new(n_atoms: usize) -> Self {
        Self {
            n_atoms,
            storage: vec![T::ZERO; 3 * n_atoms],
            reference_g: T::ZERO,
            reference_prepared: false,
        }
    }

    /// Back-compat constructor. The `_use_simd` flag is now a no-op:
    /// SIMD vs. scalar is selected at compile time via the
    /// `portable-simd` Cargo feature.
    #[inline]
    pub fn with_simd(n_atoms: usize, _use_simd: bool) -> Self {
        Self::new(n_atoms)
    }

    /// Number of atoms this workspace is configured for.
    #[inline(always)]
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Prepares the left-hand coordinates once for repeated comparisons.
    ///
    /// Column-major (Fortran-ordered) `(n_atoms, 3)` input is the
    /// preferred layout: the flat memory is the internal SoA layout
    /// `[x | y | z]` and loading is three `memcpy`s plus the squared-norm
    /// accumulator.
    pub fn prepare_reference(
        &mut self,
        left: ArrayView2<'_, T>,
    ) -> Result<(), ConformerError> {
        validate_coord_shape(left)?;
        validate_workspace_n_atoms(self.n_atoms, left.nrows())?;
        let n = self.n_atoms;
        if let Some(buf) = fortran_soa_slice(left) {
            let (lx, rest) = buf.split_at(n);
            let (ly, lz) = rest.split_at(n);
            let (rx, ry, rz) = ref_streams_mut(&mut self.storage, n);
            rx.copy_from_slice(lx);
            ry.copy_from_slice(ly);
            rz.copy_from_slice(lz);
            self.reference_g = T::squared_norm_soa(lx, ly, lz);
        } else {
            let (rx, ry, rz) = ref_streams_mut(&mut self.storage, n);
            self.reference_g = load_aos_view_to_soa(rx, ry, rz, left);
        }
        self.reference_prepared = true;
        Ok(())
    }

    /// Computes RMSD against the previously prepared reference coordinates.
    /// Returns `f64` regardless of the kernel precision.
    #[inline]
    pub fn rmsd_prepared(
        &mut self,
        right: ArrayView2<'_, T>,
    ) -> Result<f64, ConformerError> {
        if !self.reference_prepared {
            return Err(ConformerError::WorkspaceReferenceNotPrepared);
        }
        validate_coord_shape(right)?;
        validate_workspace_n_atoms(self.n_atoms, right.nrows())?;
        Ok(qcp_rmsd_against_prepared::<T>(
            &self.storage,
            self.n_atoms,
            self.reference_g,
            right,
        ))
    }

    /// Computes optimal rotational RMSD in one shot.
    #[inline]
    pub fn rmsd(
        &mut self,
        left: ArrayView2<'_, T>,
        right: ArrayView2<'_, T>,
    ) -> Result<f64, ConformerError> {
        validate_pair_shape(left, right)?;
        validate_workspace_n_atoms(self.n_atoms, left.nrows())?;
        self.prepare_reference(left)?;
        Ok(qcp_rmsd_against_prepared::<T>(
            &self.storage,
            self.n_atoms,
            self.reference_g,
            right,
        ))
    }
}

/// Back-compat alias: `f64` workspace.
pub type QcpRmsdWorkspace = QcpRmsdWorkspaceT<f64>;

/// `f32` workspace. Use for high-throughput screening where ~1e-7
/// relative precision is sufficient. The polynomial solver still runs
/// in `f64`, so only the O(N) cross-covariance accumulation is reduced
/// precision.
pub type QcpRmsdWorkspaceF32 = QcpRmsdWorkspaceT<f32>;

/// Computes optimal rotational RMSD via QCP/Theobald in `f64`.
///
/// **Allocates** a fresh [`QcpRmsdWorkspace`] per call. For repeated
/// calls on geometries of a fixed size, prefer [`QcpRmsdWorkspace::rmsd`]
/// or the prepared-reference workflow.
#[inline]
pub fn qcp_rmsd(
    left: ArrayView2<'_, f64>,
    right: ArrayView2<'_, f64>,
) -> Result<f64, ConformerError> {
    let mut workspace = QcpRmsdWorkspace::new(left.nrows());
    workspace.rmsd(left, right)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[inline(always)]
fn validate_coord_shape<T>(coords: ArrayView2<'_, T>) -> Result<(), ConformerError> {
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
fn validate_pair_shape<T>(
    left: ArrayView2<'_, T>,
    right: ArrayView2<'_, T>,
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
// Storage slicing helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn ref_streams_mut<T>(storage: &mut [T], n: usize) -> (&mut [T], &mut [T], &mut [T]) {
    let (x, rest) = storage.split_at_mut(n);
    let (y, z) = rest.split_at_mut(n);
    (x, y, z)
}

#[inline(always)]
fn ref_streams<T>(storage: &[T], n: usize) -> (&[T], &[T], &[T]) {
    (
        &storage[0..n],
        &storage[n..2 * n],
        &storage[2 * n..3 * n],
    )
}

// ---------------------------------------------------------------------------
// AoS → SoA loaders
// ---------------------------------------------------------------------------

#[inline]
fn load_aos_view_to_soa<T: QcpFloat>(
    x: &mut [T],
    y: &mut [T],
    z: &mut [T],
    coords: ArrayView2<'_, T>,
) -> T {
    let mut g = T::ZERO;
    for (i, row) in coords.rows().into_iter().enumerate() {
        let xi = row[0];
        let yi = row[1];
        let zi = row[2];
        x[i] = xi;
        y[i] = yi;
        z[i] = zi;
        g = xi.fma(xi, g);
        g = yi.fma(yi, g);
        g = zi.fma(zi, g);
    }
    g
}

/// Returns the flat SoA memory buffer when `view` is a column-major
/// (Fortran-ordered) `(n, 3)` array.
#[inline]
fn fortran_soa_slice<'a, T>(view: ArrayView2<'a, T>) -> Option<&'a [T]> {
    let n = view.nrows() as isize;
    if view.ncols() != 3 || view.strides() != [1, n] {
        return None;
    }
    view.to_slice_memory_order()
}

// ---------------------------------------------------------------------------
// Stats fusion & top-level dispatch
// ---------------------------------------------------------------------------

#[inline]
fn qcp_rmsd_against_prepared<T: QcpFloat>(
    storage: &[T],
    n_atoms: usize,
    reference_g: T,
    right: ArrayView2<'_, T>,
) -> f64 {
    if n_atoms == 0 {
        return 0.0;
    }
    let (g_l, g_r, sxx, sxy, sxz, syx, syy, syz, szx, szy, szz) =
        stats_against_prepared::<T>(storage, n_atoms, reference_g, right);

    qcp_rmsd_from_stats(
        n_atoms as f64, g_l, g_r, sxx, sxy, sxz, syx, syy, syz, szx, szy, szz,
    )
}

#[inline]
fn stats_against_prepared<T: QcpFloat>(
    storage: &[T],
    n_atoms: usize,
    reference_g: T,
    right: ArrayView2<'_, T>,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    // Fast path: F-order input → zero-copy SoA streams.
    if let Some(buf) = fortran_soa_slice(right) {
        let (rx, rest) = buf.split_at(n_atoms);
        let (ry, rz) = rest.split_at(n_atoms);
        let (lx, ly, lz) = ref_streams(storage, n_atoms);
        let g_right = T::squared_norm_soa(rx, ry, rz);
        let cc = T::cross_cov_soa(lx, ly, lz, rx, ry, rz);
        return (
            reference_g.to_f64(),
            g_right.to_f64(),
            cc[0].to_f64(),
            cc[1].to_f64(),
            cc[2].to_f64(),
            cc[3].to_f64(),
            cc[4].to_f64(),
            cc[5].to_f64(),
            cc[6].to_f64(),
            cc[7].to_f64(),
            cc[8].to_f64(),
        );
    }
    // Fallback: any non-F-order layout — single fused pass.
    stats_scalar_view::<T>(storage, n_atoms, reference_g, right)
}

#[inline]
fn stats_scalar_view<T: QcpFloat>(
    storage: &[T],
    n_atoms: usize,
    reference_g: T,
    right: ArrayView2<'_, T>,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let (ax, ay, az) = ref_streams(storage, n_atoms);
    let mut g_raw = T::ZERO;
    let mut sxx = T::ZERO;
    let mut sxy = T::ZERO;
    let mut sxz = T::ZERO;
    let mut syx = T::ZERO;
    let mut syy = T::ZERO;
    let mut syz = T::ZERO;
    let mut szx = T::ZERO;
    let mut szy = T::ZERO;
    let mut szz = T::ZERO;
    for (i, row) in right.rows().into_iter().enumerate() {
        let bx = row[0];
        let by = row[1];
        let bz = row[2];
        g_raw = bx.fma(bx, g_raw);
        g_raw = by.fma(by, g_raw);
        g_raw = bz.fma(bz, g_raw);
        let axi = ax[i];
        let ayi = ay[i];
        let azi = az[i];
        sxx = axi.fma(bx, sxx);
        sxy = axi.fma(by, sxy);
        sxz = axi.fma(bz, sxz);
        syx = ayi.fma(bx, syx);
        syy = ayi.fma(by, syy);
        syz = ayi.fma(bz, syz);
        szx = azi.fma(bx, szx);
        szy = azi.fma(by, szy);
        szz = azi.fma(bz, szz);
    }
    (
        reference_g.to_f64(),
        g_raw.to_f64(),
        sxx.to_f64(),
        sxy.to_f64(),
        sxz.to_f64(),
        syx.to_f64(),
        syy.to_f64(),
        syz.to_f64(),
        szx.to_f64(),
        szy.to_f64(),
        szz.to_f64(),
    )
}

// ---------------------------------------------------------------------------
// Eigenvalue / RMSD from cross-covariance statistics — always f64.
// ---------------------------------------------------------------------------

/// Computes the QCP RMSD from the pre-computed cross-covariance statistics.
#[inline]
pub(crate) fn qcp_rmsd_from_stats(
    divisor: f64,
    g_left: f64,
    g_right: f64,
    sxx: f64,
    sxy: f64,
    sxz: f64,
    syx: f64,
    syy: f64,
    syz: f64,
    szx: f64,
    szy: f64,
    szz: f64,
) -> f64 {
    let e0 = 0.5 * (g_left + g_right);
    if e0 <= f64::EPSILON {
        return 0.0;
    }

    let sxx2 = sxx * sxx;
    let sxy2 = sxy * sxy;
    let sxz2 = sxz * sxz;
    let syx2 = syx * syx;
    let syy2 = syy * syy;
    let syz2 = syz * syz;
    let szx2 = szx * szx;
    let szy2 = szy * szy;
    let szz2 = szz * szz;

    let c2 = -2.0
        * (sxx2 + syy2 + szz2 + sxy2 + syx2 + sxz2 + szx2 + syz2 + szy2);
    let c1 = 8.0
        * (sxx * syz * szy + syy * szx * sxz + szz * sxy * syx
            - sxx * syy * szz
            - syz * szx * sxy
            - szy * syx * sxz);

    let syzszymsyyszz2 = 2.0 * (syz * szy - syy * szz);
    let sxx2syy2szz2syz2szy2 = syy2 + szz2 - sxx2 + syz2 + szy2;
    let sxzpszx = sxz + szx;
    let syzpszy = syz + szy;
    let sxypsyx = sxy + syx;
    let syzmszy = syz - szy;
    let sxzmszx = sxz - szx;
    let sxymsyx = sxy - syx;
    let sxxpsyy = sxx + syy;
    let sxxmsyy = sxx - syy;
    let sxy2sxz2syx2szx2 = sxy2 + sxz2 - syx2 - szx2;

    let c0 = sxy2sxz2syx2szx2 * sxy2sxz2syx2szx2
        + (sxx2syy2szz2syz2szy2 + syzszymsyyszz2)
            * (sxx2syy2szz2syz2szy2 - syzszymsyyszz2)
        + (-(sxzpszx) * (syzmszy) + (sxymsyx) * (sxxmsyy - szz))
            * (-(sxzmszx) * (syzpszy) + (sxymsyx) * (sxxmsyy + szz))
        + (-(sxzpszx) * (syzpszy) - (sxypsyx) * (sxxpsyy - szz))
            * (-(sxzmszx) * (syzmszy) - (sxypsyx) * (sxxpsyy + szz))
        + ((sxypsyx) * (syzpszy) + (sxzpszx) * (sxxmsyy + szz))
            * (-(sxymsyx) * (syzmszy) + (sxzpszx) * (sxxpsyy + szz))
        + ((sxypsyx) * (syzmszy) + (sxzmszx) * (sxxmsyy - szz))
            * (-(sxymsyx) * (syzpszy) + (sxzmszx) * (sxxpsyy - szz));

    let lambda = largest_eigenvalue_newton(e0, c2, c1, c0);

    (((g_left + g_right) - 2.0 * lambda).abs() / divisor).sqrt()
}

/// Newton–Raphson solver for the largest eigenvalue of the QCP K-matrix.
#[inline]
fn largest_eigenvalue_newton(e0: f64, c2: f64, c1: f64, c0: f64) -> f64 {
    let mut lambda = e0;
    for _ in 0..50 {
        let old = lambda;
        let lambda2 = lambda * lambda;
        let b = (lambda2 + c2) * lambda;
        let a = b + c1;
        let denom = 2.0 * lambda2 * lambda + b + a;
        if denom.abs() <= f64::EPSILON {
            break;
        }
        lambda -= (a * lambda + c0) / denom;
        if (lambda - old).abs() < 1e-11 * lambda.abs() {
            break;
        }
    }
    if lambda.is_finite() { lambda } else { e0 }
}
