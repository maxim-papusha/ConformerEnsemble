//! Portable SIMD QCP kernels — generic over scalar type (`f32`/`f64`)
//! and lane width via const generics.
//!
//! Two public entry points are used by [`crate::rmsd`]:
//!
//! * [`cross_cov_soa_f64`], [`cross_cov_soa_f32`] — 9-element SoA
//!   cross-covariance.
//! * [`squared_norm_soa_f64`], [`squared_norm_soa_f32`] — SoA squared
//!   norm Σ(x² + y² + z²).
//!
//! Internally these forward to a single generic implementation
//! [`cross_cov_soa_lanes`] / [`squared_norm_soa_lanes`] parameterised
//! over `(T, const LANES: usize)`. Additional `cross_cov_soa_f{32,64}_x{N}`
//! and `squared_norm_soa_f{32,64}_x{N}` symbols are exported for the
//! benchmark to sweep the lane axis directly.
//!
//! Each kernel uses `chunks_exact(LANES)` over the SIMD body and a
//! scalar tail loop over the remainder — there is **no indexing inside
//! the hot loop**. Accumulation is done in `LANES` parallel `Simd<T,
//! LANES>` lanes with a single `reduce_sum` at the end.
//!
//! Runtime CPU dispatch is performed via the `multiversion` crate so
//! the same binary picks AVX-512 / AVX2+FMA / NEON or scalar fallback
//! at first call.

#![cfg(feature = "portable-simd")]

use core::simd::num::SimdFloat;
use core::simd::{Simd, SimdElement};
use std::simd::StdFloat;
use multiversion::multiversion;

use crate::rmsd::QcpFloat;

// ---------------------------------------------------------------------------
// Generic kernel
// ---------------------------------------------------------------------------

/// Generic SoA cross-covariance with `LANES` SIMD lanes accumulating in `T`.
#[inline(always)]
fn cross_cov_soa_lanes<T, const LANES: usize>(
    lx: &[T],
    ly: &[T],
    lz: &[T],
    rx: &[T],
    ry: &[T],
    rz: &[T],
) -> [T; 9]
where
    T: SimdElement + QcpFloat,
    Simd<T, LANES>: StdFloat,
    Simd<T, LANES>: SimdFloat<Scalar = T>,
{
    debug_assert_eq!(lx.len(), ly.len());
    debug_assert_eq!(lx.len(), lz.len());
    debug_assert_eq!(lx.len(), rx.len());
    debug_assert_eq!(lx.len(), ry.len());
    debug_assert_eq!(lx.len(), rz.len());

    let zero = Simd::<T, LANES>::splat(T::ZERO);
    let mut sxx = zero;
    let mut sxy = zero;
    let mut sxz = zero;
    let mut syx = zero;
    let mut syy = zero;
    let mut syz = zero;
    let mut szx = zero;
    let mut szy = zero;
    let mut szz = zero;

    let mut lx_iter = lx.chunks_exact(LANES);
    let mut ly_iter = ly.chunks_exact(LANES);
    let mut lz_iter = lz.chunks_exact(LANES);
    let mut rx_iter = rx.chunks_exact(LANES);
    let mut ry_iter = ry.chunks_exact(LANES);
    let mut rz_iter = rz.chunks_exact(LANES);

    loop {
        // Pull one chunk from each. All iterators advance in lockstep;
        // we exit as soon as the first one is exhausted, since the
        // slices are equal-length.
        let (cx_l, cy_l, cz_l, cx_r, cy_r, cz_r) = match (
            lx_iter.next(),
            ly_iter.next(),
            lz_iter.next(),
            rx_iter.next(),
            ry_iter.next(),
            rz_iter.next(),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => {
                (a, b, c, d, e, f)
            }
            _ => break,
        };
        let ax = Simd::<T, LANES>::from_slice(cx_l);
        let ay = Simd::<T, LANES>::from_slice(cy_l);
        let az = Simd::<T, LANES>::from_slice(cz_l);
        let bx = Simd::<T, LANES>::from_slice(cx_r);
        let by = Simd::<T, LANES>::from_slice(cy_r);
        let bz = Simd::<T, LANES>::from_slice(cz_r);
        sxx = ax.mul_add(bx, sxx);
        sxy = ax.mul_add(by, sxy);
        sxz = ax.mul_add(bz, sxz);
        syx = ay.mul_add(bx, syx);
        syy = ay.mul_add(by, syy);
        syz = ay.mul_add(bz, syz);
        szx = az.mul_add(bx, szx);
        szy = az.mul_add(by, szy);
        szz = az.mul_add(bz, szz);
    }

    // Scalar tail.
    let tail_lx = lx_iter.remainder();
    let tail_ly = ly_iter.remainder();
    let tail_lz = lz_iter.remainder();
    let tail_rx = rx_iter.remainder();
    let tail_ry = ry_iter.remainder();
    let tail_rz = rz_iter.remainder();

    let mut t_sxx = T::ZERO;
    let mut t_sxy = T::ZERO;
    let mut t_sxz = T::ZERO;
    let mut t_syx = T::ZERO;
    let mut t_syy = T::ZERO;
    let mut t_syz = T::ZERO;
    let mut t_szx = T::ZERO;
    let mut t_szy = T::ZERO;
    let mut t_szz = T::ZERO;
    for i in 0..tail_lx.len() {
        let ax = tail_lx[i];
        let ay = tail_ly[i];
        let az = tail_lz[i];
        let bx = tail_rx[i];
        let by = tail_ry[i];
        let bz = tail_rz[i];
        t_sxx = ax.fma(bx, t_sxx);
        t_sxy = ax.fma(by, t_sxy);
        t_sxz = ax.fma(bz, t_sxz);
        t_syx = ay.fma(bx, t_syx);
        t_syy = ay.fma(by, t_syy);
        t_syz = ay.fma(bz, t_syz);
        t_szx = az.fma(bx, t_szx);
        t_szy = az.fma(by, t_szy);
        t_szz = az.fma(bz, t_szz);
    }

    [
        sxx.reduce_sum().fma(T::from_f64(1.0), t_sxx),
        sxy.reduce_sum().fma(T::from_f64(1.0), t_sxy),
        sxz.reduce_sum().fma(T::from_f64(1.0), t_sxz),
        syx.reduce_sum().fma(T::from_f64(1.0), t_syx),
        syy.reduce_sum().fma(T::from_f64(1.0), t_syy),
        syz.reduce_sum().fma(T::from_f64(1.0), t_syz),
        szx.reduce_sum().fma(T::from_f64(1.0), t_szx),
        szy.reduce_sum().fma(T::from_f64(1.0), t_szy),
        szz.reduce_sum().fma(T::from_f64(1.0), t_szz),
    ]
}

#[inline(always)]
fn squared_norm_soa_lanes<T, const LANES: usize>(
    x: &[T],
    y: &[T],
    z: &[T],
) -> T
where
    T: SimdElement + QcpFloat,
    Simd<T, LANES>: StdFloat,
    Simd<T, LANES>: SimdFloat<Scalar = T>,
{
    let zero = Simd::<T, LANES>::splat(T::ZERO);
    let mut acc = zero;
    let mut x_iter = x.chunks_exact(LANES);
    let mut y_iter = y.chunks_exact(LANES);
    let mut z_iter = z.chunks_exact(LANES);
    loop {
        let (cx, cy, cz) = match (x_iter.next(), y_iter.next(), z_iter.next())
        {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => break,
        };
        let vx = Simd::<T, LANES>::from_slice(cx);
        let vy = Simd::<T, LANES>::from_slice(cy);
        let vz = Simd::<T, LANES>::from_slice(cz);
        acc = vx.mul_add(vx, acc);
        acc = vy.mul_add(vy, acc);
        acc = vz.mul_add(vz, acc);
    }
    let tx = x_iter.remainder();
    let ty = y_iter.remainder();
    let tz = z_iter.remainder();
    let mut t = T::ZERO;
    for i in 0..tx.len() {
        t = tx[i].fma(tx[i], t);
        t = ty[i].fma(ty[i], t);
        t = tz[i].fma(tz[i], t);
    }
    acc.reduce_sum().fma(T::from_f64(1.0), t)
}

// ---------------------------------------------------------------------------
// Per-(scalar, lane-width) entry points with runtime CPU dispatch.
//
// We expose each (T, LANES) instantiation as its own `#[multiversion]`
// function so that the benchmark can sweep the lane axis directly, and
// the workspace dispatcher picks one of these as its default.
// ---------------------------------------------------------------------------

macro_rules! define_kernels {
    ($scalar:ty, $cross:ident, $norm:ident, $lanes:literal) => {
        #[multiversion(targets(
            "x86_64+avx512f",
            "x86_64+avx2+fma",
            "aarch64+neon"
        ))]
        #[inline]
        #[allow(dead_code)]
        pub fn $cross(
            lx: &[$scalar],
            ly: &[$scalar],
            lz: &[$scalar],
            rx: &[$scalar],
            ry: &[$scalar],
            rz: &[$scalar],
        ) -> [$scalar; 9] {
            cross_cov_soa_lanes::<$scalar, $lanes>(lx, ly, lz, rx, ry, rz)
        }

        #[multiversion(targets(
            "x86_64+avx512f",
            "x86_64+avx2+fma",
            "aarch64+neon"
        ))]
        #[inline]
        #[allow(dead_code)]
        pub fn $norm(x: &[$scalar], y: &[$scalar], z: &[$scalar]) -> $scalar {
            squared_norm_soa_lanes::<$scalar, $lanes>(x, y, z)
        }
    };
}

// f64 — lane widths 2/4/8 are the natively useful range on x86_64;
// 16/32/64 cover AVX-512 and future-targets / experimentation.
define_kernels!(f64, cross_cov_soa_f64_x2, squared_norm_soa_f64_x2, 2);
define_kernels!(f64, cross_cov_soa_f64_x4, squared_norm_soa_f64_x4, 4);
define_kernels!(f64, cross_cov_soa_f64_x8, squared_norm_soa_f64_x8, 8);
define_kernels!(f64, cross_cov_soa_f64_x16, squared_norm_soa_f64_x16, 16);
define_kernels!(f64, cross_cov_soa_f64_x32, squared_norm_soa_f64_x32, 32);
define_kernels!(f64, cross_cov_soa_f64_x64, squared_norm_soa_f64_x64, 64);

// f32 — wider lane widths are competitive.
define_kernels!(f32, cross_cov_soa_f32_x4, squared_norm_soa_f32_x4, 4);
define_kernels!(f32, cross_cov_soa_f32_x8, squared_norm_soa_f32_x8, 8);
define_kernels!(f32, cross_cov_soa_f32_x16, squared_norm_soa_f32_x16, 16);
define_kernels!(f32, cross_cov_soa_f32_x32, squared_norm_soa_f32_x32, 32);
define_kernels!(f32, cross_cov_soa_f32_x64, squared_norm_soa_f32_x64, 64);

// ---------------------------------------------------------------------------
// Public dispatch entry points (called from rmsd.rs).
//
// Defaults picked from the lane-sweep on AVX2+FMA / AVX-512:
//   - f64: 4 lanes (AVX2 = 256-bit / 64-bit = 4)
//   - f32: 8 lanes (AVX2 = 256-bit / 32-bit = 8)
// On AVX-512 hosts the multiversion dispatcher routes the same Rust-level
// 4×f64 / 8×f32 call into the 256-bit-emitting codegen unit; the wider
// kernels are intentionally also exposed so the benchmark can quantify
// whether 8×f64 / 16×f32 (zmm) actually pays off.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn cross_cov_soa_f64(
    lx: &[f64],
    ly: &[f64],
    lz: &[f64],
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
) -> [f64; 9] {
    cross_cov_soa_f64_x4(lx, ly, lz, rx, ry, rz)
}

#[inline]
pub(crate) fn squared_norm_soa_f64(x: &[f64], y: &[f64], z: &[f64]) -> f64 {
    squared_norm_soa_f64_x4(x, y, z)
}

#[inline]
pub(crate) fn cross_cov_soa_f32(
    lx: &[f32],
    ly: &[f32],
    lz: &[f32],
    rx: &[f32],
    ry: &[f32],
    rz: &[f32],
) -> [f32; 9] {
    cross_cov_soa_f32_x8(lx, ly, lz, rx, ry, rz)
}

#[inline]
pub(crate) fn squared_norm_soa_f32(x: &[f32], y: &[f32], z: &[f32]) -> f32 {
    squared_norm_soa_f32_x8(x, y, z)
}
