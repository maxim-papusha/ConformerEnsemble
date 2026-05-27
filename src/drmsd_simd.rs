//! Explicit `std::simd` kernels for the dRMSD inner loops.
//!
//! Gated behind the `portable-simd` Cargo feature; requires a **nightly**
//! rustc because `core::simd` is unstable (`feature(portable_simd)`).
//!
//! Two row-kernels, mirroring the scalar fallbacks in [`crate::drmsd`]:
//!
//! * `pairwise_distances_row_simd` — fix one atom `i`, sweep `j > i`,
//!   write `sqrt((xj-xi)^2 + (yj-yi)^2 + (zj-zi)^2)` into `out`.
//! * `sq_dist_diff_row_simd` — same sweep, fused diff against a cached
//!   reference distance row, returning the squared-diff sum.
//!
//! Wrapped with `multiversion` for runtime CPU dispatch.

#![cfg(feature = "portable-simd")]

use core::simd::num::SimdFloat;
use core::simd::{f64x4, Simd};
use std::simd::StdFloat;

use multiversion::multiversion;

const LANES: usize = 4;
type V = f64x4;

pub(crate) fn compute_pairwise_distances_simd(
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
        pairwise_distances_row_simd(xi, yi, zi, xs, ys, zs, dst);
        offset += row_len;
    }
    debug_assert_eq!(offset, out.len());
}

pub(crate) fn sum_sq_distance_diffs_simd(
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
        ssd += sq_dist_diff_row_simd(xi, yi, zi, xs, ys, zs, rd);
        offset += row_len;
    }
    debug_assert_eq!(offset, ref_dists.len());
    ssd
}

#[multiversion(targets(
    "x86_64+avx512f",
    "x86_64+avx2+fma",
    "aarch64+neon",
))]
#[inline]
fn pairwise_distances_row_simd(
    xi: f64,
    yi: f64,
    zi: f64,
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    out: &mut [f64],
) {
    let n = xs.len();
    debug_assert_eq!(ys.len(), n);
    debug_assert_eq!(zs.len(), n);
    debug_assert_eq!(out.len(), n);

    let vxi = V::splat(xi);
    let vyi = V::splat(yi);
    let vzi = V::splat(zi);

    let chunks = n / LANES;
    for c in 0..chunks {
        let off = c * LANES;
        let dx = V::from_slice(&xs[off..off + LANES]) - vxi;
        let dy = V::from_slice(&ys[off..off + LANES]) - vyi;
        let dz = V::from_slice(&zs[off..off + LANES]) - vzi;
        let sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
        let d = sq.sqrt();
        d.copy_to_slice(&mut out[off..off + LANES]);
    }
    let base = chunks * LANES;
    for k in base..n {
        let dx = xs[k] - xi;
        let dy = ys[k] - yi;
        let dz = zs[k] - zi;
        out[k] = (dx * dx + dy * dy + dz * dz).sqrt();
    }
}

#[multiversion(targets(
    "x86_64+avx512f",
    "x86_64+avx2+fma",
    "aarch64+neon",
))]
#[inline]
fn sq_dist_diff_row_simd(
    xi: f64,
    yi: f64,
    zi: f64,
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    ref_dists: &[f64],
) -> f64 {
    let n = xs.len();
    debug_assert_eq!(ys.len(), n);
    debug_assert_eq!(zs.len(), n);
    debug_assert_eq!(ref_dists.len(), n);

    let vxi = V::splat(xi);
    let vyi = V::splat(yi);
    let vzi = V::splat(zi);
    let mut acc = V::splat(0.0);

    let chunks = n / LANES;
    for c in 0..chunks {
        let off = c * LANES;
        let dx = V::from_slice(&xs[off..off + LANES]) - vxi;
        let dy = V::from_slice(&ys[off..off + LANES]) - vyi;
        let dz = V::from_slice(&zs[off..off + LANES]) - vzi;
        let sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
        let d = sq.sqrt();
        let rd = V::from_slice(&ref_dists[off..off + LANES]);
        let diff = d - rd;
        acc = diff.mul_add(diff, acc);
    }
    let mut ssd = acc.reduce_sum();
    let base = chunks * LANES;
    for k in base..n {
        let dx = xs[k] - xi;
        let dy = ys[k] - yi;
        let dz = zs[k] - zi;
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        let diff = d - ref_dists[k];
        ssd += diff * diff;
    }
    ssd
}

#[allow(dead_code)]
fn _check_simd_type() {
    let _: Simd<f64, LANES> = V::splat(0.0);
}
