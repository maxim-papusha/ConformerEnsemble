//! QCP RMSD benchmark — sweeps lane widths, scalar types (f32/f64),
//! and `(n_atoms, comparisons)` workload sizes.
//!
//! ## Axes
//!
//! * `n_atoms ∈ {16, 64, 256, 1024}`
//! * `comparisons ∈ {1, 16, 256, 1024}` (workspace reuse count)
//! * lane width ∈ `f64: {2, 4, 8, 16, 32, 64}`,
//!                 `f32: {4, 8, 16, 32, 64}`
//!
//! All inputs are pre-built in **Fortran (column-major) order** so each
//! kernel call sees the zero-copy SoA fast path. The benchmark calls
//! the per-lane-width SIMD kernels directly (via the public
//! `simd_kernels` module), bypassing the workspace dispatcher so the
//! lane axis can be swept cleanly.
//!
//! ## Build
//!
//! ```text
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo +nightly bench --bench rmsd_bench --features portable-simd
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

const ATOM_COUNTS: &[usize] = &[16, 64, 256, 1024];
const COMPARISON_COUNTS: &[usize] = &[1, 16, 256, 1024];
const MIN_CASE_TIME: Duration = Duration::from_millis(150);

// ---------------------------------------------------------------------------
// PRNG-built F-order SoA buffers (`[x | y | z]` flat).
// ---------------------------------------------------------------------------

fn xorshift_seed(a: usize, b: usize) -> u64 {
    let mut s = 0x9E37_79B9_7F4A_7C15u64
        ^ ((a as u64) << 17)
        ^ ((b as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    if s == 0 {
        s = 1;
    }
    s
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn soa_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let u = next_u64(&mut state);
            ((u >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 10.0 - 5.0
        })
        .collect()
}

fn soa_f32(n: usize, seed: u64) -> Vec<f32> {
    soa_f64(n, seed).into_iter().map(|x| x as f32).collect()
}

// ---------------------------------------------------------------------------
// Scalar baseline (no SIMD) — reference for speedup figures.
// ---------------------------------------------------------------------------

fn cross_cov_scalar<T>(
    lx: &[T],
    ly: &[T],
    lz: &[T],
    rx: &[T],
    ry: &[T],
    rz: &[T],
) -> [T; 9]
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    let mut s = [T::default(); 9];
    for i in 0..lx.len() {
        let ax = lx[i];
        let ay = ly[i];
        let az = lz[i];
        let bx = rx[i];
        let by = ry[i];
        let bz = rz[i];
        s[0] = s[0] + ax * bx;
        s[1] = s[1] + ax * by;
        s[2] = s[2] + ax * bz;
        s[3] = s[3] + ay * bx;
        s[4] = s[4] + ay * by;
        s[5] = s[5] + ay * bz;
        s[6] = s[6] + az * bx;
        s[7] = s[7] + az * by;
        s[8] = s[8] + az * bz;
    }
    s
}

// ---------------------------------------------------------------------------
// Timing helper — runs `body` until `MIN_CASE_TIME` elapsed.
// ---------------------------------------------------------------------------

fn time_repeated<F: FnMut()>(mut body: F) -> Duration {
    body();
    let mut total = Duration::ZERO;
    let mut iters: usize = 0;
    while total < MIN_CASE_TIME {
        let t = Instant::now();
        for _ in 0..4 {
            body();
        }
        total += t.elapsed();
        iters += 4;
    }
    total / iters as u32
}

// ---------------------------------------------------------------------------
// One sweep entry.
// ---------------------------------------------------------------------------

#[cfg(feature = "portable-simd")]
mod simd_runner {
    use super::*;
    use conformerensemblers::simd_kernels::*;

    pub fn sweep_f64(
        n_atoms: usize,
        comparisons: usize,
    ) -> Vec<(&'static str, Duration)> {
        let lx = soa_f64(n_atoms, xorshift_seed(n_atoms, 0xA));
        let ly = soa_f64(n_atoms, xorshift_seed(n_atoms, 0xB));
        let lz = soa_f64(n_atoms, xorshift_seed(n_atoms, 0xC));
        let rights: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..comparisons)
            .map(|i| {
                (
                    soa_f64(n_atoms, xorshift_seed(n_atoms, 100 + i)),
                    soa_f64(n_atoms, xorshift_seed(n_atoms, 200 + i)),
                    soa_f64(n_atoms, xorshift_seed(n_atoms, 300 + i)),
                )
            })
            .collect();

        let mut out: Vec<(&'static str, Duration)> = Vec::new();
        out.push((
            "scalar",
            time_repeated(|| {
                for (rx, ry, rz) in &rights {
                    black_box(cross_cov_scalar::<f64>(&lx, &ly, &lz, rx, ry, rz));
                }
            }) / comparisons as u32,
        ));

        macro_rules! one {
            ($label:literal, $fn:ident) => {{
                let d = time_repeated(|| {
                    for (rx, ry, rz) in &rights {
                        black_box($fn(&lx, &ly, &lz, rx, ry, rz));
                    }
                }) / comparisons as u32;
                out.push(($label, d));
            }};
        }
        one!("simd-2", cross_cov_soa_f64_x2);
        one!("simd-4", cross_cov_soa_f64_x4);
        one!("simd-8", cross_cov_soa_f64_x8);
        one!("simd-16", cross_cov_soa_f64_x16);
        one!("simd-32", cross_cov_soa_f64_x32);
        one!("simd-64", cross_cov_soa_f64_x64);
        out
    }

    pub fn sweep_f32(
        n_atoms: usize,
        comparisons: usize,
    ) -> Vec<(&'static str, Duration)> {
        let lx = soa_f32(n_atoms, xorshift_seed(n_atoms, 0xA));
        let ly = soa_f32(n_atoms, xorshift_seed(n_atoms, 0xB));
        let lz = soa_f32(n_atoms, xorshift_seed(n_atoms, 0xC));
        let rights: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..comparisons)
            .map(|i| {
                (
                    soa_f32(n_atoms, xorshift_seed(n_atoms, 100 + i)),
                    soa_f32(n_atoms, xorshift_seed(n_atoms, 200 + i)),
                    soa_f32(n_atoms, xorshift_seed(n_atoms, 300 + i)),
                )
            })
            .collect();

        let mut out: Vec<(&'static str, Duration)> = Vec::new();
        out.push((
            "scalar",
            time_repeated(|| {
                for (rx, ry, rz) in &rights {
                    black_box(cross_cov_scalar::<f32>(&lx, &ly, &lz, rx, ry, rz));
                }
            }) / comparisons as u32,
        ));

        macro_rules! one {
            ($label:literal, $fn:ident) => {{
                let d = time_repeated(|| {
                    for (rx, ry, rz) in &rights {
                        black_box($fn(&lx, &ly, &lz, rx, ry, rz));
                    }
                }) / comparisons as u32;
                out.push(($label, d));
            }};
        }
        one!("simd-4", cross_cov_soa_f32_x4);
        one!("simd-8", cross_cov_soa_f32_x8);
        one!("simd-16", cross_cov_soa_f32_x16);
        one!("simd-32", cross_cov_soa_f32_x32);
        one!("simd-64", cross_cov_soa_f32_x64);
        out
    }
}

#[cfg(not(feature = "portable-simd"))]
mod simd_runner {
    use super::*;
    pub fn sweep_f64(
        n_atoms: usize,
        comparisons: usize,
    ) -> Vec<(&'static str, Duration)> {
        let lx = soa_f64(n_atoms, xorshift_seed(n_atoms, 0xA));
        let ly = soa_f64(n_atoms, xorshift_seed(n_atoms, 0xB));
        let lz = soa_f64(n_atoms, xorshift_seed(n_atoms, 0xC));
        let rights: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..comparisons)
            .map(|i| {
                (
                    soa_f64(n_atoms, xorshift_seed(n_atoms, 100 + i)),
                    soa_f64(n_atoms, xorshift_seed(n_atoms, 200 + i)),
                    soa_f64(n_atoms, xorshift_seed(n_atoms, 300 + i)),
                )
            })
            .collect();
        let d = time_repeated(|| {
            for (rx, ry, rz) in &rights {
                black_box(cross_cov_scalar::<f64>(&lx, &ly, &lz, rx, ry, rz));
            }
        }) / comparisons as u32;
        vec![("scalar", d)]
    }
    pub fn sweep_f32(
        n_atoms: usize,
        comparisons: usize,
    ) -> Vec<(&'static str, Duration)> {
        let lx = soa_f32(n_atoms, xorshift_seed(n_atoms, 0xA));
        let ly = soa_f32(n_atoms, xorshift_seed(n_atoms, 0xB));
        let lz = soa_f32(n_atoms, xorshift_seed(n_atoms, 0xC));
        let rights: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..comparisons)
            .map(|i| {
                (
                    soa_f32(n_atoms, xorshift_seed(n_atoms, 100 + i)),
                    soa_f32(n_atoms, xorshift_seed(n_atoms, 200 + i)),
                    soa_f32(n_atoms, xorshift_seed(n_atoms, 300 + i)),
                )
            })
            .collect();
        let d = time_repeated(|| {
            for (rx, ry, rz) in &rights {
                black_box(cross_cov_scalar::<f32>(&lx, &ly, &lz, rx, ry, rz));
            }
        }) / comparisons as u32;
        vec![("scalar", d)]
    }
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn fmt_dur(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    if ns < 1_000.0 {
        format!("{:>7.1} ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:>7.2} \u{00b5}s", ns / 1_000.0)
    } else {
        format!("{:>7.3} ms", ns / 1_000_000.0)
    }
}

fn print_table(
    title: &str,
    results: &[(usize, usize, Vec<(&'static str, Duration)>)],
) {
    println!();
    println!("### {title}");
    let labels: Vec<&str> = results
        .first()
        .map(|(_, _, row)| row.iter().map(|(l, _)| *l).collect())
        .unwrap_or_default();
    let mut header = String::from("| n_atoms | comparisons |");
    for l in &labels {
        header.push_str(&format!(" {:^11} |", l));
    }
    println!("{}", header);
    let mut sep = String::from("|---------|-------------|");
    for _ in &labels {
        sep.push_str("-------------|");
    }
    println!("{}", sep);
    for (n, c, row) in results {
        let mut line = format!("| {:>7} | {:>11} |", n, c);
        for (_, d) in row {
            line.push_str(&format!(" {:>11} |", fmt_dur(*d)));
        }
        println!("{}", line);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("# QCP RMSD lane-sweep benchmark");
    println!();
    #[cfg(feature = "portable-simd")]
    println!("Build: portable-simd ENABLED (nightly).");
    #[cfg(not(feature = "portable-simd"))]
    println!(
        "Build: scalar only. Rebuild with `--features portable-simd` for SIMD numbers."
    );

    let mut f64_results: Vec<(usize, usize, Vec<(&'static str, Duration)>)> =
        Vec::new();
    let mut f32_results: Vec<(usize, usize, Vec<(&'static str, Duration)>)> =
        Vec::new();

    for &n_atoms in ATOM_COUNTS {
        for &comparisons in COMPARISON_COUNTS {
            eprintln!(
                "  f64  n_atoms={n_atoms:>4}  comparisons={comparisons:>4}"
            );
            f64_results.push((
                n_atoms,
                comparisons,
                simd_runner::sweep_f64(n_atoms, comparisons),
            ));
            eprintln!(
                "  f32  n_atoms={n_atoms:>4}  comparisons={comparisons:>4}"
            );
            f32_results.push((
                n_atoms,
                comparisons,
                simd_runner::sweep_f32(n_atoms, comparisons),
            ));
        }
    }
    print_table("f64 — cross_cov_soa per comparison", &f64_results);
    print_table("f32 — cross_cov_soa per comparison", &f32_results);
}
