//! dRMSD micro-benchmark.
//!
//! Same matrix shape as `rmsd_bench`: sweeps (n_atoms × comparisons) and
//! reports time per dRMSD comparison. The dRMSD kernel is O(N^2), so
//! larger atom counts dominate the table more steeply than the QCP O(N)
//! cross-covariance kernel.
//!
//! Scalar baseline:    cargo bench --bench drmsd_bench
//! SIMD (nightly):     cargo +nightly bench --bench drmsd_bench --features portable-simd

use std::hint::black_box;
use std::time::{Duration, Instant};

use conformerensemblers::DrmsdWorkspace;
use ndarray::{Array2, ShapeBuilder};

const ATOM_COUNTS: &[usize] = &[16, 64, 256, 1024, 2048];
const COMPARISON_COUNTS: &[usize] = &[1, 16, 256];
const MIN_CASE_TIME: Duration = Duration::from_millis(250);

fn build_dataset(n_atoms: usize, n_right: usize) -> (Array2<f64>, Vec<Array2<f64>>) {
    let mut state: u64 =
        0xD1B5_4A32_D192_ED03 ^ ((n_atoms as u64) << 17) ^ (n_right as u64);
    let mut rand = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    };
    let mut alloc = |n: usize| -> Array2<f64> {
        Array2::from_shape_fn((n, 3).f(), |_| rand() * 10.0 - 5.0)
    };
    let left = alloc(n_atoms);
    let rights: Vec<Array2<f64>> = (0..n_right).map(|_| alloc(n_atoms)).collect();
    (left, rights)
}

fn measure_prepared(n_atoms: usize, n_right: usize) -> (Duration, usize) {
    let (left, rights) = build_dataset(n_atoms, n_right);
    let mut workspace = DrmsdWorkspace::new(n_atoms);

    workspace.prepare_reference(left.view()).unwrap();
    for r in &rights {
        let _ = black_box(workspace.drmsd_prepared(r.view()).unwrap());
    }

    let mut total = Duration::ZERO;
    let mut iters = 0usize;
    while total < MIN_CASE_TIME {
        workspace.prepare_reference(left.view()).unwrap();
        let t0 = Instant::now();
        for r in &rights {
            let _ = black_box(workspace.drmsd_prepared(r.view()).unwrap());
        }
        total += t0.elapsed();
        iters += 1;
    }
    (total, iters * n_right)
}

fn measure_fresh(n_atoms: usize) -> (Duration, usize) {
    let (left, rights) = build_dataset(n_atoms, 1);
    let mut ws = DrmsdWorkspace::new(n_atoms);
    let _ = black_box(ws.drmsd(left.view(), rights[0].view()).unwrap());

    let mut total = Duration::ZERO;
    let mut iters = 0usize;
    while total < MIN_CASE_TIME {
        let t0 = Instant::now();
        let mut ws = DrmsdWorkspace::new(n_atoms);
        let _ = black_box(ws.drmsd(left.view(), rights[0].view()).unwrap());
        total += t0.elapsed();
        iters += 1;
    }
    (total, iters)
}

fn fmt_ns(d: Duration, samples: usize) -> String {
    let per = d.as_secs_f64() * 1e9 / samples as f64;
    if per >= 1e6 {
        format!("{:>10.3} ms", per / 1e6)
    } else if per >= 1e3 {
        format!("{:>10.3} us", per / 1e3)
    } else {
        format!("{:>10.1} ns", per)
    }
}

fn main() {
    let feature = if cfg!(feature = "portable-simd") {
        "portable-simd (std::simd)"
    } else {
        "scalar (autovectorized)"
    };
    println!("\ndRMSD benchmark — kernel: {}", feature);
    println!("Each cell: time per dRMSD comparison\n");

    print!("{:>10}", "n_atoms");
    for c in COMPARISON_COUNTS {
        print!(" | {:>14}", format!("prep+{} cmp", c));
    }
    print!(" | {:>14}", "fresh ws");
    println!();
    print!("{:->10}", "");
    for _ in COMPARISON_COUNTS {
        print!("-+-{:->14}", "");
    }
    print!("-+-{:->14}", "");
    println!();

    for &n in ATOM_COUNTS {
        print!("{:>10}", n);
        for &c in COMPARISON_COUNTS {
            let (d, s) = measure_prepared(n, c);
            print!(" | {}", fmt_ns(d, s));
        }
        let (d, s) = measure_fresh(n);
        print!(" | {}", fmt_ns(d, s));
        println!();
    }
    println!();
}
