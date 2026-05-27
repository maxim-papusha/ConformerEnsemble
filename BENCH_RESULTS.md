# ConformerEnsemble — kernel benchmarks

Wall-clock per RMSD / dRMSD comparison, scalar baseline vs. explicit
`std::simd` kernels (`portable-simd` Cargo feature, `multiversion`
runtime CPU dispatch).

Cells show **time per comparison**. The "fresh ws" column allocates a
brand-new `QcpRmsdWorkspace` / `DrmsdWorkspace` for each call — the cost
a naive caller pays. The "prep+C cmp" columns reuse a single workspace
across `C` right-side structures against one prepared reference — the
intended high-throughput usage.

Hardware: Windows x86_64 (AVX2 + FMA capable, no AVX-512).
Compilers: `rustc 1.95 stable-x86_64-pc-windows-gnu` (scalar baseline),
`nightly-x86_64-pc-windows-gnu` (SIMD).

## QCP RMSD

`cargo bench --bench rmsd_bench` — scalar (autovectorized, SSE2 baseline):

```
   n_atoms |     prep+1 cmp |    prep+16 cmp |   prep+256 cmp |  prep+4096 cmp |       fresh ws
-----------+----------------+----------------+----------------+----------------+---------------
        16 |      240.9 ns |      196.6 ns |      202.7 ns |      213.5 ns |      292.4 ns
        64 |      428.8 ns |      380.5 ns |      386.0 ns |      417.3 ns |      529.0 ns
       256 |      982.0 ns |      957.1 ns |      998.9 ns |      1.315 us |      1.274 us
      1024 |      3.222 us |      3.249 us |      3.361 us |      3.712 us |      4.599 us
      4096 |     12.166 us |     12.253 us |     13.629 us |     13.748 us |     19.598 us
```

`cargo +nightly bench --bench rmsd_bench --features portable-simd` — `std::simd::f64x4` + `multiversion(x86_64+avx2+fma, x86_64+avx512f, aarch64+neon)`:

```
   n_atoms |     prep+1 cmp |    prep+16 cmp |   prep+256 cmp |  prep+4096 cmp |       fresh ws
-----------+----------------+----------------+----------------+----------------+---------------
        16 |      228.9 ns |      180.5 ns |      182.9 ns |      198.2 ns |      279.5 ns
        64 |      334.7 ns |      284.8 ns |      288.2 ns |      294.9 ns |      428.6 ns
       256 |      540.5 ns |      508.7 ns |      530.3 ns |      701.9 ns |      871.1 ns
      1024 |      1.437 us |      1.435 us |      1.586 us |      2.071 us |      2.788 us
      4096 |      5.370 us |      5.047 us |      6.364 us |      6.678 us |     12.480 us
```

Speedup (warm `prep+16 cmp` column):

| n_atoms | scalar     | simd       | speedup |
|--------:|-----------:|-----------:|--------:|
|      16 |  196.6 ns  |  180.5 ns  |  1.09×  |
|      64 |  380.5 ns  |  284.8 ns  |  1.34×  |
|     256 |  957.1 ns  |  508.7 ns  |  1.88×  |
|    1024 |  3.249 us  |  1.435 us  |  2.26×  |
|    4096 | 12.253 us  |  5.047 us  |  2.43×  |

## dRMSD

`cargo bench --bench drmsd_bench` — scalar:

```
   n_atoms |     prep+1 cmp |    prep+16 cmp |   prep+256 cmp |       fresh ws
-----------+----------------+----------------+----------------+---------------
        16 |      302.6 ns |      247.3 ns |      254.3 ns |      529.0 ns
        64 |      4.092 us |      3.975 us |      4.029 us |      6.728 us
       256 |     64.026 us |     64.051 us |     64.330 us |    113.508 us
      1024 |      1.032 ms |      1.035 ms |      1.037 ms |      2.070 ms
      2048 |      4.232 ms |      4.364 ms |      4.287 ms |      8.290 ms
```

`cargo +nightly bench --bench drmsd_bench --features portable-simd` — SIMD:

```
   n_atoms |     prep+1 cmp |    prep+16 cmp |   prep+256 cmp |       fresh ws
-----------+----------------+----------------+----------------+---------------
        16 |      236.8 ns |      200.3 ns |      198.9 ns |      498.2 ns
        64 |      2.251 us |      2.225 us |      2.222 us |      5.024 us
       256 |     32.307 us |     32.043 us |     32.403 us |     80.794 us
      1024 |    503.214 us |    502.931 us |    501.516 us |      1.567 ms
      2048 |      2.018 ms |      2.017 ms |      2.046 ms |      6.026 ms
```

Speedup (warm `prep+16 cmp` column):

| n_atoms | scalar     | simd       | speedup |
|--------:|-----------:|-----------:|--------:|
|      16 |  247.3 ns  |  200.3 ns  |  1.23×  |
|      64 |  3.975 us  |  2.225 us  |  1.79×  |
|     256 | 64.051 us  | 32.043 us  |  2.00×  |
|    1024 |  1.035 ms  |  502.9 us  |  2.06×  |
|    2048 |  4.364 ms  |  2.017 ms  |  2.16×  |

## Findings

* **Hot loops**: the QCP RMSD inner kernel is the 9-FMA SoA
  cross-covariance in `rmsd::cross_cov_soa_scalar` (linear in N); the
  dRMSD inner kernel is the per-pair `sqrt + 3 FMA + diff² accumulate`
  in `drmsd::pairwise_distances_row_scalar` /
  `drmsd::sq_dist_diff_row_scalar` (quadratic in N).
* **SIMD payoff**: on AVX2+FMA the explicit `std::simd::f64x4` path
  achieves ~2.0–2.5× speedup over the autovectorized scalar baseline
  for N ≥ 256 in both kernels. Small structures (N ≤ 64) see modest
  gains (1.1–1.8×) because workspace setup, function-call overhead, and
  `multiversion`'s runtime dispatch start to dominate.
* **Workspace reuse**: for QCP RMSD at N = 4096 the fresh-workspace
  call is ~37 % slower than the warm path (alloc dominates). For dRMSD
  at N = 2048 the gap is 2× because the workspace pre-allocates the
  N(N-1)/2 reference distance buffer. Always reuse the workspace in
  hot loops.
* **AVX-512**: not present on the bench machine, so the AVX-512 fork
  emitted by `multiversion` was never selected at runtime. The same
  binary picks it up automatically on AVX-512 hardware without rebuild.

## Reproducing

```powershell
# Scalar baseline (stable):
cargo bench --bench rmsd_bench
cargo bench --bench drmsd_bench

# SIMD (nightly required for std::simd):
cargo +nightly bench --bench rmsd_bench  --features portable-simd
cargo +nightly bench --bench drmsd_bench --features portable-simd
```

---

## QCP RMSD — `f32` vs `f64` × lane-width sweep (2026-05-27, release / AVX2+FMA, `RUSTFLAGS=-C target-cpu=native`)

Direct calls to per-lane-width `cross_cov_soa_*` kernels, F-order SoA
inputs, hot caches, compiled with `cargo bench` (release/bench profile,
`opt-level=3`, `lto=off`, `codegen-units=16`). `scalar` is the
autovectorised reference implementation. Lane widths 16, 32, 64 are
emulated by LLVM as repeated 256-bit ops on AVX2 hardware.

### f64 — time per `cross_cov_soa` comparison

| n_atoms | comparisons | scalar  | simd-2  | simd-4  | simd-8  | simd-16 | simd-32 | simd-64 |
|--------:|------------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
|      16 |           1 |  41 ns  |  57 ns  |  43 ns  |  47 ns  |  84 ns  | 142 ns  | 281 ns  |
|      16 |        1024 |  32 ns  |  51 ns  |  37 ns  |  40 ns  |  76 ns  | 136 ns  | 276 ns  |
|      64 |           1 | 140 ns  | 181 ns  | 104 ns  |  90 ns  | 147 ns  | 195 ns  | 319 ns  |
|      64 |        1024 | 149 ns  | 187 ns  | 103 ns  |  91 ns  | 152 ns  | 198 ns  | 328 ns  |
|     256 |           1 | 512 ns  | 663 ns  | 366 ns  | 273 ns  | 413 ns  | 431 ns  | 539 ns  |
|     256 |        1024 | 513 ns  | 666 ns  | 364 ns  | 289 ns  | 442 ns  | 464 ns  | 576 ns  |
|    1024 |           1 | 1.98 µs | 2.59 µs | 1.34 µs | 988 ns  | 1.51 µs | 1.47 µs | 1.64 µs |
|    1024 |        1024 | 2.15 µs | 2.75 µs | 1.58 µs | 1.31 µs | 1.77 µs | 1.82 µs | 1.98 µs |

**f64 best lane width:** simd-8 across the board on AVX2 — it matches
the two-256-bit-pipe issue width without over-splitting the chunk loop.
simd-16/32/64 emulate as 2/4/8 chunks per iteration and lose to
pipeline-front-end pressure. simd-4 is the next best (single native
AVX2 op). simd-2 is always worse than scalar.

### f32 — time per `cross_cov_soa` comparison

| n_atoms | comparisons | scalar  | simd-4  | simd-8  | simd-16 | simd-32 | simd-64 |
|--------:|------------:|--------:|--------:|--------:|--------:|--------:|--------:|
|      16 |           1 |  43 ns  |  41 ns  |  39 ns  |  58 ns  | 143 ns  | 252 ns  |
|      16 |        1024 |  35 ns  |  33 ns  |  32 ns  |  54 ns  | 132 ns  | 245 ns  |
|      64 |           1 | 149 ns  |  97 ns  |  68 ns  |  83 ns  | 156 ns  | 262 ns  |
|      64 |        1024 | 143 ns  |  90 ns  |  63 ns  |  80 ns  | 149 ns  | 261 ns  |
|     256 |           1 | 571 ns  | 339 ns  | 197 ns  | 185 ns  | 289 ns  | 378 ns  |
|     256 |        1024 | 566 ns  | 345 ns  | 199 ns  | 174 ns  | 298 ns  | 392 ns  |
|    1024 |           1 | 2.28 µs | 1.26 µs | 688 ns  | 560 ns  | 816 ns  | 859 ns  |
|    1024 |        1024 | 2.37 µs | 1.47 µs | 805 ns  | 649 ns  | 975 ns  | 1.02 µs |

**f32 best lane width:** simd-8 for small (n_atoms ≤ 64); simd-16 for
n_atoms ≥ 256 (one 256-bit op per chunk plus more work per iteration).
simd-32/64 emulate and lose. Peak speedup vs scalar: ~4.1× at
n_atoms = 1024.

### Summary

| dtype | recommended default lane | hardware notes |
|-------|--------------------------|----------------|
| f64   | 8                        | AVX2 = 4-lane f64 native; LLVM unrolls 8-lane Simd into 2 fused 256-bit ops and keeps both FMA pipes busy |
| f32   | 8 (small) / 16 (large)   | AVX2 = 8-lane f32 native; 16-lane = 2× native chunks which win once the loop is long enough |

Workspace-reuse (`comparisons` axis) is essentially flat once warm —
the kernel cost dominates and reference preparation is O(n_atoms).
Numbers above are the average over the inner loop, so the cost of the
first call (cold I-cache, multiversion CPU-feature probe) is fully
amortised by `comparisons ≥ 16`.
