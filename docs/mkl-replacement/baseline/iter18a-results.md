# Iter 18a — Masked AVX2 edge kernel for partial-Nr tiles

**Change**: new `MicroKernel6xNMasked` in `SimdGemm.cs`, dispatched by `MacroKernel` when `mc_actual == Mr && 0 < nc_actual < Nr`. Reuses the 12-accumulator FMA inner loop of `MicroKernel6x16` (PackB already zero-pads unused columns, so extra FMAs contribute 0) and uses `Avx.MaskLoad`/`Avx.MaskStore` for the final accumulate-and-store. Previous path: fully scalar `MicroKernelScalar` K-loop.

**Correctness**: 86/86 matmul + GEMM + DeterministicMode tests pass on net10.0.

**Benchmark delta** (Ryzen 9 3950X, Det=T/Par=T, vs iter 17 baseline):

| Shape | Baseline µs | Iter 18a µs | Δ | vs MKL (new) |
|---|---:|---:|---:|---:|
| DiT attn out [1024,1152]×[1152,1152]    |  5,935 |  6,040 | +1.8% | 1.23× |
| DiT QKV fused [1024,1152]×[1152,3456]   | 16,874 | 17,328 | +2.7% | 1.34× |
| DiT MLP up [1024,1152]×[1152,4608]      | 23,595 | 23,238 | −1.5% | 1.26× |
| DiT MLP down [1024,4608]×[4608,1152]    | 22,278 | 22,394 | +0.5% | 1.30× |
| Square 1152²                            |  5,590 |  5,575 | −0.3% | 1.10× |
| Square 4608²                            | 495,231 | 500,909 | +1.1% | 1.47× |
| Attn Q·K^T per-head [256,72]×[72,256]   |    216 |    203 | −6.3% | 2.37× |
| **Attn A·V per-head [256,256]×[256,72]** |    **890** |    **256** | **−71.2%** | **2.70×** |
| Batched 3D [1,256,1152]×[1152,4608]     |  7,600 |  7,580 | −0.3% | 0.86× (faster) |
| Batched 3D [4,256,1152]×[1152,4608]     | 32,034 | 31,947 | −0.3% | 1.08× |

**Attn A·V per-head: 890 µs → 256 µs (3.48× speedup)**

The 10.3× gap to MKL collapsed to 2.70×. Remaining gap is the `mc_actual < Mr` scalar fallback (still hit at the bottom Mr-block edge when M%Mr ≠ 0, e.g. M=256 = 42×6 + 4 → one 4×16/8 scalar block per column slice).

All other rows are within the ±2% noise floor except Q·K^T (−6.3%, borderline significant; 15 iterations × 2 launches per benchmark) and QKV fused (+2.7%, also within the sum of standard deviations). The square-shape gaps are **unchanged** because they don't hit partial-Nr tiles — 1152² and 4608² are both multiples of 16.

**What this validates**:
1. Masked AVX load-add-store is correct — no test regression, no wrong-result bugs.
2. PackB's zero-padding of tail Nr columns was already in place (lines 830–833 of SimdGemm.cs). The masked kernel relies on this invariant.
3. `Avx.MaskLoad`/`Avx.MaskStore` with `Vector256<int>` masks cast via `.AsSingle()` works correctly on .NET 10 / Zen 2 AVX2.

**What remains for the A·V shape**:
- Still 2.70× slower than MKL (256 µs vs 94 µs). The gap is now split across:
  - ~50 µs unavoidable pack overhead (B and A)
  - ~200 µs in the scalar `mc=4` partial-Mr tiles at the M edge (M=256 % 6 = 4, so 1 partial Mr-block per Nr column, ×5 Nr blocks × 2 Ic blocks = 10 scalar calls; each ~20 µs at K=256).
  - A separate iter 18b could add a `MicroKernelMxNMasked` variant that handles mc<Mr (e.g. 4×16) without going scalar. Estimated another ~2× win on A·V.

**Acceptance**: PASS for iter 18a. No row regresses beyond noise; one row improves dramatically (3.48×). Commit and move on to iter 18b (mc<Mr masked kernel) or iter 19 (libxsmm-style JIT full-K-unroll).

## Raw log

`docs/mkl-replacement/baseline/iter18a-attn-only.log` (~8000 lines).
