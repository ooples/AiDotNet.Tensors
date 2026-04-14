# feat: finish MKL replacement — DiT-XL perf + SDPA + matmul tuning

Continues [Issue #131](https://github.com/ooples/AiDotNet.Tensors/issues/131) (deterministic matmul / MKL replacement) + addresses [Issue #162](https://github.com/ooples/AiDotNet.Tensors/issues/162) (DiT-XL / VGG16BN bottleneck findings from the downstream AiDotNet CI shards).

## TL;DR

**Two wins that land immediately**:
1. **DiT-XL SDPA: 3.68× speedup** (93 ms → 25 ms on Ryzen 9 3950X, shape `[4,16,256,72]`). Saves ~2 s per forward. Routes `ScaledDotProductAttention<T>` through `BlasProvider.TryGemmEx` instead of a scalar virtual-dispatch triple-loop.
2. **Per-head attention A·V: 3.48× speedup** (890 µs → 256 µs on `[256,256]×[256,72]`) via masked AVX2 edge kernel for partial-Nr tiles — was falling to `MicroKernelScalar` (fully scalar K-loop) for every last-column tile when N isn't a multiple of 16.

**Continues the MKL-surpassing matmul kernel work**:
- Four matmul iterations (18a, 18b, 19, 20) refine `SgemmTiled` / `MicroKernel6x16` toward parity with MKL at the square shapes DiT-XL blocks hit.
- One infrastructure iteration for CI: `ParallelWorkThreshold` now scales with `Environment.ProcessorCount`, so 2-4 core CI runners stop falling to sequential on medium matmuls.

## Addresses Issue #162 hypotheses

| # | Hypothesis | Finding | Action |
|---|---|---|---|
| 1 | `TensorMatMul` hits BLAS? | **YES** (4-tier dispatch). Scalar only for non-float/double `T`. | No change needed |
| 2 | `Conv2D` uses im2col+GEMM / Winograd? | **YES** (5-tier: oneDNN → FusedConv+BLAS → Winograd → SIMD direct → im2col+BLAS). VGG 3×3-stride=1 hits Winograd. | No change needed |
| 3 | `ScaledDotProductAttention` materializes `[N,N]` via scalar loop? | **Confirmed — scalar virtual-dispatch triple-loop.** Dominated DiT-XL wall clock. | Fixed — BLAS-backed float fast path |
| 4 | Multi-threading scales with 2-4 cores on CI? | Partial — `ParallelWorkThreshold = 20M` was tuned on 16-core Ryzen; leaves 2-core boxes sequential on medium matmul. | Fixed — threshold scales with core count |

## Commits on this branch

| Commit | Change | Measured impact |
|---|---|---|
| `b8a8b81` | Phase 0: `DitXLMatMulBenchmarks` harness + 4-phase PLAN.md | — |
| `ce692dd` | Phase 1: competitive analysis (BLIS / libxsmm / OpenBLAS / oneDNN) | — |
| `c2dd21b` | Phase 0 baseline: our blocked C# vs MKL at DiT-XL shapes. Our gap: 1.31-1.52× square, 10.3× worst (A·V) | Acceptance gate |
| `6b74b75` | **iter 18a**: masked AVX2 edge kernel for partial-Nr tiles | A·V 10.3× → 2.7× (3.48×) |
| `ccff3ae` | **iter 19**: `prefetcht2` on next A panel (oneDNN pattern) | No regression |
| `cd954be` | **SDPA BLAS fast path (Issue #162)** | DiT-XL SDPA 93 ms → 25 ms (3.68×) |
| `14efda4` | **Adaptive `ParallelWorkThreshold` (Issue #162)** | 2-core CI: parallel at 128²+ |
| `249c7a6` | **iter 20**: 4-way K unroll in `MicroKernel6x16` + **iter 18b**: generalized masked edge kernel for mc<Mr | −2-3% on DiT-block shapes (noise-dominated) |
| (iter 24) | **Batched-GEMM consolidation**: single 2D GEMM for ND×2D broadcast instead of per-slice dispatch | TBD, pending benchmark |

## What's NOT in this PR (future follow-ups)

- **iter 18c**: libxsmm-style full-K-unroll JIT micro-kernel. Highest-projected win (20-40% at square ≥512²) per the competitive analysis, but requires IL emission work. Separate branch.
- **iter 21**: Per-µarch tile sizes. Needs Intel hardware for validation (dev box is Zen 2 only).
- **iter 23**: AVX-512 8×24 micro-kernel. Needs AVX-512 hardware for validation.
- **iter 25**: Write-intent prefetch on C rows. Small, speculative; will revisit after iter 18c lands.
- **iter 26**: 64-byte aligned packed A/B buffers + `vmovaps` loads. Marginal expected win on Zen 2.
- **FlashAttention** scalar path (`CpuEngine.cs:14985`). Not on DiT-XL default path.
- **Phase 3 (flip default)** and **Phase 4 (remove MKL dependency)**: gated on closing the remaining square-shape gap, which requires iter 18c.

## Testing

- **86/86** matmul + GEMM + DeterministicMode tests pass on net10.0 (iter 18a/18b/19/20/24)
- **118/118** including SDPA + Attention tests pass (after SDPA fix)
- **net471** builds clean for all changes (no intrinsic or API regressions)

## Benchmark evidence

See:
- `docs/mkl-replacement/baseline/baseline-iter17.md` — Phase 0 baseline numbers
- `docs/mkl-replacement/baseline/iter18a-results.md` — A·V 10.3× → 2.7×
- `docs/mkl-replacement/baseline/iter19-results.md` — prefetch, noise-dominated
- `docs/mkl-replacement/baseline/iter20-results.md` — K-unroll
- `docs/mkl-replacement/baseline/sdpa-results.md` — **SDPA 3.68× direct comparison**

Raw BDN logs retained under the same directory (each `*-run.log` ~8000 lines).

## Noise caveat

The dev box (Ryzen 9 3950X, 16C/32T) runs the benchmark harness while also running editors, git, terminals, etc. Run-to-run variance on the same binary (verified by watching MKL's own numbers move between runs without any MKL code change) is in the 5-20% range at DiT-XL shapes. A/B acceptance for each iter has been "no-regression beyond noise floor" rather than strict numeric improvement. A clean re-run on the CI box will give less ambiguous numbers for the marginal iters (19, 20); the big wins (18a, SDPA, adaptive threshold) are outside the noise envelope.

## Review focus areas

- `src/AiDotNet.Tensors/Engines/Simd/SimdGemm.cs` — the new masked kernel and its dispatch, the 4-way K unroll correctness (especially the scalar tail for `kc % 4 != 0`), the prefetch placement.
- `src/AiDotNet.Tensors/Engines/CpuEngine.cs` — the SDPA fast-path branch, the batched-GEMM consolidation, and the (unchanged) scalar path for non-float `T`.
- `src/AiDotNet.Tensors/Helpers/BlasProvider.cs` — no changes in this PR; the SDPA fast path uses the existing `TryGemmEx`.

## Downstream

This PR will be pulled by the AiDotNet consumer once merged. The SDPA fix and adaptive threshold are the immediately-visible wins for their 45-minute CI shards. The matmul iters 18a/18b/19/20 close part of the gap to MKL at DiT-XL shapes; iter 18c is needed to fully close it, tracked separately.
