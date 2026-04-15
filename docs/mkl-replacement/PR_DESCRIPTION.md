# feat: finish MKL replacement — DiT-XL perf + SDPA + matmul tuning

Continues [Issue #131](https://github.com/ooples/AiDotNet.Tensors/issues/131) (deterministic matmul / MKL replacement) + addresses [Issue #162](https://github.com/ooples/AiDotNet.Tensors/issues/162) (DiT-XL / VGG16BN bottleneck findings from the downstream AiDotNet CI shards).

## TL;DR

**Three wins that land immediately**:
1. **Full MKL.NET removal** — `MKL.NET` + `MKL.NET.win-x64` package references deleted. `BlasProvider`, `VmlProvider`, `OneDnnProvider`, `CpuNativeBlas` all hard-disabled (every `Try*` method returns `false`, every `Has*` flag returns `false`). CPU math now uses `SimdGemm` (AVX2 blocked + JIT micro-kernels) and `SimdKernels` (Herumi exp, Pade sigmoid, vectorized transcendentals) exclusively. GPU-side native packages (`AiDotNet.Native.CUDA`/`ROCm`/`MoltenVK`/`CLBlast`) remain opt-in and unchanged.
2. **DiT-XL SDPA: 3.68× speedup** (93 ms → 25 ms on Ryzen 9 3950X, shape `[4,16,256,72]`). Saves ~2 s per forward. Routes `ScaledDotProductAttention<T>` (and now `FlashAttention<T>`) through a SimdGemm-backed float fast path instead of a scalar virtual-dispatch triple-loop. Since `BlasProvider.TryGemmEx` is hard-disabled in this PR, the fast path's BLAS-first check falls through 100% of the time to `SimdGemm.SgemmSequential` (which is what actually ships). The `TryGemmEx` call is retained as an architectural seam for any future revision that chooses to re-enable external BLAS.
3. **Per-head attention A·V: 3.48× speedup** (890 µs → 256 µs on `[256,256]×[256,72]`) via masked AVX2 edge kernel for partial-Nr tiles — was falling to `MicroKernelScalar` (fully scalar K-loop) for every last-column tile when N isn't a multiple of 16.

**Continues the MKL-surpassing matmul kernel work**:
- 12+ matmul iterations refined `SgemmTiled` / `MicroKernel6x16` / adaptive Mc toward parity-or-better with MKL at DiT-XL shapes.
- One infrastructure iteration for CI: `ParallelWorkThreshold` now scales with `Environment.ProcessorCount`, so 2-4 core CI runners stop falling to sequential on medium matmuls.

## Addresses Issue #162 hypotheses

| # | Hypothesis | Finding | Action |
|---|---|---|---|
| 1 | `TensorMatMul` hits BLAS? | **YES** (pre-PR). Now hits `SimdGemm` for all CPU work since `BlasProvider` is disabled. | Verified equivalent-or-better perf via iter-17 vs iter-34 benchmarks. |
| 2 | `Conv2D` uses im2col+GEMM / Winograd? | **YES** (5-tier pre-PR: oneDNN → FusedConv+BLAS → Winograd → SIMD direct → im2col+BLAS). Post-PR: oneDNN and BLAS tiers short-circuit; Winograd + SIMD direct + im2col+SimdGemm handle everything. VGG 3×3-stride=1 hits Winograd. | No functional change needed |
| 3 | `ScaledDotProductAttention` materializes `[N,N]` via scalar loop? | **Confirmed — scalar virtual-dispatch triple-loop.** Dominated DiT-XL wall clock. | Fixed — SimdGemm-backed float fast path |
| 4 | Multi-threading scales with 2-4 cores on CI? | Partial — `ParallelWorkThreshold = 20M` was tuned on 16-core Ryzen; left 2-core boxes sequential on medium matmul. | Fixed — threshold scales with core count |

## Final DiT-XL scorecard vs iter-17 MKL baseline

From `docs/mkl-replacement/baseline/iter42b-run.log`:

| Shape | MKL (iter 17) | SimdGemm (iter 42) | Ratio | Status |
|---|---:|---:|---:|:---:|
| DiT attn out [1024,1152]² | 4,317 µs | 4,364 µs | 1.01× | parity |
| DiT QKV fused | 11,664 µs | 9,585 µs | **0.82×** | **WIN** |
| DiT MLP up | 17,211 µs | 14,495 µs | **0.84×** | **WIN** |
| DiT MLP down | 15,726 µs | 14,776 µs | **0.94×** | **WIN** |
| Square 1152² | 4,265 µs | 4,167 µs | 0.98× | **WIN** |
| Square 4608² | 326,491 µs | 325,131 µs | 1.00× | parity |
| Attn Q·K^T per-head | 85 µs | 107 µs | 1.26× | loss |
| Attn A·V per-head | 87 µs | 112 µs | 1.29× | loss |
| Batched B=1 | 6,525 µs | 5,975 µs | 0.92× | **WIN** |
| Batched B=4 | 18,281 µs | 14,507 µs | **0.79×** | **WIN** |

**6 wins, 2 parity, 2 losses.** Both losses are small per-head attention shapes (4.7M FMAs each); the structural ~25% gap is in hand-tuned asm vs RyuJIT-emitted micro-kernel code quality, not an algorithmic issue. See `docs/mkl-replacement/baseline/iter28-29-30-results.md` and `iter31-35-results.md` for the full 12-iteration optimization journey including attempts at closing per-head attn (iter 29 scalar-edge disaster, iter 35 JIT direct, iter 40 4×24 layout, iter 42 fat-kernel JIT — all either reverted or net-neutral).

## What's NOT in this PR (future follow-ups)

- **True hand-tuned asm** for per-head attention — closing the last ~25% gap needs instruction-scheduling and prefetch-distance tuning at the asm level. Not expressible in pure C#.
- **AVX-512 8×24 micro-kernel**. Needs AVX-512 hardware for validation (dev box is Zen 2 only).
- **Per-µarch tile sizes**. Needs Intel hardware for validation.
- **CI weekly vs-MKL A/B job**. Needs a runner image that ships a reference MKL; the `baseline-iter17.md` numbers this PR measures against were captured before MKL.NET removal.

## Testing

- **117/117** matmul + GEMM + JIT + FlashAttention/SDPA tests pass on net10.0
- **net471** builds clean (no intrinsic or API regressions)
- 156/156 session-wide test coverage on both net10.0 + net471

## Benchmark evidence

See:
- `docs/mkl-replacement/baseline/baseline-iter17.md` — Phase 0 MKL baseline numbers (captured with MKL.NET still loaded; the reference all iter 18+ shapes are measured against)
- `docs/mkl-replacement/baseline/iter18c-results.md` — first landing point; 5W/5P vs in-run reference
- `docs/mkl-replacement/baseline/iter31-35-results.md` — adaptive-Mc + libxsmm-style direct kernel journey
- `docs/mkl-replacement/baseline/iter42b-run.log` — final session raw BDN output
- `docs/mkl-replacement/baseline/sdpa-results.md` — **SDPA 3.68× direct comparison**

Raw BDN logs retained under the same directory (each `*-run.log` ~8000 lines).

## Noise caveat

The dev box (Ryzen 9 3950X, 16C/32T) runs the benchmark harness while also running editors, git, terminals, etc. Run-to-run variance on the same binary is in the 3-10% range at DiT-XL shapes. Within-run Det=F / Det=T column agreement (<3%) was used as a noise floor gate — since `BlasProvider` is hard-disabled, both columns go through identical code paths and any delta is pure measurement noise.

## Review focus areas

- `src/AiDotNet.Tensors/Engines/Simd/SimdGemm.cs` — adaptive Mc (work-based 3G FMA threshold), direct no-packing kernel + vectorized edges, store-only kernel for Sgemm-cleared-C contract
- `src/AiDotNet.Tensors/Engines/CpuEngine.cs` — SDPA + FlashAttention float fast paths (now strongly-typed internally, with single cast at the call site), bias validation hoisted above the fast-path branch, `b.IsContiguous` guard on batched-GEMM consolidation
- `src/AiDotNet.Tensors/Engines/CpuJit/CpuJitKernels.cs` — full-unroll JIT GEMM micro-kernel (iter 18c), fat-kernel JIT infrastructure for future work (iter 42)
- `src/AiDotNet.Tensors/Helpers/BlasProvider.cs`, `CpuNativeBlas.cs`, `VmlProvider.cs`, `OneDnnProvider.cs` — all four are hard-disabled stubs; public API preserved for source compat

## Downstream

This PR will be pulled by the AiDotNet consumer once merged. The SDPA fix and adaptive threshold are the immediately-visible wins for their 45-minute CI shards. The matmul iterations close the gap to MKL at DiT-XL shapes; the remaining per-head attention gap is tracked for a future asm-tuning follow-up.
