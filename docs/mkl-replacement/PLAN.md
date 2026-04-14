# Finish MKL Replacement — Feature Branch Plan

**Branch**: `feat/finish-mkl-replacement`
**Origin**: `main @ 72b01c1`
**Tracks**: Continuation of Issue #131 (deterministic matmul) / PR #134+#136 (17 perf iterations)

## Goal

Complete the replacement of MKL.NET with our own SIMD GEMM kernels such that:

1. **Our blocked C# GEMM matches or beats MKL.NET at every shape we care about** — including the square/square-ish regime (1152², 4608², [1024,1152]×[1152,1152]) where we currently trail by 1.35–1.64× per Issue #131 iter 9 numbers.
2. **MKL.NET becomes opt-in** rather than default.
3. **MKL.NET package reference is removed** from `AiDotNet.Tensors.csproj` once the flip has stabilized.

## Why now

The downstream consumer reported DiT-XL CPU forward pass shards being cancelled at the 45-minute CI budget. Investigation shows the per-block DiT-XL matmuls are concentrated in the exact "square-ish" regime where we trail MKL. Local 10-20% wins on top of the current path are not enough to bring shards under budget — we need to close the gap to MKL entirely.

The existing `DeterministicMatMulBenchmarks` already tracks HRE-tiny shapes; this branch adds DiT-XL shapes (`DitXLMatMulBenchmarks`) and treats them as the acceptance gate.

## Current baseline (from Issue #131 memory, 2026-04-11, iter 9 — pre-iter-17)

| Shape | Iter 0 (seq) | Iter 9 | vs MKL |
|---|---:|---:|---:|
| [512,512]² | 5,933 μs | 1,160 μs | 1.35× slower |
| [1024,1024]² | 48,830 μs | 5,631 μs | 1.64× slower |
| LM-head [64,128]×[128,50257] | 70,140 μs | 9,917 μs | **0.83× (17% faster)** |
| HRE shapes | unchanged | sequential | — |

After PR #136 master is at iter 17 — we re-baseline in Phase 0 before any kernel work.

## Phases

### Phase 0 — Baseline (this commit)

**Files added:**
- `tests/AiDotNet.Tensors.Benchmarks/DitXLMatMulBenchmarks.cs` — 10 DiT-XL shapes (per-block projections, square-at-hidden, per-head attention, 3D batched)
- `docs/mkl-replacement/PLAN.md` — this document
- `docs/mkl-replacement/baseline/` — will hold `baseline-iter17.md` after benchmark run

**Deliverable:** Run `dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- --dit-xl-matmul` on the Ryzen 9 3950X dev box. Record median μs + throughput (GFLOPS) for each shape under all 4 (DeterministicMode, UseParallelGemm) combos. Commit the results under `docs/mkl-replacement/baseline/baseline-iter17.md`.

**Acceptance gate:** every subsequent optimization iteration A/Bs against these numbers. Any regression > 2% at any tracked shape is reverted unless the author can show the regression is noise (p > 0.05).

### Phase 1 — Competitive analysis

**Research target:** every open-source GEMM library that might encode techniques MKL uses.

| Library | Repo / Path | What to extract |
|---|---|---|
| BLIS | `flame/blis:/config/haswell/3/bli_gemm_haswell_asm_d6x8.c` | Reference micro-kernel ASM structure. Our `MicroKernel6x16` is BLIS-derived; read their 4×16 single-precision variant and prefetch strategy. |
| OpenBLAS | `OpenMathLib/OpenBLAS:/kernel/x86_64/sgemm_kernel_16x4_haswell.S` | Goto-style kernel. Different packing layout vs BLIS. |
| libxsmm | `libxsmm/libxsmm:/src/generator_gemm_avx2_microkernel.c` | **Highest-value target.** JIT-emits size-specialized kernels at runtime. Intel engineer wrote this specifically because MKL's per-call overhead dominated small-matmul workloads. Strong candidate for MKL's small-matmul advantage. Our `CpuJitKernels.GetGemmMicroKernel` only specializes per-`kc`; libxsmm specializes fully per (mc, nc, kc). |
| oneDNN | `oneapi-src/oneDNN:/src/cpu/x64/gemm/f32/` | Direct descendant of MKL-DNN. Apache-licensed. `jit_avx2_gemm_f32.cpp` has the f32 AVX2 GEMM path; `jit_avx512_core_gemm_smalln_tn_f32.cpp` has the small-N AVX-512 path. |
| Intel Optimization Reference | `intel-optimization-manual`, `intel-intrinsics-guide` | Per-microarch recommended tile sizes, prefetch distances. |

**Deliverable:** `docs/mkl-replacement/competitive-analysis.md` — one page per library noting the specific techniques found, whether we already use them, and whether they're applicable to our AVX2 + JIT target.

### Phase 2 — Incremental optimizations

Ordered by expected impact against the **square-shape gap**. Each iteration = one commit, A/B gated against Phase 0 baseline, reverted on regression.

| Iter | Change | Expected at 1024² | Risk |
|---|---|---|---|
| 18 | **Fully-specialized JIT micro-kernel** (libxsmm-style) — emit (mc, nc, kc) as immediates, unroll k loop completely. Extends the existing `CpuJitKernels` path. | 20-40% at all shapes | Medium |
| 19 | **Register-level software prefetch** — prefetch next `Mr` rows of A during B loads, not block-level prefetch. | 5-10% | Low |
| 20 | **Per-µarch tile sizes** — runtime switch on `CpuFeatures.Microarch`. Zen 2 may prefer smaller Mc; Skylake may prefer smaller Nc. | 5-15% on non-Zen2 hardware | Low |
| 21 | **Non-temporal stores for C** — when `m*n*4 > L2_size`, use `vmovntps` on the output panel. | 10-20% at ≥1024² | Medium (correctness-sensitive) |
| 22 | **AVX-512 8×24 micro-kernel** — runtime-detected, AVX2 fallback. 2× FMA throughput on AVX-512 hardware. | 2× on AVX-512 | High (no local validation on Zen 2) |
| 23 | **Batched-GEMM consolidation** — single pack for shared B across batch slices (eliminates the per-batch BLAS dispatch in `TensorMatMulBatched`). | 1.5-3× at batched-small patterns | Medium |

### Phase 3 — Flip the default

**Trigger:** benchmarks show ≥ MKL at **every** shape in `DitXLMatMulBenchmarks` and `DeterministicMatMulBenchmarks`.

- Flip `BlasProvider._useMklNet` default to `false`. Still opt-in via `AIDOTNET_BLAS_PREFER_MKL=1` for comparison.
- Update XML docs to reflect that ours is now the reference implementation.
- Add a CI job that A/Bs our path vs MKL weekly to prevent regressions.

### Phase 4 — Remove MKL dependency

**Trigger:** 2-4 weeks of production use with the flipped default, no regression reports.

- Delete `MKLNET` package reference from `AiDotNet.Tensors.csproj`.
- Delete MKL.NET paths from `BlasProvider.cs` (keep native `cblas_sgemm` P/Invoke as an optional fallback for users who want to supply their own BLAS).
- Remove the 110 MB `MKL.NET.win-x64` binary from the published NuGet package.
- Verify net471 + net10.0 builds still pass.

## Known bottlenecks in MKL we intend to exploit

1. **Per-call MKL overhead ≈ 2-3 μs** — already mitigated by direct `calli` dispatch. Further win via `cblas_?gemm_batch`-equivalent in Phase 2 iter 23.

2. **MKL doesn't re-tune at runtime** — MKL picks a kernel at library load based on CPU ID. Our JIT path (Phase 2 iter 18) can re-tune per exact matrix shape, which is how libxsmm beats MKL at small-matrix workloads.

3. **MKL's `mkl_set_dynamic` is non-disableable from managed code** — documented in `memory/blas_native_init_crash.md`. Managed consumers cannot pin MKL to a fixed thread count through MKL.NET's public API. Our `Parallel.For` honors `Helpers.CpuParallelSettings` and composes cleanly.

4. **MKL.NET package bloat** — `MKL.NET.win-x64` is a 110 MB platform-specific binary. Removing it shrinks the published NuGet 10×.

## Scope decisions

| Question | Decision |
|---|---|
| CPU targets | Ryzen 9 3950X (Zen 2, AVX2) as dev box. AVX-512 added in iter 22 but validated on a separate runner. |
| net471 support | Kept — net471 stays on the current scalar/`Vector<T>` fallback. MKL replacement optimizations target net5+ paths only. |
| Conv2D replacement | **Out of scope for this branch.** DiT-XL's patch-embed conv is also a hot path, but Conv2D on master already has oneDNN → FusedConv BLAS → Winograd → SIMD direct → im2col. A separate branch (`feat/finish-mkl-replacement-conv`) will follow. |
| Batched-small matmul consolidation | **In scope, iter 23.** The `TensorMatMulBatched` per-slice BLAS dispatch is a known win independent of MKL vs ours. |
| Deterministic mode as default | **Yes, after Phase 3.** Matches the strategic direction documented in `memory/mkl_replacement_strategy.md`. |

## Re-run commands

```sh
# Phase 0 baseline (run once, commit results)
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- --dit-xl-matmul

# HRE + square shapes (existing harness, for historical continuity)
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- --vs-deterministic-matmul

# Run both with raw CSV output for A/B diffing
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- --dit-xl-matmul \
    > baseline-$(date +%Y%m%d-%H%M).txt
```

## Code locations touched by this plan

- `src/AiDotNet.Tensors/Engines/Simd/SimdGemm.cs` — tiled GEMM + micro-kernels (main target)
- `src/AiDotNet.Tensors/Engines/CpuJit/CpuJitKernels.cs` — JIT micro-kernel emitter (iter 18)
- `src/AiDotNet.Tensors/Helpers/MatrixMultiplyHelper.cs` — dispatch thresholds + `TryGemm` entry
- `src/AiDotNet.Tensors/Helpers/BlasProvider.cs` — MKL init + deterministic mode (Phase 3, 4)
- `src/AiDotNet.Tensors/Engines/CpuEngine.cs:TensorMatMulBatched` — batched consolidation (iter 23)
- `src/AiDotNet.Tensors/AiDotNet.Tensors.csproj` — remove `MKLNET` package reference (Phase 4)
