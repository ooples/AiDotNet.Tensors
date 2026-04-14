# Phase 0 Baseline — DiT-XL MatMul Shapes (post-iter-17)

**Date**: 2026-04-14
**Hardware**: Ryzen 9 3950X (Zen 2, AVX2, 16C/32T)
**Runtime**: .NET 10.0
**Config**: `SimpleJob(launchCount: 2, warmupCount: 5, iterationCount: 15)` × `ParamsAllValues(DeterministicMode)` × `ParamsAllValues(UseParallelGemm)` = 40 benchmarks
**Branch**: `feat/finish-mkl-replacement @ ce692dd`
**Runtime (wall)**: 7:13

## Summary

Against **MKL.NET (DeterministicMode=false)** as the reference, our best blocked-C# path **(DeterministicMode=true, UseParallelGemm=true)** is:

| Shape | MKL.NET (µs) | Ours (µs) | Ratio | Gap |
|---|---:|---:|---:|---|
| DiT attn out [1024,1152]×[1152,1152]    |   4,317 |   5,935 | **1.37×** | −31% |
| DiT QKV fused [1024,1152]×[1152,3456]   |  11,664 |  16,874 | **1.45×** | −34% |
| DiT MLP up [1024,1152]×[1152,4608]      |  17,211 |  23,595 | **1.37×** | −31% |
| DiT MLP down [1024,4608]×[4608,1152]    |  15,726 |  22,278 | **1.42×** | −30% |
| Square 1152² (DiT hidden)               |   4,265 |   5,590 | **1.31×** | −24% |
| Square 4608² (DiT MLP hidden)           | 326,491 | 495,231 | **1.52×** | −34% |
| Attn Q·K^T per-head [256,72]×[72,256]   |      85 |     216 | **2.56×** | −61% |
| Attn A·V per-head [256,256]×[256,72]    |      87 |     890 | **10.29×** | −90% |
| Batched 3D [1,256,1152]×[1152,4608]     |   6,525 |   7,600 | **1.16×** | −14% |
| Batched 3D [4,256,1152]×[1152,4608]     |  18,281 |  32,034 | **1.75×** | −43% |

**Worst gaps**: Attn A·V per-head **10.3×** slower; Square 4608² **1.52×** slower.
**Best gap**: Batched 3D [1,256,1152]×[1152,4608] only **1.16×** slower — closest to parity.

## Critical finding: Per-head attention matmul regression

| Shape | Size | MKL | Ours | Ratio |
|---|---|---:|---:|---:|
| Attn Q·K^T | [256,72]×[72,256] (4.7M FMAs) | 85 µs | 216 µs | 2.56× |
| Attn A·V   | [256,256]×[256,72] (4.7M FMAs) | 87 µs | 890 µs | **10.29×** |

The attention per-head matmuls are below our BLAS work threshold (4096) in some dimensions but well above in others — the 10.3× slowdown on `[256,256]×[256,72]` suggests we may be picking a poor dispatch path (possibly the naive scalar loop, or an unfavorable tile shape for K=256 wide but N=72 narrow). libxsmm-style JIT specialization for these exact shapes would be decisive here.

**Action item for Phase 2 iter 18**: investigate why `[256,256]×[256,72]` takes 890 µs on our path when it should be ~90 µs at 200 GFLOPS.

## Sequential-vs-parallel evidence

The `UseParallelGemm=false` deterministic path is catastrophically slow compared to parallel:

| Shape | Ours, Det=T/Par=F | Ours, Det=T/Par=T | Speedup |
|---|---:|---:|---:|
| Square 1152²      | 62,465 µs |    5,590 µs | **11.2×** |
| DiT attn out      | 56,333 µs |    5,935 µs | **9.5×** |
| Square 4608²      | 4,716,666 µs | 495,231 µs | **9.5×** |

This confirms the parallel dispatch is essential — iter 1/4/5 work (parallel-M, parallel-2D, oversubscribe to logical cores) was foundational. Any kernel-level optimization in Phase 2 must preserve the parallel path.

## Batched 3D matmul analysis

| Shape | Batch | MKL | Ours | Ratio |
|---|---|---:|---:|---:|
| Batched 3D [1,256,1152]×[1152,4608]  | 1 |  6,525 |  7,600 | 1.16× |
| Batched 3D [4,256,1152]×[1152,4608]  | 4 | 18,281 | 32,034 | **1.75×** |

**The ratio degrades sharply with batch size.** MKL's batched-GEMM path scales sublinearly with batch; ours scales linearly (we dispatch one `TryGemm` per batch slice via `Parallel.For`). This validates **Phase 2 iter 23** (batched-GEMM consolidation) as a high-impact change — the `[4,...]` case shows a 75% gap that should largely close with a single shared-B pack + multi-slice kernel dispatch.

## Where the "current path is within 10-20% of MKL" claim fails

The colleague's diagnosis was that local perf work could only shave 10-20% off. Against our Phase 0 baseline numbers:
- For **batched-small-batch** (1.16× gap), that claim is roughly correct.
- For **square ≥ 1152²** (1.31-1.52× gap), closing the gap requires 24-34% improvement — beyond the claim's envelope but achievable with libxsmm-style JIT + prefetch improvements per Phase 1 analysis.
- For **per-head attention** (2.56-10.3× gap), the claim is dramatically wrong — these shapes need completely different dispatch (small-matmul specialization, not blocked GEMM tuning).

## Allocated memory

All benchmarks allocate the output tensor only (e.g., 8.2 MB for 1024×1152×float = 8 MB + small overhead). No per-call scratch allocations — confirms prior iter 6's "pin spans directly, no memcpy" and the ArrayPool-backed packing buffers are working.

Deterministic-mode rows show a consistent +10-18 KB overhead vs MKL mode — that's the packed-A/packed-B buffers rented from the ArrayPool. Not worth optimizing in Phase 2.

## Acceptance gate for Phase 2

Each iter 18-23 must beat these numbers at **every** row above **without regressing any row by >2%** (outside BDN's reported error margin). Any regression outside noise gets the iter reverted, matching the process used in Issue #131 PR #134/#136.

**Target for end-of-Phase-2**: all ratios < 1.00× (i.e., our path beats or matches MKL at every shape). A stretch target: ratios ≤ 0.85× (our path clearly faster).

## Raw data

Full BDN output: `baseline-iter17-run.log` (253 KB, retained in `docs/mkl-replacement/baseline/`).

Reproduce with:
```sh
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --dit-xl-matmul
```
