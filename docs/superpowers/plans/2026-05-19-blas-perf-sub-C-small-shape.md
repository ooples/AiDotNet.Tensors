# Sub-issue C Implementation Plan — Small-shape pack-free streaming

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans

**Issue:** #371
**Parent:** #368
**Goal:** Eliminate the 247× worst-case gap and the cluster of 70–100× small-shape regressions by adding a tiny-shape fast path that bypasses dispatcher/autotune/options overhead.

## Baseline data analysis

Looking at ``artifacts/perf/baseline-after-B.json``, the worst-case shapes are dispatcher-overhead-dominated:

| Shape | Flops | BlasManaged time | Effective GFLOPS | Diagnosis |
|-------|------:|-----------------:|-----------------:|-----------|
| Tiny_8×6×4 FP32 | 192 | 0.07ms | 0.003 | Pure overhead (192 flops in 70μs = 65 cycles per flop) |
| Tiny_8×2×4 FP32 | 64 | similar | similar | Same |
| Tiny_32×32×32 FP32 | 32K | 0.7ms | 0.045 | Routed to PackAOnly; pack+autotune overhead dominates |
| Tiny_32×32×32 FP64 | 32K | similar | similar | Same |
| WideFat_512×512×64 FP64 | 16.7M | 6.3ms | 2.6 | Real compute, but scalar Mr=Nr=4 vs AVX2 OpenBLAS |

The first three are overhead-bound. The last is a microkernel quality issue (Sub-D).

## The fix

Add a **tiny-shape fast path** at the top of ``BlasManaged.Gemm`` that:

1. Skips ``Dispatcher.SelectStrategy`` (one branch + bool decision is "always Streaming")
2. Skips ``PickMicrokernelTile`` (no microkernel tiles needed for streaming)
3. Skips ``AutotuneDispatcher.Decide`` (no autotune cache lookup)
4. Skips ``options.PackingMode`` switch (always route to Streaming)
5. Calls ``StreamingStrategy.Run`` directly with default options
6. Still applies the epilogue if present

Threshold: ``M·N·K ≤ TinyShapeWorkThreshold`` where threshold is chosen so the bypass clearly wins. Initial choice: 100,000 flops (~46×46×46).

## Tasks

### C.1 — Tiny-shape bypass at top of `BlasManaged.Gemm`

**Files:**
- Modify: ``src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs``
- Test: ``tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/TinyShapeBypassTest.cs``

**Acceptance:**
- Tiny shapes route to StreamingStrategy with no autotune/dispatcher overhead
- Bit-exact across thread counts (Streaming inherits this from Sub-B)
- Time on Tiny_8×6×4, Tiny_32×32×32 cut by ≥2×

### C.2 — Re-baseline + assess

**Acceptance:** Worst-case ratio drops materially. New ``baseline-after-C.json`` committed.

## Out-of-scope for Sub-C

- Microkernel improvements for non-tiny small shapes (e.g., WideFat 512×512×64) — that's Sub-D
- Pack-free path for medium shapes (Streaming already handles this when ``K < 32 || M·N < 1024``)
- Vectorized pack kernels — Sub-D
