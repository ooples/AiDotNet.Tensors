# Sub-issue D Implementation Plan — Microkernel quality

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans

**Issue:** #372
**Parent:** #368

## What baseline-after-C revealed

Two root causes for the remaining 100×+ shape gaps:

### Bug #1: silent partial-M correctness regression

```csharp
// BlasManaged.cs line ~88
var (mr, nr) = PickMicrokernelTile<T>();
if (m < mr || n < nr || m % mr != 0) { mr = 4; nr = 4; }  // falls back ONCE
// ... downstream code uses (4, 4) regardless of m % 4
```

For shapes where `m % 4 != 0` (e.g., ResNet50_layer4 m=49, ViT 197), the scalar fallback doesn't help — `PackBothStrategy` still has `if (ir + mr > effectiveMc) break;` which **silently skips the partial-M trailing rows**. Output for those rows is all-zero from `c.Clear()`. No test catches this because existing tests don't assert correctness against a reference on those shapes.

### Bug #2: PackAOnly never uses AVX2

```csharp
// BlasManaged.cs PackAOnly dispatch:
// PackAOnly currently has no AVX2 strided-B kernel; always scalar Mr=Nr=4.
// TODO(Phase Cx): add Avx2 RunStridedB variants and wire them here.
PackAOnlyStrategy.Run<T>(..., mr: 4, nr: 4, options);
```

Most of the 50–80× gap shapes (ResNet50 layers, MobileNet PW, 64×64×64) route through PackAOnly because their K is small (< 128). They get scalar Mr=Nr=4 even on AVX2 hardware. OpenBLAS uses AVX2 Mr=8 Nr=8 for the same shapes → ~8× microkernel gap.

## Fix scope

### D.1 — Correctness fix for `m % 4 != 0` shapes

**Approach:** Route those shapes through `StreamingStrategy` which handles arbitrary M via its scalar (and SIMD-when-aligned) kernels without packing. Slower than a full AVX2 path but **correct**.

**Files:** ``src/AiDotNet.Tensors/Engines/BlasManaged/BlasManaged.cs``

**Test:** Compare BlasManaged output vs reference scalar implementation on shapes with m=49, m=197, m=3.

### D.2 — `Avx2Fp32_8x8.RunStridedB` microkernel

**Files:** ``src/AiDotNet.Tensors/Engines/BlasManaged/Microkernels/Avx2/Avx2Fp32_8x8.cs``

**Approach:** Add a `RunStridedB` overload that's identical to the existing packed `Run` but reads B at stride `ldb` (caller-supplied) instead of stride `Nr=8` (packed). Uses the same 8 `Vector256<float>` accumulators and `Fma.MultiplyAdd`.

**Test:** Bit-exact (or 1-ULP) against the scalar `ScalarFp32_4x4.RunStridedB` reference on aligned shapes.

### D.3 — Wire AVX2 into `PackAOnlyStrategy.DispatchStridedMicrokernel`

**Files:** ``src/AiDotNet.Tensors/Engines/BlasManaged/Strategies/PackAOnlyStrategy.cs``

**Approach:** When `Avx2Fp32_8x8.IsSupported` AND `kc` matches AVX2 layout requirements, dispatch to `Avx2Fp32_8x8.RunStridedB`. Else use scalar.

**Caller update:** `BlasManaged.Gemm` must pass `mr=8, nr=8` to PackAOnly when AVX2 is available AND shape aligns (m%8==0 AND n%8==0).

### D.4 — Re-baseline

**Files:** ``artifacts/perf/baseline-after-D.json``

**Acceptance criteria for Sub-D:**
- Correctness: any catalog shape produces output matching scalar reference within 1 ULP (FP32) or 1e-12 (FP64)
- Perf: ResNet50_layer1 / MobileNet PW shapes (m%8==0) cut ratio from 75× to ≤15×

## Out-of-scope (defer)

- FP64 strided-B AVX2 — same approach as D.2/D.3 but for FP64. Defer to D.5 follow-up if time.
- AVX-512 paths — not present on x64-amd-avx2-cpu16 hardware
- Software prefetch
- Unsafe pointer hot loops (microkernel inner is already unsafe + pointer; the strategy outer isn't)
- DynamicMethod IL emission (Phase J infra exists but activation deferred to Sub-D2)
