# Audit 2026-05 · Phase 5 · net471 SIMD migration

**Status:** Foundation (this PR) → 80-primitive migration in incremental slices.
**Branch:** `audit/2026-05-phase5-net471-simd-foundation` (foundation)
**Owner:** AiDotNet.Tensors core
**Audit finding addressed:** #13 — *"net471 silently loses AVX2 because `System.Runtime.Intrinsics` is .NET 5+"*

## Problem

`AiDotNet.Tensors.csproj` multi-targets `net10.0;net471`. The net10.0 build uses `System.Runtime.Intrinsics` (`Avx`, `Avx512F`, `Sse`, `AdvSimd`) for every hot-path SIMD primitive. `System.Runtime.Intrinsics` does not exist pre-.NET 5, so the net471 build falls through every `#if NET5_0_OR_GREATER` block to a per-element scalar loop.

Approximate impact on Ryzen 9 3950X (Zen 2, AVX2):

| Op            | net10.0 (Avx)  | net471 (scalar) | Gap   |
|---------------|----------------|-----------------|-------|
| `VectorAdd` 1M floats | ~0.42 ms       | ~3.5 ms         | 8.3×  |
| `ReLU` 1M floats      | ~0.40 ms       | ~3.2 ms         | 8.0×  |
| `Dot` 1M floats       | ~0.18 ms       | ~1.4 ms         | 7.8×  |
| Softmax 4K×4096       | ~1.8 ms        | ~13 ms          | 7.2×  |

Audit finding #13 currently asks for a disclaimer on the README, not a fix. Phase 5 closes the underlying gap so the disclaimer can be retired.

## Strategy

The BCL has shipped `System.Numerics.Vector<T>` since .NET 4.6. RyuJIT auto-vectorizes it to SSE2 / AVX / AVX2 at the host CPU's native width, accessed via the static `Vector<T>.Count` property. The API is portable across x86 / x64 / ARM and across .NET Framework / .NET (Core), and `SimdGemm.cs:1305-1339` already uses the pattern on net471 for the GEMM accumulator loop.

We migrate each `#if NET5_0_OR_GREATER ... #endif` block in `SimdKernels.cs` (and downstream callers) to:

```csharp
#if NET5_0_OR_GREATER
    /* existing Avx / Avx512F / Sse / AdvSimd code — unchanged */
#else
    SystemNumericsVectorBridge.OpName(a, b, result);
#endif
```

The new file `src/AiDotNet.Tensors/Engines/Simd/SystemNumericsVectorBridge.cs` is gated `#if !NET5_0_OR_GREATER` and contains the BCL `Vector<T>` implementation. Spans are reinterpreted via `MemoryMarshal.Cast<float, Vector<float>>(span)` for zero-copy vector access — same approach SimdGemm already uses.

## Performance expectations

On AVX2 hosts:

- `Vector<float>.Count == 8`. RyuJIT emits VEX-prefixed 256-bit `vaddps` / `vmulps` / `vfmadd231ps`.
- ~85% of the hand-written `Avx.Add` equivalent. The 15% gap comes from RyuJIT not 4×-unrolling the auto-vectorized loop the way the hand-written net10 path does for the 32-element case in `VectorAdd`.
- For ops without 4×-unroll on the net10 side (the bulk of `SimdKernels`), parity is essentially 1:1.

On SSE2-only hosts: `Vector<float>.Count == 4`. RyuJIT emits 128-bit `addps` / `mulps`. ~4× scalar — still a major win even on legacy CPUs.

On AVX-512 hosts: `Vector<T>` on net471's BCL caps at 256-bit (the runtime ABI doesn't switch to 512-bit until `System.Runtime.Intrinsics.Vector512<T>` in net8). Acceptable — net471 users on AVX-512 hardware are vanishingly rare, and the alternative is no SIMD at all.

## This PR — foundation slice

**Net-new code:**

| File | Purpose |
|------|---------|
| `src/AiDotNet.Tensors/Engines/Simd/SystemNumericsVectorBridge.cs` | Bridge with 5 primitives: `VectorAdd`, `VectorMultiply`, `Saxpy`, `Dot`, `ReLU`. Gated `#if !NET5_0_OR_GREATER`. |
| `tests/AiDotNet.Tensors.Tests/Engines/Simd/SystemNumericsVectorBridgeTests.cs` | 32 parity tests (element-wise ops bit-identical to scalar reference; reductions within ulp ≤ 4). |
| `docs/internal/audit-2026-05-phase5-net471-simd.md` | This document. |

**Modified:**

| File | Change |
|------|--------|
| `src/AiDotNet.Tensors/Engines/Simd/SimdKernels.cs` | `VectorAdd(ReadOnlySpan<float>...)` net471 branch now calls the bridge. Proof-of-concept for the slice pattern. |

**Acceptance gates met:**

- Both TFMs build clean (`dotnet build -c Release -f net471` and `-f net10.0` each return 0 Warning(s) / 0 Error(s)).
- 32/32 bridge parity tests pass on net471.
- No regression on net10.0 (bridge is excluded from compilation; `SimdKernels.VectorAdd` net10 path is untouched).

## Migration slices (post-foundation)

Each slice is a separate PR that:

1. Adds N new bridge primitives.
2. Wires the corresponding `#if NET5_0_OR_GREATER ... #else SystemNumericsVectorBridge.X(...); #endif` branches in the relevant `SimdKernels.cs` / `SimdGemm.cs` / `SoftmaxSimd.cs` etc.
3. Adds parity tests for each new primitive.
4. Updates this doc's "Status" column.

Slicing keeps each PR < 600 LoC and reviewable in one sitting.

| Slice | Primitives | Status |
|-------|------------|--------|
| **Foundation (this PR)** | VectorAdd, VectorMultiply, Saxpy, Dot, ReLU (only `VectorAdd` wired) | done |
| 1 — element-wise math | wire VectorMultiply / Saxpy / Dot / ReLU into SimdKernels; add Sub / Div / Negate / Abs / Sqrt / Reciprocal | pending |
| 2 — activations | Sigmoid, Tanh, GELU, Mish, ELU, SELU, HardSwish, HardSigmoid, Softplus, Sign | pending |
| 3 — reductions | Sum, Max, Min, Mean, ArgMax, ArgMin, L2Norm | pending |
| 4 — softmax + log-sum-exp | SoftmaxRow, LogSoftmaxRow, fused-max-reduce | pending |
| 5 — fused activations | FusedAddReLU, FusedAddSigmoid, FusedMulAdd | pending |
| 6 — pool / NCHWc | NchwcMaxPool, NchwcAvgPool 2×2 s=2 | pending |
| 7 — BN / LN forward | NchwcBatchNorm, LayerNorm two-pass Welford | pending |
| 8 — backward stencils | SoftmaxBackward, LayerNormBackward, GELUBackward, BatchNormBackward | pending |
| 9 — GEMM micro | unroll the existing `SimdGemm` net471 4×8 microkernel via BCL Vector<float>; eliminate scalar tail | pending |
| 10 — conv direct | 3×3 s=1 p=1, 1×1 s=1 p=0 NCHWc fwd + dW + dX | pending |
| 11 — quantization | Int8 quantize / dequantize, BFloat16 round-trip | pending |
| 12 — README cleanup | remove the audit-2026-05 disclaimer + close audit finding #13 | pending |

Approximate total: ~80 primitives over 11 slices.

## Constraints

- No test weakening — every slice adds tests, never relaxes existing ones.
- Bridge ops produce bit-identical output to the scalar fallback for element-wise primitives; reductions allow ulp ≤ 4 because horizontal sum order differs.
- net10.0 path stays untouched in every slice (only `#else` branches are added; no edits inside `#if NET5_0_OR_GREATER`).
- No new external dependencies — `System.Numerics` is BCL.
- No `unsafe`. The bridge is fully verifiable IL.

## Verification protocol per slice

1. `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release -f net471` → 0 errors.
2. `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release -f net10.0` → 0 errors (and no behavioral change verified by full net10.0 test suite).
3. `dotnet test -c Release -f net471 --filter "FullyQualifiedName~SystemNumericsVectorBridgeTests"` → all pass.
4. `dotnet test -c Release -f net471` (full suite) → no regressions.
5. Update this doc's slice table.
