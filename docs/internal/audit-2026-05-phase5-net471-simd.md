# Audit 2026-05 · Phase 5 · net471 SIMD migration

**Status:** Foundation + slices 1–3 wired → the dense-layer, activation, pooling, batch-norm, and convolution hot paths all use BCL SIMD on net471.
**Branch:** `audit/2026-05-phase5-net471-simd-foundation`
**Owner:** AiDotNet.Tensors core
**Audit finding addressed:** #13 — *"net471 silently loses AVX2 because `System.Runtime.Intrinsics` is .NET 5+"*

## Slice ledger (this branch)

| Slice | Scope | Status |
|-------|-------|--------|
| Foundation | bridge skeleton + `VectorAdd` proof-of-concept | ✅ |
| 1 | SimdKernels elementwise / scalar / reduction (34 methods, float+double) | ✅ |
| 2 | transcendentals: Cephes `Vector<float>` exp/log → Exp, Log, Sigmoid, Tanh, ELU, GELU (Swish/Mish transitive) | ✅ |
| 3a | NchwcPool: MaxPool / AvgPool / GlobalAvgPool | ✅ |
| 3b | NchwcBatchNorm: NCHWc8 + NCHW inference | ✅ |
| 3c | NchwcConv2D: forward (was hard-disabled on net471) | ✅ |
| 3d | FusedKernels: Swish/GELU/Mish/AddReLU/SigmoidMul/RMSNorm/LayerNorm/Softmax/LogSoftmax/BatchNorm | ✅ |
| 4 | SimdGemm micro-kernel | already vectorized — see §C |

## Problem

`AiDotNet.Tensors.csproj` multi-targets `net10.0;net471`. The net10.0 build uses `System.Runtime.Intrinsics` (`Avx`, `Avx512F`, `Sse`, `AdvSimd`) for every hot-path SIMD primitive. `System.Runtime.Intrinsics` does not exist pre-.NET 5, so the net471 build used to fall through every `#if NET5_0_OR_GREATER` block to a per-element scalar loop.

Approximate impact on Ryzen 9 3950X (Zen 2, AVX2):

| Op            | net10.0 (Avx)  | net471 pre-phase5 (scalar) | Gap   |
|---------------|----------------|----------------------------|-------|
| `VectorAdd` 1M floats | ~0.42 ms       | ~3.5 ms                    | 8.3×  |
| `ReLU` 1M floats      | ~0.40 ms       | ~3.2 ms                    | 8.0×  |
| `Dot` 1M floats       | ~0.18 ms       | ~1.4 ms                    | 7.8×  |
| Softmax 4K×4096       | ~1.8 ms        | ~13 ms                     | 7.2×  |

## Strategy

The BCL has shipped `System.Numerics.Vector<T>` since .NET 4.6. RyuJIT auto-vectorizes it to SSE2 / AVX / AVX2 at the host CPU's native width, accessed via the static `Vector<T>.Count` property. The API is portable across x86 / x64 / ARM and across .NET Framework / .NET (Core), and `SimdGemm.cs:1305-1339` already uses the pattern on net471 for the GEMM accumulator loop.

Migration pattern, applied to every `#if NET5_0_OR_GREATER ... #endif` block:

```csharp
#if NET5_0_OR_GREATER
    /* existing Avx / Avx512F / Sse / AdvSimd code — unchanged */
#else
    SystemNumericsVectorBridge.OpName(a, b, result);
#endif
```

`SystemNumericsVectorBridge` (gated `#if !NET5_0_OR_GREATER`) implements each primitive via `MemoryMarshal.Cast<T, Vector<T>>` to reinterpret spans as Vector<T> spans, then loops + element-wise op. Same approach SimdGemm uses at line 1329.

## Performance expectations

On AVX2 hosts (Vector<float>.Count == 8):
- ~85% of the hand-written `Avx.Add` equivalent. The 15% gap comes from RyuJIT not 4×-unrolling the auto-vectorized loop the way the hand-written net10 path does.
- For ops without 4×-unroll on net10, parity is essentially 1:1.

On SSE2-only hosts (Vector<float>.Count == 4): ~4× scalar — still a major win.

On AVX-512 hosts: `Vector<T>` on net471 caps at 256-bit. Acceptable — the alternative is no SIMD at all.

## What lands in this PR

### `src/AiDotNet.Tensors/Engines/Simd/SystemNumericsVectorBridge.cs`

31 primitives covering the BLAS-1 / activation / reduction core of SimdKernels.cs:

| Family | float | double |
|--------|-------|--------|
| Binary element-wise | VectorAdd, VectorSubtract, VectorMultiply, VectorDivide | same |
| Scalar broadcast | AddScalar, SubtractScalar, MultiplyScalar, DivideScalar | same |
| Fused mul-add | ScalarMultiplyAdd, Saxpy | ScalarMultiplyAdd |
| Unary | Sqrt, Abs, Negate, Clamp | same |
| Activations | ReLU, LeakyReLU | same |
| Reductions | Sum, Max, Min, Dot | same |

Gated `#if !NET5_0_OR_GREATER`; `internal` class; ~700 LoC.

### `src/AiDotNet.Tensors/Engines/Simd/SimdKernels.cs` — wired methods

Every method in the table below now calls the bridge on net471 instead of the per-element scalar loop. The net10.0 path is **completely unchanged**.

**float:** VectorAdd, VectorSubtract, VectorMultiply, VectorDivide, AddScalar, MultiplyScalar, ScalarMultiplyAdd, DotProduct, ReLU, LeakyReLU, Sum, Max, Min, Sqrt, Abs, Negate, Clamp

**double:** VectorAdd, VectorSubtract, VectorMultiply, VectorDivide, AddScalar, MultiplyScalar, ScalarMultiplyAdd, Sum, DotProduct, Max, Min, Sqrt, Abs, Negate, Clamp, ReLU, LeakyReLU

That's 34 method wirings in total.

### `tests/AiDotNet.Tensors.Tests/Engines/Simd/SystemNumericsVectorBridgeTests.cs`

109 parity tests (gated `#if !NET5_0_OR_GREATER`). Element-wise ops asserted bit-identical to scalar reference; reductions tolerated within ulp ≤ 4 because horizontal-sum order differs.

### `docs/internal/audit-2026-05-phase5-net471-simd.md`

This document.

## Acceptance gates

- `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release -f net471` → 0 warning / 0 error ✅
- `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release -f net10.0` → 0 warning / 0 error ✅
- `dotnet test -c Release -f net471 --filter "FullyQualifiedName~SystemNumericsVectorBridgeTests"` → 109/109 pass ✅

## Resolved since the original deferral

The foundation PR deferred transcendentals and the NCHWc8 kernels claiming both were blocked. **Both claims were wrong** and have since been delivered:

### A. Transcendentals — DONE (slice 2)

The original deferral claimed an accurate `Vector<float>` exp was impossible because the IEEE `2^n` reconstruction needs a vector integer left-shift (`<<23`) unavailable before .NET 7's `Vector.ShiftLeft`. **That was incorrect**: the shift is expressible as integer **multiply** by `2^23` (and the log right-shift as integer **divide** by `2^23` for non-negative operands), both supported on `Vector<int>`, plus `Vector.AsVectorSingle` / `Vector.AsVectorInt32` for the bit-reinterpret — all present in the netstandard2.0 `System.Numerics.Vectors` net471 references.

`FastExp` and `FastLog` are now faithful `Vector<float>` ports of the net10 `FastExp256` / `FastLog256` polynomials. Sigmoid, Tanh, ELU, GELU are wired; Swish and Mish vectorize transitively (they compose Sigmoid/Tanh/Exp/Log/VectorMultiply). Accuracy is asserted to the same fast-poly class the net10 path holds (~2e-4 rel for exp, ~1e-4 abs for log/sigmoid/tanh/elu), not weakened tolerance.

### B. NCHWc8 kernels — DONE (slice 3)

The original deferral claimed the 8-channel-block assumption blocked a portable path. It only blocks the SSE2-only (4-lane) host; on the common AVX2 net471 host `Vector<float>.Count == 8 == CBlock`, so `new Vector<float>(arr, idx)` loads the block directly. All three NCHWc8 reducers (Pool/BatchNorm/Conv2D) now gate the BCL path on `Vector<float>.Count == CBlock` and fall through to the existing scalar path on narrower SIMD — zero correctness risk. The NCHW (non-blocked) BatchNorm path is lane-width-agnostic and accelerates on SSE2 too.

## Still deferred — with concrete technical reasons

### C. GEMM micro-kernel (SimdGemm / SimdGemmDouble) — already vectorized

Investigated: SimdGemm's net471 path (`SgemmVector`, line 1313) is **already** a fully `Vector<float>`-vectorized axpy-over-N kernel with a 4×-unrolled inner loop — a prior PR closed this (the method's own doc-comment notes net471 previously fell to scalar and now reaches "~8× over scalar on AVX2"). The net5 hand-unrolled 6×16 / 4×8 micro-kernels are an *additional* optimization on top, gated on `Avx2.IsSupported`, that targets specific small-matmul shapes; porting their exact unroll geometry to BCL `Vector<T>` would yield marginal gains over `SgemmVector` and risks regressing the tuned net5 path. No action needed — GEMM is not a net471 scalar-fallback gap.

### D. Math primitives Floor / Ceiling / Frac / Sin / Cos / Log2 / Pow

`Floor`/`Ceiling`/`Frac` have no `Vector<T>` BCL primitive before .NET 7 (`Vector.Floor` is .NET 7+) and the integer-truncate-and-correct trick used for exp's range reduction would need per-call masking that erodes the win. `Sin`/`Cos` would need their own polynomial ports (the bridge now has the exp/log machinery to model them on); `Pow` = `exp(y·log(x))` could compose the new bridge primitives. These are the lowest-call-volume ops in training and remain scalar on net471 for now.

### E. Backward-pass primitives (SoftmaxBackward, LayerNormBackward, GELUBackward, BatchNormBackward, *_Half variants)

Now unblocked by slice 2's transcendentals; wire in a follow-up. The `*_Half` variants additionally need a `Vector<float>`-from-`Half` widening path.

## Cumulative impact (delivered on this branch)

| Slice | What it added | Net471 coverage |
|-------|---------------|-----------------|
| Pre-phase 5 | — | 0% (everything scalar except GEMM) |
| Foundation + 1 | 34 SimdKernels elementwise/scalar/reduction ops | dense-layer hot path |
| 2 | exp/log/sigmoid/tanh/elu/gelu (+swish/mish transitive) | + all activations |
| 3a–3d | pool, batchnorm, conv2d, fused pointwise/norm/softmax | + CNN + transformer blocks |

Combined, the wired surface covers the full forward inference path of CNNs (conv → BN → ReLU/activation → pool) and transformers (matmul → LayerNorm → softmax → fused activations), plus the dense-layer training inner loop (Add/Multiply/ScalarMultiplyAdd/Sum/Dot). GEMM was already vectorized (§C). The net471 build now uses AVX2 throughput across essentially every op a typical model exercises at inference, and the dense + activation ops at training.

## Constraints honored

- ✅ No test weakening — every slice adds tests, never relaxes existing ones.
- ✅ Bridge ops produce bit-identical output to scalar fallback for element-wise primitives; reductions allow ulp ≤ 4.
- ✅ net10.0 path stays untouched — only `#else` branches are added.
- ✅ No new external dependencies — `System.Numerics` is BCL.
- ✅ No `unsafe` — bridge is fully verifiable IL.

## Verification protocol per follow-up slice

1. `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release -f net471` → 0 errors.
2. `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release -f net10.0` → 0 errors.
3. `dotnet test -c Release -f net471 --filter "FullyQualifiedName~SystemNumericsVectorBridgeTests"` → all pass.
4. `dotnet test -c Release -f net471` (full suite) → no regressions.
5. Update this doc's "Cumulative impact" table.
