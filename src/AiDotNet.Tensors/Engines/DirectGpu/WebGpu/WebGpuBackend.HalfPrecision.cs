// Copyright (c) AiDotNet. All rights reserved.
// IGpuHalfPrecisionBackend implementation for the WebGPU backend (issue #560):
// FP16 GEMM so MatrixMultiply<Half> runs on the GPU instead of dropping to a
// scalar CPU fallback.


// WebGPU is only available on .NET 7+ (Blazor WebAssembly / Dawn); the rest of
// the backend is compiled out on net471, so this partial must be too.
#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Half-precision GEMM dispatch for the <see cref="IGpuHalfPrecisionBackend"/>
/// capability interface, so the backend-agnostic
/// <c>DirectGpuTensorEngine.MatrixMultiply</c> FP16 path (issue #560) runs on
/// WebGPU devices instead of falling back to the scalar CPU loop.
/// <para>
/// WebGPU has no native 16-bit storage by default — the optional
/// <c>shader-f16</c> feature is not used here, and all WebGPU buffers are f32.
/// This backend therefore models FP16 as f32 with a truncated 10-bit mantissa
/// (see <see cref="ConvertToFp16"/>). Under that model an FP16 GEMM is exactly
/// the existing real WGSL f32 GEMM run on the truncated-precision inputs: the
/// multiply sees FP16-precision operands, the accumulation is full FP32, and the
/// f32 output flows on to downstream ops — which is precisely the
/// <see cref="GemmFp16In32fOut"/> (mixed-precision) contract. No separate FP16
/// kernel is needed because the data is already f32.
/// </para>
/// </summary>
public sealed partial class WebGpuBackend : IGpuHalfPrecisionBackend
{
    /// <inheritdoc/>
    /// <remarks>
    /// True whenever the backend is initialized: the real WGSL GEMM that backs
    /// the FP16 path is always present (WebGPU has no CPU-fallback GEMM), so the
    /// only gate is device availability.
    /// </remarks>
    public bool SupportsHgemm => IsAvailable;

    /// <inheritdoc/>
    /// <remarks>The fused backward routes through the real WGSL transposed GEMMs (always present on WebGPU,
    /// which has no CPU-fallback GEMM), so it tracks device availability like <see cref="SupportsHgemm"/>.</remarks>
    public bool SupportsFp16FusedBackward => IsAvailable;

    /// <inheritdoc/>
    /// <remarks>
    /// WebGPU models FP16 as f32 (truncated mantissa — see class remarks), so the inputs already carry FP16
    /// precision and the backward is two real f32 transposed GEMMs with no materialized transpose:
    /// gradA[M,K] = gradC·Bᵀ via the rhs-transposed kernel (<see cref="MatMulTransposed"/>) and gradB[K,N] = Aᵀ·gradC
    /// via the lhs-transposed kernel (<see cref="MatMulLhsTransposedAsync"/>). FP16-output grads run into an f32
    /// scratch then <c>ConvertToFp16</c> (mirrors <see cref="Hgemm"/>); the accumulate is f32 throughout.
    /// </remarks>
    public void MatMulBackwardFp16Fused(
        IGpuBuffer gradCFp16, IGpuBuffer aFp16, IGpuBuffer bFp16,
        IGpuBuffer gradAOut, IGpuBuffer gradBOut,
        int m, int n, int k, bool gradOutHalf)
    {
        if (gradCFp16 is null) throw new ArgumentNullException(nameof(gradCFp16));
        if (aFp16 is null) throw new ArgumentNullException(nameof(aFp16));
        if (bFp16 is null) throw new ArgumentNullException(nameof(bFp16));
        if (gradAOut is null) throw new ArgumentNullException(nameof(gradAOut));
        if (gradBOut is null) throw new ArgumentNullException(nameof(gradBOut));
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");

        // gradA[M,K] = gradC[M,N] · Bᵀ  (b stored [K,N] → read transposed as [N,K]).
        DispatchBackward(scratchElems: m * k, gradOutHalf, gradAOut, dest =>
            MatMulTransposed(gradCFp16, bFp16, dest, m, k, n));
        // gradB[K,N] = Aᵀ · gradC  (a stored [M,K] → contraction over leading dim M).
        DispatchBackward(scratchElems: k * n, gradOutHalf, gradBOut, dest =>
            MatMulLhsTransposedAsync(aFp16, gradCFp16, dest, k, n, m).GetAwaiter().GetResult());
    }

    /// <summary>Runs one backward GEMM into <paramref name="gradOut"/> (FP32) or, when
    /// <paramref name="gradOutHalf"/>, into an f32 scratch then packs to FP16 via <c>ConvertToFp16</c>.</summary>
    private void DispatchBackward(int scratchElems, bool gradOutHalf, IGpuBuffer gradOut, Action<IGpuBuffer> gemm)
    {
        if (!gradOutHalf)
        {
            gemm(gradOut);
            return;
        }
        using var scratch = AllocateBuffer(scratchElems);
        gemm(scratch);
        ConvertToFp16(scratch, gradOut, scratchElems);
    }

    /// <inheritdoc/>
    public void GemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
        int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp32, m, n, k);
        // Inputs already carry FP16 precision (f32 storage, truncated mantissa);
        // the WGSL GEMM accumulates in f32 and writes an f32 result.
        Gemm(aFp16, bFp16, cFp32, m, n, k);
    }

    /// <inheritdoc/>
    public void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16,
        int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp16, m, n, k);
        // Accumulate in f32 into a scratch buffer, then round the result down to
        // FP16 precision so the output buffer carries half-precision values
        // (matches cublasHgemm's FP16 output). A separate scratch avoids binding
        // the same buffer as both GEMM output and convert input.
        using var scratch = AllocateBuffer(m * n);
        Gemm(aFp16, bFp16, scratch, m, n, k);
        ConvertToFp16(scratch, cFp16, m * n);
    }

    private static void ValidateHalfGemmArgs(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int m, int n, int k)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (c is null) throw new ArgumentNullException(nameof(c));
        // Report the offending dimension by name (mirrors the CUDA backend).
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");
    }
}

#endif
