// Copyright (c) AiDotNet. All rights reserved.
// IGpuHalfPrecisionBackend implementation for the HIP/ROCm backend (issue #560):
// FP16 GEMM so MatrixMultiply<Half> runs on AMD GPUs instead of dropping to a
// scalar CPU fallback.

using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// Half-precision GEMM dispatch for the <see cref="IGpuHalfPrecisionBackend"/>
/// capability interface, so the backend-agnostic
/// <c>DirectGpuTensorEngine.MatrixMultiply</c> FP16 path (issue #560) runs on
/// AMD ROCm GPUs. Mirrors the CUDA backend's contract and column-major
/// derivation, but through hipBLAS — the cuBLAS-compatible AMD library.
/// <para>
/// Both paths reuse the exact column-major argument ordering the backend's
/// existing <c>hipblasSgemm</c> GEMM uses (row-major <c>C = A·B</c> dispatched
/// as <c>C^T = B^T·A^T</c> with swapped m/n and B before A), so they inherit a
/// proven layout rather than re-deriving it.
/// </para>
/// </summary>
public sealed partial class HipBackend : IGpuHalfPrecisionBackend
{
    /// <inheritdoc/>
    /// <remarks>
    /// True when hipBLAS loaded and a handle was created — both
    /// <see cref="Hgemm"/> (rocBLAS hgemm) and
    /// <see cref="GemmFp16In32fOut"/> (hipblasGemmEx, FP16→FP32) are hipBLAS
    /// calls, so device + library availability is the only gate.
    /// </remarks>
    public bool SupportsHgemm => _hipblasAvailable && _hipblasHandle != IntPtr.Zero;

    /// <inheritdoc/>
    /// <remarks>The fused backward is two <c>hipblasGemmEx</c> launches with transpose flags — same library +
    /// device gate as the forward GEMM, so it tracks <see cref="SupportsHgemm"/>.</remarks>
    public bool SupportsFp16FusedBackward => SupportsHgemm;

    /// <inheritdoc/>
    /// <remarks>
    /// rocBLAS counterpart of <see cref="CUDA.CudaBackend.MatMulBackwardFp16Fused"/>: two transpose-free
    /// <c>hipblasGemmEx</c> launches (FP16 in, FP32 accumulate) with the same column-major derivation hipBLAS
    /// shares with cuBLAS. gradA[M,K] = gradC·Bᵀ → col-major (op=T on B, op=N on gradC; m=K,n=M,k=N);
    /// gradB[K,N] = Aᵀ·gradC → col-major (op=N on gradC, op=T on A; m=N,n=K,k=M).
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
        EnsureHalfGemmSupported();

        var outType = gradOutHalf ? HipBlasNative.HipBlasDatatype.R_16F : HipBlasNative.HipBlasDatatype.R_32F;
        float alpha = 1.0f, beta = 0.0f;

        // gradA[M,K] = gradC · Bᵀ  →  col-major gradA[K,M] = Bᵀ_col · gradC_col.
        var sA = HipBlasNative.hipblasGemmEx(
            _hipblasHandle,
            HipBlasNative.HipBlasOperation.Transpose, HipBlasNative.HipBlasOperation.None,
            k, m, n,
            ref alpha,
            ((HipGpuBuffer)bFp16).Handle, HipBlasNative.HipBlasDatatype.R_16F, n,
            ((HipGpuBuffer)gradCFp16).Handle, HipBlasNative.HipBlasDatatype.R_16F, n,
            ref beta,
            ((HipGpuBuffer)gradAOut).Handle, outType, k,
            HipBlasNative.HipBlasDatatype.R_32F, HipBlasNative.HipBlasGemmAlgo.Default);
        if (sA != HipBlasNative.HipBlasStatus.Success)
            throw new InvalidOperationException($"hipblasGemmEx(MatMulBackwardFp16Fused gradA) failed: {sA}");

        // gradB[K,N] = Aᵀ · gradC  →  col-major gradB[N,K] = gradC_col · Aᵀ_col.
        var sB = HipBlasNative.hipblasGemmEx(
            _hipblasHandle,
            HipBlasNative.HipBlasOperation.None, HipBlasNative.HipBlasOperation.Transpose,
            n, k, m,
            ref alpha,
            ((HipGpuBuffer)gradCFp16).Handle, HipBlasNative.HipBlasDatatype.R_16F, n,
            ((HipGpuBuffer)aFp16).Handle, HipBlasNative.HipBlasDatatype.R_16F, k,
            ref beta,
            ((HipGpuBuffer)gradBOut).Handle, outType, n,
            HipBlasNative.HipBlasDatatype.R_32F, HipBlasNative.HipBlasGemmAlgo.Default);
        if (sB != HipBlasNative.HipBlasStatus.Success)
            throw new InvalidOperationException($"hipblasGemmEx(MatMulBackwardFp16Fused gradB) failed: {sB}");

        SyncHalfGemm();
    }

    /// <inheritdoc/>
    public void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16,
        int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp16, m, n, k);
        EnsureHalfGemmSupported();

        // alpha = 1.0, beta = 0.0 as IEEE-754 binary16 bit patterns.
        ushort alpha = 0x3C00;
        ushort beta = 0x0000;

        var status = HipBlasNative.hipblasHgemm(
            _hipblasHandle,
            HipBlasNative.HipBlasOperation.None,
            HipBlasNative.HipBlasOperation.None,
            n, m, k,
            ref alpha,
            ((HipGpuBuffer)bFp16).Handle, n,
            ((HipGpuBuffer)aFp16).Handle, k,
            ref beta,
            ((HipGpuBuffer)cFp16).Handle, n);
        if (status != HipBlasNative.HipBlasStatus.Success)
            throw new InvalidOperationException($"hipblasHgemm failed: {status}");

        SyncHalfGemm();
    }

    /// <inheritdoc/>
    public void GemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
        int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp32, m, n, k);
        EnsureHalfGemmSupported();

        // FP32 alpha/beta (compute type = FP32). The accumulator is FP32 even
        // though A/B are FP16 — the standard AMP mixed-precision matmul.
        float alpha = 1.0f, beta = 0.0f;

        var status = HipBlasNative.hipblasGemmEx(
            _hipblasHandle,
            HipBlasNative.HipBlasOperation.None,
            HipBlasNative.HipBlasOperation.None,
            n, m, k,
            ref alpha,
            ((HipGpuBuffer)bFp16).Handle, HipBlasNative.HipBlasDatatype.R_16F, n,
            ((HipGpuBuffer)aFp16).Handle, HipBlasNative.HipBlasDatatype.R_16F, k,
            ref beta,
            ((HipGpuBuffer)cFp32).Handle, HipBlasNative.HipBlasDatatype.R_32F, n,
            HipBlasNative.HipBlasDatatype.R_32F,
            HipBlasNative.HipBlasGemmAlgo.Default);
        if (status != HipBlasNative.HipBlasStatus.Success)
            throw new InvalidOperationException($"hipblasGemmEx(FP16 in / FP32 out) failed: {status}");

        SyncHalfGemm();
    }

    private void EnsureHalfGemmSupported()
    {
        if (!SupportsHgemm)
            throw new NotSupportedException(
                "Half-precision GEMM requires hipBLAS, which is not available on this device.");
    }

    private void SyncHalfGemm()
    {
        // Match the backend's synchronous GEMM contract: callers expect the
        // result ready on return, not in-flight on the stream.
        var sync = HipNativeBindings.hipStreamSynchronize(_stream);
        HipNativeBindings.CheckError(sync, "hipStreamSynchronize (hipBLAS half GEMM)");
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
    // #1650/#638 FP16 conv im2col — STUB pending the HIP kernel port (dispatch gates on Fp16Im2colAvailable
    // so Im2colKNFp16 is never invoked while this returns false; the engine keeps the FP32 conv on HIP).
    /// <inheritdoc/>
    public bool Fp16Im2colAvailable => false;
    /// <inheritdoc/>
    public void Im2colKNFp16(IGpuBuffer input, IGpuBuffer outputHalf,
        int batch, int channels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => throw new System.NotSupportedException("FP16 im2col (Im2colKNFp16) is not yet ported to the HIP backend.");

}
