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
}
