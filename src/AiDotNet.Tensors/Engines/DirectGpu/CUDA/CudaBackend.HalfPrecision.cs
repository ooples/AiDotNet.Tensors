// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Issue #337: half-precision GEMM dispatch for the
/// <see cref="IGpuHalfPrecisionBackend"/> capability interface.
/// <para>
/// Two paths:
/// </para>
/// <list type="bullet">
/// <item><see cref="Hgemm"/> — pure FP16 multiply, FP16 accumulate.
/// Cheapest path; precision can drift on long matmul chains so it's
/// reserved for forward-only inference where ~3 ULP error is fine.</item>
/// <item><see cref="GemmFp16In32fOut"/> — FP16 inputs, FP32 accumulate.
/// The AMP standard: matmul reads compressed FP16 from memory (half
/// the bandwidth) but accumulates at full FP32 precision. Compatible
/// with AutocastScope and lossless on training loss curves.</item>
/// </list>
/// </summary>
public sealed partial class CudaBackend
{
    /// <inheritdoc/>
    public bool SupportsHgemm => IsAvailable && _ccMajor >= 5;

    /// <inheritdoc/>
    public unsafe void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16,
        int m, int n, int k)
    {
        if (!SupportsHgemm)
            throw new NotSupportedException(
                "Hgemm requires CUDA compute capability >= 5.0 (Maxwell+).");
        if (aFp16 is null) throw new ArgumentNullException(nameof(aFp16));
        if (bFp16 is null) throw new ArgumentNullException(nameof(bFp16));
        if (cFp16 is null) throw new ArgumentNullException(nameof(cFp16));
        // PR #346 review: report the offending dimension by name so
        // failures are diagnosable. Reporting nameof(m) for every case
        // sent debugging down the wrong path.
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");

        using var _ = PushContext();
        // alpha=1, beta=0 as FP16 bit patterns.
        // 1.0 in IEEE 754 binary16 = 0x3C00. 0.0 = 0x0000.
        ushort alpha = 0x3C00;
        ushort beta = 0x0000;

        // cuBLAS is column-major. To compute row-major C = A · B we
        // dispatch as C^T = B^T · A^T → cublas args (B, A, B->ldb=n,
        // A->lda=k, C->ldc=n) with swapped m/n.
        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasHgemm(
                _cublasHandle,
                CublasOperation.None, CublasOperation.None,
                n, m, k,
                ref alpha,
                bFp16.Handle, n,
                aFp16.Handle, k,
                ref beta,
                cFp16.Handle, n),
            "cublasHgemm");
    }

    /// <inheritdoc/>
    public unsafe void GemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
        int m, int n, int k)
    {
        if (!SupportsHgemm)
            throw new NotSupportedException(
                "GemmFp16In32fOut requires CUDA compute capability >= 5.0 (Maxwell+).");
        if (aFp16 is null) throw new ArgumentNullException(nameof(aFp16));
        if (bFp16 is null) throw new ArgumentNullException(nameof(bFp16));
        if (cFp32 is null) throw new ArgumentNullException(nameof(cFp32));
        // PR #346 review: report the offending dimension by name so
        // failures are diagnosable. Reporting nameof(m) for every case
        // sent debugging down the wrong path.
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");

        using var _ = PushContext();
        // GemmEx needs alpha/beta as POINTERS to scalars in the
        // accumulator type — FP32 here (computeType = COMPUTE_32F).
        float alphaF = 1.0f, betaF = 0.0f;
        IntPtr alphaPtr = (IntPtr)(&alphaF);
        IntPtr betaPtr = (IntPtr)(&betaF);

        // Same row-major-via-col-major swap pattern as Hgemm.
        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasGemmEx(
                _cublasHandle,
                CublasOperation.None, CublasOperation.None,
                n, m, k,
                alphaPtr,
                bFp16.Handle, CuBlasNative.CUDA_R_16F, n,
                aFp16.Handle, CuBlasNative.CUDA_R_16F, k,
                betaPtr,
                cFp32.Handle, CuBlasNative.CUDA_R_32F, n,
                CuBlasNative.CUBLAS_COMPUTE_32F,
                0 /* CUBLAS_GEMM_DEFAULT — let cuBLAS pick algorithm */),
            "cublasGemmEx(FP16 in / FP32 out)");
    }
}
