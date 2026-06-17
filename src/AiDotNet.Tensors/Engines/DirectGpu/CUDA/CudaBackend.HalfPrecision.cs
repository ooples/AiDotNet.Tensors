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

    /// <summary>
    /// Fused FP16 matmul BACKWARD for the forward <c>C[M,N] = A[M,K] · B[K,N]</c>. Computes both gradients in
    /// two Tensor-Core <c>cublasGemmEx</c> launches (FP16 inputs, FP32 accumulate + FP32 output) directly from
    /// the FP16-resident activations — with NO materialized transpose and NO FP32 up-cast scratch:
    /// <list type="bullet">
    /// <item><c>gradA[M,K] = gradC[M,N] · Bᵀ</c></item>
    /// <item><c>gradB[K,N] = Aᵀ · gradC[M,N]</c></item>
    /// </list>
    /// This replaces the eager backward's two <c>TensorTranspose</c> + two <c>TensorMatMul</c> dispatches (four
    /// FP32 device allocations per matmul backward — the measured dominant cost of the FP16 hetero path, see
    /// Fp16ActivationDominated_ReducesGpuPeakBytes) with two transpose-free Half GEMMs into caller-owned
    /// (reused) output buffers. cuBLAS is column-major, so each gradient is derived via the same
    /// memory-equivalence transposition as <see cref="MatMulTransposed"/>:
    /// <para>gradA_col[K,M] = Bᵀ_col · gradC_col → (op=T on B, op=N on gradC; m=K,n=M,k=N; ldB=N,ldgradC=N,ldgradA=K).</para>
    /// <para>gradB_col[N,K] = gradC_col · Aᵀ_col → (op=N on gradC, op=T on A; m=N,n=K,k=M; ldgradC=N,ldA=K,ldgradB=N).</para>
    /// All buffers are device-resident; the caller owns allocation + stream (capture-safe: no alloc/free here).
    /// </summary>
    /// <param name="gradCHalf">FP16 upstream gradient, shape [M,N].</param>
    /// <param name="aHalf">FP16 forward input A, shape [M,K].</param>
    /// <param name="bHalf">FP16 forward input B, shape [K,N].</param>
    /// <param name="gradAOut">Output buffer for gradA, shape [M,K] (caller-owned, reused). FP32 or FP16 per <paramref name="gradOutHalf"/>.</param>
    /// <param name="gradBOut">Output buffer for gradB, shape [K,N] (caller-owned, reused). FP32 or FP16 per <paramref name="gradOutHalf"/>.</param>
    /// <param name="gradOutHalf">When true the gradient outputs are FP16 (CUDA_R_16F) — the dtype the FP16
    /// hetero backward needs so the grads flow on as Half with no FP32 down-cast scratch; the accumulate stays
    /// FP32 (COMPUTE_32F) either way, so only the stored output is Half. When false the outputs are FP32.</param>
    public unsafe void MatMulBackwardFp16Fused(
        IGpuBuffer gradCHalf, IGpuBuffer aHalf, IGpuBuffer bHalf,
        IGpuBuffer gradAOut, IGpuBuffer gradBOut,
        int M, int N, int K, bool gradOutHalf = false)
    {
        if (!SupportsHgemm)
            throw new NotSupportedException("MatMulBackwardFp16Fused requires CUDA compute capability >= 5.0 (Maxwell+).");
        if (gradCHalf is null) throw new ArgumentNullException(nameof(gradCHalf));
        if (aHalf is null) throw new ArgumentNullException(nameof(aHalf));
        if (bHalf is null) throw new ArgumentNullException(nameof(bHalf));
        if (gradAOut is null) throw new ArgumentNullException(nameof(gradAOut));
        if (gradBOut is null) throw new ArgumentNullException(nameof(gradBOut));
        if (M <= 0) throw new ArgumentOutOfRangeException(nameof(M), "Dimensions must be positive.");
        if (N <= 0) throw new ArgumentOutOfRangeException(nameof(N), "Dimensions must be positive.");
        if (K <= 0) throw new ArgumentOutOfRangeException(nameof(K), "Dimensions must be positive.");
        int outType = gradOutHalf ? CuBlasNative.CUDA_R_16F : CuBlasNative.CUDA_R_32F;
        long outElemBytes = gradOutHalf ? 2 : sizeof(float);
        // Byte-aware buffer-size guards (half = 2 bytes, fp32 = 4) — the element-count view would misread.
        if (gradCHalf.SizeInBytes < (long)M * N * 2) throw new ArgumentException($"gradC half buffer too small: {gradCHalf.SizeInBytes} < {(long)M * N * 2}.");
        if (aHalf.SizeInBytes < (long)M * K * 2) throw new ArgumentException($"A half buffer too small: {aHalf.SizeInBytes} < {(long)M * K * 2}.");
        if (bHalf.SizeInBytes < (long)K * N * 2) throw new ArgumentException($"B half buffer too small: {bHalf.SizeInBytes} < {(long)K * N * 2}.");
        if (gradAOut.SizeInBytes < (long)M * K * outElemBytes) throw new ArgumentException($"gradA buffer too small: {gradAOut.SizeInBytes} < {(long)M * K * outElemBytes}.");
        if (gradBOut.SizeInBytes < (long)K * N * outElemBytes) throw new ArgumentException($"gradB buffer too small: {gradBOut.SizeInBytes} < {(long)K * N * outElemBytes}.");

        using var _ = PushContext();
        float alphaF = 1.0f, betaF = 0.0f;
        IntPtr alphaPtr = (IntPtr)(&alphaF);
        IntPtr betaPtr = (IntPtr)(&betaF);

        // gradA[M,K] = gradC · Bᵀ  →  col-major gradA[K,M] = Bᵀ_col · gradC_col.
        var sA = CuBlasNative.cublasGemmEx(
            _cublasHandle,
            CublasOperation.Transpose, CublasOperation.None,
            K, M, N,
            alphaPtr,
            bHalf.Handle, CuBlasNative.CUDA_R_16F, N,
            gradCHalf.Handle, CuBlasNative.CUDA_R_16F, N,
            betaPtr,
            gradAOut.Handle, outType, K,
            CuBlasNative.CUBLAS_COMPUTE_32F, 0);
        if (sA == AiDotNet.Tensors.Engines.CublasStatus.NotSupported)
            throw new NotSupportedException($"FP16 Tensor-Core GEMM not supported on this device (cc {_ccMajor}.{_ccMinor}).");
        CuBlasNative.CheckCublasStatus(sA, "cublasGemmEx(MatMulBackwardFp16Fused gradA)");

        // gradB[K,N] = Aᵀ · gradC  →  col-major gradB[N,K] = gradC_col · Aᵀ_col.
        var sB = CuBlasNative.cublasGemmEx(
            _cublasHandle,
            CublasOperation.None, CublasOperation.Transpose,
            N, K, M,
            alphaPtr,
            gradCHalf.Handle, CuBlasNative.CUDA_R_16F, N,
            aHalf.Handle, CuBlasNative.CUDA_R_16F, K,
            betaPtr,
            gradBOut.Handle, outType, N,
            CuBlasNative.CUBLAS_COMPUTE_32F, 0);
        if (sB == AiDotNet.Tensors.Engines.CublasStatus.NotSupported)
            throw new NotSupportedException($"FP16 Tensor-Core GEMM not supported on this device (cc {_ccMajor}.{_ccMinor}).");
        CuBlasNative.CheckCublasStatus(sB, "cublasGemmEx(MatMulBackwardFp16Fused gradB)");
    }
}
