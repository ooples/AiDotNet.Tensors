using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// Row-major C[M,N] = alpha · A[K,M]ᵀ · B[K,N] + beta · C: the HIP port of CudaBackend.MatMulTransposedA.
public sealed partial class HipBackend : ITransposedAGemm
{
    public void MatMulTransposedA(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        TransposedAGemmArgs.Validate(A, B, C, M, N, K);
        if (!_hipblasAvailable || _hipblasHandle == IntPtr.Zero)
        {
            // No vendor BLAS: materialize Aᵀ [M,K] and run the regular GEMM (same result, one more launch).
            using var aT = TransposeBufferRowMajor(A, K, M);
            Gemm(aT, B, C, M, N, K, alpha, beta);
            return;
        }
        float alphaVal = alpha;
        float betaVal = beta;
        // Column-major views of the row-major buffers, as in the CUDA path:
        //   A_row[K,M] = A_col[M,K] (ld M), B_row[K,N] = B_col[N,K] (ld N), C_row[M,N] = C_col[N,M] (ld N).
        //   C_col[N,M] = B_col[N,K] · A_col[M,K]ᵀ  =>  op(B) = None, op(A) = Transpose, m = N, n = M, k = K.
        var status = HipBlasNative.hipblasSgemm(
            _hipblasHandle,
            HipBlasNative.HipBlasOperation.None,
            HipBlasNative.HipBlasOperation.Transpose,
            N, M, K,
            ref alphaVal,
            ((HipGpuBuffer)B).Handle, N,
            ((HipGpuBuffer)A).Handle, M,
            ref betaVal,
            ((HipGpuBuffer)C).Handle, N);
        if (status != HipBlasNative.HipBlasStatus.Success)
            throw new InvalidOperationException($"hipblasSgemm(MatMulTransposedA) failed: {status}");
        // Match Gemm's synchronous contract: C is ready on return.
        HipNativeBindings.CheckError(HipNativeBindings.hipStreamSynchronize(_stream), "hipStreamSynchronize (hipBLAS MatMulTransposedA)");
    }
}