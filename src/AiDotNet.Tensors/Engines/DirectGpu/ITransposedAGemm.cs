using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Row-major <c>C[M,N] = alpha · A[K,M]ᵀ · B[K,N] + beta · C</c> without materializing Aᵀ: the weight gradient of a
/// 2-D matmul (<c>dB = Aᵀ · dY</c>). With <see cref="IDirectGpuBackend.MatMulTransposed"/> it lets the engine form both
/// matmul gradients as one GEMM each instead of a transpose kernel and a GEMM per gradient.
/// </summary>
/// <remarks>Implemented by all six GPU backends (CUDA, HIP, Metal, OpenCL, Vulkan, WebGPU).</remarks>
internal interface ITransposedAGemm
{
    void MatMulTransposedA(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);
}

/// <summary>Argument checks every <see cref="ITransposedAGemm"/> implementation shares.</summary>
internal static class TransposedAGemmArgs
{
    internal static void Validate(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K)
    {
        if (A is null) throw new ArgumentNullException(nameof(A));
        if (B is null) throw new ArgumentNullException(nameof(B));
        if (C is null) throw new ArgumentNullException(nameof(C));
        if (M <= 0 || N <= 0 || K <= 0)
            throw new ArgumentOutOfRangeException(nameof(M), "Matrix dimensions M, N, K must all be positive.");
        if ((long)A.Size < (long)K * M)
            throw new ArgumentException($"A.Size {A.Size} < K*M = {(long)K * M}.", nameof(A));
        if ((long)B.Size < (long)K * N)
            throw new ArgumentException($"B.Size {B.Size} < K*N = {(long)K * N}.", nameof(B));
        if ((long)C.Size < (long)M * N)
            throw new ArgumentException($"C.Size {C.Size} < M*N = {(long)M * N}.", nameof(C));
    }
}