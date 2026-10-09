using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// Row-major C[M,N] = alpha · A[K,M]ᵀ · B[K,N] + beta · C: the Metal port of CudaBackend.MatMulTransposedA
// (matmul_transposed_a in the resident library).
public sealed partial class MetalBackend : ITransposedAGemm
{
    public void MatMulTransposedA(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        ThrowIfDisposed();
        TransposedAGemmArgs.Validate(A, B, C, M, N, K);
        DispatchResidentMetal("matmul_transposed_a", checked(M * N), new[] { A, B, C },
            (uint)M, (uint)N, (uint)K,
            unchecked((uint)SingleToInt32BitsCompat(alpha)),
            unchecked((uint)SingleToInt32BitsCompat(beta)));
    }
}