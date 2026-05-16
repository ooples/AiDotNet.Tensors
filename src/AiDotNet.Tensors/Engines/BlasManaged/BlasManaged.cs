using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// BLIS-style managed GEMM kernel. Replaces Avx512Sgemm + SimdGemm as the
/// codebase's primary GEMM path. See docs/superpowers/specs/2026-05-16-blas-managed-design.md.
/// </summary>
public static class BlasManaged
{
    /// <summary>
    /// Computes C = alpha*op(A)*op(B) + beta*C where op(X) is X or X^T.
    /// Scalars alpha and beta and an optional fused epilogue (bias, activation, skip,
    /// dropout, output scale) are specified via <see cref="BlasOptions{T}"/>. Defaults
    /// when options are omitted: alpha = 1, beta = 0, no epilogue.
    /// </summary>
    public static void Gemm<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        throw new NotImplementedException("BlasManaged.Gemm: filled in by Phase B.");
    }
}
