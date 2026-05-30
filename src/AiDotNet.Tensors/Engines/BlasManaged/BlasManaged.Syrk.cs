using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Symmetric rank-k update: C = α·op(A)·op(A)ᵀ + β·C, writing only the
    /// <paramref name="uplo"/> triangle of the n×n matrix C. op(A) is A (trans=false,
    /// A is n×k) or Aᵀ (trans=true, A is k×n). Drop-in for cblas_ssyrk/cblas_dsyrk.
    /// </summary>
    public static void Syrk<T>(
        Uplo uplo, bool trans,
        int n, int k, T alpha,
        ReadOnlySpan<T> a, int lda, T beta,
        Span<T> c, int ldc,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (n <= 0) return;
        var ops = MathHelper.GetNumericOperations<T>();

        // full = op(A)·op(A)ᵀ  (n×n) via the existing GEMM core.
        // trans=false: A(n×k) · A(n×k)ᵀ → Gemm(a=A, transA=false, b=A, transB=true, k=k)
        // trans=true:  Aᵀ(n×k) · A(k×n) → Gemm(a=A, transA=true, b=A, transB=false, k=k)
        T[] full = new T[n * n];
        if (k > 0)
        {
            var gemmOpts = new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode };
            Gemm<T>(a, lda, trans, a, lda, !trans, full, n, n, n, k, gemmOpts);
        }

        // Write C[uplo] = α·full + β·C[uplo]; leave the other triangle untouched.
        for (int i = 0; i < n; i++)
        {
            int lo = uplo == Uplo.Lower ? 0 : i;
            int hi = uplo == Uplo.Lower ? i : n - 1;
            for (int j = lo; j <= hi; j++)
            {
                int ci = i * ldc + j;
                T scaled = ops.Multiply(alpha, full[i * n + j]);
                c[ci] = ops.Add(scaled, ops.Multiply(beta, c[ci]));
            }
        }
    }
}
