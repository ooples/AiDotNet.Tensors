using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Triangular solve. Computes op(A)·X = α·B (Left) or X·op(A) = α·B (Right),
    /// overwriting B with X. A is the <paramref name="uplo"/> triangle of an
    /// m×m (Left) or n×n (Right) matrix; op(A) is A or Aᵀ per <paramref name="transA"/>.
    /// Drop-in for cblas_strsm/cblas_dtrsm (row-major; order is fixed RowMajor).
    /// </summary>
    public static void Trsm<T>(
        Side side, Uplo uplo, bool transA, Diag diag,
        int m, int n, T alpha,
        ReadOnlySpan<T> a, int lda,
        Span<T> b, int ldb,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m <= 0 || n <= 0) return;
        var ops = MathHelper.GetNumericOperations<T>();

        // Scale B by alpha (BLAS semantics: solve against alpha·B).
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                int idx = i * ldb + j;
                b[idx] = ops.Multiply(b[idx], alpha);
            }

        if (side == Side.Left)
            TrsmLeftScalar(uplo, transA, diag, m, n, a, lda, b, ldb, ops);
        else
            TrsmRightScalar(uplo, transA, diag, m, n, a, lda, b, ldb, ops);
    }

    // A(r,c) honoring transpose. Inlined as a static helper (local functions
    // cannot capture the ref-like ReadOnlySpan<T> parameter).
    private static T At<T>(ReadOnlySpan<T> a, int lda, bool transA, int r, int c) where T : unmanaged
        => transA ? a[c * lda + r] : a[r * lda + c];

    private static void TrsmLeftScalar<T>(
        Uplo uplo, bool transA, Diag diag, int m, int n,
        ReadOnlySpan<T> a, int lda, Span<T> b, int ldb,
        INumericOperations<T> ops) where T : unmanaged
    {
        // Effective triangle after optional transpose.
        bool lower = (uplo == Uplo.Lower) ^ transA;

        if (lower)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    T sum = b[i * ldb + j];
                    for (int kk = 0; kk < i; kk++)
                        sum = ops.Subtract(sum, ops.Multiply(At(a, lda, transA, i, kk), b[kk * ldb + j]));
                    b[i * ldb + j] = diag == Diag.Unit ? sum : ops.Divide(sum, At(a, lda, transA, i, i));
                }
        }
        else
        {
            for (int i = m - 1; i >= 0; i--)
                for (int j = 0; j < n; j++)
                {
                    T sum = b[i * ldb + j];
                    for (int kk = i + 1; kk < m; kk++)
                        sum = ops.Subtract(sum, ops.Multiply(At(a, lda, transA, i, kk), b[kk * ldb + j]));
                    b[i * ldb + j] = diag == Diag.Unit ? sum : ops.Divide(sum, At(a, lda, transA, i, i));
                }
        }
    }

    private static void TrsmRightScalar<T>(
        Uplo uplo, bool transA, Diag diag, int m, int n,
        ReadOnlySpan<T> a, int lda, Span<T> b, int ldb,
        INumericOperations<T> ops) where T : unmanaged
    {
        // Right solve X·op(A) = B with A n×n. Effective triangle after transpose.
        bool lower = (uplo == Uplo.Lower) ^ transA;

        if (!lower)
        {
            // X·U = B : columns left-to-right.
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                {
                    T sum = b[i * ldb + j];
                    for (int kk = 0; kk < j; kk++)
                        sum = ops.Subtract(sum, ops.Multiply(b[i * ldb + kk], At(a, lda, transA, kk, j)));
                    b[i * ldb + j] = diag == Diag.Unit ? sum : ops.Divide(sum, At(a, lda, transA, j, j));
                }
        }
        else
        {
            // X·L = B : columns right-to-left.
            for (int j = n - 1; j >= 0; j--)
                for (int i = 0; i < m; i++)
                {
                    T sum = b[i * ldb + j];
                    for (int kk = j + 1; kk < n; kk++)
                        sum = ops.Subtract(sum, ops.Multiply(b[i * ldb + kk], At(a, lda, transA, kk, j)));
                    b[i * ldb + j] = diag == Diag.Unit ? sum : ops.Divide(sum, At(a, lda, transA, j, j));
                }
        }
    }
}
