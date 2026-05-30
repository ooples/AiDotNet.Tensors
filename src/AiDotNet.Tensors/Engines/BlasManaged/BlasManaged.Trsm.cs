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
        {
            if (m > TrsmBlock)
                TrsmLeftBlocked(uplo, transA, diag, m, n, a, lda, b, ldb, options, ops);
            else
                TrsmLeftScalar(uplo, transA, diag, m, n, a, lda, b, ldb, ops);
        }
        else
            TrsmRightScalar(uplo, transA, diag, m, n, a, lda, b, ldb, ops);
    }

    // Block size for the diagonal-solve / trailing-update split. 64 keeps the
    // diagonal block in L1 while giving the GEMM macrokernel a worthwhile panel.
    private const int TrsmBlock = 64;

    /// <summary>
    /// Blocked Left triangular solve for all four (uplo × transA) combinations.
    /// Each diagonal block is solved with the scalar kernel; the trailing block
    /// rows are updated via the GEMM macrokernel (B -= op(A)_offdiag · X_block).
    /// Effective-lower (forward, top→bottom) updates rows below; effective-upper
    /// (backward, bottom→top) updates rows above.
    /// </summary>
    private static void TrsmLeftBlocked<T>(
        Uplo uplo, bool transA, Diag diag, int m, int n,
        ReadOnlySpan<T> a, int lda, Span<T> b, int ldb,
        in BlasOptions<T> options,
        INumericOperations<T> ops) where T : unmanaged
    {
        // Effective triangle after transpose decides substitution direction.
        bool lower = (uplo == Uplo.Lower) ^ transA;
        var gemmOpts = new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode };

        if (lower)
        {
            for (int i0 = 0; i0 < m; i0 += TrsmBlock)
            {
                int bm = Math.Min(TrsmBlock, m - i0);
                TrsmLeftScalar(uplo, transA, diag, bm, n, a.Slice(i0 * lda + i0), lda, b.Slice(i0 * ldb), ldb, ops);
                // Update rows below: [i0+bm, m).
                int rlo = i0 + bm, rhi = m;
                if (rhi > rlo)
                    TrsmTrailingUpdate(transA, rlo, rhi, i0, bm, n, a, lda, b, ldb, gemmOpts, ops);
            }
        }
        else
        {
            // Backward: process diagonal blocks bottom→top.
            int nb = (m + TrsmBlock - 1) / TrsmBlock;
            for (int blk = nb - 1; blk >= 0; blk--)
            {
                int i0 = blk * TrsmBlock;
                int bm = Math.Min(TrsmBlock, m - i0);
                TrsmLeftScalar(uplo, transA, diag, bm, n, a.Slice(i0 * lda + i0), lda, b.Slice(i0 * ldb), ldb, ops);
                // Update rows above: [0, i0).
                if (i0 > 0)
                    TrsmTrailingUpdate(transA, 0, i0, i0, bm, n, a, lda, b, ldb, gemmOpts, ops);
            }
        }
    }

    /// <summary>
    /// B[rlo:rhi, :] -= op(A)[rlo:rhi, i0:i0+bm] · X[i0:i0+bm, :], via the GEMM macrokernel.
    /// For transA the off-diagonal panel of op(A) is the transpose of A[i0:i0+bm, rlo:rhi].
    /// </summary>
    private static void TrsmTrailingUpdate<T>(
        bool transA, int rlo, int rhi, int i0, int bm, int n,
        ReadOnlySpan<T> a, int lda, Span<T> b, int ldb,
        in BlasOptions<T> gemmOpts, INumericOperations<T> ops) where T : unmanaged
    {
        int rRows = rhi - rlo;
        T[] scratch = new T[rRows * n];
        var xBlk = b.Slice(i0 * ldb); // bm × n (just solved)
        if (!transA)
            // op(A)[rlo:rhi, i0:i0+bm] = A[rlo:rhi, i0:i0+bm] (rRows × bm, NoTrans).
            Gemm<T>(a.Slice(rlo * lda + i0), lda, false, xBlk, ldb, false, scratch, n, rRows, n, bm, gemmOpts);
        else
            // op(A)[rlo:rhi, i0:i0+bm] = (A[i0:i0+bm, rlo:rhi])ᵀ — pass that block with transA=true.
            Gemm<T>(a.Slice(i0 * lda + rlo), lda, true, xBlk, ldb, false, scratch, n, rRows, n, bm, gemmOpts);

        var bRem = b.Slice(rlo * ldb);
        for (int r = 0; r < rRows; r++)
            for (int c = 0; c < n; c++)
            {
                int bi = r * ldb + c;
                bRem[bi] = ops.Subtract(bRem[bi], scratch[r * n + c]);
            }
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
