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

        // Tile-skip path for large n: compute only tiles intersecting the requested
        // triangle (~half the FLOPs of a dense GEMM — the real SYRK win).
        if (n > SyrkBlock)
        {
            SyrkBlocked(uplo, trans, n, k, alpha, a, lda, beta, c, ldc, options, ops);
            return;
        }

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

    private const int SyrkBlock = 64;

    private static void SyrkBlocked<T>(
        Uplo uplo, bool trans, int n, int k, T alpha, ReadOnlySpan<T> a, int lda, T beta,
        Span<T> c, int ldc, in BlasOptions<T> options, INumericOperations<T> ops) where T : unmanaged
    {
        var gemmOpts = new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode };
        for (int i0 = 0; i0 < n; i0 += SyrkBlock)
        {
            int bm = Math.Min(SyrkBlock, n - i0);
            for (int j0 = 0; j0 < n; j0 += SyrkBlock)
            {
                int bn = Math.Min(SyrkBlock, n - j0);
                // Skip tiles wholly outside the requested triangle.
                if (uplo == Uplo.Lower) { if (j0 > i0 + bm - 1) continue; }
                else                    { if (j0 + bn - 1 < i0) continue; }

                // tile = op(A)[i0:i0+bm, :] · op(A)[j0:j0+bn, :]ᵀ  (bm×bn)
                T[] tile = new T[bm * bn];
                if (k > 0)
                {
                    if (!trans)
                    {
                        // op(A) row r = a[r*lda + p]; sub-blocks at rows i0.., j0..
                        var aI = a.Slice(i0 * lda);          // bm rows × k, lda-strided
                        var aJ = a.Slice(j0 * lda);          // bn rows × k, lda-strided
                        Gemm<T>(aI, lda, false, aJ, lda, true, tile, bn, bm, bn, k, gemmOpts);
                    }
                    else
                    {
                        // trans=true: op(A) = Aᵀ, A is k×n. tile = op(A)[i0:,:]·op(A)[j0:,:]ᵀ.
                        // op(A)[i0:i0+bm,:] = (A[:, i0:i0+bm])ᵀ → pass A[:, i0:] (k×bm) with transA=true;
                        // op(A)[j0:j0+bn,:]ᵀ = A[:, j0:j0+bn] (k×bn) → pass with transB=false.
                        Gemm<T>(a.Slice(i0), lda, true, a.Slice(j0), lda, false, tile, bn, bm, bn, k, gemmOpts);
                    }
                }

                // Write tile into C[uplo] with alpha/beta, masking the diagonal tile.
                for (int ii = 0; ii < bm; ii++)
                {
                    int gi = i0 + ii;
                    for (int jj = 0; jj < bn; jj++)
                    {
                        int gj = j0 + jj;
                        bool inTri = uplo == Uplo.Lower ? gj <= gi : gj >= gi;
                        if (!inTri) continue;
                        int ci = gi * ldc + gj;
                        c[ci] = ops.Add(ops.Multiply(alpha, tile[ii * bn + jj]), ops.Multiply(beta, c[ci]));
                    }
                }
            }
        }
    }
}
