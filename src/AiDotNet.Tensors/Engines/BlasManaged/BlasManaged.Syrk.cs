using System;
using System.Runtime.InteropServices;
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
        // The whole n×n product is one [0,0] block of stride n.
        SyrkWriteTriangle(uplo, 0, 0, n, n, alpha, full, n, beta, c, ldc, ops);
    }

    /// <summary>
    /// C[uplo] tile += α·src + β·C over the (i0,j0)-offset bm×bn block, masked to the
    /// requested triangle. float/double take a typed, JIT-vectorizable AXPBY (the generic
    /// <see cref="INumericOperations{T}"/> path is per-element interface-dispatched + boxed);
    /// the per-row triangle bounds replace the old inner "if (inTri) continue" branch.
    /// </summary>
    private static void SyrkWriteTriangle<T>(
        Uplo uplo, int i0, int j0, int bm, int bn, T alpha,
        ReadOnlySpan<T> src, int srcStride, T beta, Span<T> c, int ldc,
        INumericOperations<T> ops) where T : unmanaged
    {
        if (typeof(T) == typeof(float))
            SyrkWriteTriangleFloat(uplo, i0, j0, bm, bn, (float)(object)alpha,
                MemoryMarshal.Cast<T, float>(src), srcStride, (float)(object)beta,
                MemoryMarshal.Cast<T, float>(c), ldc);
        else if (typeof(T) == typeof(double))
            SyrkWriteTriangleDouble(uplo, i0, j0, bm, bn, (double)(object)alpha,
                MemoryMarshal.Cast<T, double>(src), srcStride, (double)(object)beta,
                MemoryMarshal.Cast<T, double>(c), ldc);
        else
            for (int ii = 0; ii < bm; ii++)
            {
                int gi = i0 + ii;
                int lo = uplo == Uplo.Lower ? 0 : Math.Max(gi - j0, 0);
                int hi = uplo == Uplo.Lower ? Math.Min(gi - j0, bn - 1) : bn - 1;
                int sr = ii * srcStride, cr = gi * ldc;
                for (int jj = lo; jj <= hi; jj++)
                {
                    int ci = cr + j0 + jj;
                    c[ci] = ops.Add(ops.Multiply(alpha, src[sr + jj]), ops.Multiply(beta, c[ci]));
                }
            }
    }

    private static void SyrkWriteTriangleFloat(
        Uplo uplo, int i0, int j0, int bm, int bn, float alpha,
        ReadOnlySpan<float> src, int srcStride, float beta, Span<float> c, int ldc)
    {
        for (int ii = 0; ii < bm; ii++)
        {
            int gi = i0 + ii;
            int lo = uplo == Uplo.Lower ? 0 : Math.Max(gi - j0, 0);
            int hi = uplo == Uplo.Lower ? Math.Min(gi - j0, bn - 1) : bn - 1;
            int sr = ii * srcStride, cr = gi * ldc + j0;
            for (int jj = lo; jj <= hi; jj++)
                c[cr + jj] = alpha * src[sr + jj] + beta * c[cr + jj];
        }
    }

    private static void SyrkWriteTriangleDouble(
        Uplo uplo, int i0, int j0, int bm, int bn, double alpha,
        ReadOnlySpan<double> src, int srcStride, double beta, Span<double> c, int ldc)
    {
        for (int ii = 0; ii < bm; ii++)
        {
            int gi = i0 + ii;
            int lo = uplo == Uplo.Lower ? 0 : Math.Max(gi - j0, 0);
            int hi = uplo == Uplo.Lower ? Math.Min(gi - j0, bn - 1) : bn - 1;
            int sr = ii * srcStride, cr = gi * ldc + j0;
            for (int jj = lo; jj <= hi; jj++)
                c[cr + jj] = alpha * src[sr + jj] + beta * c[cr + jj];
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
                SyrkWriteTriangle(uplo, i0, j0, bm, bn, alpha, tile, bn, beta, c, ldc, ops);
            }
        }
    }
}
