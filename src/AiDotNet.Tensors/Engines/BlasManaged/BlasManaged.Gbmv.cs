using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Banded matrix-vector multiply: y = α·op(A)·x + β·y, where A is an m×n band
    /// matrix with <paramref name="kl"/> sub- and <paramref name="ku"/> super-diagonals,
    /// stored in the LAPACK band layout (lda ≥ kl+ku+1, column-major band: the logical
    /// element A(i,j) lives at a[j·lda + (ku - j + i)]). Drop-in for cblas_sgbmv/cblas_dgbmv.
    /// </summary>
    public static void Gbmv<T>(
        bool transA, int m, int n, int kl, int ku, T alpha,
        ReadOnlySpan<T> a, int lda,
        ReadOnlySpan<T> x, int incx, T beta,
        Span<T> y, int incy,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m <= 0 || n <= 0) return;
        var ops = MathHelper.GetNumericOperations<T>();

        int lenY = transA ? n : m;
        // Scale y by beta.
        for (int o = 0; o < lenY; o++)
        {
            int yo = o * incy;
            y[yo] = ops.Multiply(beta, y[yo]);
        }

        if (!transA)
        {
            // y[i] += α · Σ_j A(i,j)·x[j], j over the band [max(0,i-kl), min(n-1,i+ku)].
            for (int i = 0; i < m; i++)
            {
                int jlo = Math.Max(0, i - kl);
                int jhi = Math.Min(n - 1, i + ku);
                T acc = ops.Zero;
                for (int j = jlo; j <= jhi; j++)
                {
                    T aval = a[j * lda + (ku - j + i)];
                    acc = ops.Add(acc, ops.Multiply(aval, x[j * incx]));
                }
                int yi = i * incy;
                y[yi] = ops.Add(y[yi], ops.Multiply(alpha, acc));
            }
        }
        else
        {
            // y[j] += α · Σ_i A(i,j)·x[i], i over the band [max(0,j-ku), min(m-1,j+kl)].
            for (int j = 0; j < n; j++)
            {
                int ilo = Math.Max(0, j - ku);
                int ihi = Math.Min(m - 1, j + kl);
                T acc = ops.Zero;
                for (int i = ilo; i <= ihi; i++)
                {
                    T aval = a[j * lda + (ku - j + i)];
                    acc = ops.Add(acc, ops.Multiply(aval, x[i * incx]));
                }
                int yj = j * incy;
                y[yj] = ops.Add(y[yj], ops.Multiply(alpha, acc));
            }
        }
    }
}
