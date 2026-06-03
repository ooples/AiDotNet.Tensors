using System;
using System.Runtime.InteropServices;
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
        if (kl < 0 || ku < 0)
            throw new ArgumentOutOfRangeException(nameof(kl), "kl and ku must be non-negative.");
        if (incx <= 0 || incy <= 0)
            throw new ArgumentOutOfRangeException(nameof(incx), "incx and incy must be positive.");
        if (lda < kl + ku + 1)
            throw new ArgumentOutOfRangeException(nameof(lda), "lda must be >= kl + ku + 1.");

        int lenX = transA ? m : n;   // length of x (column index range)
        int lenY = transA ? n : m;   // length of y (row index range)
        long needA = (long)lda * n;
        long needX = 1L + (long)(lenX - 1) * incx;
        long needY = 1L + (long)(lenY - 1) * incy;
        if (a.Length < needA)
            throw new ArgumentException("Span 'a' is too short for lda*n band storage.", nameof(a));
        if (x.Length < needX)
            throw new ArgumentException("Span 'x' is too short for the given length/incx.", nameof(x));
        if (y.Length < needY)
            throw new ArgumentException("Span 'y' is too short for the given length/incy.", nameof(y));

        var ops = MathHelper.GetNumericOperations<T>();

        // float/double take a typed band matvec (typed AXPY accumulation over the band +
        // typed β-scale); the generic INumericOperations path is kept for other types.
        if (typeof(T) == typeof(float))
        {
            GbmvFloat(transA, m, n, kl, ku, (float)(object)alpha, MemoryMarshal.Cast<T, float>(a), lda,
                MemoryMarshal.Cast<T, float>(x), incx, (float)(object)beta, MemoryMarshal.Cast<T, float>(y), incy, lenY);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            GbmvDouble(transA, m, n, kl, ku, (double)(object)alpha, MemoryMarshal.Cast<T, double>(a), lda,
                MemoryMarshal.Cast<T, double>(x), incx, (double)(object)beta, MemoryMarshal.Cast<T, double>(y), incy, lenY);
            return;
        }

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

    private static void GbmvFloat(
        bool transA, int m, int n, int kl, int ku, float alpha,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> x, int incx, float beta,
        Span<float> y, int incy, int lenY)
    {
        if (beta != 1f)
            for (int o = 0; o < lenY; o++) y[o * incy] *= beta;
        if (!transA)
            for (int i = 0; i < m; i++)
            {
                int jlo = Math.Max(0, i - kl), jhi = Math.Min(n - 1, i + ku);
                float acc = 0f;
                for (int j = jlo; j <= jhi; j++)
                    acc += a[j * lda + (ku - j + i)] * x[j * incx];
                y[i * incy] += alpha * acc;
            }
        else
            for (int j = 0; j < n; j++)
            {
                int ilo = Math.Max(0, j - ku), ihi = Math.Min(m - 1, j + kl);
                float acc = 0f;
                for (int i = ilo; i <= ihi; i++)
                    acc += a[j * lda + (ku - j + i)] * x[i * incx];
                y[j * incy] += alpha * acc;
            }
    }

    private static void GbmvDouble(
        bool transA, int m, int n, int kl, int ku, double alpha,
        ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, int incx, double beta,
        Span<double> y, int incy, int lenY)
    {
        if (beta != 1.0)
            for (int o = 0; o < lenY; o++) y[o * incy] *= beta;
        if (!transA)
            for (int i = 0; i < m; i++)
            {
                int jlo = Math.Max(0, i - kl), jhi = Math.Min(n - 1, i + ku);
                double acc = 0.0;
                for (int j = jlo; j <= jhi; j++)
                    acc += a[j * lda + (ku - j + i)] * x[j * incx];
                y[i * incy] += alpha * acc;
            }
        else
            for (int j = 0; j < n; j++)
            {
                int ilo = Math.Max(0, j - ku), ihi = Math.Min(m - 1, j + kl);
                double acc = 0.0;
                for (int i = ilo; i <= ihi; i++)
                    acc += a[j * lda + (ku - j + i)] * x[i * incx];
                y[j * incy] += alpha * acc;
            }
    }
}
