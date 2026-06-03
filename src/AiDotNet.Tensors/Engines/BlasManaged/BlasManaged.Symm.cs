using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Symmetric matrix multiply. Side.Left: C = α·A·B + β·C (A is m×m symmetric).
    /// Side.Right: C = α·B·A + β·C (A is n×n symmetric). A is stored in the
    /// <paramref name="uplo"/> triangle. Drop-in for cblas_ssymm/cblas_dsymm.
    /// </summary>
    public static void Symm<T>(
        Side side, Uplo uplo,
        int m, int n, T alpha,
        ReadOnlySpan<T> a, int lda,
        ReadOnlySpan<T> b, int ldb, T beta,
        Span<T> c, int ldc,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m < 0) throw new ArgumentOutOfRangeException(nameof(m), "m must be non-negative.");
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "n must be non-negative.");
        if (m == 0 || n == 0) return;
        if (side is not Side.Left and not Side.Right)
            throw new ArgumentOutOfRangeException(nameof(side));
        if (uplo is not Uplo.Lower and not Uplo.Upper)
            throw new ArgumentOutOfRangeException(nameof(uplo));

        var ops = MathHelper.GetNumericOperations<T>();
        int s = side == Side.Left ? m : n;   // dimension of square symmetric A
        if (lda < s) throw new ArgumentOutOfRangeException(nameof(lda), "lda must be >= the symmetric matrix dimension.");
        if (ldb < n) throw new ArgumentOutOfRangeException(nameof(ldb), "ldb must be >= n.");
        if (ldc < n) throw new ArgumentOutOfRangeException(nameof(ldc), "ldc must be >= n.");
        if (a.Length < (long)(s - 1) * lda + s)
            throw new ArgumentException("Span 'a' is too short for the symmetric matrix storage.", nameof(a));
        if (b.Length < (long)(m - 1) * ldb + n)
            throw new ArgumentException("Span 'b' is too short.", nameof(b));
        if (c.Length < (long)(m - 1) * ldc + n)
            throw new ArgumentException("Span 'c' is too short.", nameof(c));

        // Materialize full symmetric A (s×s) from the uplo triangle (mirror).
        T[] full = new T[s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
            {
                bool stored = uplo == Uplo.Lower ? j <= i : j >= i;
                full[i * s + j] = stored ? a[i * lda + j] : a[j * lda + i];
            }

        // result = A·B (Left, m×n) or B·A (Right, m×n) via the existing GEMM core.
        T[] result = new T[m * n];
        var gemmOpts = new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode };
        if (side == Side.Left)
            Gemm<T>(full, s, false, b, ldb, false, result, n, m, n, s, gemmOpts); // (m×m)(m×n)
        else
            Gemm<T>(b, ldb, false, full, s, false, result, n, m, n, s, gemmOpts); // (m×n)(n×n)

        // C = α·result + β·C. float/double take a typed, JIT-vectorizable per-row AXPBY;
        // the generic INumericOperations path is per-element interface-dispatched + boxed.
        if (typeof(T) == typeof(float))
            SymmEpilogueFloat(m, n, (float)(object)alpha, MemoryMarshal.Cast<T, float>(result),
                (float)(object)beta, MemoryMarshal.Cast<T, float>(c), ldc);
        else if (typeof(T) == typeof(double))
            SymmEpilogueDouble(m, n, (double)(object)alpha, MemoryMarshal.Cast<T, double>(result),
                (double)(object)beta, MemoryMarshal.Cast<T, double>(c), ldc);
        else
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    int ci = i * ldc + j;
                    c[ci] = ops.Add(ops.Multiply(alpha, result[i * n + j]), ops.Multiply(beta, c[ci]));
                }
    }

    private static void SymmEpilogueFloat(
        int m, int n, float alpha, ReadOnlySpan<float> result, float beta, Span<float> c, int ldc)
    {
        for (int i = 0; i < m; i++)
        {
            var r = result.Slice(i * n, n);
            var cc = c.Slice(i * ldc, n);
            for (int j = 0; j < n; j++)
                cc[j] = alpha * r[j] + beta * cc[j];
        }
    }

    private static void SymmEpilogueDouble(
        int m, int n, double alpha, ReadOnlySpan<double> result, double beta, Span<double> c, int ldc)
    {
        for (int i = 0; i < m; i++)
        {
            var r = result.Slice(i * n, n);
            var cc = c.Slice(i * ldc, n);
            for (int j = 0; j < n; j++)
                cc[j] = alpha * r[j] + beta * cc[j];
        }
    }
}
