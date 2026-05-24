using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Adds a skip-connection tensor element-wise to C: <c>C[i, j] += skip[i, j]</c>.
/// The skip tensor is expected to be (m × n) row-major contiguous (lds = n).
/// </summary>
internal static class SkipEpilogue
{
    public static void Apply<T>(Span<T> c, int ldc, int m, int n, ReadOnlySpan<T> skip)
        where T : unmanaged
    {
        // PR #402 CodeRabbit fix: use long arithmetic so the guard doesn't
        // silently overflow on large shapes (m*n with int can wrap negative
        // for m=n=~50000), and validate c.Length up front so the inner
        // cd[i*ldc + j] indexer can't OOB on a too-small c. The cd/sf cast
        // operates on the live Span so MemoryMarshal.Cast preserves the
        // length but doesn't bounds-check the row-major access pattern.
        if (m < 0 || n < 0)
            throw new ArgumentOutOfRangeException(m < 0 ? nameof(m) : nameof(n),
                "m and n must be non-negative.");
        if (ldc < n)
            throw new ArgumentOutOfRangeException(nameof(ldc), $"ldc ({ldc}) must be >= n ({n}).");

        long requiredSkip = (long)m * n;
        if (requiredSkip > skip.Length)
            throw new ArgumentException(
                $"skip length ({skip.Length}) less than m*n ({requiredSkip}).", nameof(skip));

        // Max index accessed is (m-1)*ldc + (n-1), so c must hold at least
        // (m-1)*ldc + n elements. Zero-area case requires no storage.
        long requiredC = (m == 0 || n == 0) ? 0L : ((long)(m - 1) * ldc + n);
        if (requiredC > c.Length)
            throw new ArgumentException(
                $"c length ({c.Length}) less than required ({requiredC}) for m={m} n={n} ldc={ldc}.",
                nameof(c));

        if (typeof(T) == typeof(double))
        {
            var cd = MemoryMarshal.Cast<T, double>(c);
            var sd = MemoryMarshal.Cast<T, double>(skip);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    cd[i * ldc + j] += sd[i * n + j];
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var cf = MemoryMarshal.Cast<T, float>(c);
            var sf = MemoryMarshal.Cast<T, float>(skip);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    cf[i * ldc + j] += sf[i * n + j];
            return;
        }
        throw new NotSupportedException($"SkipEpilogue does not support T={typeof(T).Name}.");
    }
}
