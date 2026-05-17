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
        if (skip.Length < m * n)
            throw new ArgumentException($"skip length ({skip.Length}) less than m*n ({m * n}).", nameof(skip));

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
