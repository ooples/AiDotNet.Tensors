using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Adds a bias vector to each row of the C output matrix.
/// <c>C[i, j] += bias[j]</c> for all (i, j) in [0, m) × [0, n).
/// </summary>
internal static class BiasEpilogue
{
    public static void Apply<T>(Span<T> c, int ldc, int m, int n, ReadOnlySpan<T> bias)
        where T : unmanaged
    {
        if (bias.Length < n)
            throw new ArgumentException($"bias length ({bias.Length}) less than n ({n}).", nameof(bias));

        if (typeof(T) == typeof(double))
        {
            var cd = MemoryMarshal.Cast<T, double>(c);
            var bd = MemoryMarshal.Cast<T, double>(bias);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    cd[i * ldc + j] += bd[j];
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var cf = MemoryMarshal.Cast<T, float>(c);
            var bf = MemoryMarshal.Cast<T, float>(bias);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    cf[i * ldc + j] += bf[j];
            return;
        }
        throw new NotSupportedException($"BiasEpilogue does not support T={typeof(T).Name}.");
    }
}
