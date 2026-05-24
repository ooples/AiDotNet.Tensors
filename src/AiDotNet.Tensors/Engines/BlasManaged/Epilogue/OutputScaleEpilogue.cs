using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Multiplies every cell of C by a scalar: <c>C[i, j] *= scale</c>.
/// </summary>
internal static class OutputScaleEpilogue
{
    public static void Apply<T>(Span<T> c, int ldc, int m, int n, T scale)
        where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            var cd = MemoryMarshal.Cast<T, double>(c);
            double s = (double)(object)scale;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    cd[i * ldc + j] *= s;
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var cf = MemoryMarshal.Cast<T, float>(c);
            float s = (float)(object)scale;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    cf[i * ldc + j] *= s;
            return;
        }
        throw new NotSupportedException($"OutputScaleEpilogue does not support T={typeof(T).Name}.");
    }
}
