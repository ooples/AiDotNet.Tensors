using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Applies a deterministic dropout mask to C — given a uint seed, generates
/// a bit-mask per element via a per-cell xoshiro-style PRG and zeroes out
/// cells where the mask bit is 0. The dropout rate is fixed at 0.5 for now
/// (50% of elements zeroed); future refinement can parameterize the rate.
///
/// <para>
/// Determinism: same seed produces the same mask regardless of thread count
/// or call ordering. Each (row, col) cell hashes (seed, row, col) into a
/// uint and zeros the cell if the hash &amp; 1 == 0.
/// </para>
/// </summary>
internal static class DropoutEpilogue
{
    public static void Apply<T>(Span<T> c, int ldc, int m, int n, uint seed)
        where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            var cd = MemoryMarshal.Cast<T, double>(c);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    if (HashKeep(seed, (uint)i, (uint)j))
                        cd[i * ldc + j] = 2.0 * cd[i * ldc + j];  // Scale by 1/(1-rate) = 2.
                    else
                        cd[i * ldc + j] = 0.0;
                }
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var cf = MemoryMarshal.Cast<T, float>(c);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    if (HashKeep(seed, (uint)i, (uint)j))
                        cf[i * ldc + j] = 2.0f * cf[i * ldc + j];
                    else
                        cf[i * ldc + j] = 0.0f;
                }
            return;
        }
        throw new NotSupportedException($"DropoutEpilogue does not support T={typeof(T).Name}.");
    }

    private static bool HashKeep(uint seed, uint row, uint col)
    {
        // Simple xorshift-style mixer: deterministic + cheap.
        uint h = seed ^ (row * 0x9E3779B1u) ^ (col * 0x85EBCA77u);
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        // Keep if low bit is 1 (~50% rate).
        return (h & 1u) == 1u;
    }
}
