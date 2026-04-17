namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Minimal XXHash64 implementation for plan-file integrity checking.
/// Pure C#, no dependencies, works on net471. This is a corruption
/// detector, not a cryptographic hash — XXHash64 is chosen for speed
/// (GB/s throughput) since the check runs on every Load.
/// </summary>
internal static class XXHash64
{
    private const ulong Prime1 = 0x9E3779B185EBCA87UL;
    private const ulong Prime2 = 0x14DEF9DEA2F79CD6UL;
    private const ulong Prime3 = 0x0000000165EBCA87UL;
    private const ulong Prime4 = 0x9E3779B185EBCA87UL;
    private const ulong Prime5 = 0x27D4EB2F165B7D76UL;

    /// <summary>
    /// Computes the XXHash64 of the given byte span.
    /// </summary>
    internal static ulong Compute(byte[] data, int offset, int length, ulong seed = 0)
    {
        ulong hash;
        int remaining = length;
        int index = offset;

        if (length >= 32)
        {
            ulong v1 = seed + Prime1 + Prime2;
            ulong v2 = seed + Prime2;
            ulong v3 = seed;
            ulong v4 = seed - Prime1;

            do
            {
                v1 = Round(v1, ReadU64(data, index));      index += 8;
                v2 = Round(v2, ReadU64(data, index));      index += 8;
                v3 = Round(v3, ReadU64(data, index));      index += 8;
                v4 = Round(v4, ReadU64(data, index));      index += 8;
                remaining -= 32;
            } while (remaining >= 32);

            hash = RotateLeft(v1, 1) + RotateLeft(v2, 7) + RotateLeft(v3, 12) + RotateLeft(v4, 18);
            hash = MergeRound(hash, v1);
            hash = MergeRound(hash, v2);
            hash = MergeRound(hash, v3);
            hash = MergeRound(hash, v4);
        }
        else
        {
            hash = seed + Prime5;
        }

        hash += (ulong)length;

        while (remaining >= 8)
        {
            hash ^= Round(0, ReadU64(data, index));
            hash = RotateLeft(hash, 27) * Prime1 + Prime4;
            index += 8;
            remaining -= 8;
        }

        while (remaining >= 4)
        {
            hash ^= ReadU32(data, index) * Prime1;
            hash = RotateLeft(hash, 23) * Prime2 + Prime3;
            index += 4;
            remaining -= 4;
        }

        while (remaining > 0)
        {
            hash ^= data[index] * Prime5;
            hash = RotateLeft(hash, 11) * Prime1;
            index++;
            remaining--;
        }

        hash ^= hash >> 33;
        hash *= Prime2;
        hash ^= hash >> 29;
        hash *= Prime3;
        hash ^= hash >> 32;

        return hash;
    }

    private static ulong Round(ulong acc, ulong input)
    {
        acc += input * Prime2;
        acc = RotateLeft(acc, 31);
        acc *= Prime1;
        return acc;
    }

    private static ulong MergeRound(ulong acc, ulong val)
    {
        val = Round(0, val);
        acc ^= val;
        acc = acc * Prime1 + Prime4;
        return acc;
    }

    private static ulong RotateLeft(ulong value, int count)
        => (value << count) | (value >> (64 - count));

    private static ulong ReadU64(byte[] buf, int offset)
        => BitConverter.ToUInt64(buf, offset);

    private static ulong ReadU32(byte[] buf, int offset)
        => BitConverter.ToUInt32(buf, offset);
}
