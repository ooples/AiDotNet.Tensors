using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// 2-bit packed storage — 4 signed values per byte, range [-2, 1].
/// Used by GGUF Q2_K and similar sub-1-bpw formats when paired with
/// a super-block scale + min-value pair.
///
/// <para><b>Layout:</b> lanes 0..3 occupy bits [0:1], [2:3], [4:5], [6:7].
/// Two's-complement signed 2-bit means the mapped values are:
/// <c>00 → 0, 01 → 1, 10 → -2, 11 → -1</c>.</para>
///
/// <para>Lower fidelity than Int4 — the accuracy cost is absorbed by
/// per-group scales (group size 16–32 typical). On commodity hardware
/// Int2 lets 7B-parameter models fit in &lt; 2 GB.</para>
/// </summary>
public readonly struct PackedInt2 : IEquatable<PackedInt2>
{
    private readonly byte _raw;

    public const int ValuesPerByte = 4;
    public const int MinValue = -2;
    public const int MaxValue = 1;

    public byte RawValue => _raw;

    public PackedInt2(byte raw) { _raw = raw; }

    public static PackedInt2 FromInts(int a, int b, int c, int d)
    {
        Check(a, nameof(a)); Check(b, nameof(b));
        Check(c, nameof(c)); Check(d, nameof(d));
        return new PackedInt2((byte)(
            (a & 0x03) | ((b & 0x03) << 2) | ((c & 0x03) << 4) | ((d & 0x03) << 6)));
    }

    private static void Check(int v, string name)
    {
        if (v < MinValue || v > MaxValue)
            throw new ArgumentOutOfRangeException(name,
                $"int2 value out of range [-2, 1]: {v}");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetLane(int index)
    {
        if ((uint)index >= ValuesPerByte)
            throw new ArgumentOutOfRangeException(nameof(index));
        int raw = (_raw >> (index * 2)) & 0x03;
        // Sign-extend 2 bits → int: 0,1 pass through; 2,3 become -2,-1.
        return (raw & 0x01) - (raw & 0x02);
    }

    public bool Equals(PackedInt2 other) => _raw == other._raw;
    public override bool Equals(object? obj) => obj is PackedInt2 o && Equals(o);
    public override int GetHashCode() => _raw.GetHashCode();
    public static bool operator ==(PackedInt2 a, PackedInt2 b) => a.Equals(b);
    public static bool operator !=(PackedInt2 a, PackedInt2 b) => !a.Equals(b);
}
