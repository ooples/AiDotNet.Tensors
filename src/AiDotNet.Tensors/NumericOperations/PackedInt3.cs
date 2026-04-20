namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// 3-bit packed storage — 8 signed values per 3 bytes, range [-4, 3].
/// Used by GGUF Q3_K for 3-bit weight-only quantization of LLMs.
///
/// <para><b>Block layout:</b> 8 consecutive 3-bit values occupy 24 bits,
/// stored as <c>{ bytes[0..2] }</c>. Lane i (0..7) reads from bit index
/// <c>3 × i</c> within the 24-bit little-endian concatenation. Unlike
/// Int2/Int4 this does NOT pack cleanly into a single-byte struct, so
/// the API is static helpers on a byte array; the struct holds a 3-byte
/// block for convenience.</para>
///
/// <para>Two's-complement signed 3-bit: 0,1,2,3 positive; 4,5,6,7 →
/// -4,-3,-2,-1 respectively.</para>
/// </summary>
public readonly struct PackedInt3Block : IEquatable<PackedInt3Block>
{
    private readonly byte _b0, _b1, _b2;

    public const int ValuesPerBlock = 8;
    public const int BytesPerBlock = 3;
    public const int MinValue = -4;
    public const int MaxValue = 3;

    public byte B0 => _b0;
    public byte B1 => _b1;
    public byte B2 => _b2;

    public PackedInt3Block(byte b0, byte b1, byte b2)
    {
        _b0 = b0; _b1 = b1; _b2 = b2;
    }

    /// <summary>
    /// Build a 3-byte block from 8 int3 values (-4..3).
    /// </summary>
    public static PackedInt3Block FromInts(ReadOnlySpan<int> values)
    {
        if (values.Length != ValuesPerBlock)
            throw new ArgumentException(
                $"Expected {ValuesPerBlock} values, got {values.Length}.", nameof(values));
        uint acc = 0;
        for (int i = 0; i < ValuesPerBlock; i++)
        {
            int v = values[i];
            if (v < MinValue || v > MaxValue)
                throw new ArgumentOutOfRangeException(nameof(values),
                    $"int3 value at lane {i} out of range [-4, 3]: {v}");
            acc |= (uint)(v & 0x07) << (i * 3);
        }
        return new PackedInt3Block((byte)(acc & 0xFF), (byte)((acc >> 8) & 0xFF), (byte)((acc >> 16) & 0xFF));
    }

    /// <summary>
    /// Unpack lane <paramref name="index"/> (0..7) as a signed int.
    /// </summary>
    public int GetLane(int index)
    {
        if ((uint)index >= ValuesPerBlock)
            throw new ArgumentOutOfRangeException(nameof(index));
        uint combined = (uint)_b0 | ((uint)_b1 << 8) | ((uint)_b2 << 16);
        int raw = (int)((combined >> (index * 3)) & 0x07);
        // Sign-extend 3 bits: bit 2 is the sign.
        return (raw & 0x03) - (raw & 0x04);
    }

    public bool Equals(PackedInt3Block other) => _b0 == other._b0 && _b1 == other._b1 && _b2 == other._b2;
    public override bool Equals(object? obj) => obj is PackedInt3Block o && Equals(o);
    public override int GetHashCode() => (_b0 << 16) | (_b1 << 8) | _b2;
    public static bool operator ==(PackedInt3Block a, PackedInt3Block b) => a.Equals(b);
    public static bool operator !=(PackedInt3Block a, PackedInt3Block b) => !a.Equals(b);
}
