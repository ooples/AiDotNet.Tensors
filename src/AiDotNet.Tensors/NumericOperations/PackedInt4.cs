using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Bit-packed storage for 4-bit quantized values. Each byte holds two
/// signed nibbles in the range [-8, 7] — two's-complement 4-bit ints,
/// the dominant weight-only quantization format for open-source LLM
/// inference (llama.cpp Q4_0, AWQ, GPTQ).
///
/// <para><b>Layout:</b> low nibble of byte N is element 2N; high nibble is
/// element 2N+1. Matches llama.cpp's Q4_0 block layout so a GGUF reader
/// can copy block weights byte-for-byte into this type.</para>
///
/// <para><b>Scaling:</b> Int4 on its own is low-fidelity; real accuracy
/// comes from pairing it with per-group float16 scales (group-wise
/// quantization, block size 32 or 128). See
/// <see cref="QuantizationScale"/> for the companion scale tensor.</para>
/// </summary>
public readonly struct PackedInt4 : IEquatable<PackedInt4>
{
    private readonly byte _raw;

    /// <summary>Number of 4-bit values packed per byte.</summary>
    public const int ValuesPerByte = 2;

    /// <summary>Minimum representable value (two's-complement int4).</summary>
    public const int MinValue = -8;

    /// <summary>Maximum representable value.</summary>
    public const int MaxValue = 7;

    /// <summary>Raw 8-bit payload (low nibble = lane 0, high = lane 1).</summary>
    public byte RawValue => _raw;

    public PackedInt4(byte raw) { _raw = raw; }

    /// <summary>
    /// Pack two signed int4 values (-8..7) into a byte.
    /// </summary>
    public static PackedInt4 FromInts(int lo, int hi)
    {
        if (lo < MinValue || lo > MaxValue)
            throw new ArgumentOutOfRangeException(nameof(lo),
                $"int4 lo out of range [-8, 7]: {lo}");
        if (hi < MinValue || hi > MaxValue)
            throw new ArgumentOutOfRangeException(nameof(hi),
                $"int4 hi out of range [-8, 7]: {hi}");
        return new PackedInt4((byte)((lo & 0x0F) | ((hi & 0x0F) << 4)));
    }

    /// <summary>Unpack the low nibble as a signed int4.</summary>
    public int LoNibble
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => SignExtendNibble(_raw & 0x0F);
    }

    /// <summary>Unpack the high nibble as a signed int4.</summary>
    public int HiNibble
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => SignExtendNibble((_raw >> 4) & 0x0F);
    }

    /// <summary>Retrieve lane 0 (low nibble) or 1 (high nibble) as an int.</summary>
    public int GetLane(int index) => index switch
    {
        0 => LoNibble,
        1 => HiNibble,
        _ => throw new ArgumentOutOfRangeException(nameof(index)),
    };

    /// <summary>
    /// Sign-extends a 4-bit two's-complement value stored in the low nibble.
    /// Input nibble &gt;= 8 is interpreted as a negative value.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SignExtendNibble(int nibble)
    {
        // Branchless: nibble 0..7 passes through, 8..15 becomes -8..-1.
        // Bit 3 is the sign bit of int4.
        return (nibble & 0x07) - (nibble & 0x08);
    }

    public bool Equals(PackedInt4 other) => _raw == other._raw;
    public override bool Equals(object? obj) => obj is PackedInt4 o && Equals(o);
    public override int GetHashCode() => _raw.GetHashCode();
    public static bool operator ==(PackedInt4 a, PackedInt4 b) => a.Equals(b);
    public static bool operator !=(PackedInt4 a, PackedInt4 b) => !a.Equals(b);
}
