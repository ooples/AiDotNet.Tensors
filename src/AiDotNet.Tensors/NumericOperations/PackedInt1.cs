using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Bit-packed storage for 1-bit quantized values. Each byte holds 8
/// signed-bit values: bit i encodes value +1 when set, -1 when clear
/// (XNOR-Net / BitNet convention). Matmul collapses to popcount + bit-
/// XOR — orders of magnitude faster than any FP format on binary nets.
///
/// <para><b>Encoding:</b> input ≥ 0 → bit 1, input &lt; 0 → bit 0.
/// Popcount gives the number of agreeing signs across a row×col dot
/// product; result = 2 × popcount(a XNOR b) − bitCount. That maps
/// directly to the {-1, +1} inner-product algebra BitNet trains under.</para>
///
/// <para><b>Storage layout:</b> LSB-first — bit 0 of byte 0 is the
/// first element. A 16-element vector packs into 2 bytes:
/// <c>[b0b1b2b3b4b5b6b7][b8b9b10b11b12b13b14b15]</c>.</para>
/// </summary>
public readonly struct PackedInt1 : IEquatable<PackedInt1>
{
    private readonly byte _raw;

    /// <summary>Number of 1-bit values packed per byte.</summary>
    public const int ValuesPerByte = 8;

    /// <summary>Raw 8-bit payload.</summary>
    public byte RawValue => _raw;

    public PackedInt1(byte raw) { _raw = raw; }

    /// <summary>Read the 1-bit value at lane <paramref name="index"/> (0–7)
    /// as ±1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public sbyte GetLane(int index)
    {
        if ((uint)index >= ValuesPerByte)
            throw new ArgumentOutOfRangeException(nameof(index));
        return ((_raw >> index) & 0x1) == 0 ? (sbyte)-1 : (sbyte)1;
    }

    /// <summary>Pack <paramref name="signs"/> (length 8, ±1) into a byte.</summary>
    public static PackedInt1 FromSigns(ReadOnlySpan<sbyte> signs)
    {
        if (signs.Length != ValuesPerByte)
            throw new ArgumentException(
                $"Expected {ValuesPerByte} signs, got {signs.Length}.", nameof(signs));
        byte raw = 0;
        for (int i = 0; i < ValuesPerByte; i++)
            if (signs[i] >= 0) raw |= (byte)(1 << i);
        return new PackedInt1(raw);
    }

    /// <summary>Quantize <paramref name="values"/> (length 8, any float) to
    /// the sign-encoded byte. Positive → 1, zero → 1, negative → -1 —
    /// matches BitNet's <c>sign()</c> with the <c>sign(0) = +1</c> tiebreak.</summary>
    public static PackedInt1 FromFloat(ReadOnlySpan<float> values)
    {
        if (values.Length != ValuesPerByte)
            throw new ArgumentException(
                $"Expected {ValuesPerByte} values, got {values.Length}.", nameof(values));
        byte raw = 0;
        for (int i = 0; i < ValuesPerByte; i++)
            if (values[i] >= 0f) raw |= (byte)(1 << i);
        return new PackedInt1(raw);
    }

    public bool Equals(PackedInt1 other) => _raw == other._raw;
    public override bool Equals(object? obj) => obj is PackedInt1 o && Equals(o);
    public override int GetHashCode() => _raw.GetHashCode();
    public static bool operator ==(PackedInt1 a, PackedInt1 b) => a.Equals(b);
    public static bool operator !=(PackedInt1 a, PackedInt1 b) => !a.Equals(b);
}
