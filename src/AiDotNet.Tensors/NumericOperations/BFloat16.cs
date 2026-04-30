// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// IEEE-754-style brain-float 16: 1 sign bit, 8 exponent bits (same range as <see cref="float"/>),
/// 7 mantissa bits. The dynamic range matches single precision so transformer training does not
/// underflow/overflow the way <see cref="System.Half"/> can — but with half the storage cost.
///
/// <para>Conversion to/from <see cref="float"/> is the upper-16-bit truncation pattern with
/// round-to-nearest-even (the same code AVX-512's VCVTNEPS2BF16 emits in hardware).
/// All arithmetic is implemented as <c>BFloat16 → float → op → BFloat16</c> round-trips for
/// numerical stability; the win comes from 50% storage reduction and the matmul / kernel
/// codegen that the SIMD layer wires per shape.</para>
///
/// <para>Reference: Intel "BFLOAT16 - Hardware Numerics Definition" (Nov 2018), and
/// Micikevicius et al. "Mixed Precision Training" (ICLR 2018) for the float-accumulate
/// gradient pattern that callers should pair with this type.</para>
/// </summary>
[StructLayout(LayoutKind.Sequential, Size = 2)]
public readonly struct BFloat16 : IEquatable<BFloat16>, IComparable<BFloat16>, IFormattable
{
    /// <summary>Raw 16-bit representation (sign | exp | mantissa[7]).</summary>
    public readonly ushort RawValue;

    private BFloat16(ushort raw) { RawValue = raw; }

    /// <summary>Constructs from raw bits — used by SIMD codegen and serializers.</summary>
    public static BFloat16 FromRawBits(ushort raw) => new(raw);

    /// <summary>Zero (+0).</summary>
    public static BFloat16 Zero => new(0x0000);
    /// <summary>One (1.0).</summary>
    public static BFloat16 One => FromFloat(1f);
    /// <summary>+Infinity.</summary>
    public static BFloat16 PositiveInfinity => new(0x7F80);
    /// <summary>-Infinity.</summary>
    public static BFloat16 NegativeInfinity => new(0xFF80);
    /// <summary>NaN.</summary>
    public static BFloat16 NaN => new(0x7FC0);
    /// <summary>Largest representable finite value (~3.39e38, same exponent range as float).</summary>
    public static BFloat16 MaxValue => new(0x7F7F);
    /// <summary>Most-negative representable finite value.</summary>
    public static BFloat16 MinValue => new(0xFF7F);
    /// <summary>Smallest positive normal value (~1.18e-38).</summary>
    public static BFloat16 Epsilon => new(0x0080);

    // ── Conversions ─────────────────────────────────────────────

    /// <summary>Converts a <see cref="float"/> to <see cref="BFloat16"/> with round-to-nearest-even.
    /// NaN preserves a quiet bit; ±Inf maps to the bf16 ±Inf encodings.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static BFloat16 FromFloat(float value)
    {
        // Bit-cast float → uint32, take upper 16 bits with RNE rounding.
        uint bits = SingleToUInt32Bits(value);

        // NaN: preserve quiet-bit so the bf16 round-trip stays a NaN.
        // Bit pattern: exponent all-ones, mantissa non-zero.
        if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0)
        {
            // Quiet NaN: top mantissa bit set in the bf16 result.
            return new(unchecked((ushort)((bits >> 16) | 0x0040)));
        }

        // Round-to-nearest-even: add the rounding bias 0x7FFF + LSB-of-target, truncate.
        uint lsb = (bits >> 16) & 1u;
        uint roundingBias = 0x7FFFu + lsb;
        uint rounded = bits + roundingBias;
        return new(unchecked((ushort)(rounded >> 16)));
    }

    /// <summary>Converts <see cref="BFloat16"/> back to <see cref="float"/>. Lossless.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToFloat(BFloat16 value)
    {
        uint bits = ((uint)value.RawValue) << 16;
        return UInt32BitsToSingle(bits);
    }

    /// <summary>Implicit float → bfloat16 (PyTorch parity: bf16 is a strict subset of float in range).</summary>
    public static explicit operator BFloat16(float v) => FromFloat(v);
    /// <summary>Explicit double → bfloat16 (lossy).</summary>
    public static explicit operator BFloat16(double v) => FromFloat((float)v);
    /// <summary>Implicit bfloat16 → float (lossless).</summary>
    public static implicit operator float(BFloat16 v) => ToFloat(v);
    /// <summary>Implicit bfloat16 → double (lossless via float).</summary>
    public static implicit operator double(BFloat16 v) => (double)ToFloat(v);

    // ── Arithmetic operators (round-trip via float) ────────────

    public static BFloat16 operator +(BFloat16 a, BFloat16 b) => FromFloat(ToFloat(a) + ToFloat(b));
    public static BFloat16 operator -(BFloat16 a, BFloat16 b) => FromFloat(ToFloat(a) - ToFloat(b));
    public static BFloat16 operator *(BFloat16 a, BFloat16 b) => FromFloat(ToFloat(a) * ToFloat(b));
    public static BFloat16 operator /(BFloat16 a, BFloat16 b) => FromFloat(ToFloat(a) / ToFloat(b));
    public static BFloat16 operator -(BFloat16 a) => new((ushort)(a.RawValue ^ 0x8000));

    // ── Comparison ─────────────────────────────────────────────

    public static bool operator ==(BFloat16 a, BFloat16 b) => ToFloat(a) == ToFloat(b);
    public static bool operator !=(BFloat16 a, BFloat16 b) => !(a == b);
    public static bool operator <(BFloat16 a, BFloat16 b) => ToFloat(a) < ToFloat(b);
    public static bool operator >(BFloat16 a, BFloat16 b) => ToFloat(a) > ToFloat(b);
    public static bool operator <=(BFloat16 a, BFloat16 b) => ToFloat(a) <= ToFloat(b);
    public static bool operator >=(BFloat16 a, BFloat16 b) => ToFloat(a) >= ToFloat(b);

    public bool Equals(BFloat16 other) => RawValue == other.RawValue || ToFloat(this) == ToFloat(other);
    public override bool Equals(object? obj) => obj is BFloat16 b && Equals(b);
    public override int GetHashCode() => RawValue.GetHashCode();

    public int CompareTo(BFloat16 other) => ToFloat(this).CompareTo(ToFloat(other));

    // ── Classification ──────────────────────────────────────────

    /// <summary>True iff value is NaN.</summary>
    public static bool IsNaN(BFloat16 v) => (v.RawValue & 0x7F80) == 0x7F80 && (v.RawValue & 0x007F) != 0;
    /// <summary>True iff value is ±Infinity.</summary>
    public static bool IsInfinity(BFloat16 v) => (v.RawValue & 0x7FFF) == 0x7F80;
    /// <summary>True iff value is finite (not NaN, not ±Inf).</summary>
    public static bool IsFinite(BFloat16 v) => (v.RawValue & 0x7F80) != 0x7F80;

    public override string ToString() => ToFloat(this).ToString();
    public string ToString(string? format, IFormatProvider? formatProvider) =>
        ToFloat(this).ToString(format, formatProvider);

    // ── BitConverter helpers (no NET5+ dependency) ─────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe uint SingleToUInt32Bits(float v) => *(uint*)&v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float UInt32BitsToSingle(uint b) => *(float*)&b;
}
