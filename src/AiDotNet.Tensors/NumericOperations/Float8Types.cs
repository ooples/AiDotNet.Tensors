using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Shared bit-twiddling helpers — <c>BitConverter.SingleToUInt32Bits</c>
/// only exists on net6+, we target net471 as well.
/// </summary>
internal static class FloatBits
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static uint SingleToUInt32Bits(float value)
        => Unsafe.As<float, uint>(ref value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float UInt32BitsToSingle(uint value)
        => Unsafe.As<uint, float>(ref value);
}

/// <summary>
/// 8-bit floating point type with 1 sign + 4 exponent + 3 mantissa bits
/// (OCP FP8 E4M3). NVIDIA H100 / Blackwell forward-pass default.
/// Representable range ≈ ±448 with ≈ 7-bit precision in the mantissa.
/// </summary>
/// <remarks>
/// <para>FP8 types differ from IEEE-754 half-precision in two ways: (1) no
/// subnormal gap — smallest normal is exactly 2^-6 for E4M3; (2) saturating
/// overflow to MaxFinite instead of +Inf, per the OCP FP8 spec and NVIDIA's
/// H100 hardware implementation. Operations that would yield Inf are instead
/// clamped to ±MaxFinite, preserving training stability without requiring
/// gradient clipping everywhere FP8 is used.</para>
///
/// <para>Only <see cref="float.NaN"/> propagates through — Inf inputs
/// saturate on conversion rather than mapping to an FP8 Inf (E4M3 has no
/// Inf encoding reserved; all exponent-max values are finite numbers).</para>
/// </remarks>
public readonly struct Float8E4M3 : IEquatable<Float8E4M3>, IComparable<Float8E4M3>
{
    private readonly byte _raw;

    // OCP FP8 E4M3 encoding: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
    // Exponent-max (0b1111) with mantissa-max (0b111) encodes NaN (0x7F / 0xFF).
    // All other exponent-max encodings are finite numbers saturating near ±448.
    private const int ExponentBias = 7;
    private const byte ExponentMask = 0b0111_1000;
    private const byte MantissaMask = 0b0000_0111;
    private const byte SignMask     = 0b1000_0000;
    private const byte NaNRawPos    = 0x7F; // 0_1111_111
    private const byte NaNRawNeg    = 0xFF; // 1_1111_111

    /// <summary>Raw byte representation. 1 bit sign, 4 bit exp, 3 bit mantissa.</summary>
    public byte RawValue => _raw;

    private Float8E4M3(byte raw) { _raw = raw; }

    /// <summary>Maximum representable finite value (≈ 448).</summary>
    public static Float8E4M3 MaxFinite => new((byte)0x7E); // 0_1111_110

    /// <summary>Minimum (most negative) representable finite value (≈ -448).</summary>
    public static Float8E4M3 MinFinite => new((byte)0xFE); // 1_1111_110

    /// <summary>Positive zero.</summary>
    public static Float8E4M3 Zero => new((byte)0x00);

    /// <summary>NaN encoding (exp-max + mantissa-max, positive sign).</summary>
    public static Float8E4M3 NaN => new(NaNRawPos);

    /// <summary>True iff this value encodes NaN.</summary>
    public bool IsNaN
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => (_raw & 0x7F) == NaNRawPos;
    }

    /// <summary>Converts an fp32 value to FP8 E4M3 with saturating overflow.</summary>
    public static Float8E4M3 FromFloat(float value)
    {
        if (float.IsNaN(value)) return NaN;

        // Preserve the sign of zero. `value == 0f` is true for both
        // +0 and -0, so we must inspect the raw bit pattern rather
        // than collapsing everything to the positive-zero encoding.
        uint bits = FloatBits.SingleToUInt32Bits(value);
        if ((bits & 0x7FFF_FFFFu) == 0)
            return (bits & 0x8000_0000u) == 0 ? Zero : new Float8E4M3((byte)SignMask);

        // Saturate on Inf — E4M3 has no Inf encoding, clamp to MaxFinite.
        if (float.IsInfinity(value))
            return value > 0 ? MaxFinite : MinFinite;

        uint sign = (bits >> 31) & 0x1;
        int exp = (int)((bits >> 23) & 0xFF) - 127 + ExponentBias; // re-bias 127 → 7
        uint mantissa23 = bits & 0x7FFFFF;

        // Round-to-nearest-even on mantissa truncation (23 → 3 bits).
        // Keep top 3 bits, round based on the dropped 20 bits + LSB.
        uint mantissa3 = mantissa23 >> 20;
        uint roundingBits = mantissa23 & 0xFFFFF;   // lower 20 bits
        uint half = 1u << 19;                        // 0x80000
        if (roundingBits > half || (roundingBits == half && (mantissa3 & 1u) == 1u))
        {
            mantissa3++;
            if (mantissa3 >= 8) { mantissa3 = 0; exp++; }
        }

        // Saturate overflow — anything beyond MaxFinite's exponent range clamps.
        if (exp >= 0b1111)
        {
            // Exponent-max saturates unless it would be NaN (mantissa all-ones).
            if (exp > 0b1111 || mantissa3 == 0b111)
                return sign == 0 ? MaxFinite : MinFinite;
        }

        // Underflow to zero — anything below the smallest normal collapses.
        if (exp <= 0)
            return sign == 0 ? Zero : new Float8E4M3((byte)SignMask);

        byte raw = (byte)((sign << 7) | ((uint)(exp & 0xF) << 3) | (mantissa3 & 0x7));
        return new Float8E4M3(raw);
    }

    /// <summary>Converts this FP8 E4M3 value back to fp32.</summary>
    public float ToFloat()
    {
        if (IsNaN) return float.NaN;
        if ((_raw & 0x7F) == 0) return (_raw & SignMask) != 0 ? -0f : 0f;

        uint sign = (uint)((_raw & SignMask) >> 7);
        uint exp4 = (uint)((_raw & ExponentMask) >> 3);
        uint m3   = (uint)(_raw & MantissaMask);

        int exp32 = (int)exp4 - ExponentBias + 127;
        uint bits = (sign << 31) | ((uint)(exp32 & 0xFF) << 23) | (m3 << 20);
        return FloatBits.UInt32BitsToSingle(bits);
    }

    public bool Equals(Float8E4M3 other) => _raw == other._raw;
    public override bool Equals(object? obj) => obj is Float8E4M3 o && Equals(o);
    public override int GetHashCode() => _raw.GetHashCode();
    public override string ToString() => ToFloat().ToString("R");

    public int CompareTo(Float8E4M3 other) => ToFloat().CompareTo(other.ToFloat());

    public static bool operator ==(Float8E4M3 a, Float8E4M3 b) => a.Equals(b);
    public static bool operator !=(Float8E4M3 a, Float8E4M3 b) => !a.Equals(b);

    public static explicit operator float(Float8E4M3 v) => v.ToFloat();
    public static explicit operator Float8E4M3(float v) => FromFloat(v);
}

/// <summary>
/// 8-bit floating point type with 1 sign + 5 exponent + 2 mantissa bits
/// (OCP FP8 E5M2). NVIDIA H100 / Blackwell backward-pass default.
/// Wider range (≈ ±57344) than E4M3 but less precision — the gradient
/// landscape typically has wider range than the forward activations,
/// hence this is preferred for gradient storage during backward pass.
/// </summary>
/// <remarks>
/// E5M2 DOES reserve Inf / NaN encodings (exp-max with mantissa zero is
/// Inf; exp-max with non-zero mantissa is NaN) — matching IEEE-754
/// semantics more closely than E4M3. Overflow to Inf is therefore a valid
/// encoding here (the saturating behaviour is kept as an option but
/// defaults to IEEE-style in FromFloat below).
/// </remarks>
public readonly struct Float8E5M2 : IEquatable<Float8E5M2>, IComparable<Float8E5M2>
{
    private readonly byte _raw;

    private const int ExponentBias = 15;
    private const byte ExponentMask = 0b0111_1100;
    private const byte MantissaMask = 0b0000_0011;
    private const byte SignMask     = 0b1000_0000;

    public byte RawValue => _raw;
    private Float8E5M2(byte raw) { _raw = raw; }

    /// <summary>Maximum representable finite value (≈ 57344).</summary>
    public static Float8E5M2 MaxFinite => new((byte)0x7B); // 0_11110_11

    /// <summary>Minimum (most negative) representable finite value.</summary>
    public static Float8E5M2 MinFinite => new((byte)0xFB);

    /// <summary>Positive zero.</summary>
    public static Float8E5M2 Zero => new((byte)0x00);

    /// <summary>Positive infinity (exp-max, mantissa 0).</summary>
    public static Float8E5M2 PositiveInfinity => new((byte)0x7C);

    /// <summary>Negative infinity (exp-max, mantissa 0).</summary>
    public static Float8E5M2 NegativeInfinity => new((byte)0xFC);

    /// <summary>NaN encoding.</summary>
    public static Float8E5M2 NaN => new((byte)0x7F);

    public bool IsNaN
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => (((_raw & 0x7C) == 0x7C) && ((_raw & 0x03) != 0));
    }

    public bool IsInfinity
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ((_raw & 0x7C) == 0x7C) && ((_raw & 0x03) == 0);
    }

    /// <summary>Converts an fp32 value to FP8 E5M2.</summary>
    public static Float8E5M2 FromFloat(float value)
    {
        if (float.IsNaN(value)) return NaN;

        // Preserve the sign of zero — E5M2 has both +0 and -0 encodings.
        uint bits = FloatBits.SingleToUInt32Bits(value);
        if ((bits & 0x7FFF_FFFFu) == 0)
            return (bits & 0x8000_0000u) == 0 ? Zero : new Float8E5M2((byte)SignMask);

        if (float.IsInfinity(value))
            return value > 0 ? PositiveInfinity : NegativeInfinity;

        uint sign = (bits >> 31) & 0x1;
        int exp = (int)((bits >> 23) & 0xFF) - 127 + ExponentBias;
        uint mantissa23 = bits & 0x7FFFFF;

        // Round-to-nearest-even on mantissa truncation (23 → 2 bits).
        uint mantissa2 = mantissa23 >> 21;
        uint roundingBits = mantissa23 & 0x1FFFFF;  // lower 21 bits
        uint half = 1u << 20;                        // 0x100000
        if (roundingBits > half || (roundingBits == half && (mantissa2 & 1u) == 1u))
        {
            mantissa2++;
            if (mantissa2 >= 4) { mantissa2 = 0; exp++; }
        }

        if (exp >= 0b11111)
            return sign == 0 ? PositiveInfinity : NegativeInfinity;
        if (exp <= 0)
            return sign == 0 ? Zero : new Float8E5M2((byte)SignMask);

        byte raw = (byte)((sign << 7) | ((uint)(exp & 0x1F) << 2) | (mantissa2 & 0x3));
        return new Float8E5M2(raw);
    }

    public float ToFloat()
    {
        if (IsNaN) return float.NaN;
        if (IsInfinity) return (_raw & SignMask) != 0 ? float.NegativeInfinity : float.PositiveInfinity;
        if ((_raw & 0x7F) == 0) return (_raw & SignMask) != 0 ? -0f : 0f;

        uint sign = (uint)((_raw & SignMask) >> 7);
        uint exp5 = (uint)((_raw & ExponentMask) >> 2);
        uint m2   = (uint)(_raw & MantissaMask);

        int exp32 = (int)exp5 - ExponentBias + 127;
        uint bits = (sign << 31) | ((uint)(exp32 & 0xFF) << 23) | (m2 << 21);
        return FloatBits.UInt32BitsToSingle(bits);
    }

    public bool Equals(Float8E5M2 other) => _raw == other._raw;
    public override bool Equals(object? obj) => obj is Float8E5M2 o && Equals(o);
    public override int GetHashCode() => _raw.GetHashCode();
    public override string ToString() => ToFloat().ToString("R");

    public int CompareTo(Float8E5M2 other) => ToFloat().CompareTo(other.ToFloat());

    public static bool operator ==(Float8E5M2 a, Float8E5M2 b) => a.Equals(b);
    public static bool operator !=(Float8E5M2 a, Float8E5M2 b) => !a.Equals(b);

    public static explicit operator float(Float8E5M2 v) => v.ToFloat();
    public static explicit operator Float8E5M2(float v) => FromFloat(v);
}
