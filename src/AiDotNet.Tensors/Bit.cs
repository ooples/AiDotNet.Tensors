using System;

namespace AiDotNet.Tensors;

/// <summary>
/// A value type wrapping a single byte (0 = false, 1 = true) that enables
/// <c>Tensor&lt;Bit&gt;</c> for boolean mask and conditional operations across
/// all supported .NET versions, including net471.
/// </summary>
public readonly struct Bit : IEquatable<Bit>, IComparable<Bit>
{
    private readonly byte _value;

    private Bit(byte value)
    {
        _value = (byte)(value != 0 ? 1 : 0);
    }

    /// <summary>A Bit representing logical true (1).</summary>
    public static readonly Bit True = new Bit(1);

    /// <summary>A Bit representing logical false (0).</summary>
    public static readonly Bit False = new Bit(0);

    // ----------------------------------------------------------------
    // Conversions
    // ----------------------------------------------------------------

    /// <summary>Implicitly converts a bool to a Bit.</summary>
    public static implicit operator Bit(bool value) => value ? True : False;

    /// <summary>Implicitly converts a Bit to a bool.</summary>
    public static implicit operator bool(Bit bit) => bit._value != 0;

    /// <summary>Implicitly converts a byte (0 or non-zero) to a Bit.</summary>
    public static implicit operator Bit(byte value) => new Bit(value);

    /// <summary>Implicitly converts a Bit to its underlying byte (0 or 1).</summary>
    public static implicit operator byte(Bit bit) => bit._value;

    // ----------------------------------------------------------------
    // Logical operators
    // ----------------------------------------------------------------

    /// <summary>Bitwise AND (logical AND for bits).</summary>
    public static Bit operator &(Bit a, Bit b) => new Bit((byte)(a._value & b._value));

    /// <summary>Bitwise OR (logical OR for bits).</summary>
    public static Bit operator |(Bit a, Bit b) => new Bit((byte)(a._value | b._value));

    /// <summary>Bitwise XOR (logical XOR for bits).</summary>
    public static Bit operator ^(Bit a, Bit b) => new Bit((byte)(a._value ^ b._value));

    /// <summary>Logical NOT.</summary>
    public static Bit operator !(Bit a) => a._value == 0 ? True : False;

    // ----------------------------------------------------------------
    // Comparison operators
    // ----------------------------------------------------------------

    /// <summary>Equality comparison.</summary>
    public static bool operator ==(Bit a, Bit b) => a._value == b._value;

    /// <summary>Inequality comparison.</summary>
    public static bool operator !=(Bit a, Bit b) => a._value != b._value;

    /// <summary>Less-than comparison.</summary>
    public static bool operator <(Bit a, Bit b) => a._value < b._value;

    /// <summary>Greater-than comparison.</summary>
    public static bool operator >(Bit a, Bit b) => a._value > b._value;

    /// <summary>Less-than-or-equal comparison.</summary>
    public static bool operator <=(Bit a, Bit b) => a._value <= b._value;

    /// <summary>Greater-than-or-equal comparison.</summary>
    public static bool operator >=(Bit a, Bit b) => a._value >= b._value;

    // ----------------------------------------------------------------
    // IEquatable<Bit>
    // ----------------------------------------------------------------

    /// <inheritdoc/>
    public bool Equals(Bit other) => _value == other._value;

    /// <inheritdoc/>
    public override bool Equals(
#if NET5_0_OR_GREATER
        object? obj
#else
        object obj
#endif
        )
    {
        if (obj is Bit other)
            return Equals(other);
        return false;
    }

    /// <inheritdoc/>
    public override int GetHashCode() => _value.GetHashCode();

    // ----------------------------------------------------------------
    // IComparable<Bit>
    // ----------------------------------------------------------------

    /// <inheritdoc/>
    public int CompareTo(Bit other) => _value.CompareTo(other._value);

    // ----------------------------------------------------------------
    // ToString
    // ----------------------------------------------------------------

    /// <summary>Returns "True" or "False".</summary>
    public override string ToString() => _value != 0 ? "True" : "False";
}
