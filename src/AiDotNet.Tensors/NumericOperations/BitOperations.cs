using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides numeric operations for the <see cref="Bit"/> type, enabling
/// <c>Tensor&lt;Bit&gt;</c> for boolean mask and conditional operations across
/// all supported .NET versions, including net471.
/// </summary>
/// <remarks>
/// <para>
/// Arithmetic semantics for Bit follow boolean algebra:
/// Add = OR, Multiply = AND, Subtract = XOR, Negate = NOT.
/// Zero is <see cref="Bit.False"/> and One is <see cref="Bit.True"/>.
/// </para>
/// </remarks>
public class BitOperations : INumericOperations<Bit>
{
    private static readonly BitOperations _instance = new BitOperations();

    // ----------------------------------------------------------------
    // Scalar arithmetic
    // ----------------------------------------------------------------

    /// <summary>Add = logical OR.</summary>
    public Bit Add(Bit a, Bit b) => a | b;

    /// <summary>Subtract = logical XOR.</summary>
    public Bit Subtract(Bit a, Bit b) => a ^ b;

    /// <summary>Multiply = logical AND.</summary>
    public Bit Multiply(Bit a, Bit b) => a & b;

    /// <summary>Divide: b==False returns One; otherwise returns a AND b.</summary>
    public Bit Divide(Bit a, Bit b) => b == Bit.False ? Bit.True : a & b;

    /// <summary>Negate = logical NOT.</summary>
    public Bit Negate(Bit a) => !a;

    /// <summary>Zero element: Bit.False.</summary>
    public Bit Zero => Bit.False;

    /// <summary>One element: Bit.True.</summary>
    public Bit One => Bit.True;

    // ----------------------------------------------------------------
    // Math functions — return Zero or One as semantically meaningful approximations
    // ----------------------------------------------------------------

    /// <summary>Square root of a Bit is the identity (sqrt(0)=0, sqrt(1)=1).</summary>
    public Bit Sqrt(Bit value) => value;

    /// <summary>Converts a double to a Bit (non-zero = True).</summary>
    public Bit FromDouble(double value) => value != 0.0 ? Bit.True : Bit.False;

    /// <summary>Converts a float to a Bit (non-zero = True).</summary>
    public Bit FromFloat(float value) => value != 0f ? Bit.True : Bit.False;

    /// <summary>Converts a Bit to float (0.0 or 1.0).</summary>
    public float ToFloat(Bit value) => (bool)value ? 1f : 0f;

    /// <summary>Converts a Bit to double (0.0 or 1.0).</summary>
    public double ToDouble(Bit value) => (bool)value ? 1.0 : 0.0;

    /// <summary>Converts a Bit to Half (0 or 1).</summary>
    public Half ToHalf(Bit value) => (bool)value ? (Half)1f : (Half)0f;

    /// <summary>Converts a Half to a Bit (non-zero = True).</summary>
    public Bit FromHalf(Half value) => (float)value != 0f ? Bit.True : Bit.False;

    /// <summary>Converts a Bit to int32 (0 or 1).</summary>
    public int ToInt32(Bit value) => (bool)value ? 1 : 0;

    // ----------------------------------------------------------------
    // Comparison
    // ----------------------------------------------------------------

    /// <summary>Greater than by underlying byte value.</summary>
    public bool GreaterThan(Bit a, Bit b) => (byte)a > (byte)b;

    /// <summary>Less than by underlying byte value.</summary>
    public bool LessThan(Bit a, Bit b) => (byte)a < (byte)b;

    /// <summary>Greater than or equal by underlying byte value.</summary>
    public bool GreaterThanOrEquals(Bit a, Bit b) => (byte)a >= (byte)b;

    /// <summary>Less than or equal by underlying byte value.</summary>
    public bool LessThanOrEquals(Bit a, Bit b) => (byte)a <= (byte)b;

    /// <summary>Equality by underlying byte value.</summary>
    public bool Equals(Bit a, Bit b) => a == b;

    /// <summary>Compares two Bit values.</summary>
    public int Compare(Bit a, Bit b) => a.CompareTo(b);

    // ----------------------------------------------------------------
    // Min / Max / Abs / Square
    // ----------------------------------------------------------------

    /// <summary>Minimum value: Bit.False.</summary>
    public Bit MinValue => Bit.False;

    /// <summary>Maximum value: Bit.True.</summary>
    public Bit MaxValue => Bit.True;

    /// <summary>Absolute value is identity for Bit.</summary>
    public Bit Abs(Bit value) => value;

    /// <summary>Square is identity (AND with self).</summary>
    public Bit Square(Bit value) => value & value;

    /// <summary>Sign: True returns One, False returns Zero.</summary>
    public Bit SignOrZero(Bit value) => value;

    // ----------------------------------------------------------------
    // Math functions that cannot be meaningfully represented as Bit
    // — return closest boolean approximation
    // ----------------------------------------------------------------

    /// <summary>Exp: e^0=1 (False→True), e^1≈2.718 (True→True). Always returns True for defined input.</summary>
    public Bit Exp(Bit value) => Bit.True;

    /// <summary>Log: log(1)=0 (True→False), log(0)=undefined (False→False). Returns False.</summary>
    public Bit Log(Bit value) => Bit.False;

    /// <summary>Power: a^b semantics follow AND for Bits.</summary>
    public Bit Power(Bit baseValue, Bit exponent) => baseValue & exponent;

    /// <summary>Round is identity for Bit.</summary>
    public Bit Round(Bit value) => value;

    /// <summary>Floor is identity for Bit.</summary>
    public Bit Floor(Bit value) => value;

    /// <summary>Ceiling is identity for Bit.</summary>
    public Bit Ceiling(Bit value) => value;

    /// <summary>Fractional part of an integer bit is always zero (False).</summary>
    public Bit Frac(Bit value) => Bit.False;

    /// <summary>Sin is not supported for Bit. Always returns False.</summary>
    public Bit Sin(Bit value) => Bit.False;

    /// <summary>Cos is not supported for Bit. Always returns False.</summary>
    public Bit Cos(Bit value) => Bit.False;

    // ----------------------------------------------------------------
    // Special value predicates
    // ----------------------------------------------------------------

    /// <summary>Bit values are never NaN.</summary>
    public bool IsNaN(Bit value) => false;

    /// <summary>Bit values are never infinity.</summary>
    public bool IsInfinity(Bit value) => false;

    // ----------------------------------------------------------------
    // Precision / acceleration
    // ----------------------------------------------------------------

    /// <summary>Bit has 1 bit of precision.</summary>
    public int PrecisionBits => 1;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => false;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    // ----------------------------------------------------------------
    // IVectorizedOperations<Bit> span overloads
    // ----------------------------------------------------------------

    /// <summary>Element-wise OR into destination.</summary>
    public void Add(ReadOnlySpan<Bit> x, ReadOnlySpan<Bit> y, Span<Bit> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = x[i] | y[i];
    }

    /// <summary>Element-wise XOR into destination.</summary>
    public void Subtract(ReadOnlySpan<Bit> x, ReadOnlySpan<Bit> y, Span<Bit> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = x[i] ^ y[i];
    }

    /// <summary>Element-wise AND into destination.</summary>
    public void Multiply(ReadOnlySpan<Bit> x, ReadOnlySpan<Bit> y, Span<Bit> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = x[i] & y[i];
    }

    /// <summary>Element-wise divide (b==False→True, else a AND b).</summary>
    public void Divide(ReadOnlySpan<Bit> x, ReadOnlySpan<Bit> y, Span<Bit> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = Divide(x[i], y[i]);
    }

    /// <summary>Dot product = count of positions where both are True.</summary>
    public Bit Dot(ReadOnlySpan<Bit> x, ReadOnlySpan<Bit> y)
        => VectorizedOperationsFallback.Dot(_instance, x, y);

    /// <summary>Sum = OR reduction (any True → True).</summary>
    public Bit Sum(ReadOnlySpan<Bit> x)
        => VectorizedOperationsFallback.Sum(_instance, x);

    /// <summary>Max = OR reduction (any True → True).</summary>
    public Bit Max(ReadOnlySpan<Bit> x)
        => VectorizedOperationsFallback.Max(_instance, x);

    /// <summary>Min = AND reduction (all True → True).</summary>
    public Bit Min(ReadOnlySpan<Bit> x)
        => VectorizedOperationsFallback.Min(_instance, x);

    /// <summary>Exp span — always sets destination to True.</summary>
    public void Exp(ReadOnlySpan<Bit> x, Span<Bit> destination) => destination.Fill(Bit.True);

    /// <summary>Log span — always sets destination to False.</summary>
    public void Log(ReadOnlySpan<Bit> x, Span<Bit> destination) => destination.Fill(Bit.False);

    /// <summary>Tanh span — copy source to destination (tanh(1)≈1, tanh(0)=0).</summary>
    public void Tanh(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Sigmoid span — copy source (sigmoid(1)≈0.73→True, sigmoid(0)=0.5→True is borderline, use identity).</summary>
    public void Sigmoid(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Log2 span — always sets destination to False.</summary>
    public void Log2(ReadOnlySpan<Bit> x, Span<Bit> destination) => destination.Fill(Bit.False);

    /// <summary>SoftMax is not meaningful for Bit — copies source as identity.</summary>
    public void SoftMax(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Cosine similarity between two Bit spans.</summary>
    public Bit CosineSimilarity(ReadOnlySpan<Bit> x, ReadOnlySpan<Bit> y)
        => VectorizedOperationsFallback.CosineSimilarity(_instance, x, y);

    /// <summary>Fills the destination span with the given value.</summary>
    public void Fill(Span<Bit> destination, Bit value) => destination.Fill(value);

    /// <summary>Scalar multiply (AND) each element.</summary>
    public void MultiplyScalar(ReadOnlySpan<Bit> x, Bit scalar, Span<Bit> destination)
        => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);

    /// <summary>Scalar divide each element.</summary>
    public void DivideScalar(ReadOnlySpan<Bit> x, Bit scalar, Span<Bit> destination)
        => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);

    /// <summary>Scalar add (OR) each element.</summary>
    public void AddScalar(ReadOnlySpan<Bit> x, Bit scalar, Span<Bit> destination)
        => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);

    /// <summary>Scalar subtract (XOR) each element.</summary>
    public void SubtractScalar(ReadOnlySpan<Bit> x, Bit scalar, Span<Bit> destination)
        => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);

    /// <summary>Sqrt span — identity for Bit.</summary>
    public void Sqrt(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Abs span — identity for Bit.</summary>
    public void Abs(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Negate span — logical NOT of each element.</summary>
    public void Negate(ReadOnlySpan<Bit> x, Span<Bit> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = !x[i];
    }

    /// <summary>Clip span.</summary>
    public void Clip(ReadOnlySpan<Bit> x, Bit min, Bit max, Span<Bit> destination)
        => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);

    /// <summary>Pow span.</summary>
    public void Pow(ReadOnlySpan<Bit> x, Bit power, Span<Bit> destination)
        => VectorizedOperationsFallback.Pow(_instance, x, power, destination);

    /// <summary>Copy span.</summary>
    public void Copy(ReadOnlySpan<Bit> source, Span<Bit> destination) => source.CopyTo(destination);

    /// <summary>Floor span — identity for Bit.</summary>
    public void Floor(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Ceiling span — identity for Bit.</summary>
    public void Ceiling(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Frac span — always False for integer-like Bit.</summary>
    public void Frac(ReadOnlySpan<Bit> x, Span<Bit> destination) => destination.Fill(Bit.False);

    /// <summary>Sin span — always False.</summary>
    public void Sin(ReadOnlySpan<Bit> x, Span<Bit> destination) => destination.Fill(Bit.False);

    /// <summary>Cos span — always False.</summary>
    public void Cos(ReadOnlySpan<Bit> x, Span<Bit> destination) => destination.Fill(Bit.False);

    /// <summary>Multiply-add span.</summary>
    public void MultiplyAdd(ReadOnlySpan<Bit> x, ReadOnlySpan<Bit> y, Bit scalar, Span<Bit> destination)
        => VectorizedOperationsFallback.MultiplyAdd(_instance, x, y, scalar, destination);

    /// <summary>Converts Bit span to float span.</summary>
    public void ToFloatSpan(ReadOnlySpan<Bit> source, Span<float> destination)
        => VectorizedOperationsFallback.ToFloatSpan(_instance, source, destination);

    /// <summary>Converts float span to Bit span.</summary>
    public void FromFloatSpan(ReadOnlySpan<float> source, Span<Bit> destination)
        => VectorizedOperationsFallback.FromFloatSpan(_instance, source, destination);

    /// <summary>Converts Bit span to Half span.</summary>
    public void ToHalfSpan(ReadOnlySpan<Bit> source, Span<Half> destination)
        => VectorizedOperationsFallback.ToHalfSpan(_instance, source, destination);

    /// <summary>Converts Half span to Bit span.</summary>
    public void FromHalfSpan(ReadOnlySpan<Half> source, Span<Bit> destination)
        => VectorizedOperationsFallback.FromHalfSpan(_instance, source, destination);

    /// <summary>Leaky ReLU — identity for Bit (no negative values).</summary>
    public void LeakyReLU(ReadOnlySpan<Bit> x, Bit alpha, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>GELU — identity for Bit (0→0, 1→1).</summary>
    public void GELU(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Mish — identity for Bit.</summary>
    public void Mish(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>Swish — identity for Bit.</summary>
    public void Swish(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>ELU — identity for Bit (no negative values).</summary>
    public void ELU(ReadOnlySpan<Bit> x, Bit alpha, Span<Bit> destination) => x.CopyTo(destination);

    /// <summary>ReLU — identity for Bit (no negative values).</summary>
    public void ReLU(ReadOnlySpan<Bit> x, Span<Bit> destination) => x.CopyTo(destination);
}
