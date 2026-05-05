using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides numeric operations for the <see cref="bool"/> type, enabling
/// <c>Tensor&lt;bool&gt;</c> for boolean mask + conditional operations.
/// Mirrors <see cref="BitOperations"/>'s semantics — <see cref="bool"/> is
/// the public-API surface (no special wrapper struct needed) while
/// <see cref="Bit"/> is the Memory&lt;byte&gt;-friendly internal form.
/// </summary>
/// <remarks>
/// <para>Arithmetic semantics for bool follow boolean algebra to match
/// <see cref="BitOperations"/>: Add = OR, Multiply = AND, Subtract = XOR,
/// Negate = NOT. Zero is <c>false</c>; One is <c>true</c>.</para>
/// <para>Required for <see cref="LinearAlgebra.TensorBase{T}"/>'s static
/// initializer to succeed — the static <c>_numOps = MathHelper.GetNumericOperations&lt;T&gt;()</c>
/// runs at type init for <c>Tensor&lt;bool&gt;</c>, and would otherwise
/// throw <c>NotSupportedException</c> the moment any test or consumer
/// touched <c>Tensor&lt;bool&gt;</c>. That's exactly what
/// SafetensorsRoundTripTests + TensorEmbeddingLookupTapeTests hit on
/// net471.</para>
/// </remarks>
public class BoolOperations : INumericOperations<bool>
{
    private static readonly BoolOperations _instance = new BoolOperations();

    // ---- Scalar arithmetic ----
    public bool Add(bool a, bool b) => a | b;
    public bool Subtract(bool a, bool b) => a ^ b;
    public bool Multiply(bool a, bool b) => a & b;
    public bool Divide(bool a, bool b) => !b || (a & b);
    public bool Negate(bool a) => !a;
    public bool Zero => false;
    public bool One => true;

    // ---- Math ----
    public bool Sqrt(bool value) => value;
    public bool FromDouble(double value) => value != 0.0;
    public bool FromFloat(float value) => value != 0f;
    public float ToFloat(bool value) => value ? 1f : 0f;
    public double ToDouble(bool value) => value ? 1.0 : 0.0;
    public Half ToHalf(bool value) => value ? (Half)1f : (Half)0f;
    public bool FromHalf(Half value) => (float)value != 0f;
    public int ToInt32(bool value) => value ? 1 : 0;

    // ---- Comparison ----
    public bool GreaterThan(bool a, bool b) => a && !b;
    public bool LessThan(bool a, bool b) => !a && b;
    public bool GreaterThanOrEquals(bool a, bool b) => a || !b;
    public bool LessThanOrEquals(bool a, bool b) => !a || b;
    public bool Equals(bool a, bool b) => a == b;
    public int Compare(bool a, bool b) => a.CompareTo(b);

    // ---- Reductions / extrema ----
    public bool MinValue => false;
    public bool MaxValue => true;
    public bool Abs(bool value) => value;
    public bool Square(bool value) => value;
    public bool SignOrZero(bool value) => value;

    // ---- Functions that don't have a clean bool semantics ----
    public bool Exp(bool value) => true;            // e^0=1 (true), e^1≈2.7 (true)
    public bool Log(bool value) => false;            // log(0) undefined, log(1)=0
    public bool Power(bool b, bool e) => b & e;
    public bool Round(bool value) => value;
    public bool Floor(bool value) => value;
    public bool Ceiling(bool value) => value;
    public bool Frac(bool value) => false;
    public bool Sin(bool value) => false;
    public bool Cos(bool value) => false;

    // ---- Special-value predicates ----
    public bool IsNaN(bool value) => false;
    public bool IsInfinity(bool value) => false;

    // ---- Precision / acceleration ----
    public int PrecisionBits => 1;
    public bool SupportsCpuAcceleration => false;
    public bool SupportsGpuAcceleration => false;

    // ---- IVectorizedOperations<bool> overloads ----
    public void Add(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++) destination[i] = x[i] | y[i];
    }
    public void Subtract(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++) destination[i] = x[i] ^ y[i];
    }
    public void Multiply(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++) destination[i] = x[i] & y[i];
    }
    public void Divide(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++) destination[i] = Divide(x[i], y[i]);
    }
    public bool Dot(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y) => VectorizedOperationsFallback.Dot(_instance, x, y);
    public bool Sum(ReadOnlySpan<bool> x) => VectorizedOperationsFallback.Sum(_instance, x);
    public bool Max(ReadOnlySpan<bool> x) => VectorizedOperationsFallback.Max(_instance, x);
    public bool Min(ReadOnlySpan<bool> x) => VectorizedOperationsFallback.Min(_instance, x);
    public void Exp(ReadOnlySpan<bool> x, Span<bool> destination) => destination.Fill(true);
    public void Log(ReadOnlySpan<bool> x, Span<bool> destination) => destination.Fill(false);
    public void Tanh(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void Sigmoid(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void Log2(ReadOnlySpan<bool> x, Span<bool> destination) => destination.Fill(false);
    public void SoftMax(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public bool CosineSimilarity(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y) => VectorizedOperationsFallback.CosineSimilarity(_instance, x, y);
    public void Fill(Span<bool> destination, bool value) => destination.Fill(value);
    public void MultiplyScalar(ReadOnlySpan<bool> x, bool scalar, Span<bool> destination) => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);
    public void DivideScalar(ReadOnlySpan<bool> x, bool scalar, Span<bool> destination) => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);
    public void AddScalar(ReadOnlySpan<bool> x, bool scalar, Span<bool> destination) => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);
    public void SubtractScalar(ReadOnlySpan<bool> x, bool scalar, Span<bool> destination) => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);
    public void Sqrt(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void Abs(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void Negate(ReadOnlySpan<bool> x, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++) destination[i] = !x[i];
    }
    public void Clip(ReadOnlySpan<bool> x, bool min, bool max, Span<bool> destination) => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);
    public void Pow(ReadOnlySpan<bool> x, bool power, Span<bool> destination) => VectorizedOperationsFallback.Pow(_instance, x, power, destination);
    public void Copy(ReadOnlySpan<bool> source, Span<bool> destination) => source.CopyTo(destination);
    public void Floor(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void Ceiling(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void Frac(ReadOnlySpan<bool> x, Span<bool> destination) => destination.Fill(false);
    public void Sin(ReadOnlySpan<bool> x, Span<bool> destination) => destination.Fill(false);
    public void Cos(ReadOnlySpan<bool> x, Span<bool> destination) => destination.Fill(false);
    public void MultiplyAdd(ReadOnlySpan<bool> x, ReadOnlySpan<bool> y, bool scalar, Span<bool> destination) => VectorizedOperationsFallback.MultiplyAdd(_instance, x, y, scalar, destination);
    public void ToFloatSpan(ReadOnlySpan<bool> source, Span<float> destination) => VectorizedOperationsFallback.ToFloatSpan(_instance, source, destination);
    public void FromFloatSpan(ReadOnlySpan<float> source, Span<bool> destination) => VectorizedOperationsFallback.FromFloatSpan(_instance, source, destination);
    public void ToHalfSpan(ReadOnlySpan<bool> source, Span<Half> destination) => VectorizedOperationsFallback.ToHalfSpan(_instance, source, destination);
    public void FromHalfSpan(ReadOnlySpan<Half> source, Span<bool> destination) => VectorizedOperationsFallback.FromHalfSpan(_instance, source, destination);
    public void LeakyReLU(ReadOnlySpan<bool> x, bool alpha, Span<bool> destination) => x.CopyTo(destination);
    public void GELU(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void Mish(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void Swish(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
    public void ELU(ReadOnlySpan<bool> x, bool alpha, Span<bool> destination) => x.CopyTo(destination);
    public void ReLU(ReadOnlySpan<bool> x, Span<bool> destination) => x.CopyTo(destination);
}
