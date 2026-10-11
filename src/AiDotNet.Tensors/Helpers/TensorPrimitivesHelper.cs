using System;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using static AiDotNet.Tensors.ErrorMessages;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides type-safe wrappers around vectorized operations for generic type T.
/// Uses SIMD-optimized implementations when available (float, double), falls back to sequential loops otherwise.
/// </summary>
/// <typeparam name="T">The numeric type for tensor operations.</typeparam>
/// <remarks>
/// <para>
/// This helper class leverages the polymorphic IVectorizedOperations interface to provide
/// hardware-accelerated operations. Float and double types use TensorPrimitives for SIMD
/// acceleration (SSE, AVX, AVX2, AVX-512), while other types use sequential fallback implementations.
/// </para>
/// <para><b>Performance Characteristics:</b>
/// - float/double: 5-15x speedup via SIMD (TensorPrimitives)
/// - Other types: Sequential loops (no SIMD)
///
/// <b>Design:</b>
/// The dispatch is handled via polymorphism through INumericOperations, which extends
/// IVectorizedOperations. Each numeric type implementation provides its own optimized
/// vectorized operations, following the Open/Closed principle.
/// </para>
/// </remarks>
public static class TensorPrimitivesHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    #region Vector Operations

    /// <summary>
    /// Performs element-wise addition.
    /// </summary>
    public static Vector<T> Add(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var result = new Vector<T>(x.Length);
        NumOps.Add(x.AsSpan(), y.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Performs element-wise subtraction.
    /// </summary>
    public static Vector<T> Subtract(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var result = new Vector<T>(x.Length);
        NumOps.Subtract(x.AsSpan(), y.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Performs element-wise multiplication.
    /// </summary>
    public static Vector<T> Multiply(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var result = new Vector<T>(x.Length);
        NumOps.Multiply(x.AsSpan(), y.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Performs element-wise division.
    /// </summary>
    public static Vector<T> Divide(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var result = new Vector<T>(x.Length);
        NumOps.Divide(x.AsSpan(), y.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes dot product: sum(x[i] * y[i]).
    /// </summary>
    public static T Dot(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        return NumOps.Dot(x.AsSpan(), y.AsSpan());
    }

    /// <summary>
    /// Computes sum of all elements.
    /// </summary>
    public static T Sum(Vector<T> x)
    {
        return NumOps.Sum(x.AsSpan());
    }

    /// <summary>
    /// Finds maximum value.
    /// </summary>
    public static T Max(Vector<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException(VectorCannotBeEmpty);

        return NumOps.Max(x.AsSpan());
    }

    /// <summary>
    /// Finds minimum value.
    /// </summary>
    public static T Min(Vector<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException(VectorCannotBeEmpty);

        return NumOps.Min(x.AsSpan());
    }

    /// <summary>
    /// Computes exponential element-wise: exp(x).
    /// </summary>
    public static Vector<T> Exp(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.Exp(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes natural logarithm element-wise: log(x).
    /// </summary>
    public static Vector<T> Log(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.Log(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes square root element-wise: sqrt(x).
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types via TensorPrimitives.
    /// </remarks>
    public static Vector<T> Sqrt(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.Sqrt(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes hyperbolic tangent element-wise: tanh(x).
    /// </summary>
    public static Vector<T> Tanh(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.Tanh(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes sigmoid element-wise: 1 / (1 + exp(-x)).
    /// </summary>
    public static Vector<T> Sigmoid(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.Sigmoid(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes LeakyReLU element-wise: x if x > 0, alpha * x otherwise.
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types via hardware intrinsics.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    /// <param name="alpha">Negative slope coefficient (typically 0.01).</param>
    public static Vector<T> LeakyReLU(Vector<T> x, double alpha = 0.01)
    {
        var result = new Vector<T>(x.Length);
        T alphaT = NumOps.FromDouble(alpha);
        NumOps.LeakyReLU(x.AsSpan(), alphaT, result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes GELU (Gaussian Error Linear Unit) element-wise.
    /// Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    public static Vector<T> GELU(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.GELU(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes Mish activation element-wise: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    public static Vector<T> Mish(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.Mish(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes Swish/SiLU activation element-wise: x * sigmoid(x) = x / (1 + exp(-x)).
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    public static Vector<T> Swish(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.Swish(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes ELU (Exponential Linear Unit) element-wise: x if x > 0, alpha * (exp(x) - 1) otherwise.
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types via hardware intrinsics.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    /// <param name="alpha">Scale factor for negative values (typically 1.0).</param>
    public static Vector<T> ELU(Vector<T> x, double alpha = 1.0)
    {
        var result = new Vector<T>(x.Length);
        T alphaT = NumOps.FromDouble(alpha);
        NumOps.ELU(x.AsSpan(), alphaT, result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes base-2 logarithm element-wise: log2(x).
    /// </summary>
    public static Vector<T> Log2(Vector<T> x)
    {
        var result = new Vector<T>(x.Length);
        NumOps.Log2(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes softmax: exp(x - max) / sum(exp(x - max)).
    /// </summary>
    public static Vector<T> Softmax(Vector<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Vector cannot be empty", nameof(x));

        var result = new Vector<T>(x.Length);
        NumOps.SoftMax(x.AsSpan(), result.AsWritableSpan());

        return result;
    }

    /// <summary>
    /// Computes cosine similarity: dot(a, b) / (norm(a) * norm(b)).
    /// </summary>
    public static T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException(VectorsSameLength);

        return NumOps.CosineSimilarity(a.AsSpan(), b.AsSpan());
    }

    #endregion
}
