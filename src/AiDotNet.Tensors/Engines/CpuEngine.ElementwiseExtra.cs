using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Tensor-level unary maths that until now existed only over spans and <c>Vector&lt;T&gt;</c>.
/// </summary>
/// <remarks>
/// <para>
/// <c>torch.acosh</c>, <c>asinh</c>, <c>exp2</c>, <c>expm1</c>, <c>log10</c>, <c>log1p</c>, <c>log2</c>,
/// <c>rsqrt</c>, <c>tan</c> and <c>trunc</c> were recorded as covered by the <c>Vector&lt;T&gt;</c> overloads of the
/// same name, which take no tensor and sit outside autodiff, so a tensor caller had nothing to call and a gradient
/// through them did not exist. These record on the tape like <see cref="TensorAtanh{T}"/>, and each has an in-place
/// form (<c>acosh_</c> ...).
/// </para>
/// </remarks>
public partial class CpuEngine
{
    private const double Ln2 = 0.69314718055994530942;
    private const double Ln10 = 2.30258509299404568402;

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAcosh<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorAcosh", AcoshScalar,
            static (eng, t) => eng.TensorAcosh(t), BackwardFunctions<T>.AcoshBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAsinh<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorAsinh", AsinhScalar,
            static (eng, t) => eng.TensorAsinh(t), BackwardFunctions<T>.AsinhBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorExp2<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorExp2", static x => Math.Pow(2.0, x),
            static (eng, t) => eng.TensorExp2(t), BackwardFunctions<T>.Exp2Backward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorExpm1<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorExpm1", Expm1Scalar,
            static (eng, t) => eng.TensorExpm1(t), BackwardFunctions<T>.Expm1Backward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLog10<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorLog10", Math.Log10,
            static (eng, t) => eng.TensorLog10(t), BackwardFunctions<T>.Log10Backward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLog1p<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorLog1p", Log1pScalar,
            static (eng, t) => eng.TensorLog1p(t), BackwardFunctions<T>.Log1pBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLog2<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorLog2", Log2Scalar,
            static (eng, t) => eng.TensorLog2(t), BackwardFunctions<T>.Log2Backward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRsqrt<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorRsqrt", static x => 1.0 / Math.Sqrt(x),
            static (eng, t) => eng.TensorRsqrt(t), BackwardFunctions<T>.RsqrtBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTan<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorTan", Math.Tan,
            static (eng, t) => eng.TensorTan(t), BackwardFunctions<T>.TanBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTrunc<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorTrunc", Math.Truncate,
            static (eng, t) => eng.TensorTrunc(t), BackwardFunctions<T>.TruncBackward);

    /// <summary>acosh with torch's edges: NaN below 1, +infinity at +infinity.</summary>
    private static double AcoshScalar(double x)
    {
#if NET5_0_OR_GREATER
        return Math.Acosh(x);
#else
        if (x < 1.0 || double.IsNaN(x)) return double.NaN;
        // log(x + sqrt(x^2 - 1)) overflows x^2 for large x; there acosh(x) = log(2x) to double precision.
        if (x > 1e8) return Math.Log(x) + Ln2;
        double t = x - 1.0;
        // log1p keeps the digits of x just above 1, where x + sqrt(x^2 - 1) is 1 + (small).
        return Log1pScalar(t + Math.Sqrt(2.0 * t + t * t));
#endif
    }

    /// <summary>asinh, odd and accurate for tiny and huge |x|.</summary>
    private static double AsinhScalar(double x)
    {
#if NET5_0_OR_GREATER
        return Math.Asinh(x);
#else
        if (double.IsNaN(x) || double.IsInfinity(x)) return x;
        double a = Math.Abs(x);
        double r = a > 1e8
            ? Math.Log(a) + Ln2
            : Log1pScalar(a + a * a / (1.0 + Math.Sqrt(1.0 + a * a)));
        return x < 0 ? -r : r;
#endif
    }

    /// <summary>
    /// log(1 + x) without losing a tiny x to the rounding of 1 + x (Kahan: the ratio x / (u - 1) corrects for exactly
    /// the rounding error u carries).
    /// </summary>
    private static double Log1pScalar(double x)
    {
        double u = 1.0 + x;
        if (u == 1.0) return x;
        if (double.IsInfinity(u)) return u;
        // The ratio first: Math.Log(u) * x overflows for x near double.MaxValue although the result is finite.
        return Math.Log(u) * (x / (u - 1.0));
    }

    /// <summary>exp(x) - 1 without cancellation near 0 (Kahan's correction, as in <see cref="Log1pScalar"/>).</summary>
    private static double Expm1Scalar(double x)
    {
        double u = Math.Exp(x);
        if (u == 1.0) return x;
        double um1 = u - 1.0;
        if (um1 == -1.0) return -1.0;
        if (double.IsInfinity(u)) return u;
        // The ratio first: for x just under ln(double.MaxValue), um1 * x overflows although the result is finite.
        return um1 * (x / Math.Log(u));
    }

    private static double Log2Scalar(double x)
    {
#if NETCOREAPP3_0_OR_GREATER
        return Math.Log2(x);
#else
        return Math.Log(x) / Ln2;
#endif
    }
}
