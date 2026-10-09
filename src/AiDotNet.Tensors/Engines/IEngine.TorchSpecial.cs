using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Element-wise math, comparison, bitwise and special functions matching their PyTorch counterparts
/// (<c>torch.*</c> and <c>torch.special.*</c>). Binary ops take tensors of equal shape. Comparisons return 1/0 in
/// <typeparamref name="T"/>, as the existing comparison ops do. Special functions are evaluated in double precision.
/// </summary>
public partial interface IEngine
{
    /// <summary>The error function erf(x) (<c>torch.erf</c>).</summary>
    Tensor<T> TensorErf<T>(Tensor<T> tensor);

    /// <summary>log(x / (1 - x)) (<c>torch.logit</c>); with <paramref name="eps"/>, x is first clamped to [eps, 1 - eps].</summary>
    Tensor<T> TensorLogit<T>(Tensor<T> tensor, double? eps = null);

    /// <summary>The normalized sinc, sin(πx)/(πx), with sinc(0) = 1 (<c>torch.sinc</c>).</summary>
    Tensor<T> TensorSinc<T>(Tensor<T> tensor);

    /// <summary>Degrees to radians (<c>torch.deg2rad</c>).</summary>
    Tensor<T> TensorDeg2Rad<T>(Tensor<T> tensor);

    /// <summary>Radians to degrees (<c>torch.rad2deg</c>).</summary>
    Tensor<T> TensorRad2Deg<T>(Tensor<T> tensor);

    /// <summary>The input itself (<c>torch.positive</c>).</summary>
    Tensor<T> TensorPositive<T>(Tensor<T> tensor);

    /// <summary>1 where the sign bit is set (negative values, -0.0 and negative NaN payloads aside), else 0 (<c>torch.signbit</c>).</summary>
    Tensor<T> TensorSignbit<T>(Tensor<T> tensor);

    /// <summary>1 where the value is +∞, else 0 (<c>torch.isposinf</c>).</summary>
    Tensor<T> TensorIsPosInf<T>(Tensor<T> tensor);

    /// <summary>1 where the value is -∞, else 0 (<c>torch.isneginf</c>).</summary>
    Tensor<T> TensorIsNegInf<T>(Tensor<T> tensor);

    /// <summary>1 where the value is real: every element of a real-valued tensor (<c>torch.isreal</c>).</summary>
    Tensor<T> TensorIsReal<T>(Tensor<T> tensor);

    /// <summary>1 where a ≥ b, else 0 (<c>torch.ge</c> / <c>greater_equal</c>).</summary>
    Tensor<T> TensorGreaterEqual<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>1 where a ≤ b, else 0 (<c>torch.le</c> / <c>less_equal</c>).</summary>
    Tensor<T> TensorLessEqual<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>The Heaviside step: 0 for x &lt; 0, <paramref name="values"/> for x = 0, 1 for x &gt; 0 (<c>torch.heaviside</c>).</summary>
    Tensor<T> TensorHeaviside<T>(Tensor<T> tensor, Tensor<T> values);

    /// <summary>The element-wise maximum that ignores NaN: NaN only where both are NaN (<c>torch.fmax</c>).</summary>
    Tensor<T> TensorFmax<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>The element-wise minimum that ignores NaN: NaN only where both are NaN (<c>torch.fmin</c>).</summary>
    Tensor<T> TensorFmin<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>floor(a / b) (<c>torch.floor_divide</c>).</summary>
    Tensor<T> TensorFloorDivide<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>The greatest common divisor of integer-valued tensors (<c>torch.gcd</c>).</summary>
    Tensor<T> TensorGcd<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>The least common multiple of integer-valued tensors (<c>torch.lcm</c>).</summary>
    Tensor<T> TensorLcm<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Bitwise AND of integer tensors (<c>torch.bitwise_and</c>).</summary>
    Tensor<T> TensorBitwiseAnd<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Bitwise OR of integer tensors (<c>torch.bitwise_or</c>).</summary>
    Tensor<T> TensorBitwiseOr<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Bitwise XOR of integer tensors (<c>torch.bitwise_xor</c>).</summary>
    Tensor<T> TensorBitwiseXor<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Bitwise NOT of an integer tensor (<c>torch.bitwise_not</c>).</summary>
    Tensor<T> TensorBitwiseNot<T>(Tensor<T> tensor);

    /// <summary>a shifted left by b bits (<c>torch.bitwise_left_shift</c>).</summary>
    Tensor<T> TensorBitwiseLeftShift<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>a shifted right (arithmetically) by b bits (<c>torch.bitwise_right_shift</c>).</summary>
    Tensor<T> TensorBitwiseRightShift<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>The sum over <paramref name="axes"/> (all when null) treating NaN as 0 (<c>torch.nansum</c>).</summary>
    Tensor<T> TensorNanSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false);

    /// <summary>The mean over <paramref name="axes"/> (all when null) of the non-NaN values (<c>torch.nanmean</c>).</summary>
    Tensor<T> TensorNanMean<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false);

    /// <summary>The regularized lower incomplete gamma function P(a, x) (<c>torch.special.gammainc</c> / <c>igamma</c>).</summary>
    Tensor<T> TensorIgamma<T>(Tensor<T> a, Tensor<T> x);

    /// <summary>The regularized upper incomplete gamma function Q(a, x) (<c>torch.special.gammaincc</c> / <c>igammac</c>).</summary>
    Tensor<T> TensorIgammac<T>(Tensor<T> a, Tensor<T> x);

    /// <summary>The multivariate log-gamma function of dimension <paramref name="p"/> (<c>torch.mvlgamma</c>).</summary>
    Tensor<T> TensorMvlgamma<T>(Tensor<T> tensor, int p);

    /// <summary>-x·ln(x) for x &gt; 0, 0 at 0, -∞ below (<c>torch.special.entr</c>).</summary>
    Tensor<T> TensorEntr<T>(Tensor<T> tensor);

    /// <summary>The scaled complementary error function exp(x²)·erfc(x) (<c>torch.special.erfcx</c>).</summary>
    Tensor<T> TensorErfcx<T>(Tensor<T> tensor);

    /// <summary>The standard normal CDF (<c>torch.special.ndtr</c>).</summary>
    Tensor<T> TensorNdtr<T>(Tensor<T> tensor);

    /// <summary>The log of the standard normal CDF (<c>torch.special.log_ndtr</c>).</summary>
    Tensor<T> TensorLogNdtr<T>(Tensor<T> tensor);

    /// <summary>The standard normal quantile function (<c>torch.special.ndtri</c>).</summary>
    Tensor<T> TensorNdtri<T>(Tensor<T> tensor);

    /// <summary>The Bessel function of the first kind of order 0 (<c>torch.special.bessel_j0</c>).</summary>
    Tensor<T> TensorBesselJ0<T>(Tensor<T> tensor);

    /// <summary>The Bessel function of the first kind of order 1 (<c>torch.special.bessel_j1</c>).</summary>
    Tensor<T> TensorBesselJ1<T>(Tensor<T> tensor);

    /// <summary>The Bessel function of the second kind of order 0 (<c>torch.special.bessel_y0</c>).</summary>
    Tensor<T> TensorBesselY0<T>(Tensor<T> tensor);

    /// <summary>The Bessel function of the second kind of order 1 (<c>torch.special.bessel_y1</c>).</summary>
    Tensor<T> TensorBesselY1<T>(Tensor<T> tensor);

    /// <summary>The modified Bessel function of the first kind of order 0 (<c>torch.special.modified_bessel_i0</c>).</summary>
    Tensor<T> TensorModifiedBesselI0<T>(Tensor<T> tensor);

    /// <summary>The modified Bessel function of the first kind of order 1 (<c>torch.special.modified_bessel_i1</c>).</summary>
    Tensor<T> TensorModifiedBesselI1<T>(Tensor<T> tensor);

    /// <summary>The modified Bessel function of the second kind of order 0 (<c>torch.special.modified_bessel_k0</c>).</summary>
    Tensor<T> TensorModifiedBesselK0<T>(Tensor<T> tensor);

    /// <summary>The modified Bessel function of the second kind of order 1 (<c>torch.special.modified_bessel_k1</c>).</summary>
    Tensor<T> TensorModifiedBesselK1<T>(Tensor<T> tensor);

    /// <summary>exp(x)·K₀(x) (<c>torch.special.scaled_modified_bessel_k0</c>).</summary>
    Tensor<T> TensorScaledModifiedBesselK0<T>(Tensor<T> tensor);

    /// <summary>exp(x)·K₁(x) (<c>torch.special.scaled_modified_bessel_k1</c>).</summary>
    Tensor<T> TensorScaledModifiedBesselK1<T>(Tensor<T> tensor);

    /// <summary>The spherical Bessel function of the first kind of order 0, sin(x)/x (<c>torch.special.spherical_bessel_j0</c>).</summary>
    Tensor<T> TensorSphericalBesselJ0<T>(Tensor<T> tensor);

    /// <summary>The Airy function Ai(x) (<c>torch.special.airy_ai</c>).</summary>
    Tensor<T> TensorAiryAi<T>(Tensor<T> tensor);

    /// <summary>The Chebyshev polynomial of the first kind Tₙ(x), degree per element of <paramref name="n"/>.</summary>
    Tensor<T> TensorChebyshevPolynomialT<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The Chebyshev polynomial of the second kind Uₙ(x).</summary>
    Tensor<T> TensorChebyshevPolynomialU<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The Chebyshev polynomial of the third kind Vₙ(x).</summary>
    Tensor<T> TensorChebyshevPolynomialV<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The Chebyshev polynomial of the fourth kind Wₙ(x).</summary>
    Tensor<T> TensorChebyshevPolynomialW<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The shifted Chebyshev polynomial of the first kind Tₙ(2x - 1).</summary>
    Tensor<T> TensorShiftedChebyshevPolynomialT<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The shifted Chebyshev polynomial of the second kind Uₙ(2x - 1).</summary>
    Tensor<T> TensorShiftedChebyshevPolynomialU<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The shifted Chebyshev polynomial of the third kind Vₙ(2x - 1).</summary>
    Tensor<T> TensorShiftedChebyshevPolynomialV<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The shifted Chebyshev polynomial of the fourth kind Wₙ(2x - 1).</summary>
    Tensor<T> TensorShiftedChebyshevPolynomialW<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The physicists' Hermite polynomial Hₙ(x).</summary>
    Tensor<T> TensorHermitePolynomialH<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The probabilists' Hermite polynomial Heₙ(x).</summary>
    Tensor<T> TensorHermitePolynomialHe<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The Laguerre polynomial Lₙ(x).</summary>
    Tensor<T> TensorLaguerrePolynomialL<T>(Tensor<T> x, Tensor<T> n);

    /// <summary>The Legendre polynomial Pₙ(x).</summary>
    Tensor<T> TensorLegendrePolynomialP<T>(Tensor<T> x, Tensor<T> n);
}
