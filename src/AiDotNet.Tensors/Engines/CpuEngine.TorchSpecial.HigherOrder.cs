using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using static AiDotNet.Tensors.Helpers.SpecialFunctions;

namespace AiDotNet.Tensors.Engines;

/// <summary>The partials of a binary element-wise op as recorded engine ops of (engine, a, b, y); null = none.</summary>
internal sealed class RecordedBinaryDerivatives<T>
{
    internal RecordedBinaryDerivatives(Func<IEngine, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>>? withRespectToA,
        Func<IEngine, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>>? withRespectToB)
    {
        WithRespectToA = withRespectToA;
        WithRespectToB = withRespectToB;
    }

    internal Func<IEngine, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>>? WithRespectToA { get; }

    internal Func<IEngine, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>>? WithRespectToB { get; }
}

/// <summary>
/// The derivatives of the element-wise special functions and activations as recorded engine ops, used by their
/// backwards under create-graph (double backward, Hessians). Each is the same function as the host derivative the
/// first-order backward evaluates, so first-order values agree; written as recorded ops, the gradient it scales is
/// differentiable again, to any order whose ops are. Piecewise definitions select between branches with constant masks
/// (zero derivative of the switch, as in torch), and a branch that is singular where it is masked out reads a safe
/// replacement argument there so it stays finite.
/// </summary>
public partial class CpuEngine
{
    private static readonly double SqrtTwoPi = Math.Sqrt(2 * Math.PI);

    private static Tensor<T> DErf<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => BackwardFunctions<T>.Scale(e, e.TensorExp(e.TensorNegate(BackwardFunctions<T>.Squared(e, x))), TwoOverSqrtPi);

    private static Tensor<T> DNdtr<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => BackwardFunctions<T>.Scale(e, e.TensorExp(BackwardFunctions<T>.Scale(e, BackwardFunctions<T>.Squared(e, x), -0.5)), InvSqrtTwoPi);

    private static Tensor<T> DLogNdtr<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => BackwardFunctions<T>.Scale(e,
            e.TensorExp(e.TensorSubtract(BackwardFunctions<T>.Scale(e, BackwardFunctions<T>.Squared(e, x), -0.5), y)), InvSqrtTwoPi);

    private static Tensor<T> DNdtri<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => BackwardFunctions<T>.Scale(e, e.TensorExp(BackwardFunctions<T>.Scale(e, BackwardFunctions<T>.Squared(e, y), 0.5)), SqrtTwoPi);

    private static Tensor<T> DErfcx<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => BackwardFunctions<T>.Shift(e, BackwardFunctions<T>.Scale(e, e.TensorMultiply(x, y), 2), -TwoOverSqrtPi);

    private static Tensor<T> DEntr<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => e.TensorNegate(BackwardFunctions<T>.Shift(e, e.TensorLog(x), 1));

    // 1 / (x(1 - x)) inside the clamp range, 0 outside it (the forward is constant there).
    private static Tensor<T> DLogit<T>(IEngine e, Tensor<T> x, double lo, double hi)
    {
        var outside = BackwardFunctions<T>.HostMask(x, v => v < lo || v > hi);
        var xs = BackwardFunctions<T>.SafeArgument(e, x, outside, 0.5);
        var d = e.TensorReciprocal(e.TensorSubtract(xs, BackwardFunctions<T>.Squared(e, xs)));
        return BackwardFunctions<T>.Select(e, outside, BackwardFunctions<T>.ConstantLike(x, 0), d);
    }

    // (cos(πx)·πx - sin(πx)) / (πx²), masked to 0 at x = 0 (and ±∞), as torch's sinc backward.
    private static Tensor<T> DSinc<T>(IEngine e, Tensor<T> x, Tensor<T> y)
    {
        var singular = BackwardFunctions<T>.HostMask(x, v => v == 0 || double.IsInfinity(v));
        var xs = BackwardFunctions<T>.SafeArgument(e, x, singular, 1);
        var px = BackwardFunctions<T>.Scale(e, xs, Math.PI);
        var numerator = e.TensorSubtract(e.TensorMultiply(e.TensorCos(px), px), e.TensorSin(px));
        var d = e.TensorDivide(numerator, BackwardFunctions<T>.Scale(e, BackwardFunctions<T>.Squared(e, xs), Math.PI));
        return BackwardFunctions<T>.Select(e, singular, BackwardFunctions<T>.ConstantLike(x, 0), d);
    }

    // y / x with x = 0 replaced (the caller masks that element).
    private static Tensor<T> OutputOverInput<T>(IEngine e, Tensor<T> x, Tensor<T> y, Tensor<T> zeroMask)
        => e.TensorDivide(y, BackwardFunctions<T>.SafeArgument(e, x, zeroMask, 1));

    // f'(x) = g(x) - y/x, with the limit `atZero` at x = 0.
    private static Tensor<T> MinusOutputOverInput<T>(IEngine e, Tensor<T> x, Tensor<T> y, Tensor<T> g, double atZero)
    {
        var zero = BackwardFunctions<T>.HostMask(x, v => v == 0);
        var d = e.TensorSubtract(g, OutputOverInput(e, x, y, zero));
        return BackwardFunctions<T>.Select(e, zero, BackwardFunctions<T>.ConstantLike(x, atZero), d);
    }

    private static Tensor<T> DBesselJ1<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => MinusOutputOverInput(e, x, y, e.TensorBesselJ0(x), 0.5);

    private static Tensor<T> DBesselY1<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => e.TensorSubtract(e.TensorBesselY0(x), e.TensorDivide(y, x));

    private static Tensor<T> DModifiedBesselI1<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => MinusOutputOverInput(e, x, y, e.TensorModifiedBesselI0(x), 0.5);

    private static Tensor<T> DModifiedBesselK1<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => e.TensorSubtract(e.TensorNegate(e.TensorModifiedBesselK0(x)), e.TensorDivide(y, x));

    private static Tensor<T> DScaledModifiedBesselK0<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => e.TensorSubtract(y, e.TensorScaledModifiedBesselK1(x));

    private static Tensor<T> DScaledModifiedBesselK1<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => e.TensorSubtract(e.TensorSubtract(y, e.TensorScaledModifiedBesselK0(x)), e.TensorDivide(y, x));

    // (x cos x - sin x) / x², and its series -x/3 near 0 (as the host derivative).
    private static Tensor<T> DSphericalBesselJ0<T>(IEngine e, Tensor<T> x, Tensor<T> y)
    {
        var small = BackwardFunctions<T>.HostMask(x, v => Math.Abs(v) < 1e-4);
        var xs = BackwardFunctions<T>.SafeArgument(e, x, small, 1);
        var d = e.TensorDivide(e.TensorSubtract(e.TensorMultiply(xs, e.TensorCos(xs)), e.TensorSin(xs)),
            BackwardFunctions<T>.Squared(e, xs));
        return BackwardFunctions<T>.Select(e, small, BackwardFunctions<T>.Scale(e, x, -1.0 / 3), d);
    }

    // Ai'(x) has no closed form in the other ops, so it is its own recorded node: its value is the host derivative,
    // and its derivative Ai''(x) = x·Ai(x) is recorded, so every order stays connected.
    private static Tensor<T> AiryAiPrime<T>(Tensor<T> x)
        => SpecialUnary("TensorAiryAiPrime", x, AiryAiDerivative, (v, _) => v * AiryAi(v), e => AiryAiPrime(x),
            (e, v, _) => e.TensorMultiply(v, e.TensorAiryAi(v)));

    private static Tensor<T> DMvlgamma<T>(IEngine e, Tensor<T> x, int p)
    {
        Tensor<T>? sum = null;
        for (int j = 0; j < p; j++)
        {
            var term = e.TensorDigamma(j == 0 ? x : BackwardFunctions<T>.Shift(e, x, -j / 2.0));
            sum = sum is null ? term : e.TensorAdd(sum, term);
        }
        return sum!;
    }

    private static Tensor<T> DCelu<T>(IEngine e, Tensor<T> x, double alpha)
        => BackwardFunctions<T>.Select(e, BackwardFunctions<T>.HostMask(x, v => v > 0), BackwardFunctions<T>.ConstantLike(x, 1),
            e.TensorExp(BackwardFunctions<T>.Scale(e, x, 1 / alpha)));

    private static Tensor<T> DSoftsign<T>(IEngine e, Tensor<T> x, Tensor<T> y)
        => e.TensorReciprocal(BackwardFunctions<T>.Squared(e, BackwardFunctions<T>.Shift(e, e.TensorAbs(x), 1)));

    private static Tensor<T> DSurrogateSpike<T>(IEngine e, Tensor<T> v, double threshold, double beta)
    {
        var s = e.TensorSigmoid(BackwardFunctions<T>.Scale(e, BackwardFunctions<T>.Shift(e, v, -threshold), beta));
        return BackwardFunctions<T>.Scale(e, e.TensorMultiply(s, BackwardFunctions<T>.OneMinus(e, s)), beta);
    }

    // The fused-activation kernels' derivatives (CpuFusedOperations.GetDoubleActivationDerivative) as recorded ops,
    // with the same (single-precision) parameters the kernels read.
    private static Func<IEngine, Tensor<T>, Tensor<T>, Tensor<T>>? FusedKernelRecordedDerivative<T>(FusedActivationType type,
        FusedActivationParams? p)
    {
        switch (type)
        {
            case FusedActivationType.LiSHT:
                return (e, x, _) =>
                {
                    var t = e.Tanh(x);
                    return e.TensorAdd(t, e.TensorMultiply(x, BackwardFunctions<T>.OneMinus(e, BackwardFunctions<T>.Squared(e, t))));
                };
            case FusedActivationType.BentIdentity:
                return (e, x, _) => BackwardFunctions<T>.Shift(e,
                    e.TensorDivide(x, BackwardFunctions<T>.Scale(e, e.TensorSqrt(BackwardFunctions<T>.Shift(e, BackwardFunctions<T>.Squared(e, x), 1)), 2)), 1);
            case FusedActivationType.Gaussian:
                return (e, x, _) => BackwardFunctions<T>.Scale(e,
                    e.TensorMultiply(x, e.TensorExp(e.TensorNegate(BackwardFunctions<T>.Squared(e, x)))), -2);
            case FusedActivationType.ScaledTanh:
            {
                double a = p?.Alpha ?? 1.0, b = p?.Beta ?? 1.0;
                return (e, x, _) =>
                {
                    var th = e.Tanh(BackwardFunctions<T>.Scale(e, x, b));
                    return BackwardFunctions<T>.Scale(e, BackwardFunctions<T>.OneMinus(e, BackwardFunctions<T>.Squared(e, th)), a * b);
                };
            }
            case FusedActivationType.ThresholdedReLU:
            {
                double t = p?.Theta ?? 1.0;
                return (_, x, _) => BackwardFunctions<T>.HostMask(x, v => v > t);
            }
            case FusedActivationType.ISRU:
            {
                double a = p?.Alpha ?? 1.0;
                return (e, x, _) => e.TensorPow(BackwardFunctions<T>.Shift(e, BackwardFunctions<T>.Scale(e, BackwardFunctions<T>.Squared(e, x), a), 1),
                    MathHelper.GetNumericOperations<T>().FromDouble(-1.5));
            }
            case FusedActivationType.SQRBF:
            {
                double b = p?.Beta ?? 1.0;
                return (e, x, _) => BackwardFunctions<T>.Scale(e,
                    e.TensorMultiply(x, e.TensorExp(BackwardFunctions<T>.Scale(e, BackwardFunctions<T>.Squared(e, x), -b))), -2 * b);
            }
            default:
                return null;
        }
    }

    // fmax/fmin route the gradient to whichever operand was selected: constant selection masks.
    private static RecordedBinaryDerivatives<T> FminFmaxRecorded<T>(bool max)
    {
        Func<double, double, bool> takesA = max
            ? (x, y) => x >= y || double.IsNaN(y)
            : (x, y) => x <= y || double.IsNaN(y);
        return new RecordedBinaryDerivatives<T>(
            (_, a, b, _) => BackwardFunctions<T>.HostMask(a, b, takesA),
            (_, a, b, _) => BackwardFunctions<T>.HostMask(a, b, (x, y) => !takesA(x, y)));
    }

    // ∂P(a, x)/∂x = e^{-x} x^{a-1} / Γ(a) as recorded ops of x; a enters as a constant (no gradient flows to a, as in
    // PyTorch), and x ≤ 0 gives 0 like the host density.
    private static Tensor<T> DIgammaX<T>(IEngine e, Tensor<T> a, Tensor<T> x, double sign)
    {
        var nonPositive = BackwardFunctions<T>.HostMask(x, v => v <= 0);
        var xs = BackwardFunctions<T>.SafeArgument(e, x, nonPositive, 1);
        var aMinusOne = BackwardFunctions<T>.HostMap(a, v => v - 1);
        var logGammaA = BackwardFunctions<T>.HostMap(a, LogGamma);
        var exponent = e.TensorSubtract(e.TensorSubtract(e.TensorMultiply(aMinusOne, e.TensorLog(xs)), xs), logGammaA);
        var d = BackwardFunctions<T>.Scale(e, e.TensorExp(exponent), sign);
        return BackwardFunctions<T>.Select(e, nonPositive, BackwardFunctions<T>.ConstantLike(x, 0), d);
    }
}
