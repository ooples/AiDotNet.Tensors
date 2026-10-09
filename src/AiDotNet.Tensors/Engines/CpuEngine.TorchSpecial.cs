using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using static AiDotNet.Tensors.Helpers.SpecialFunctions;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The PyTorch element-wise math, comparison, bitwise and torch.special ops (see the matching
/// <see cref="IEngine"/> members). Values are evaluated in double precision; a differentiable op records a backward
/// built from its derivative.
/// </summary>
public partial class CpuEngine
{
    // y = f(x) element-wise; dy/dx = derivative(x, y) when the op is differentiable.
    private static Tensor<T> SpecialUnary<T>(string opName, Tensor<T> tensor, Func<double, double> f,
        Func<double, double, double>? derivative, Func<IEngine, Tensor<T>> replay)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        BackwardFunction<T>? backward = derivative is null ? null : SpecialBackward<T>.Unary;
        object[]? saved = derivative is null ? null : new object[] { derivative };
        var lazy = TryRecordLazyUnary(opName, tensor, replay, backward, saved);
        if (lazy != null) return lazy;
        var ops = MathHelper.GetNumericOperations<T>();
        var result = ElementwiseUnary(tensor, v => ops.FromDouble(f(ops.ToDouble(v))), opName);
        if (backward != null) DifferentiableOps.RecordUnary(opName, result, tensor, backward, saved);
        return result;
    }

    // y = f(a, b) element-wise over equal shapes; ∂y/∂a and ∂y/∂b = da/db(a, b, y) when given.
    private static Tensor<T> SpecialBinary<T>(string opName, Tensor<T> a, Tensor<T> b, Func<double, double, double> f,
        Func<double, double, double, double>? da, Func<double, double, double, double>? db, Func<IEngine, Tensor<T>> replay)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        bool differentiable = da is not null || db is not null;
        BackwardFunction<T>? backward = differentiable ? SpecialBackward<T>.Binary : null;
        object[]? saved = differentiable ? new object[] { new BinaryDerivatives(da, db) } : null;
        var lazy = TryRecordLazyBinary(opName, a, b, replay, backward, saved);
        if (lazy != null) return lazy;
        var ops = MathHelper.GetNumericOperations<T>();
        var result = ElementwiseBinary(a, b, (x, y) => ops.FromDouble(f(ops.ToDouble(x), ops.ToDouble(y))), opName);
        if (backward != null) DifferentiableOps.RecordBinary(opName, result, a, b, backward, saved);
        return result;
    }

    private static bool IsIntegral<T>()
        => typeof(T) == typeof(int) || typeof(T) == typeof(long) || typeof(T) == typeof(short) || typeof(T) == typeof(sbyte)
           || typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) || typeof(T) == typeof(uint) || typeof(T) == typeof(ulong);

    // An integer-only binary op (bitwise, gcd/lcm), as in PyTorch: floating-point tensors are refused.
    private static Tensor<T> IntegerBinary<T>(string opName, Tensor<T> a, Tensor<T> b, Func<long, long, long> f,
        Func<IEngine, Tensor<T>> replay)
    {
        if (!IsIntegral<T>())
            throw new NotSupportedException($"{opName} is defined for integer tensors only; got {typeof(T).Name}.");
        var lazy = TryRecordLazyBinary(opName, a, b, replay);
        if (lazy != null) return lazy;
        return ElementwiseBinary(a, b,
            (x, y) => FromLong<T>(f(ToLong(x), ToLong(y))), opName);
    }

    private static Tensor<T> Indicator<T>(string opName, Tensor<T> tensor, Func<double, bool> predicate,
        Func<IEngine, Tensor<T>> replay) => SpecialUnary(opName, tensor, v => predicate(v) ? 1.0 : 0.0, null, replay);

    private static readonly double TwoOverSqrtPi = 2.0 / Math.Sqrt(Math.PI);
    private static readonly double InvSqrtTwoPi = 1.0 / Math.Sqrt(2 * Math.PI);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorErf<T>(Tensor<T> tensor)
        => SpecialUnary("TensorErf", tensor, Erf, (x, _) => TwoOverSqrtPi * Math.Exp(-x * x), e => e.TensorErf(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogit<T>(Tensor<T> tensor, double? eps = null)
    {
        double lo = eps ?? double.NegativeInfinity, hi = eps.HasValue ? 1 - eps.Value : double.PositiveInfinity;
        return SpecialUnary("TensorLogit", tensor,
            x => { double c = Math.Min(Math.Max(x, lo), hi); return Math.Log(c / (1 - c)); },
            (x, _) => x < lo || x > hi ? 0 : 1 / (x * (1 - x)),
            e => e.TensorLogit(tensor, eps));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSinc<T>(Tensor<T> tensor)
        => SpecialUnary("TensorSinc", tensor,
            x => x == 0 ? 1 : double.IsInfinity(x) ? 0 : SpecialFunctions.SinPi(x) / (Math.PI * x),
            (x, _) => x == 0 || double.IsInfinity(x) ? 0
                : (SpecialFunctions.CosPi(x) * Math.PI * x - SpecialFunctions.SinPi(x)) / (Math.PI * x * x),
            e => e.TensorSinc(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDeg2Rad<T>(Tensor<T> tensor)
        => SpecialUnary("TensorDeg2Rad", tensor, x => x * (Math.PI / 180), (_, _) => Math.PI / 180, e => e.TensorDeg2Rad(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRad2Deg<T>(Tensor<T> tensor)
        => SpecialUnary("TensorRad2Deg", tensor, x => x * (180 / Math.PI), (_, _) => 180 / Math.PI, e => e.TensorRad2Deg(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPositive<T>(Tensor<T> tensor)
        => tensor ?? throw new ArgumentNullException(nameof(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSignbit<T>(Tensor<T> tensor)
        => Indicator("TensorSignbit", tensor, x => BitConverter.DoubleToInt64Bits(x) < 0, e => e.TensorSignbit(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIsPosInf<T>(Tensor<T> tensor)
        => Indicator("TensorIsPosInf", tensor, double.IsPositiveInfinity, e => e.TensorIsPosInf(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIsNegInf<T>(Tensor<T> tensor)
        => Indicator("TensorIsNegInf", tensor, double.IsNegativeInfinity, e => e.TensorIsNegInf(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIsReal<T>(Tensor<T> tensor)
        => Indicator("TensorIsReal", tensor, _ => true, e => e.TensorIsReal(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorGreaterEqual<T>(Tensor<T> a, Tensor<T> b)
        => SpecialBinary("TensorGreaterEqual", a, b, (x, y) => x >= y ? 1 : 0, null, null, e => e.TensorGreaterEqual(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLessEqual<T>(Tensor<T> a, Tensor<T> b)
        => SpecialBinary("TensorLessEqual", a, b, (x, y) => x <= y ? 1 : 0, null, null, e => e.TensorLessEqual(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHeaviside<T>(Tensor<T> tensor, Tensor<T> values)
        => SpecialBinary("TensorHeaviside", tensor, values, (x, v) => double.IsNaN(x) ? x : x < 0 ? 0 : x > 0 ? 1 : v,
            null, null, e => e.TensorHeaviside(tensor, values));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFmax<T>(Tensor<T> a, Tensor<T> b)
        => SpecialBinary("TensorFmax", a, b, (x, y) => double.IsNaN(x) ? y : double.IsNaN(y) ? x : Math.Max(x, y),
            (x, y, _) => x >= y || double.IsNaN(y) ? 1 : 0, (x, y, _) => x >= y || double.IsNaN(y) ? 0 : 1,
            e => e.TensorFmax(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFmin<T>(Tensor<T> a, Tensor<T> b)
        => SpecialBinary("TensorFmin", a, b, (x, y) => double.IsNaN(x) ? y : double.IsNaN(y) ? x : Math.Min(x, y),
            (x, y, _) => x <= y || double.IsNaN(y) ? 1 : 0, (x, y, _) => x <= y || double.IsNaN(y) ? 0 : 1,
            e => e.TensorFmin(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFloorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (IsIntegral<T>())
            return IntegerBinary("TensorFloorDivide", a, b, (x, y) => FloorDiv(x, y), e => e.TensorFloorDivide(a, b));
        return SpecialBinary("TensorFloorDivide", a, b, (x, y) => Math.Floor(x / y), null, null, e => e.TensorFloorDivide(a, b));
    }

    private static long FloorDiv(long x, long y)
    {
        if (y == 0) throw new DivideByZeroException("TensorFloorDivide: integer division by zero.");
        long q = x / y;
        return (x % y != 0) && ((x < 0) != (y < 0)) ? q - 1 : q;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorGcd<T>(Tensor<T> a, Tensor<T> b)
        => IntegerBinary("TensorGcd", a, b, Gcd, e => e.TensorGcd(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLcm<T>(Tensor<T> a, Tensor<T> b)
        => IntegerBinary("TensorLcm", a, b, (x, y) => { long g = Gcd(x, y); return g == 0 ? 0 : Math.Abs(x / g * y); },
            e => e.TensorLcm(a, b));

    private static long Gcd(long x, long y)
    {
        x = Math.Abs(x);
        y = Math.Abs(y);
        while (y != 0) { long t = x % y; x = y; y = t; }
        return x;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBitwiseAnd<T>(Tensor<T> a, Tensor<T> b)
        => IntegerBinary("TensorBitwiseAnd", a, b, (x, y) => x & y, e => e.TensorBitwiseAnd(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBitwiseOr<T>(Tensor<T> a, Tensor<T> b)
        => IntegerBinary("TensorBitwiseOr", a, b, (x, y) => x | y, e => e.TensorBitwiseOr(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBitwiseXor<T>(Tensor<T> a, Tensor<T> b)
        => IntegerBinary("TensorBitwiseXor", a, b, (x, y) => x ^ y, e => e.TensorBitwiseXor(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBitwiseNot<T>(Tensor<T> tensor)
    {
        if (!IsIntegral<T>())
            throw new NotSupportedException($"TensorBitwiseNot is defined for integer tensors only; got {typeof(T).Name}.");
        var lazy = TryRecordLazyUnary("TensorBitwiseNot", tensor, e => e.TensorBitwiseNot(tensor));
        if (lazy != null) return lazy;
        // ~x in the element type's own width (unsigned types wrap within it).
        return ElementwiseUnary(tensor, v => FromLong<T>(Mask<T>(~ToLong(v))), "TensorBitwiseNot");
    }

    private static long ToLong<T>(T value) => value switch
    {
        ulong u => unchecked((long)u),
        _ => Convert.ToInt64(value),
    };

    // The element type's value of a 64-bit result, wrapping (unchecked) as the type's own arithmetic would.
    private static T FromLong<T>(long value)
    {
        object boxed = unchecked(Type.GetTypeCode(typeof(T)) switch
        {
            TypeCode.SByte => (object)(sbyte)value,
            TypeCode.Byte => (byte)value,
            TypeCode.Int16 => (short)value,
            TypeCode.UInt16 => (ushort)value,
            TypeCode.Int32 => (int)value,
            TypeCode.UInt32 => (uint)value,
            TypeCode.UInt64 => (ulong)value,
            _ => value,
        });
        return (T)boxed;
    }

    // Reduces a 64-bit result to the element type's range (two's complement wrap for unsigned types).
    private static long Mask<T>(long value)
    {
        if (typeof(T) == typeof(byte)) return value & 0xFF;
        if (typeof(T) == typeof(ushort)) return value & 0xFFFF;
        if (typeof(T) == typeof(uint)) return value & 0xFFFFFFFFL;
        if (typeof(T) == typeof(sbyte)) return (sbyte)value;
        if (typeof(T) == typeof(short)) return (short)value;
        if (typeof(T) == typeof(int)) return (int)value;
        return value;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBitwiseLeftShift<T>(Tensor<T> a, Tensor<T> b)
        => IntegerBinary("TensorBitwiseLeftShift", a, b, (x, y) => Mask<T>(y >= 64 ? 0 : x << (int)y), e => e.TensorBitwiseLeftShift(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBitwiseRightShift<T>(Tensor<T> a, Tensor<T> b)
        => IntegerBinary("TensorBitwiseRightShift", a, b, (x, y) => y >= 64 ? (x < 0 ? -1 : 0) : x >> (int)y, e => e.TensorBitwiseRightShift(a, b));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNanSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        // where(isnan(x), 0, x) then sum: the recorded where routes the gradient to the non-NaN elements only.
        var (kept, _) = NonNanParts(tensor);
        return ReduceSum(kept, axes, keepDims);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNanMean<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var (kept, present) = NonNanParts(tensor);
        Tensor<T> count;
        using (new NoGradScope<T>()) count = ReduceSum(present, axes, keepDims);
        return TensorDivide(ReduceSum(kept, axes, keepDims), count);
    }

    // (x with NaN replaced by 0 through a recorded where, 1/0 indicator of the non-NaN elements).
    private (Tensor<T> Kept, Tensor<T> Present) NonNanParts<T>(Tensor<T> tensor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var source = tensor.IsContiguous ? tensor : tensor.Contiguous();
        var present = new Tensor<T>(source._shape);
        var span = source.AsSpan();
        var mask = present.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) mask[i] = double.IsNaN(ops.ToDouble(span[i])) ? ops.Zero : ops.One;
        return (TensorWhere(present, tensor, new Tensor<T>(source._shape)), present);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIgamma<T>(Tensor<T> a, Tensor<T> x)
        => SpecialBinary("TensorIgamma", a, x, Igamma, null, (s, v, _) => IgammaDensity(s, v), e => e.TensorIgamma(a, x));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIgammac<T>(Tensor<T> a, Tensor<T> x)
        => SpecialBinary("TensorIgammac", a, x, Igammac, null, (s, v, _) => -IgammaDensity(s, v), e => e.TensorIgammac(a, x));

    // ∂P(a, x)/∂x = e^{-x} x^{a-1} / Γ(a). As in PyTorch, no gradient flows to a.
    private static double IgammaDensity(double a, double x) => x <= 0 ? 0 : Math.Exp(-x + (a - 1) * Math.Log(x) - LogGamma(a));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMvlgamma<T>(Tensor<T> tensor, int p)
    {
        if (p < 1) throw new ArgumentOutOfRangeException(nameof(p), "p must be at least 1.");
        return SpecialUnary("TensorMvlgamma", tensor,
            x => x <= (p - 1) / 2.0 ? double.NaN : Mvlgamma(x, p),
            (x, _) => MvlgammaDerivative(x, p), e => e.TensorMvlgamma(tensor, p));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorEntr<T>(Tensor<T> tensor)
        => SpecialUnary("TensorEntr", tensor,
            x => double.IsNaN(x) ? x : x > 0 ? -x * Math.Log(x) : x == 0 ? 0 : double.NegativeInfinity,
            (x, _) => -(Math.Log(x) + 1), e => e.TensorEntr(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorErfcx<T>(Tensor<T> tensor)
        => SpecialUnary("TensorErfcx", tensor, Erfcx, (x, y) => 2 * x * y - TwoOverSqrtPi, e => e.TensorErfcx(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNdtr<T>(Tensor<T> tensor)
        => SpecialUnary("TensorNdtr", tensor, Ndtr, (x, _) => InvSqrtTwoPi * Math.Exp(-0.5 * x * x), e => e.TensorNdtr(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogNdtr<T>(Tensor<T> tensor)
        => SpecialUnary("TensorLogNdtr", tensor, LogNdtr, (x, y) => InvSqrtTwoPi * Math.Exp(-0.5 * x * x - y), e => e.TensorLogNdtr(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNdtri<T>(Tensor<T> tensor)
        => SpecialUnary("TensorNdtri", tensor, Ndtri, (_, y) => Math.Sqrt(2 * Math.PI) * Math.Exp(0.5 * y * y), e => e.TensorNdtri(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBesselJ0<T>(Tensor<T> tensor)
        => SpecialUnary("TensorBesselJ0", tensor, x => BesselJ(0, x), (x, _) => -BesselJ(1, x), e => e.TensorBesselJ0(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBesselJ1<T>(Tensor<T> tensor)
        => SpecialUnary("TensorBesselJ1", tensor, x => BesselJ(1, x),
            (x, y) => x == 0 ? 0.5 : BesselJ(0, x) - y / x, e => e.TensorBesselJ1(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBesselY0<T>(Tensor<T> tensor)
        => SpecialUnary("TensorBesselY0", tensor, x => BesselY(0, x), (x, _) => -BesselY(1, x), e => e.TensorBesselY0(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBesselY1<T>(Tensor<T> tensor)
        => SpecialUnary("TensorBesselY1", tensor, x => BesselY(1, x), (x, y) => BesselY(0, x) - y / x, e => e.TensorBesselY1(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorModifiedBesselI0<T>(Tensor<T> tensor)
        => SpecialUnary("TensorModifiedBesselI0", tensor, x => BesselI(0, x), (x, _) => BesselI(1, x), e => e.TensorModifiedBesselI0(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorModifiedBesselI1<T>(Tensor<T> tensor)
        => SpecialUnary("TensorModifiedBesselI1", tensor, x => BesselI(1, x),
            (x, y) => x == 0 ? 0.5 : BesselI(0, x) - y / x, e => e.TensorModifiedBesselI1(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorModifiedBesselK0<T>(Tensor<T> tensor)
        => SpecialUnary("TensorModifiedBesselK0", tensor, x => BesselK(0, x), (x, _) => -BesselK(1, x), e => e.TensorModifiedBesselK0(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorModifiedBesselK1<T>(Tensor<T> tensor)
        => SpecialUnary("TensorModifiedBesselK1", tensor, x => BesselK(1, x), (x, y) => -BesselK(0, x) - y / x, e => e.TensorModifiedBesselK1(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorScaledModifiedBesselK0<T>(Tensor<T> tensor)
        => SpecialUnary("TensorScaledModifiedBesselK0", tensor, x => BesselKScaled(0, x),
            (x, y) => y - BesselKScaled(1, x), e => e.TensorScaledModifiedBesselK0(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorScaledModifiedBesselK1<T>(Tensor<T> tensor)
        => SpecialUnary("TensorScaledModifiedBesselK1", tensor, x => BesselKScaled(1, x),
            (x, y) => y - BesselKScaled(0, x) - y / x, e => e.TensorScaledModifiedBesselK1(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSphericalBesselJ0<T>(Tensor<T> tensor)
        => SpecialUnary("TensorSphericalBesselJ0", tensor, SphericalBesselJ0,
            (x, _) => Math.Abs(x) < 1e-4 ? -x / 3 : (x * Math.Cos(x) - Math.Sin(x)) / (x * x), e => e.TensorSphericalBesselJ0(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAiryAi<T>(Tensor<T> tensor)
        => SpecialUnary("TensorAiryAi", tensor, AiryAi, (x, _) => AiryAiDerivative(x), e => e.TensorAiryAi(tensor));

    // Pₙ(x) per element, with the degree taken from n (rounded, as PyTorch truncates an integral n tensor).
    private static Tensor<T> PolynomialOp<T>(string opName, PolynomialKind kind, bool shifted, Tensor<T> x, Tensor<T> n,
        Func<IEngine, Tensor<T>> replay)
        => SpecialBinary(opName, x, n, (v, d) => double.IsNaN(d) ? double.NaN : Polynomial(kind, shifted ? 2 * v - 1 : v, PolynomialDegree(d)), null, null, replay);

    // The recurrence runs once per degree for each element; 2^20 steps is about a millisecond per element.
    private const int MaxPolynomialDegree = 1 << 20;

    // PyTorch casts the degree with static_cast<int64_t>, truncating toward zero (2.7 -> 2, -0.7 -> 0); a negative degree
    // evaluates to 0. A non-finite or larger degree is rejected: the cast is undefined for it, and the recurrence would not
    // finish in practical time.
    private static int PolynomialDegree(double d)
    {
        if (double.IsInfinity(d) || d > MaxPolynomialDegree)
            throw new ArgumentOutOfRangeException(nameof(d), $"polynomial degree must be finite and at most {MaxPolynomialDegree}, got {d}.");
        return d <= int.MinValue ? int.MinValue : (int)Math.Truncate(d);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorChebyshevPolynomialT<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorChebyshevPolynomialT", PolynomialKind.ChebyshevT, false, x, n, e => e.TensorChebyshevPolynomialT(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorChebyshevPolynomialU<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorChebyshevPolynomialU", PolynomialKind.ChebyshevU, false, x, n, e => e.TensorChebyshevPolynomialU(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorChebyshevPolynomialV<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorChebyshevPolynomialV", PolynomialKind.ChebyshevV, false, x, n, e => e.TensorChebyshevPolynomialV(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorChebyshevPolynomialW<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorChebyshevPolynomialW", PolynomialKind.ChebyshevW, false, x, n, e => e.TensorChebyshevPolynomialW(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorShiftedChebyshevPolynomialT<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorShiftedChebyshevPolynomialT", PolynomialKind.ChebyshevT, true, x, n, e => e.TensorShiftedChebyshevPolynomialT(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorShiftedChebyshevPolynomialU<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorShiftedChebyshevPolynomialU", PolynomialKind.ChebyshevU, true, x, n, e => e.TensorShiftedChebyshevPolynomialU(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorShiftedChebyshevPolynomialV<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorShiftedChebyshevPolynomialV", PolynomialKind.ChebyshevV, true, x, n, e => e.TensorShiftedChebyshevPolynomialV(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorShiftedChebyshevPolynomialW<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorShiftedChebyshevPolynomialW", PolynomialKind.ChebyshevW, true, x, n, e => e.TensorShiftedChebyshevPolynomialW(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHermitePolynomialH<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorHermitePolynomialH", PolynomialKind.HermiteH, false, x, n, e => e.TensorHermitePolynomialH(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHermitePolynomialHe<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorHermitePolynomialHe", PolynomialKind.HermiteHe, false, x, n, e => e.TensorHermitePolynomialHe(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLaguerrePolynomialL<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorLaguerrePolynomialL", PolynomialKind.LaguerreL, false, x, n, e => e.TensorLaguerrePolynomialL(x, n));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLegendrePolynomialP<T>(Tensor<T> x, Tensor<T> n)
        => PolynomialOp("TensorLegendrePolynomialP", PolynomialKind.LegendreP, false, x, n, e => e.TensorLegendrePolynomialP(x, n));

    /// <summary>The two partial derivatives of a binary op, ∂y/∂a and ∂y/∂b as functions of (a, b, y); null = none.</summary>
    private sealed record BinaryDerivatives(
        Func<double, double, double, double>? WithRespectToA, Func<double, double, double, double>? WithRespectToB);

    /// <summary>Backward passes of the element-wise ops above, from their derivatives.</summary>
    private static class SpecialBackward<T>
    {
        public static void Unary(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
            object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
        {
            var derivative = (Func<double, double, double>)savedState[0];
            var ops = MathHelper.GetNumericOperations<T>();
            var x = inputs[0].IsContiguous ? inputs[0] : inputs[0].Contiguous();
            var y = output.IsContiguous ? output : output.Contiguous();
            var dy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var gx = new Tensor<T>(x._shape);
            var xs = x.AsSpan();
            var ys = y.AsSpan();
            var gs = dy.AsSpan();
            var dst = gx.AsWritableSpan();
            for (int i = 0; i < dst.Length; i++)
                dst[i] = ops.FromDouble(ops.ToDouble(gs[i]) * derivative(ops.ToDouble(xs[i]), ops.ToDouble(ys[i])));
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }

        public static void Binary(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
            object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
        {
            var derivatives = (BinaryDerivatives)savedState[0];
            var ops = MathHelper.GetNumericOperations<T>();
            var a = inputs[0].IsContiguous ? inputs[0] : inputs[0].Contiguous();
            var b = inputs[1].IsContiguous ? inputs[1] : inputs[1].Contiguous();
            var y = output.IsContiguous ? output : output.Contiguous();
            var dy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            if (derivatives.WithRespectToA is { } da)
                DifferentiableOps.AccumulateGrad(grads, inputs[0], Partial(da, a, b, y, dy, ops), engine);
            if (derivatives.WithRespectToB is { } db)
                DifferentiableOps.AccumulateGrad(grads, inputs[1], Partial(db, a, b, y, dy, ops), engine);
        }

        private static Tensor<T> Partial(Func<double, double, double, double> d, Tensor<T> a, Tensor<T> b, Tensor<T> y,
            Tensor<T> dy, Interfaces.INumericOperations<T> ops)
        {
            var g = new Tensor<T>(a._shape);
            var av = a.AsSpan();
            var bv = b.AsSpan();
            var yv = y.AsSpan();
            var gv = dy.AsSpan();
            var dst = g.AsWritableSpan();
            for (int i = 0; i < dst.Length; i++)
                dst[i] = ops.FromDouble(ops.ToDouble(gv[i]) * d(ops.ToDouble(av[i]), ops.ToDouble(bv[i]), ops.ToDouble(yv[i])));
            return g;
        }
    }
}
