using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device gradients for the torch-parity element-wise kernels. The CPU engine differentiates these ops with a host
/// derivative lambda, so under a tape the GPU engine used to hand the whole op to the CPU engine: the forward read its
/// input back to the host and the backward evaluated the derivative there. Each derivative below is the CPU lambda's
/// formula written with engine ops instead, so a taped forward stays on the device and its backward runs on the device
/// too. An op missing here (AiryAi, whose derivative has no device kernel) keeps the CPU route.
/// </summary>
public partial class DirectGpuTensorEngine
{
    private const double TwoOverSqrtPi = 1.1283791670955126;
    private const double InvSqrtTwoPi = 0.3989422804014327;
    private const double SqrtTwoPi = 2.5066282746310002;

    /// <summary>Ops whose CPU forward records no tape node: under a tape the device forward is already the whole op.</summary>
    private static bool IsNonDifferentiableParityOp(TorchParityUnaryOp op)
        => op is TorchParityUnaryOp.Signbit or TorchParityUnaryOp.IsPosInf or TorchParityUnaryOp.IsNegInf;

    private static bool IsNonDifferentiableParityOp(TorchParityBinaryOp op)
        => op is TorchParityBinaryOp.GreaterEqual or TorchParityBinaryOp.LessEqual
            or TorchParityBinaryOp.Heaviside or TorchParityBinaryOp.FloorDivide;

    private static bool HasDeviceDerivative(TorchParityUnaryOp op) => op switch
    {
        TorchParityUnaryOp.Celu or TorchParityUnaryOp.Hardtanh or TorchParityUnaryOp.LogSigmoid
            or TorchParityUnaryOp.Softsign or TorchParityUnaryOp.Erf or TorchParityUnaryOp.Logit
            or TorchParityUnaryOp.Sinc or TorchParityUnaryOp.Mvlgamma or TorchParityUnaryOp.Entr
            or TorchParityUnaryOp.Erfcx or TorchParityUnaryOp.Ndtr or TorchParityUnaryOp.LogNdtr
            or TorchParityUnaryOp.Ndtri or TorchParityUnaryOp.BesselJ0 or TorchParityUnaryOp.BesselJ1
            or TorchParityUnaryOp.BesselY0 or TorchParityUnaryOp.BesselY1 or TorchParityUnaryOp.ModifiedBesselI0
            or TorchParityUnaryOp.ModifiedBesselI1 or TorchParityUnaryOp.ModifiedBesselK0
            or TorchParityUnaryOp.ModifiedBesselK1 or TorchParityUnaryOp.ScaledModifiedBesselK0
            or TorchParityUnaryOp.ScaledModifiedBesselK1 or TorchParityUnaryOp.SphericalBesselJ0 => true,
        _ => false,
    };

    private static bool HasDeviceDerivative(TorchParityBinaryOp op)
        => op is TorchParityBinaryOp.Fmax or TorchParityBinaryOp.Fmin
            or TorchParityBinaryOp.Igamma or TorchParityBinaryOp.Igammac;

    /// <summary>Records the CPU engine's node name ("Tensor" + op) with a backward that differentiates on the device.</summary>
    private static void RecordParityUnary<T>(TorchParityUnaryOp op, Tensor<T> result, Tensor<T> input, float p0, float p1)
    {
        if (IsNonDifferentiableParityOp(op)) return;
        DifferentiableOps.RecordUnary("Tensor" + op, result, input, ParityUnaryBackward<T>, new object[] { op, p0, p1 });
    }

    private static void RecordParityBinary<T>(TorchParityBinaryOp op, Tensor<T> result, Tensor<T> a, Tensor<T> b)
    {
        if (IsNonDifferentiableParityOp(op)) return;
        DifferentiableOps.RecordBinary("Tensor" + op, result, a, b, ParityBinaryBackward<T>, new object[] { op });
    }

    private static void ParityUnaryBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var derivative = ParityUnaryDerivative(engine, (TorchParityUnaryOp)savedState[0], inputs[0], output,
            (float)savedState[1], (float)savedState[2]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], engine.TensorMultiply(gradOutput, derivative), engine);
    }

    internal static void ParityBinaryBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var op = (TorchParityBinaryOp)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> a = inputs[0], b = inputs[1];
        switch (op)
        {
            case TorchParityBinaryOp.Fmax:
            case TorchParityBinaryOp.Fmin:
            {
                // da = 1 where a wins (a >= b for fmax, a <= b for fmin, or b is NaN), else 0; db = 1 - da.
                var wins = op == TorchParityBinaryOp.Fmax ? engine.TensorGreaterThan(a, b) : engine.TensorLessThan(a, b);
                var da = engine.TensorMaskedFill(engine.TensorAdd(wins, engine.TensorEquals(a, b)),
                    engine.TensorIsNan(b), ops.One);
                var db = engine.TensorAddScalar(engine.TensorNegate(da), ops.One);
                DifferentiableOps.AccumulateGrad(grads, a, engine.TensorMultiply(gradOutput, da), engine);
                DifferentiableOps.AccumulateGrad(grads, b, engine.TensorMultiply(gradOutput, db), engine);
                return;
            }
            case TorchParityBinaryOp.Igamma:
            case TorchParityBinaryOp.Igammac:
            {
                // ∂P(a, x)/∂x = e^{-x} x^{a-1} / Γ(a) for x > 0, else 0; no gradient flows to a (as in PyTorch).
                var density = engine.TensorExp(engine.TensorSubtract(
                    engine.TensorAdd(engine.TensorNegate(b), engine.TensorMultiply(engine.TensorAddScalar(a, ops.FromDouble(-1)), engine.TensorLog(b))),
                    engine.TensorLgamma(a)));
                density = engine.TensorMaskedFill(density, NotGreaterThan(engine, b, 0), ops.Zero);
                if (op == TorchParityBinaryOp.Igammac) density = engine.TensorNegate(density);
                DifferentiableOps.AccumulateGrad(grads, b, engine.TensorMultiply(gradOutput, density), engine);
                return;
            }
            default:
                throw new InvalidOperationException($"No device derivative for {op}.");
        }
    }

    // 1-bit masks from the engine's 0/1 comparisons, so a select never multiplies an infinite branch by zero.
    private static Tensor<Bit> GreaterThanMask<T>(IEngine e, Tensor<T> x, double v)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return e.TensorEqScalar(e.TensorGreaterThan(x, ops.FromDouble(v)), ops.One);
    }

    private static Tensor<Bit> NotGreaterThan<T>(IEngine e, Tensor<T> x, double v)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return e.TensorEqScalar(e.TensorGreaterThan(x, ops.FromDouble(v)), ops.Zero);
    }

    /// <summary>dy/dx of a parity unary op, from its input x and output y, as engine ops (the CPU lambda's formula).</summary>
    internal static Tensor<T> ParityUnaryDerivative<T>(IEngine e, TorchParityUnaryOp op, Tensor<T> x, Tensor<T> y, float p0, float p1)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T C(double v) => ops.FromDouble(v);
        Tensor<T> Scale(Tensor<T> t, double v) => e.TensorMultiplyScalar(t, C(v));
        Tensor<T> Shift(Tensor<T> t, double v) => e.TensorAddScalar(t, C(v));
        Tensor<T> YOverX() => e.TensorDivide(y, x);
        switch (op)
        {
            case TorchParityUnaryOp.Celu:          // x > 0 ? 1 : exp(x / alpha)
                return e.TensorMaskedFill(e.TensorExp(Scale(x, 1.0 / p0)), GreaterThanMask(e, x, 0), ops.One);
            case TorchParityUnaryOp.Hardtanh:      // min < x < max ? 1 : 0
                return e.TensorMultiply(e.TensorGreaterThan(x, C(p0)), e.TensorLessThan(x, C(p1)));
            case TorchParityUnaryOp.LogSigmoid:    // 1 / (1 + e^x)
                return e.TensorReciprocal(Shift(e.TensorExp(x), 1));
            case TorchParityUnaryOp.Softsign:      // 1 / (1 + |x|)^2
                return e.TensorReciprocal(e.TensorSquare(Shift(e.TensorAbs(x), 1)));
            case TorchParityUnaryOp.Erf:           // 2/sqrt(pi) e^{-x^2}
                return Scale(e.TensorExp(e.TensorNegate(e.TensorSquare(x))), TwoOverSqrtPi);
            case TorchParityUnaryOp.Logit:         // x outside [lo, hi] ? 0 : 1 / (x (1 - x))
            {
                var inner = e.TensorReciprocal(e.TensorMultiply(x, Shift(e.TensorNegate(x), 1)));
                var outside = e.TensorAdd(e.TensorLessThan(x, C(p0)), e.TensorGreaterThan(x, C(p1)));
                return e.TensorMaskedFill(inner, e.TensorEqScalar(outside, ops.One), ops.Zero);
            }
            case TorchParityUnaryOp.Sinc:          // x == 0 or ±inf ? 0 : (cos(pi x) pi x - sin(pi x)) / (pi x^2)
            {
                var px = Scale(x, Math.PI);
                var d = e.TensorDivide(e.TensorSubtract(e.TensorMultiply(e.TensorCos(px), px), e.TensorSin(px)),
                    Scale(e.TensorSquare(x), Math.PI));
                d = e.TensorMaskedFill(d, e.TensorEqScalar(x, ops.Zero), ops.Zero);
                return e.TensorMaskedFill(d, e.TensorIsInf(x), ops.Zero);
            }
            case TorchParityUnaryOp.Mvlgamma:      // sum_{j<p} digamma(x - j/2)
            {
                int p = (int)p0;
                var sum = e.TensorDigamma(x);
                for (int j = 1; j < p; j++) sum = e.TensorAdd(sum, e.TensorDigamma(Shift(x, -j / 2.0)));
                return sum;
            }
            case TorchParityUnaryOp.Entr:          // -(log x + 1)
                return e.TensorNegate(Shift(e.TensorLog(x), 1));
            case TorchParityUnaryOp.Erfcx:         // 2 x y - 2/sqrt(pi)
                return Shift(Scale(e.TensorMultiply(x, y), 2), -TwoOverSqrtPi);
            case TorchParityUnaryOp.Ndtr:          // e^{-x^2/2} / sqrt(2 pi)
                return Scale(e.TensorExp(Scale(e.TensorSquare(x), -0.5)), InvSqrtTwoPi);
            case TorchParityUnaryOp.LogNdtr:       // e^{-x^2/2 - y} / sqrt(2 pi)
                return Scale(e.TensorExp(e.TensorSubtract(Scale(e.TensorSquare(x), -0.5), y)), InvSqrtTwoPi);
            case TorchParityUnaryOp.Ndtri:         // sqrt(2 pi) e^{y^2/2}
                return Scale(e.TensorExp(Scale(e.TensorSquare(y), 0.5)), SqrtTwoPi);
            case TorchParityUnaryOp.BesselJ0:      // -J1
                return e.TensorNegate(e.TensorBesselJ1(x));
            case TorchParityUnaryOp.BesselJ1:      // x == 0 ? 1/2 : J0 - y / x
                return e.TensorMaskedFill(e.TensorSubtract(e.TensorBesselJ0(x), YOverX()), e.TensorEqScalar(x, ops.Zero), C(0.5));
            case TorchParityUnaryOp.BesselY0:      // -Y1
                return e.TensorNegate(e.TensorBesselY1(x));
            case TorchParityUnaryOp.BesselY1:      // Y0 - y / x
                return e.TensorSubtract(e.TensorBesselY0(x), YOverX());
            case TorchParityUnaryOp.ModifiedBesselI0:  // I1
                return e.TensorModifiedBesselI1(x);
            case TorchParityUnaryOp.ModifiedBesselI1:  // x == 0 ? 1/2 : I0 - y / x
                return e.TensorMaskedFill(e.TensorSubtract(e.TensorModifiedBesselI0(x), YOverX()), e.TensorEqScalar(x, ops.Zero), C(0.5));
            case TorchParityUnaryOp.ModifiedBesselK0:  // -K1
                return e.TensorNegate(e.TensorModifiedBesselK1(x));
            case TorchParityUnaryOp.ModifiedBesselK1:  // -K0 - y / x
                return e.TensorSubtract(e.TensorNegate(e.TensorModifiedBesselK0(x)), YOverX());
            case TorchParityUnaryOp.ScaledModifiedBesselK0:  // y - K1e
                return e.TensorSubtract(y, e.TensorScaledModifiedBesselK1(x));
            case TorchParityUnaryOp.ScaledModifiedBesselK1:  // y - K0e - y / x
                return e.TensorSubtract(e.TensorSubtract(y, e.TensorScaledModifiedBesselK0(x)), YOverX());
            case TorchParityUnaryOp.SphericalBesselJ0:  // |x| < 1e-4 ? -x/3 : (x cos x - sin x) / x^2
            {
                var d = e.TensorDivide(e.TensorSubtract(e.TensorMultiply(x, e.TensorCos(x)), e.TensorSin(x)), e.TensorSquare(x));
                var small = e.TensorEqScalar(e.TensorLessThan(e.TensorAbs(x), C(1e-4)), ops.One);
                return e.TensorWhere(small, Scale(x, -1.0 / 3.0), d);
            }
            default:
                throw new InvalidOperationException($"No device derivative for {op}.");
        }
    }
}
