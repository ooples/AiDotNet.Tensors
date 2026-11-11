using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// The fused-activation epilogue <paramref name="activation"/> (with its <see cref="FusedActivationParams"/>) as
    /// recorded, twice-differentiable engine ops, for create-graph backwards of the fused ops (FusedLinear, fused
    /// convolutions, fused batch norm). The fused kernels apply the activation with host math (ActivationHandler),
    /// which records nothing, so a create-graph backward through them returned a gradient the second backward saw as a
    /// constant: the Hessian of a DenseLayer in eval mode was zero.
    /// </summary>
    /// <remarks>
    /// Each case computes the same function as the fused forward with the same defaults and parameters; the
    /// FusedActivationRecordedParityTests pin that, value for value, for every <see cref="FusedActivationType"/>.
    /// Gumbel-Softmax is stochastic; its designed gradient is the temperature-1 softmax relaxation's (as
    /// GumbelSoftmaxActivationHandler.ApplyBackward), so that is what it differentiates. Sign and BinarySpiking are
    /// piecewise constant: their derivative is zero, as in the fused backward, so their value enters as a constant.
    /// </remarks>
    /// <summary>
    /// Backward of a result copied into a caller's destination (FusedLinearInto under a tape): the destination is an
    /// identity of the recorded result, so its gradient passes through unchanged (and stays connected at any order).
    /// </summary>
    internal static void CopyIntoBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
        => DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, engine);

    /// <summary>A constant 1/0 tensor of x's shape: 1 where <paramref name="predicate"/> holds for x's value.</summary>
    private static Tensor<T> ConstantMask(Tensor<T> x, Func<double, bool> predicate)
    {
        var values = RoIAlignPlan.ToDoubles(x);
        var mask = new double[values.Length];
        for (int i = 0; i < mask.Length; i++) mask[i] = predicate(values[i]) ? 1.0 : 0.0;
        return RoIAlignPlan.FromDoubles<T>(mask, (int[])x._shape.Clone());
    }

    private const double SeluAlpha = 1.6732632423543772, SeluScale = 1.0507009873554805;

    /// <summary>d selu/dx = x &gt; 0 ? λ : λα·eˣ as recorded ops (the region mask is a constant).</summary>
    internal static Tensor<T> SeluDerivativeRecorded(IEngine e, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var positive = ConstantMask(x, v => v > 0);
        var negativeSlope = e.TensorMultiplyScalar(e.TensorExp(x), ops.FromDouble(SeluScale * SeluAlpha));
        return e.TensorAdd(e.TensorMultiplyScalar(positive, ops.FromDouble(SeluScale)),
            e.TensorMultiply(OneMinus(e, positive), negativeSlope));
    }

    /// <summary>d hardswish/dx = 0 (x &lt; -3), x/3 + 1/2 (-3 ≤ x ≤ 3), 1 (x &gt; 3) as recorded ops.</summary>
    internal static Tensor<T> HardSwishDerivativeRecorded(IEngine e, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var ramp = ConstantMask(x, v => v >= -3 && v <= 3);
        var above = ConstantMask(x, v => v > 3);
        var slope = e.TensorAddScalar(e.TensorMultiplyScalar(x, ops.FromDouble(1.0 / 3.0)), ops.FromDouble(0.5));
        return e.TensorAdd(e.TensorMultiply(ramp, slope), above);
    }

    /// <summary>
    /// PReLU as recorded ops: x ≥ 0 ? x : α̃·x, where α̃ routes the slope to each element as the forward does (one
    /// slope per element, one shared slope, or one per channel of an NCHW-style layout).
    /// </summary>
    internal static Tensor<T> PReluRecorded(IEngine e, Tensor<T> x, Tensor<T> alpha)
    {
        int n = x.Length, channels = alpha.Length;
        int spatial = x.Rank >= 4 ? x._shape[x.Rank - 2] * x._shape[x.Rank - 1] : 1;
        var to = new int[n];
        var from = new int[n];
        for (int i = 0; i < n; i++)
        {
            to[i] = i;
            from[i] = channels == n ? i : channels == 1 ? 0 : (i / spatial) % channels;
        }
        var routedSlope = SparseMap(alpha, (int[])x._shape.Clone(), to, from, null);
        var nonNegative = ConstantMask(x, v => v >= 0);
        return e.TensorAdd(e.TensorMultiply(nonNegative, x),
            e.TensorMultiply(OneMinus(e, nonNegative), e.TensorMultiply(routedSlope, x)));
    }

    /// <summary>x·W (+ b broadcast over the leading axes) as recorded ops: a FusedLinear's pre-activation.</summary>
    internal static Tensor<T> FusedLinearPreActivationRecorded(IEngine e, Tensor<T>[] inputs)
    {
        var z = e.TensorMatMul(inputs[0], inputs[1]);
        if (inputs.Length > 2 && inputs[2] is { } bias)
        {
            var row = new int[z.Rank];
            for (int i = 0; i < row.Length; i++) row[i] = 1;
            row[row.Length - 1] = bias.Length;
            z = e.TensorAdd(z, e.TensorBroadcastTo(e.Reshape(bias, row), z._shape));
        }
        return z;
    }

    internal static Tensor<T> ApplyFusedActivationRecorded(IEngine e, Tensor<T> x, FusedActivationType activation,
        FusedActivationParams? p)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        switch (activation)
        {
            case FusedActivationType.None: return x;
            case FusedActivationType.ReLU: return e.ReLU(x);
            case FusedActivationType.GELU: return e.GELU(x);
            case FusedActivationType.Sigmoid: return e.TensorSigmoid(x);
            case FusedActivationType.Tanh: return e.Tanh(x);
            // The handler's default slope is the double 0.01 (a float 0.01f would differ in the 9th digit).
            case FusedActivationType.LeakyReLU: return e.LeakyReLU(x, ops.FromDouble(p?.Alpha is float slope ? slope : 0.01));
            case FusedActivationType.Swish: return e.Swish(x);
            case FusedActivationType.Softmax: return e.Softmax(x, -1);
            case FusedActivationType.ELU: return e.ELU(x, p?.Alpha ?? 1f);
            case FusedActivationType.SELU: return e.TensorSELU(x);
            case FusedActivationType.Softplus: return e.Softplus(x);
            case FusedActivationType.Mish: return e.Mish(x);
            case FusedActivationType.HardSwish: return e.TensorHardSwish(x);
            case FusedActivationType.HardSigmoid: return e.TensorHardSigmoid(x);
            case FusedActivationType.HardTanh: return e.TensorHardtanh(x, -1.0, 1.0);
            case FusedActivationType.ReLU6: return e.TensorReLU6(x);
            case FusedActivationType.SoftSign: return e.TensorSoftsign(x);
            case FusedActivationType.CELU: return e.TensorCelu(x, p?.Alpha ?? 1f);
            case FusedActivationType.ThresholdedReLU: return e.TensorThresholdedRelu(x, p?.Theta ?? 1f);
            case FusedActivationType.ScaledTanh: return e.TensorScaledTanh(x, p?.Alpha ?? 1f, p?.Beta ?? 1f);
            case FusedActivationType.PReLU:
                return e.LeakyReLU(x, ops.FromDouble(p?.PReluSlope is { Length: > 0 } s ? s[0] : 0.25));
            case FusedActivationType.RReLU: return e.LeakyReLU(x, ops.FromDouble(p?.Alpha ?? 0.22916667f));
            case FusedActivationType.Softmin: return e.TensorSoftmin(x, -1);
            case FusedActivationType.BentIdentity: return e.TensorBentIdentity(x);
            case FusedActivationType.Gaussian: return e.TensorGaussian(x);
            case FusedActivationType.LiSHT: return e.TensorLisht(x);
            case FusedActivationType.ISRU: return e.TensorIsru(x, p?.Alpha ?? 1f);
            case FusedActivationType.SQRBF: return e.TensorSqrbf(x, p?.Beta ?? 1f);
            case FusedActivationType.LogSoftmax: return e.TensorLogSoftmax(x, -1);
            case FusedActivationType.LogSoftmin: return e.TensorLogSoftmin(x, -1);
            // The fused forms of these two differ from the engine ops of the same name (no max shift for Taylor; the
            // fused spherical softmax normalizes by the row norm), so they are composed from primitives to match.
            case FusedActivationType.SphericalSoftmax:
            {
                var norm = e.TensorSqrt(e.ReduceSum(e.TensorMultiply(x, x), new[] { x.Rank - 1 }, keepDims: true));
                return e.Softmax(e.TensorDivide(x, e.TensorBroadcastTo(norm, x._shape)), -1);
            }
            case FusedActivationType.TaylorSoftmax:
            {
                var t = e.TensorAddScalar(e.TensorAdd(x, e.TensorMultiplyScalar(e.TensorMultiply(x, x), ops.FromDouble(0.5))), ops.One);
                var sum = e.ReduceSum(t, new[] { x.Rank - 1 }, keepDims: true);
                return e.TensorDivide(t, e.TensorBroadcastTo(sum, x._shape));
            }
            case FusedActivationType.GumbelSoftmax: return e.Softmax(x, -1);
            case FusedActivationType.Sparsemax: return e.Sparsemax(x, -1);
            case FusedActivationType.Squash:
            {
                // x·‖x‖/(1 + ‖x‖²) over the last axis, as the fused forward.
                var r2 = e.ReduceSum(e.TensorMultiply(x, x), new[] { x.Rank - 1 }, keepDims: true);
                var scale = e.TensorDivide(e.TensorSqrt(r2), e.TensorAddScalar(r2, ops.One));
                return e.TensorMultiply(x, e.TensorBroadcastTo(scale, x._shape));
            }
            case FusedActivationType.Sign:
            case FusedActivationType.BinarySpiking:
            {
                // Zero derivative everywhere it is defined: a constant of the fused forward's value.
                var engine = e.As<CpuEngine>() ?? new CpuEngine();
                var handler = ActivationRegistry.Get(activation)
                    ?? throw new NotSupportedException($"No fused activation handler is registered for {activation}.");
                Tensor<T> value;
                using (new NoGradScope<T>()) value = handler.Apply(engine, x, p);
                // Kept on the graph as x·0 + value, so the input still receives its (all-zero) gradient, as the fused
                // backward and torch's sign backward give, instead of none.
                return e.TensorAdd(e.TensorMultiplyScalar(x, ops.Zero), value);
            }
            default:
                throw new NotSupportedException($"No recorded form of the fused activation {activation}.");
        }
    }
}
