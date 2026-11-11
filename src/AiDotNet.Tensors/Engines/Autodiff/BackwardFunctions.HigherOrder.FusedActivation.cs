using System;
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
            case FusedActivationType.LeakyReLU: return e.LeakyReLU(x, ops.FromDouble(p?.Alpha ?? 0.01f));
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
            case FusedActivationType.SphericalSoftmax: return e.SphericalSoftmax(x, -1);
            case FusedActivationType.TaylorSoftmax: return e.TaylorSoftmax(x, 2, -1);
            case FusedActivationType.GumbelSoftmax: return e.Softmax(x, -1);
            case FusedActivationType.Sparsemax: return e.Sparsemax(x, -1);
            case FusedActivationType.Squash: return e.TensorSquash(x, -1);
            case FusedActivationType.Sign:
            case FusedActivationType.BinarySpiking:
            {
                // Zero derivative everywhere it is defined: a constant of the fused forward's value.
                var engine = e.As<CpuEngine>() ?? new CpuEngine();
                var handler = ActivationRegistry.Get(activation)
                    ?? throw new NotSupportedException($"No fused activation handler is registered for {activation}.");
                using (new NoGradScope<T>())
                    return handler.Apply(engine, x, p);
            }
            default:
                throw new NotSupportedException($"No recorded form of the fused activation {activation}.");
        }
    }
}
