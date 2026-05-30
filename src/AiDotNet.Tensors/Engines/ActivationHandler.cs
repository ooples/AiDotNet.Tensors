using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Base class for fused activation handlers. Each activation type is a separate
/// implementation — adding new activations requires only a new class and one
/// registration line (Open/Closed Principle).
/// </summary>
internal abstract class ActivationHandler
{
    /// <summary>Applies this activation, returning a new tensor.</summary>
    public abstract Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input);

    /// <summary>
    /// Applies this activation in-place on the tensor.
    /// Default implementation allocates + copies back. Override for true in-place.
    /// </summary>
    public virtual void ApplyInPlace<T>(CpuEngine engine, Tensor<T> input)
    {
        var result = Apply(engine, input);
        result.AsSpan().CopyTo(input.AsWritableSpan());
    }

    /// <summary>
    /// Computes the backward pass: gradOutput * activation'(input).
    /// <paramref name="input"/> is the pre-activation tensor (before activation was applied).
    /// Activations that only need post-activation (Sigmoid, Tanh) can compute it
    /// from the pre-activation input, or override to accept it via other means.
    /// </summary>
    public abstract Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input);
}

/// <summary>
/// Registry mapping <see cref="FusedActivationType"/> to <see cref="ActivationHandler"/>.
/// To add a new activation: create a handler class and add one entry here.
/// No existing code needs to be modified.
/// </summary>
internal static class ActivationRegistry
{
    private static readonly ConcurrentDictionary<FusedActivationType, ActivationHandler> _handlers = new(
        new[]
        {
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.ReLU, new ReLUActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.GELU, new GELUActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Sigmoid, new SigmoidActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Tanh, new TanhActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.LeakyReLU, new LeakyReLUActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Swish, new SwishActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Softmax, new SoftmaxActivationHandler()),

            // #499: backward handlers for every remaining FusedActivationType so the
            // fused-linear backward pass never throws "No handler registered". Pointwise
            // activations reuse the canonical fused forward delegate + its exact analytic
            // derivative (CpuFusedOperations.GetXxxActivation[Derivative]); row-wise /
            // Jacobian families reuse the engine's dedicated backward kernels.
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.ELU, new PointwiseFusedActivationHandler(FusedActivationType.ELU)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.SELU, new PointwiseFusedActivationHandler(FusedActivationType.SELU)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Softplus, new PointwiseFusedActivationHandler(FusedActivationType.Softplus)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Mish, new PointwiseFusedActivationHandler(FusedActivationType.Mish)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.HardSwish, new PointwiseFusedActivationHandler(FusedActivationType.HardSwish)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.HardSigmoid, new PointwiseFusedActivationHandler(FusedActivationType.HardSigmoid)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.HardTanh, new PointwiseFusedActivationHandler(FusedActivationType.HardTanh)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.ReLU6, new PointwiseFusedActivationHandler(FusedActivationType.ReLU6)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.SoftSign, new PointwiseFusedActivationHandler(FusedActivationType.SoftSign)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.CELU, new PointwiseFusedActivationHandler(FusedActivationType.CELU)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.ThresholdedReLU, new PointwiseFusedActivationHandler(FusedActivationType.ThresholdedReLU)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.ScaledTanh, new PointwiseFusedActivationHandler(FusedActivationType.ScaledTanh)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.RReLU, new PointwiseFusedActivationHandler(FusedActivationType.RReLU)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Sign, new PointwiseFusedActivationHandler(FusedActivationType.Sign)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.BentIdentity, new PointwiseFusedActivationHandler(FusedActivationType.BentIdentity)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Gaussian, new PointwiseFusedActivationHandler(FusedActivationType.Gaussian)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.LiSHT, new PointwiseFusedActivationHandler(FusedActivationType.LiSHT)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.ISRU, new PointwiseFusedActivationHandler(FusedActivationType.ISRU)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.SQRBF, new PointwiseFusedActivationHandler(FusedActivationType.SQRBF)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.BinarySpiking, new PointwiseFusedActivationHandler(FusedActivationType.BinarySpiking)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.PReLU, new PReLUFusedActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Softmin, new SoftminActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.LogSoftmax, new LogSoftmaxActivationHandler(negate: false)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.LogSoftmin, new LogSoftmaxActivationHandler(negate: true)),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.SphericalSoftmax, new SphericalSoftmaxActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.TaylorSoftmax, new TaylorSoftmaxActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.GumbelSoftmax, new GumbelSoftmaxActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Sparsemax, new SparsemaxActivationHandler()),
            new KeyValuePair<FusedActivationType, ActivationHandler>(FusedActivationType.Squash, new SquashActivationHandler()),
        });

    /// <summary>Gets the handler for a given activation type, or null for None.</summary>
    public static ActivationHandler? Get(FusedActivationType activation)
    {
        if (activation == FusedActivationType.None)
            return null;
        if (_handlers.TryGetValue(activation, out var handler))
            return handler;
        throw new ArgumentException($"No handler registered for activation type: {activation}");
    }

    /// <summary>Register a new activation handler at runtime (for extensibility).</summary>
    public static void Register(FusedActivationType type, ActivationHandler handler)
    {
        _handlers[type] = handler;
    }

    /// <summary>
    /// Maps a FusedActivationType to its corresponding fused LazyNodeType.
    /// OCP-compliant: add new mappings here when registering new activations.
    /// </summary>
    private static readonly ConcurrentDictionary<FusedActivationType, Compilation.LazyNodeType> _lazyNodeMap = new(
        new[]
        {
            new KeyValuePair<FusedActivationType, Compilation.LazyNodeType>(FusedActivationType.ReLU, Compilation.LazyNodeType.FusedLinearReLU),
            new KeyValuePair<FusedActivationType, Compilation.LazyNodeType>(FusedActivationType.GELU, Compilation.LazyNodeType.FusedLinearGELU),
            new KeyValuePair<FusedActivationType, Compilation.LazyNodeType>(FusedActivationType.Sigmoid, Compilation.LazyNodeType.FusedLinearSigmoid),
        });

    /// <summary>Gets the fused LazyNodeType for an activation, defaulting to FusedLinear for unknown.</summary>
    public static Compilation.LazyNodeType GetFusedLinearNodeType(FusedActivationType activation)
    {
        if (_lazyNodeMap.TryGetValue(activation, out var nodeType))
            return nodeType;
        return Compilation.LazyNodeType.FusedLinear;
    }

    /// <summary>
    /// Maps a LazyNodeType (activation op) to its corresponding FusedActivationType.
    /// Returns null if the node type is not a recognized activation.
    /// OCP-compliant: add new mappings here when registering new activations.
    /// </summary>
    private static readonly ConcurrentDictionary<Compilation.LazyNodeType, FusedActivationType> _nodeToActivation = new(
        new[]
        {
            new KeyValuePair<Compilation.LazyNodeType, FusedActivationType>(Compilation.LazyNodeType.ReLU, FusedActivationType.ReLU),
            new KeyValuePair<Compilation.LazyNodeType, FusedActivationType>(Compilation.LazyNodeType.GELU, FusedActivationType.GELU),
            new KeyValuePair<Compilation.LazyNodeType, FusedActivationType>(Compilation.LazyNodeType.Sigmoid, FusedActivationType.Sigmoid),
            new KeyValuePair<Compilation.LazyNodeType, FusedActivationType>(Compilation.LazyNodeType.Tanh, FusedActivationType.Tanh),
            new KeyValuePair<Compilation.LazyNodeType, FusedActivationType>(Compilation.LazyNodeType.Swish, FusedActivationType.Swish),
            new KeyValuePair<Compilation.LazyNodeType, FusedActivationType>(Compilation.LazyNodeType.LeakyReLU, FusedActivationType.LeakyReLU),
            new KeyValuePair<Compilation.LazyNodeType, FusedActivationType>(Compilation.LazyNodeType.Softmax, FusedActivationType.Softmax),
        });

    /// <summary>Tries to map a LazyNodeType to a FusedActivationType. Returns false for non-activation ops.</summary>
    public static bool TryGetActivationType(Compilation.LazyNodeType nodeType, out FusedActivationType activation)
    {
        return _nodeToActivation.TryGetValue(nodeType, out activation);
    }
}

// ==================== Concrete Handlers ====================
// Each handler is a standalone class. Adding a new activation = new class + one dictionary entry.

internal sealed class ReLUActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.ReLU(input);
    public override void ApplyInPlace<T>(CpuEngine engine, Tensor<T> input) => engine.ReLUInPlace(input);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
        => engine.ReluBackward(gradOutput, input);
}

internal sealed class GELUActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.GELU(input);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
        => engine.GeluBackward(gradOutput, input);
}

internal sealed class SigmoidActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Sigmoid(input);
    public override void ApplyInPlace<T>(CpuEngine engine, Tensor<T> input) => engine.SigmoidInPlace(input);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
    {
        // SigmoidBackward expects post-activation output: grad * s * (1 - s)
        var sigmoidOutput = engine.Sigmoid(input);
        return engine.SigmoidBackward(gradOutput, sigmoidOutput);
    }
}

internal sealed class TanhActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Tanh(input);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
    {
        // TanhBackward expects post-activation output: grad * (1 - tanh^2)
        var tanhOutput = engine.Tanh(input);
        return engine.TanhBackward(gradOutput, tanhOutput);
    }
}

internal sealed class LeakyReLUActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
        => engine.LeakyReLU(input, Helpers.MathHelper.GetNumericOperations<T>().FromDouble(0.01));
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
        => engine.LeakyReluBackward(gradOutput, input, 0.01);
}

internal sealed class SwishActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Swish(input);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
        => engine.SwishBackward(gradOutput, input);
}

internal sealed class SoftmaxActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Softmax(input, -1);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
    {
        // SoftmaxBackward expects post-activation output
        var softmaxOutput = engine.Softmax(input, -1);
        return engine.SoftmaxBackward(gradOutput, softmaxOutput, -1);
    }
}

// ==================== #499 backward handlers for the remaining activations ====================

/// <summary>
/// Pointwise fused activation backward: forward via the canonical fused delegate
/// (<see cref="Helpers.CpuFusedOperations.GetFloatActivation"/>) and backward =
/// gradOutput ⊙ f'(preActivation) via its exact analytic derivative. float/double only.
/// </summary>
internal sealed class PointwiseFusedActivationHandler : ActivationHandler
{
    private readonly FusedActivationType _type;
    public PointwiseFusedActivationHandler(FusedActivationType type) => _type = type;

    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
    {
        var result = new Tensor<T>(input._shape);
        int n = input.Length;
        if (typeof(T) == typeof(float))
        {
            var f = Helpers.CpuFusedOperations.GetFloatActivation(_type);
            var xi = (float[])(object)input.GetDataArray();
            var ro = (float[])(object)result.GetDataArray();
            for (int i = 0; i < n; i++) ro[i] = f(xi[i]);
        }
        else if (typeof(T) == typeof(double))
        {
            var f = Helpers.CpuFusedOperations.GetDoubleActivation(_type);
            var xi = (double[])(object)input.GetDataArray();
            var ro = (double[])(object)result.GetDataArray();
            for (int i = 0; i < n; i++) ro[i] = f(xi[i]);
        }
        else throw new NotSupportedException($"Fused activation backward supports float/double, not {typeof(T).Name}.");
        return result;
    }

    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
    {
        var result = new Tensor<T>(input._shape);
        int n = input.Length;
        if (typeof(T) == typeof(float))
        {
            var d = Helpers.CpuFusedOperations.GetFloatActivationDerivative(_type);
            var xi = (float[])(object)input.GetDataArray();
            var gi = (float[])(object)gradOutput.GetDataArray();
            var ro = (float[])(object)result.GetDataArray();
            for (int i = 0; i < n; i++) ro[i] = gi[i] * d(xi[i]);
        }
        else if (typeof(T) == typeof(double))
        {
            var d = Helpers.CpuFusedOperations.GetDoubleActivationDerivative(_type);
            var xi = (double[])(object)input.GetDataArray();
            var gi = (double[])(object)gradOutput.GetDataArray();
            var ro = (double[])(object)result.GetDataArray();
            for (int i = 0; i < n; i++) ro[i] = gi[i] * d(xi[i]);
        }
        else throw new NotSupportedException($"Fused activation backward supports float/double, not {typeof(T).Name}.");
        return result;
    }
}

/// <summary>PReLU with the canonical default slope 0.25: f = x&gt;0 ? x : 0.25·x; f' = x&gt;0 ? 1 : 0.25.</summary>
internal sealed class PReLUFusedActivationHandler : ActivationHandler
{
    private const double DefaultSlope = 0.25;
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        var slope = ops.FromDouble(DefaultSlope);
        var result = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var ro = result.AsWritableSpan();
        for (int i = 0; i < xi.Length; i++)
            ro[i] = ops.ToDouble(xi[i]) > 0.0 ? xi[i] : ops.Multiply(slope, xi[i]);
        return result;
    }
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        var slope = ops.FromDouble(DefaultSlope);
        var result = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var gi = gradOutput.AsSpan(); var ro = result.AsWritableSpan();
        for (int i = 0; i < xi.Length; i++)
            ro[i] = ops.Multiply(gi[i], ops.ToDouble(xi[i]) > 0.0 ? ops.One : slope);
        return result;
    }
}

/// <summary>Softmin = softmax(−x). Backward chains through the negation:
/// dx = −Jₛₒₓₜₘₐₓ(−x)ᵀ·g.</summary>
internal sealed class SoftminActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
        => engine.Softmax(FusedActivationBackwardMath.Negate(input), -1);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
    {
        var o = engine.Softmax(FusedActivationBackwardMath.Negate(input), -1);
        var g = engine.SoftmaxBackward(gradOutput, o, -1);
        return FusedActivationBackwardMath.Negate(g);
    }
}

/// <summary>LogSoftmax (and LogSoftmin when <c>negate</c>) over the last axis.
/// o = u − logΣe^u with u = ±x; VJP dx = ±(g − softmax(u)·Σg).</summary>
internal sealed class LogSoftmaxActivationHandler : ActivationHandler
{
    private readonly bool _negate;
    public LogSoftmaxActivationHandler(bool negate) => _negate = negate;
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
        => FusedActivationBackwardMath.LogSoftmaxForward(input, _negate);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
        => FusedActivationBackwardMath.LogSoftmaxBackward(gradOutput, input, _negate);
}

/// <summary>Spherical softmax = softmax(x/‖x‖₂), matching the fused forward. Backward
/// chains the softmax VJP through the L2-normalization Jacobian.</summary>
internal sealed class SphericalSoftmaxActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
        => FusedActivationBackwardMath.SphericalSoftmaxForward(input);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
        => FusedActivationBackwardMath.SphericalSoftmaxBackward(gradOutput, input);
}

/// <summary>Taylor softmax (2nd order): yᵢ = T(xᵢ)/ΣT, T(x)=1+x+x²/2, matching the fused
/// forward. VJP: dxₖ = (T'(xₖ)/S)(gₖ − Σᵢgᵢyᵢ), T'(x)=1+x.</summary>
internal sealed class TaylorSoftmaxActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
        => FusedActivationBackwardMath.TaylorSoftmaxForward(input);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
        => FusedActivationBackwardMath.TaylorSoftmaxBackward(gradOutput, input);
}

internal sealed class SparsemaxActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Sparsemax(input, -1);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
    {
        var output = engine.Sparsemax(input, -1);
        return engine.SparsemaxBackward(gradOutput, output, -1);
    }
}

/// <summary>Capsule squash yᵢ = xᵢ·‖x‖/(1+‖x‖²), matching the fused forward.</summary>
internal sealed class SquashActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
        => FusedActivationBackwardMath.SquashForward(input);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
        => FusedActivationBackwardMath.SquashBackward(gradOutput, input);
}

/// <summary>Gumbel-Softmax: the forward draws Gumbel noise (stochastic), so the fused
/// backward uses the deterministic softmax-temperature relaxation (the standard
/// reparameterization gradient at temperature 1) — gradients flow through the softmax,
/// not the non-differentiable noise.</summary>
internal sealed class GumbelSoftmaxActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.GumbelSoftmax(input, 1.0, false, -1);
    public override Tensor<T> ApplyBackward<T>(CpuEngine engine, Tensor<T> gradOutput, Tensor<T> input)
    {
        // Deterministic relaxation at temperature 1: J_softmax(x)ᵀ·g.
        var relaxed = engine.Softmax(input, -1);
        return engine.SoftmaxBackward(gradOutput, relaxed, -1);
    }
}

/// <summary>Shared generic math for the row-wise (last-axis) softmax-family backward
/// handlers. Computes in double precision internally for numerical stability.</summary>
internal static class FusedActivationBackwardMath
{
    public static Tensor<T> Negate<T>(Tensor<T> t)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        var r = new Tensor<T>(t._shape);
        var s = t.AsSpan(); var d = r.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) d[i] = ops.Negate(s[i]);
        return r;
    }

    public static Tensor<T> LogSoftmaxForward<T>(Tensor<T> input, bool negate)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int m = input._shape[input._shape.Length - 1];
        int rows = m == 0 ? 0 : input.Length / m;
        var r = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var ro = r.AsWritableSpan();
        for (int row = 0; row < rows; row++)
        {
            int off = row * m;
            double max = double.NegativeInfinity;
            for (int j = 0; j < m; j++)
            {
                double u = ops.ToDouble(xi[off + j]); if (negate) u = -u;
                if (u > max) max = u;
            }
            double sum = 0.0;
            for (int j = 0; j < m; j++)
            {
                double u = ops.ToDouble(xi[off + j]); if (negate) u = -u;
                sum += Math.Exp(u - max);
            }
            double lse = max + Math.Log(sum);
            for (int j = 0; j < m; j++)
            {
                double u = ops.ToDouble(xi[off + j]); if (negate) u = -u;
                ro[off + j] = ops.FromDouble(u - lse);
            }
        }
        return r;
    }

    public static Tensor<T> LogSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, bool negate)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int m = input._shape[input._shape.Length - 1];
        int rows = m == 0 ? 0 : input.Length / m;
        var r = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var gi = gradOutput.AsSpan(); var ro = r.AsWritableSpan();
        for (int row = 0; row < rows; row++)
        {
            int off = row * m;
            double max = double.NegativeInfinity;
            for (int j = 0; j < m; j++)
            {
                double u = ops.ToDouble(xi[off + j]); if (negate) u = -u;
                if (u > max) max = u;
            }
            double z = 0.0, sg = 0.0;
            for (int j = 0; j < m; j++)
            {
                double u = ops.ToDouble(xi[off + j]); if (negate) u = -u;
                z += Math.Exp(u - max);
                sg += ops.ToDouble(gi[off + j]);
            }
            for (int j = 0; j < m; j++)
            {
                double u = ops.ToDouble(xi[off + j]); if (negate) u = -u;
                double p = Math.Exp(u - max) / z;                 // softmax(u)_j
                double baseGrad = ops.ToDouble(gi[off + j]) - p * sg;
                ro[off + j] = ops.FromDouble(negate ? -baseGrad : baseGrad);
            }
        }
        return r;
    }

    // ---- Taylor softmax (2nd order): T(x)=1+x+x²/2, yᵢ=Tᵢ/ΣT ----
    public static Tensor<T> TaylorSoftmaxForward<T>(Tensor<T> input)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int m = input._shape[input._shape.Length - 1];
        int rows = m == 0 ? 0 : input.Length / m;
        var r = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var ro = r.AsWritableSpan();
        for (int row = 0; row < rows; row++)
        {
            int off = row * m;
            double s = 0.0;
            for (int j = 0; j < m; j++) { double x = ops.ToDouble(xi[off + j]); s += 1.0 + x + 0.5 * x * x; }
            for (int j = 0; j < m; j++) { double x = ops.ToDouble(xi[off + j]); ro[off + j] = ops.FromDouble((1.0 + x + 0.5 * x * x) / s); }
        }
        return r;
    }

    public static Tensor<T> TaylorSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int m = input._shape[input._shape.Length - 1];
        int rows = m == 0 ? 0 : input.Length / m;
        var r = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var gi = gradOutput.AsSpan(); var ro = r.AsWritableSpan();
        for (int row = 0; row < rows; row++)
        {
            int off = row * m;
            double s = 0.0, gy = 0.0;
            for (int j = 0; j < m; j++) { double x = ops.ToDouble(xi[off + j]); s += 1.0 + x + 0.5 * x * x; }
            for (int j = 0; j < m; j++)
            {
                double x = ops.ToDouble(xi[off + j]);
                double y = (1.0 + x + 0.5 * x * x) / s;
                gy += ops.ToDouble(gi[off + j]) * y;
            }
            for (int j = 0; j < m; j++)
            {
                double x = ops.ToDouble(xi[off + j]);
                double tPrime = 1.0 + x;                          // T'(x)
                ro[off + j] = ops.FromDouble((tPrime / s) * (ops.ToDouble(gi[off + j]) - gy));
            }
        }
        return r;
    }

    // ---- Spherical softmax: softmax(x/‖x‖₂) ----
    public static Tensor<T> SphericalSoftmaxForward<T>(Tensor<T> input)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int m = input._shape[input._shape.Length - 1];
        int rows = m == 0 ? 0 : input.Length / m;
        var r = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var ro = r.AsWritableSpan();
        var u = new double[m];
        for (int row = 0; row < rows; row++)
        {
            int off = row * m;
            double r2 = 0.0;
            for (int j = 0; j < m; j++) { double x = ops.ToDouble(xi[off + j]); r2 += x * x; }
            double norm = Math.Sqrt(r2);
            double max = double.NegativeInfinity;
            for (int j = 0; j < m; j++) { u[j] = norm > 0.0 ? ops.ToDouble(xi[off + j]) / norm : ops.ToDouble(xi[off + j]); if (u[j] > max) max = u[j]; }
            double z = 0.0;
            for (int j = 0; j < m; j++) z += Math.Exp(u[j] - max);
            for (int j = 0; j < m; j++) ro[off + j] = ops.FromDouble(Math.Exp(u[j] - max) / z);
        }
        return r;
    }

    public static Tensor<T> SphericalSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int m = input._shape[input._shape.Length - 1];
        int rows = m == 0 ? 0 : input.Length / m;
        var r = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var gi = gradOutput.AsSpan(); var ro = r.AsWritableSpan();
        var p = new double[m]; var gu = new double[m];
        for (int row = 0; row < rows; row++)
        {
            int off = row * m;
            double r2 = 0.0;
            for (int j = 0; j < m; j++) { double x = ops.ToDouble(xi[off + j]); r2 += x * x; }
            double norm = Math.Sqrt(r2);
            double max = double.NegativeInfinity;
            for (int j = 0; j < m; j++) { double uj = norm > 0.0 ? ops.ToDouble(xi[off + j]) / norm : ops.ToDouble(xi[off + j]); p[j] = uj; if (uj > max) max = uj; }
            double z = 0.0;
            for (int j = 0; j < m; j++) { p[j] = Math.Exp(p[j] - max); z += p[j]; }
            double sp = 0.0;
            for (int j = 0; j < m; j++) { p[j] /= z; sp += ops.ToDouble(gi[off + j]) * p[j]; }
            // grad w.r.t. u = softmax argument
            double gudotx = 0.0;
            for (int j = 0; j < m; j++) { gu[j] = p[j] * (ops.ToDouble(gi[off + j]) - sp); gudotx += gu[j] * ops.ToDouble(xi[off + j]); }
            // chain through u = x / ‖x‖:  dxₖ = guₖ/‖x‖ − xₖ·(Σ guᵢxᵢ)/‖x‖³
            for (int j = 0; j < m; j++)
            {
                double dx = norm > 0.0
                    ? gu[j] / norm - ops.ToDouble(xi[off + j]) * gudotx / (r2 * norm)
                    : gu[j];
                ro[off + j] = ops.FromDouble(dx);
            }
        }
        return r;
    }

    // ---- Capsule squash: yᵢ = xᵢ·‖x‖/(1+‖x‖²) ----
    public static Tensor<T> SquashForward<T>(Tensor<T> input)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int m = input._shape[input._shape.Length - 1];
        int rows = m == 0 ? 0 : input.Length / m;
        var r = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var ro = r.AsWritableSpan();
        for (int row = 0; row < rows; row++)
        {
            int off = row * m;
            double r2 = 0.0;
            for (int j = 0; j < m; j++) { double x = ops.ToDouble(xi[off + j]); r2 += x * x; }
            double norm = Math.Sqrt(r2);
            double a = norm > 0.0 ? norm / (1.0 + r2) : 0.0;
            for (int j = 0; j < m; j++) ro[off + j] = ops.FromDouble(a * ops.ToDouble(xi[off + j]));
        }
        return r;
    }

    public static Tensor<T> SquashBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int m = input._shape[input._shape.Length - 1];
        int rows = m == 0 ? 0 : input.Length / m;
        var r = new Tensor<T>(input._shape);
        var xi = input.AsSpan(); var gi = gradOutput.AsSpan(); var ro = r.AsWritableSpan();
        for (int row = 0; row < rows; row++)
        {
            int off = row * m;
            double r2 = 0.0, gdotx = 0.0;
            for (int j = 0; j < m; j++) { double x = ops.ToDouble(xi[off + j]); r2 += x * x; gdotx += ops.ToDouble(gi[off + j]) * x; }
            double norm = Math.Sqrt(r2);
            double a = norm > 0.0 ? norm / (1.0 + r2) : 0.0;
            // a = ‖x‖/(1+‖x‖²); ∂a/∂xₖ = (1−‖x‖²)/(1+‖x‖²)² · xₖ/‖x‖
            double coef = norm > 0.0 ? (1.0 - r2) / ((1.0 + r2) * (1.0 + r2) * norm) : 0.0;
            for (int j = 0; j < m; j++)
            {
                double dx = a * ops.ToDouble(gi[off + j]) + ops.ToDouble(xi[off + j]) * gdotx * coef;
                ro[off + j] = ops.FromDouble(dx);
            }
        }
        return r;
    }
}
