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
