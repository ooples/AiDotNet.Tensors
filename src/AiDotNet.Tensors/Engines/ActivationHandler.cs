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
}

// ==================== Concrete Handlers ====================
// Each handler is a standalone class. Adding a new activation = new class + one dictionary entry.

internal sealed class ReLUActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.ReLU(input);
    public override void ApplyInPlace<T>(CpuEngine engine, Tensor<T> input) => engine.ReLUInPlace(input);
}

internal sealed class GELUActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.GELU(input);
}

internal sealed class SigmoidActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Sigmoid(input);
    public override void ApplyInPlace<T>(CpuEngine engine, Tensor<T> input) => engine.SigmoidInPlace(input);
}

internal sealed class TanhActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Tanh(input);
}

internal sealed class LeakyReLUActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input)
        => engine.LeakyReLU(input, Helpers.MathHelper.GetNumericOperations<T>().FromDouble(0.01));
}

internal sealed class SwishActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Swish(input);
}

internal sealed class SoftmaxActivationHandler : ActivationHandler
{
    public override Tensor<T> Apply<T>(CpuEngine engine, Tensor<T> input) => engine.Softmax(input, -1);
}
