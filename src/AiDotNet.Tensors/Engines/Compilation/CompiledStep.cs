using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A single step in a compiled execution plan. Pre-resolved: no graph traversal,
/// no dictionary lookups, no shape validation at execution time.
/// Straight-line delegate invocations for zero-overhead replay.
/// </summary>
internal sealed class CompiledStep<T>
{
    /// <summary>The execute delegate — writes result into OutputBuffer.</summary>
    internal readonly Action<IEngine, Tensor<T>> Execute;

    /// <summary>Pre-allocated output buffer. Reused across replays.</summary>
    internal readonly Tensor<T> OutputBuffer;

    /// <summary>Backward function for gradient computation (null for inference-only).</summary>
    internal readonly BackwardFunction<T>? BackwardFn;

    /// <summary>Input tensors for backward pass.</summary>
    internal readonly Tensor<T>[] Inputs;

    /// <summary>Saved state for backward (activation type, indices, etc.).</summary>
    internal readonly object[]? SavedState;

    /// <summary>Operation name for diagnostics.</summary>
    internal readonly string OpName;

    internal CompiledStep(
        string opName,
        Action<IEngine, Tensor<T>> execute,
        Tensor<T> outputBuffer,
        Tensor<T>[] inputs,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        OpName = opName;
        Execute = execute;
        OutputBuffer = outputBuffer;
        Inputs = inputs;
        BackwardFn = backwardFn;
        SavedState = savedState;
    }
}
