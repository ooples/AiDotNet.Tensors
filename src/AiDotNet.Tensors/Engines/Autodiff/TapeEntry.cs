using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Delegate for backward functions that propagate gradients through a recorded operation.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <param name="gradOutput">The gradient flowing back from the output of this operation.</param>
/// <param name="inputs">The input tensors that were passed to the forward operation.</param>
/// <param name="output">The output tensor produced by the forward operation.</param>
/// <param name="savedState">Optional extra state saved during the forward pass (e.g., masks, indices).</param>
/// <param name="engine">The engine to use for gradient computation.</param>
/// <param name="gradAccumulator">Dictionary mapping each tensor to its accumulated gradient.
/// Backward functions should add their computed input gradients into this accumulator.</param>
public delegate void BackwardFunction<T>(
    Tensor<T> gradOutput,
    Tensor<T>[] inputs,
    Tensor<T> output,
    object[]? savedState,
    IEngine engine,
    Dictionary<Tensor<T>, Tensor<T>> gradAccumulator);

/// <summary>
/// Records a single differentiable operation on the gradient tape.
/// Stores the operation name, input/output tensors, and the backward function
/// needed for reverse-mode automatic differentiation.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
public sealed class TapeEntry<T>
{
    /// <summary>
    /// Name of the operation (for debugging and profiling).
    /// </summary>
    public string OperationName { get; }

    /// <summary>
    /// The input tensors to this operation.
    /// </summary>
    public Tensor<T>[] Inputs { get; }

    /// <summary>
    /// The output tensor produced by this operation.
    /// </summary>
    public Tensor<T> Output { get; }

    /// <summary>
    /// The backward function that computes input gradients given the output gradient.
    /// </summary>
    public BackwardFunction<T> Backward { get; }

    /// <summary>
    /// Optional extra state saved during the forward pass (e.g., dropout mask, max indices).
    /// </summary>
    public object[]? SavedState { get; }

    /// <summary>
    /// Creates a new tape entry.
    /// </summary>
    public TapeEntry(
        string operationName,
        Tensor<T>[] inputs,
        Tensor<T> output,
        BackwardFunction<T> backward,
        object[]? savedState = null)
    {
        OperationName = operationName;
        Inputs = inputs;
        Output = output;
        Backward = backward;
        SavedState = savedState;
    }
}
