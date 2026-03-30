using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Delegate for backward functions that propagate gradients through a recorded operation.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <param name="gradOutput">The gradient flowing back from the output of this operation.</param>
/// <param name="inputs">The input tensors that were passed to the forward operation.</param>
/// <param name="output">The output tensor produced by the forward operation.</param>
/// <param name="savedState">Extra state saved during the forward pass (e.g., masks, indices). Never null at invocation time.</param>
/// <param name="engine">The engine to use for gradient computation.</param>
/// <param name="gradAccumulator">Dictionary mapping each tensor to its accumulated gradient.
/// Backward functions should add their computed input gradients into this accumulator.</param>
public delegate void BackwardFunction<T>(
    Tensor<T> gradOutput,
    Tensor<T>[] inputs,
    Tensor<T> output,
    object[] savedState,
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
    /// Version counters of each input tensor at recording time.
    /// Used to detect in-place mutation after recording (which corrupts gradients).
    /// </summary>
    public int[] InputVersions { get; }

    /// <summary>
    /// Creates a new tape entry. Captures input tensor version counters for mutation detection.
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

        // Snapshot version counters so backward can detect post-record mutations
        InputVersions = new int[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
            InputVersions[i] = inputs[i].Version;
    }

    /// <summary>
    /// Validates that no input tensor has been mutated since this entry was recorded.
    /// Throws if any input's version counter has changed.
    /// </summary>
    public void ValidateInputVersions()
    {
        for (int i = 0; i < Inputs.Length; i++)
        {
            if (Inputs[i].Version != InputVersions[i])
            {
                throw new InvalidOperationException(
                    $"Tensor input {i} of operation '{OperationName}' was mutated in-place after being recorded on the gradient tape. " +
                    $"This produces incorrect gradients. Use non-in-place operations, or ensure in-place mutations happen before recording.");
            }
        }
    }
}
