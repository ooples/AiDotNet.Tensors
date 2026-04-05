using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// A node in the computation graph, stored directly on each tensor.
/// When a tape-recorded operation produces a tensor, the output tensor's
/// GradFn points to this node. Backward traversal walks GradFn pointers
/// instead of a tape — O(1) lookup, no dictionary, no hash computation.
/// This is the same architecture as PyTorch's autograd.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class GradNode<T>
{
    /// <summary>The backward function that computes input gradients.</summary>
    public readonly BackwardFunction<T> Backward;

    /// <summary>Input tensors to the operation (1-3 inline, overflow for 4+).</summary>
    public readonly Tensor<T> Input0;
    public readonly Tensor<T>? Input1;
    public readonly Tensor<T>? Input2;
    public readonly Tensor<T>[]? InputsOverflow;
    public readonly byte InputCount;

    /// <summary>The output tensor (for gradient seeding during backward).</summary>
    public readonly Tensor<T> Output;

    /// <summary>Optional saved state for backward (dropout mask, max indices, etc.).</summary>
    public readonly object[]? SavedState;

    /// <summary>Accumulated gradient for this node's output tensor.</summary>
    public Tensor<T>? Gradient;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GradNode(
        BackwardFunction<T> backward,
        Tensor<T> output,
        Tensor<T> input0,
        Tensor<T>? input1 = null,
        Tensor<T>? input2 = null,
        Tensor<T>[]? inputsOverflow = null,
        byte inputCount = 1,
        object[]? savedState = null)
    {
        Backward = backward;
        Output = output;
        Input0 = input0;
        Input1 = input1;
        Input2 = input2;
        InputsOverflow = inputsOverflow;
        InputCount = inputCount;
        SavedState = savedState;
    }

    /// <summary>Gets the input tensors as an array for backward function compatibility.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T>[] GetInputsArray()
    {
        if (InputsOverflow is not null) return InputsOverflow;
        return InputCount switch
        {
            1 => new[] { Input0 },
            2 => new[] { Input0, Input1! },
            3 => new[] { Input0, Input1!, Input2! },
            _ => new[] { Input0 }
        };
    }
}
