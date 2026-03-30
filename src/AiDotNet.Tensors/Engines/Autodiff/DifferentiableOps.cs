using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Provides the tape recording hook that engine operations call after computing their result.
/// The <see cref="RecordIfActive{T}"/> method is marked AggressiveInlining so the JIT
/// can eliminate the null check entirely when no tape is active.
/// </summary>
internal static class DifferentiableOps
{
    /// <summary>
    /// Records an operation to the current gradient tape if one is active.
    /// This is a no-op (zero overhead) when no tape is active.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="opName">The name of the operation (for debugging/profiling).</param>
    /// <param name="output">The result tensor from the forward operation.</param>
    /// <param name="inputs">The input tensors to the forward operation.</param>
    /// <param name="backward">The backward function that computes input gradients.</param>
    /// <param name="savedState">Optional extra state needed by the backward function.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordIfActive<T>(
        string opName,
        Tensor<T> output,
        Tensor<T>[] inputs,
        BackwardFunction<T> backward,
        object[]? savedState = null)
    {
        var tape = GradientTape<T>.Current;
        if (tape is null) return;

        tape.Record(new TapeEntry<T>(opName, inputs, output, backward, savedState));
    }

    /// <summary>
    /// Convenience overload for unary operations (single input).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordUnary<T>(
        string opName,
        Tensor<T> output,
        Tensor<T> input,
        BackwardFunction<T> backward,
        object[]? savedState = null)
    {
        var tape = GradientTape<T>.Current;
        if (tape is null) return;

        tape.Record(new TapeEntry<T>(opName, new[] { input }, output, backward, savedState));
    }

    /// <summary>
    /// Lazy overload: defers savedState allocation until tape is confirmed active.
    /// Use this when savedState requires a new array allocation to avoid heap allocation on the no-grad path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordUnaryLazy<T>(
        string opName,
        Tensor<T> output,
        Tensor<T> input,
        BackwardFunction<T> backward,
        Func<object[]> savedStateFactory)
    {
        var tape = GradientTape<T>.Current;
        if (tape is null) return;

        tape.Record(new TapeEntry<T>(opName, new[] { input }, output, backward, savedStateFactory()));
    }

    /// <summary>
    /// Lazy overload for binary operations: defers savedState allocation until tape is confirmed active.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordBinaryLazy<T>(
        string opName,
        Tensor<T> output,
        Tensor<T> a,
        Tensor<T> b,
        BackwardFunction<T> backward,
        Func<object[]> savedStateFactory)
    {
        var tape = GradientTape<T>.Current;
        if (tape is null) return;

        tape.Record(new TapeEntry<T>(opName, new[] { a, b }, output, backward, savedStateFactory()));
    }

    /// <summary>
    /// Convenience overload for binary operations (two inputs).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordBinary<T>(
        string opName,
        Tensor<T> output,
        Tensor<T> a,
        Tensor<T> b,
        BackwardFunction<T> backward,
        object[]? savedState = null)
    {
        var tape = GradientTape<T>.Current;
        if (tape is null) return;

        tape.Record(new TapeEntry<T>(opName, new[] { a, b }, output, backward, savedState));
    }

    /// <summary>
    /// Accumulates a gradient for a tensor in the gradient dictionary.
    /// If the tensor already has a gradient, the new gradient is added to it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void AccumulateGrad<T>(
        Dictionary<Tensor<T>, Tensor<T>> grads,
        Tensor<T> tensor,
        Tensor<T> grad,
        IEngine engine)
    {
        if (grads.TryGetValue(tensor, out var existing))
        {
            engine.TensorAddInPlace(existing, grad);
        }
        else
        {
            grads[tensor] = grad.Clone();
        }
    }
}
