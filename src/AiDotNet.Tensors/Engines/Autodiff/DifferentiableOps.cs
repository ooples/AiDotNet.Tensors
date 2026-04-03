using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Provides the tape recording hook that engine operations call after computing their result.
/// All Record methods are AggressiveInlining so the JIT eliminates the null check entirely
/// when no tape is active (~2ns on the inference hot path).
/// </summary>
/// <remarks>
/// <para><b>Zero-allocation recording:</b> The Record methods construct <see cref="TapeEntry{T}"/>
/// structs with inline input fields (no <c>Tensor&lt;T&gt;[]</c> or <c>int[]</c> allocation).
/// The struct is passed by value to <see cref="GradientTape{T}.Record"/> which stores it
/// in a pre-allocated list. Only the SavedState array (when present) allocates.</para>
/// </remarks>
internal static class DifferentiableOps
{
    /// <summary>
    /// Returns true if a gradient tape is active and not suppressed.
    /// Use this to guard savedState allocation: only create new object[]
    /// when IsRecording is true, avoiding unnecessary GC pressure during inference.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsRecording<T>()
    {
        return GradientTape<T>.Current is not null && !NoGradScope<T>.IsSuppressed;
    }

    /// <summary>
    /// Records a variadic operation (4+ inputs) to the current gradient tape if one is active.
    /// The caller must provide the pre-allocated inputs array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RecordIfActive<T>(
        string opName,
        Tensor<T> output,
        Tensor<T>[] inputs,
        BackwardFunction<T> backward,
        object[]? savedState = null)
    {
        if (NoGradScope<T>.IsSuppressed) return;
        var tape = GradientTape<T>.Current;
        if (tape is null) return;

        ref var slot = ref tape.RecordSlot();
        slot.OperationName = opName;
        slot.Output = output;
        slot.Backward = backward;
        slot.SavedState = savedState;

        if (inputs.Length <= 3)
        {
            // Use inline fields for 1-3 inputs (avoids overflow array)
            slot.InputCount = (byte)inputs.Length;
            slot.Input0 = inputs.Length > 0 ? inputs[0] : null!;
            slot.Input1 = inputs.Length > 1 ? inputs[1] : null;
            slot.Input2 = inputs.Length > 2 ? inputs[2] : null;
            slot.Version0 = inputs.Length > 0 ? inputs[0].Version : 0;
            slot.Version1 = inputs.Length > 1 ? inputs[1].Version : 0;
            slot.Version2 = inputs.Length > 2 ? inputs[2].Version : 0;
        }
        else
        {
            // Overflow for 4+ inputs
            var versions = new int[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                versions[i] = inputs[i].Version;
            slot.InputsOverflow = inputs;
            slot.InputVersionsOverflow = versions;
            slot.InputCount = 0xFF;
            slot.Input0 = inputs[0];
        }
    }

    /// <summary>
    /// Records a unary operation (single input). Zero heap allocation, zero struct copy.
    /// Writes directly into the arena slot via ref return.
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
        if (tape is null || NoGradScope<T>.IsSuppressed) return;

        ref var slot = ref tape.RecordSlot();
        slot.OperationName = opName;
        slot.Output = output;
        slot.Backward = backward;
        slot.SavedState = savedState;
        slot.Input0 = input;
        slot.InputCount = 1;
        slot.Version0 = input.Version;
    }

    /// <summary>
    /// Records a binary operation (two inputs). Zero heap allocation, zero struct copy.
    /// Writes directly into the arena slot via ref return.
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
        if (tape is null || NoGradScope<T>.IsSuppressed) return;

        ref var slot = ref tape.RecordSlot();
        slot.OperationName = opName;
        slot.Output = output;
        slot.Backward = backward;
        slot.SavedState = savedState;
        slot.Input0 = a;
        slot.Input1 = b;
        slot.InputCount = 2;
        slot.Version0 = a.Version;
        slot.Version1 = b.Version;
    }

    /// <summary>
    /// Accumulates a gradient for a tensor in the gradient dictionary.
    /// If the tensor already has a gradient, the new gradient is added to it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining
#if !NETFRAMEWORK
        | MethodImplOptions.AggressiveOptimization
#endif
    )]
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
            // Store directly on first access — no clone needed since backward
            // functions create fresh gradient tensors that aren't shared.
            grads[tensor] = grad;
        }
    }
}
