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
    /// Global flag: true when ANY thread has an active gradient tape.
    /// Checked first in Record methods — when false, the entire method
    /// is skipped without even reading ThreadStatic (saves ~5ns/op).
    /// Set by GradientTape constructor/Dispose.
    /// </summary>
    internal static volatile int _anyTapeActive;

    /// <summary>Fast check: is any tape active on any thread?</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool AnyTapeActive() => _anyTapeActive != 0;

    /// <summary>
    /// Thread-local check: is a tape active on the calling thread for the
    /// given numeric T? Use this when an op needs to switch dispatch paths
    /// (e.g. take a slower tape-aware branch) — using the cross-thread
    /// <see cref="AnyTapeActive"/> would incorrectly trigger the slow path
    /// for a thread that has no tape but happens to share a process with
    /// a thread that does.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsTapeActiveForThread<T>()
        => _anyTapeActive != 0
           && GradientTape<T>.Current is not null
           && !NoGradScope<T>.IsSuppressed;

    // Indexed gradient array: set by ComputeGradients before backward walk,
    // read by AccumulateGrad for O(1) access instead of dictionary hash lookup.
    // ThreadStatic because backward is single-threaded per tape.
    [ThreadStatic]
    internal static object?[]? _indexedGrads;

    /// <summary>Sets the indexed gradient array for the current backward pass.</summary>
    internal static void SetIndexedGrads(object?[] grads) => _indexedGrads = grads;

    /// <summary>Clears the indexed gradient array after backward completes.</summary>
    internal static void ClearIndexedGrads() => _indexedGrads = null;

    /// <summary>
    /// True when a backward pass is running with <c>createGraph=true</c> —
    /// backward ops are themselves recorded on the tape for higher-order
    /// differentiation. While this flag is set, <see cref="AccumulateGrad{T}"/>
    /// uses out-of-place <c>TensorAdd</c> instead of <c>TensorAddInPlace</c>
    /// so the gradient tensor identity stays connected to its producing
    /// op through the graph. In-place mutation records an entry keyed on
    /// a <c>savedA.Clone()</c> input, which severs the double-backward
    /// graph — the second <see cref="GradientTape{T}.ComputeGradients"/>
    /// call would observe a disconnected gradient tensor and return an
    /// incomplete result.
    /// </summary>
    /// <remarks>
    /// ThreadStatic — a nested inner tape running backward with
    /// <c>createGraph=false</c> while an outer pass has the flag set
    /// should still see the flag, because the thread is the same. This
    /// is correct: if we're in a backward that records, we're generating
    /// tape entries for the outer higher-order pass and in-place on a
    /// fresh gradient tensor would still sever that graph. The flag is
    /// only cleared in the <c>finally</c> block of the top-level call.
    /// </remarks>
    [ThreadStatic]
    internal static bool _isBackwardCreateGraph;

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
        if (_anyTapeActive == 0) return;
        if (NoGradScope<T>.IsSuppressed) return;
        var tape = GradientTape<T>.Current;
        if (tape is null) return;
        // Note: we always record during forward passes even when a compiled backward exists.
        // This ensures GradFn is set on outputs (needed for createGraph: true) and the tape
        // is populated as a fallback. The compiled backward is used at ComputeGradients time
        // (see GradientTape.ComputeGradients), not here.

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

        // Set GradFn for graph-based backward
        output.GradFn = inputs.Length switch
        {
            1 => new GradNode<T>(backward, output, inputs[0], savedState: savedState),
            2 => new GradNode<T>(backward, output, inputs[0], inputs[1], savedState: savedState, inputCount: 2),
            3 => new GradNode<T>(backward, output, inputs[0], inputs[1], inputs[2], savedState: savedState, inputCount: 3),
            _ => new GradNode<T>(backward, output, inputs[0], inputsOverflow: inputs, inputCount: 0xFF, savedState: savedState)
        };
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
        // Fast path: skip ThreadStatic read entirely when no tape exists globally
        if (_anyTapeActive == 0) return;
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

        // Set GradFn on output for O(1) graph-based backward traversal
        output.GradFn = new GradNode<T>(backward, output, input, savedState: savedState);
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
        if (_anyTapeActive == 0) return;
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

        output.GradFn = new GradNode<T>(backward, output, a, b, savedState: savedState, inputCount: 2);
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
        // Higher-order AD: in-place add records a "TensorAddInPlace"
        // entry whose saved input is a *clone* of the existing gradient,
        // which severs the graph the second backward pass needs to
        // walk. Use out-of-place add so the new gradient tensor is
        // produced by a "TensorAdd" entry whose inputs ARE the original
        // tensor references — the graph stays connected.
        bool needsOutOfPlace = _isBackwardCreateGraph;

        // CONTIGUITY INVARIANT (issue #274): every backward op that
        // permutes / reshapes / slices its incoming grad produces a
        // non-contiguous view tensor (e.g. PermuteBackward returns
        // engine.TensorPermute(...) which is a stride-rewrite, not a
        // contiguous copy). If we store that view as the first
        // gradient and a later AccumulateGrad call hits TensorAddInPlace
        // on it, the engine throws "In-place add requires contiguous
        // target tensor." Materialize at storage time so every grad
        // accumulator slot holds a contiguous buffer. The
        // .Contiguous() call no-ops when already contiguous, so the
        // hot path (Add / Mul / MatMul backward producing contiguous
        // grads) pays nothing.
        var gradContig = grad.IsContiguous ? grad : grad.Contiguous();

        // Fast path: use indexed array when grad indices are assigned (avoids hash lookup)
        int idx = tensor._gradIndex;
        if (idx >= 0 && _indexedGrads != null && idx < _indexedGrads.Length)
        {
            var existing = (Tensor<T>?)_indexedGrads[idx];
            if (existing != null)
            {
                Tensor<T> accumulated;
                if (needsOutOfPlace)
                {
                    accumulated = engine.TensorAdd(existing, gradContig);
                }
                else
                {
                    // Defensive: if the existing slot is somehow
                    // non-contiguous (e.g. populated outside this
                    // method), materialize before in-place add.
                    if (!existing.IsContiguous) existing = existing.Contiguous();
                    engine.TensorAddInPlace(existing, gradContig);
                    accumulated = existing;
                }
                _indexedGrads[idx] = accumulated;
                tensor.Grad = accumulated;
            }
            else
            {
                _indexedGrads[idx] = gradContig;
                tensor.Grad = gradContig;
            }
            grads[tensor] = tensor.Grad!;
            return;
        }

        // Fallback: dictionary path (for ops outside tape or during non-indexed backward)
        if (grads.TryGetValue(tensor, out var existingDict))
        {
            if (needsOutOfPlace)
            {
                var accumulated = engine.TensorAdd(existingDict, gradContig);
                grads[tensor] = accumulated;
                tensor.Grad = accumulated;
            }
            else
            {
                if (!existingDict.IsContiguous)
                {
                    existingDict = existingDict.Contiguous();
                    grads[tensor] = existingDict;
                }
                engine.TensorAddInPlace(existingDict, gradContig);
                tensor.Grad = existingDict;
            }
        }
        else
        {
            grads[tensor] = gradContig;
            tensor.Grad = gradContig;
        }
    }
}
