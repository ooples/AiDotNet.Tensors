using System.Runtime.CompilerServices;
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
/// Uses inline fields for inputs (up to 3) to avoid per-record heap allocation.
/// The input array is constructed lazily only during backward (the slow path).
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para><b>Performance:</b> This is a struct with inline input slots covering 99%+ of ops
/// (unary=1, binary=2, ternary=3). The forward recording path is zero-alloc — no
/// <c>Tensor&lt;T&gt;[]</c> or <c>int[]</c> is heap-allocated. The backward path constructs
/// the input array on demand since it's already dominated by matrix math.</para>
/// <para><b>For variadic ops (4+ inputs):</b> Use the <c>InputsOverflow</c> field, which
/// stores a pre-allocated array. This path is rare (TensorAddMany, Concat) and the array
/// is passed in by the caller — still no per-record allocation in DifferentiableOps.</para>
/// </remarks>
public struct TapeEntry<T>
{
    // ═══ HOT FIELDS (accessed every backward step — keep together for L1 cache) ═══

    /// <summary>The output tensor produced by this operation.</summary>
    public Tensor<T> Output;

    /// <summary>The backward function delegate.</summary>
    public BackwardFunction<T> Backward;

    /// <summary>First input tensor (always present).</summary>
    public Tensor<T> Input0;

    /// <summary>Second input tensor (binary/ternary ops), or null for unary.</summary>
    public Tensor<T>? Input1;

    /// <summary>Third input tensor (ternary ops), or null for unary/binary.</summary>
    public Tensor<T>? Input2;

    /// <summary>Number of inputs: 1, 2, 3, or 0xFF if using <see cref="InputsOverflow"/>.</summary>
    public byte InputCount;

    // ═══ COLD FIELDS (accessed rarely — validation, profiling, overflow) ═══

    /// <summary>Optional extra state saved during the forward pass (e.g., dropout mask, max indices).</summary>
    public object[]? SavedState;

    /// <summary>Name of the operation (for debugging and profiling).</summary>
    public string OperationName;

    /// <summary>Version counter of Input0 at recording time.</summary>
    public int Version0;

    /// <summary>Version counter of Input1 at recording time.</summary>
    public int Version1;

    /// <summary>Version counter of Input2 at recording time.</summary>
    public int Version2;

    /// <summary>
    /// Overflow array for ops with 4+ inputs (e.g., TensorAddMany, Concat, Stack).
    /// </summary>
    public Tensor<T>[]? InputsOverflow;

    /// <summary>
    /// Version counters for overflow inputs. Snapshotted at recording time.
    /// </summary>
    public int[]? InputVersionsOverflow;

    // ── Backward-path helpers (array construction is deferred) ──────────

    /// <summary>
    /// Constructs the input array for the backward function. This allocates, but is only
    /// called during backward (which is already dominated by matrix operations).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T>[] GetInputsArray()
    {
        if (InputsOverflow is not null) return InputsOverflow;
        // Allocate small arrays per call — avoids ThreadStatic reuse issues with
        // re-entrant backward (gradient checkpointing calls backward inside backward).
        // For 1-3 element arrays, the allocation is tiny (~40 bytes) and the JIT
        // may stack-allocate or pool them.
        return InputCount switch
        {
            1 => new[] { Input0 },
            2 => new[] { Input0, Input1! },
            3 => new[] { Input0, Input1!, Input2! },
            _ => new[] { Input0 }
        };
    }

    /// <summary>
    /// Validates that no input tensor has been mutated since this entry was recorded.
    /// Throws if any input's version counter has changed.
    /// </summary>
    /// <remarks>
    /// Skipped when an <see cref="InferenceModeScope{T}"/> is active —
    /// inference mode does not record tape entries that need to be
    /// replayed, and even if a stale entry remains from before the
    /// scope opened, in-place mutations during the scope deliberately
    /// don't bump the version counter (see
    /// <see cref="LinearAlgebra.TensorBase.IncrementVersion"/>), so
    /// running validation here would still pass — the early-out is
    /// purely a hot-path optimisation.
    /// </remarks>
    public void ValidateInputVersions()
    {
        if (InferenceModeFlag.IsActive) return;

        if (InputsOverflow is not null)
        {
            // Overflow path uses InputVersionsOverflow if available
            if (InputVersionsOverflow is not null)
            {
                for (int i = 0; i < InputsOverflow.Length && i < InputVersionsOverflow.Length; i++)
                {
                    if (InputsOverflow[i].Version != InputVersionsOverflow[i])
                        ThrowMutation(i);
                }
            }
            return;
        }

        if (Input0.Version != Version0)
            ThrowMutation(0);
        if (InputCount >= 2 && Input1 is not null && Input1.Version != Version1)
            ThrowMutation(1);
        if (InputCount >= 3 && Input2 is not null && Input2.Version != Version2)
            ThrowMutation(2);
    }

    private void ThrowMutation(int index)
    {
        throw new InvalidOperationException(
            $"Tensor input {index} of operation '{OperationName}' was mutated in-place after being recorded on the gradient tape. " +
            $"This produces incorrect gradients. Use non-in-place operations, or ensure in-place mutations happen before recording.");
    }

    // ── Legacy compatibility ────────────────────────────────────────────

    /// <summary>
    /// Gets the input tensors as an array. Property retained for backward compatibility
    /// with code that accesses <c>entry.Inputs</c>.
    /// </summary>
    public Tensor<T>[] Inputs => GetInputsArray();

    /// <summary>
    /// Gets the version counters as an array. Property retained for backward compatibility.
    /// </summary>
    public int[] InputVersions => InputCount switch
    {
        1 => new[] { Version0 },
        2 => new[] { Version0, Version1 },
        3 => new[] { Version0, Version1, Version2 },
        _ => InputsOverflow is not null
            ? Array.Empty<int>() // overflow path doesn't track versions inline
            : new[] { Version0 }
    };
}
