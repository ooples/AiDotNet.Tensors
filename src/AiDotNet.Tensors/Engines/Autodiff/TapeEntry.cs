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
    // ── Inline input slots (zero-alloc for 1-3 inputs) ──────────────────

    /// <summary>First input tensor (always present).</summary>
    public Tensor<T> Input0;

    /// <summary>Second input tensor (binary/ternary ops), or null for unary.</summary>
    public Tensor<T>? Input1;

    /// <summary>Third input tensor (ternary ops), or null for unary/binary.</summary>
    public Tensor<T>? Input2;

    /// <summary>Number of inputs: 1, 2, 3, or 0xFF if using <see cref="InputsOverflow"/>.</summary>
    public byte InputCount;

    /// <summary>
    /// Overflow array for ops with 4+ inputs (e.g., TensorAddMany, Concat, Stack).
    /// Null for 1-3 input ops. When set, <see cref="InputCount"/> is 0xFF.
    /// The caller passes the array — DifferentiableOps does not allocate it.
    /// </summary>
    public Tensor<T>[]? InputsOverflow;

    /// <summary>
    /// Version counters for overflow inputs. Snapshotted at recording time.
    /// </summary>
    public int[]? InputVersionsOverflow;

    // ── Inline version counters (zero-alloc) ────────────────────────────

    /// <summary>Version counter of Input0 at recording time.</summary>
    public int Version0;

    /// <summary>Version counter of Input1 at recording time.</summary>
    public int Version1;

    /// <summary>Version counter of Input2 at recording time.</summary>
    public int Version2;

    // ── Core fields ─────────────────────────────────────────────────────

    /// <summary>Name of the operation (for debugging and profiling).</summary>
    public string OperationName;

    /// <summary>The output tensor produced by this operation.</summary>
    public Tensor<T> Output;

    /// <summary>The backward function that computes input gradients given the output gradient.</summary>
    public BackwardFunction<T> Backward;

    /// <summary>Optional extra state saved during the forward pass (e.g., dropout mask, max indices).</summary>
    public object[]? SavedState;

    // ── Backward-path helpers (array construction is deferred) ──────────

    /// <summary>
    /// Constructs the input array for the backward function. This allocates, but is only
    /// called during backward (which is already dominated by matrix operations).
    /// </summary>
    // Thread-local reusable input arrays to avoid per-backward-step allocation.
    // Safe because backward is single-threaded per tape and the array is consumed
    // before the next GetInputsArray call in the same backward walk.
    [ThreadStatic] private static Tensor<T>[]? _reuse1;
    [ThreadStatic] private static Tensor<T>[]? _reuse2;
    [ThreadStatic] private static Tensor<T>[]? _reuse3;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T>[] GetInputsArray()
    {
        if (InputsOverflow is not null) return InputsOverflow;
        switch (InputCount)
        {
            case 1:
                var arr1 = _reuse1 ??= new Tensor<T>[1];
                arr1[0] = Input0;
                return arr1;
            case 2:
                var arr2 = _reuse2 ??= new Tensor<T>[2];
                arr2[0] = Input0; arr2[1] = Input1!;
                return arr2;
            case 3:
                var arr3 = _reuse3 ??= new Tensor<T>[3];
                arr3[0] = Input0; arr3[1] = Input1!; arr3[2] = Input2!;
                return arr3;
            default:
                var arrD = _reuse1 ??= new Tensor<T>[1];
                arrD[0] = Input0;
                return arrD;
        }
    }

    /// <summary>
    /// Validates that no input tensor has been mutated since this entry was recorded.
    /// Throws if any input's version counter has changed.
    /// </summary>
    public void ValidateInputVersions()
    {
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
