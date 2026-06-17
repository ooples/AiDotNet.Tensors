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
/// <remarks>
/// <para>
/// Issue #319 Phase 3: GradNode instances are POOLED via
/// <see cref="GradNodePool{T}"/>. The previous design allocated one
/// per <c>RecordUnary</c> / <c>RecordBinary</c> / <c>RecordIfActive</c>
/// call (~80 bytes each); on a ViT-Base training step that's hundreds
/// of allocations per second of pure Gen-0 GC pressure. Fields are no
/// longer readonly so the pool can recycle them — callers must treat
/// them as conceptually immutable for the lifetime of one backward
/// pass.
/// </para>
/// </remarks>
internal sealed class GradNode<T>
{
    /// <summary>The backward function that computes input gradients. Null after
    /// the node has been released (pool return, or streaming-backward activation
    /// release once its backward has run).</summary>
    public BackwardFunction<T>? Backward;

    /// <summary>Input tensors to the operation (1-3 inline, overflow for 4+).</summary>
    public Tensor<T> Input0 = null!;
    public Tensor<T>? Input1;
    public Tensor<T>? Input2;
    public Tensor<T>[]? InputsOverflow;
    public byte InputCount;

    /// <summary>The output tensor (for gradient seeding during backward). Null
    /// after the node has been released (pool return, or streaming-backward
    /// activation release once its backward has run).</summary>
    public Tensor<T>? Output;

    /// <summary>Optional saved state for backward (dropout mask, max indices, etc.).</summary>
    public object[]? SavedState;

    /// <summary>Accumulated gradient for this node's output tensor.</summary>
    public Tensor<T>? Gradient;

    /// <summary>
    /// Identity of the GradientTape that recorded this node. Used by
    /// the cleanup walk to decide whether a node it encounters in its
    /// topological order is safe to return to <see cref="GradNodePool{T}"/>.
    /// Inner tapes (e.g. <c>GradientCheckpointing</c>'s recompute pass)
    /// see outer-tape nodes via shared input tensors during topo sort
    /// — those must NOT be pooled by the inner cleanup, because the
    /// outer's cleanup will still walk over them later.
    /// </summary>
    internal object? OwningTape;

    /// <summary>
    /// Internal default constructor for pool-based allocation. Production
    /// callers should use <see cref="GradNodePool{T}.Rent"/>; tests and
    /// fallback paths may use this directly.
    /// </summary>
    public GradNode() { }

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

    /// <summary>
    /// Clears all reference-holding fields so the node can be safely
    /// returned to the pool without extending tensor lifetimes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void ClearForPoolReturn()
    {
        Backward = null;
        Input0 = null!;
        Input1 = null;
        Input2 = null;
        InputsOverflow = null;
        InputCount = 0;
        Output = null;
        SavedState = null;
        Gradient = null;
        OwningTape = null;
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

/// <summary>
/// Thread-local pool of <see cref="GradNode{T}"/> instances. The
/// per-op allocation in <c>DifferentiableOps.RecordUnary</c> /
/// <c>RecordBinary</c> / <c>RecordIfActive</c> was the highest-frequency
/// heap allocation on the training hot path (one per recorded forward
/// op); pooling brings it to zero in steady state.
/// </summary>
/// <remarks>
/// <para>
/// <b>Lifecycle:</b> nodes are rented when a forward op records and
/// assigned to the output tensor's <c>GradFn</c>. They're returned to
/// the pool at the end of the backward pass in
/// <c>GradientTape.ComputeGradientsViaGraphCore</c>, when the cleanup
/// walk nulls out the GradFn references on intermediates.
/// </para>
/// <para>
/// <b>Returns are gated on:</b> non-persistent tape (persistent
/// tapes keep their chain across backwards, so the nodes must stay
/// alive), <b>and</b> the <c>DifferentiableOps._isBackwardCreateGraph</c>
/// flag is false (a higher-order AD pass records backward ops on
/// the tape; those new ops point to GradNodes that the next
/// backward needs to walk).
/// </para>
/// <para>
/// <b>Capacity:</b> the pool is a simple <c>Stack&lt;GradNode&lt;T&gt;&gt;</c>
/// with a soft cap of 4096 nodes per thread (enough for ~256-layer
/// transformer at ViT-Base — well above any realistic workload).
/// Beyond cap, additional Returns are dropped on the floor; the
/// GC reclaims them normally.
/// </para>
/// </remarks>
internal static class GradNodePool<T>
{
    [ThreadStatic] private static Stack<GradNode<T>>? _pool;

    private const int SoftCap = 4096;

    /// <summary>
    /// Rents a node from the pool, allocating a fresh one if the pool
    /// is empty. Caller MUST populate every field before the node is
    /// visible to backward traversal.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static GradNode<T> Rent()
    {
        var pool = _pool;
        if (pool is not null && pool.Count > 0)
        {
            return pool.Pop();
        }
        return new GradNode<T>();
    }

    /// <summary>
    /// Returns a node to the pool after backward completes. Must
    /// only be called when the node is unreachable from any live
    /// computation graph (the caller is responsible for proving
    /// this — see <c>ComputeGradientsViaGraphCore</c>'s cleanup
    /// walk for the canonical pattern).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Return(GradNode<T> node)
    {
        if (node is null) return;
        var pool = _pool ??= new Stack<GradNode<T>>(64);
        if (pool.Count >= SoftCap) return;
        node.ClearForPoolReturn();
        pool.Push(node);
    }
}
