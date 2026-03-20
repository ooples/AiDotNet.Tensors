using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Pre-allocated memory workspace for zero-allocation tensor operations.
/// Surpasses PyTorch's caching allocator by pre-computing the exact memory layout
/// at construction time — zero overhead, zero fragmentation, zero GC pressure during forward passes.
/// </summary>
/// <remarks>
/// <para>
/// PyTorch's caching allocator has per-operation overhead (free list lookup, block splitting,
/// thread synchronization). TensorWorkspace eliminates ALL of this by:
/// 1. Registering all needed tensor shapes upfront
/// 2. Computing a single contiguous memory block with offsets for each slot
/// 3. Returning pre-mapped Tensor views during forward passes via simple pointer arithmetic
/// </para>
/// <para>
/// Usage pattern:
/// <code>
/// // At construction time:
/// var workspace = new TensorWorkspace&lt;float&gt;();
/// int conv1Out = workspace.Register([1, 256, 64, 64]);  // returns slot ID
/// int normOut  = workspace.Register([1, 256, 64, 64]);
/// int conv2Out = workspace.Register([1, 512, 32, 32]);
/// workspace.Allocate();  // single allocation for all slots
///
/// // During forward pass (zero allocation):
/// var output1 = workspace.Get(conv1Out);  // returns pre-allocated tensor view
/// Engine.Conv2DInto(output1, input, kernels, ...);
/// var output2 = workspace.Get(normOut);
/// Engine.GroupNormInto(output2, output1, ...);
/// </code>
/// </para>
/// <para><b>For Beginners:</b> Think of this like reserving seats in a theater before the show.
/// Instead of finding a seat every time someone arrives (PyTorch), we assign ALL seats at once
/// before the show starts. During the show, everyone goes directly to their assigned seat — no
/// searching, no waiting, no confusion.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor elements.</typeparam>
public sealed class TensorWorkspace<T> : IDisposable
{
    private readonly List<int[]> _shapes = new();
    private readonly List<int> _offsets = new();
    private T[]? _buffer;
    private bool _isAllocated;
    private bool _disposed;

    /// <summary>
    /// Gets the total number of elements in the workspace.
    /// </summary>
    public int TotalElements { get; private set; }

    /// <summary>
    /// Gets the number of registered tensor slots.
    /// </summary>
    public int SlotCount => _shapes.Count;

    /// <summary>
    /// Gets whether the workspace has been allocated.
    /// </summary>
    public bool IsAllocated => _isAllocated;

    /// <summary>
    /// Registers a tensor shape and returns a slot ID for later retrieval.
    /// Must be called before <see cref="Allocate"/>.
    /// </summary>
    /// <param name="shape">The shape of the tensor to reserve space for.</param>
    /// <returns>The slot ID to use with <see cref="Get"/>.</returns>
    /// <exception cref="InvalidOperationException">If workspace is already allocated.</exception>
    public int Register(int[] shape)
    {
        if (_isAllocated)
            throw new InvalidOperationException("Cannot register new shapes after allocation. Call Reset() first.");

        int id = _shapes.Count;
        _shapes.Add((int[])shape.Clone());
        return id;
    }

    /// <summary>
    /// Allocates the workspace buffer. Call after all shapes are registered.
    /// Uses a single contiguous allocation for all tensor slots.
    /// </summary>
    public void Allocate()
    {
        if (_isAllocated) return;

        // Compute offsets and total size
        _offsets.Clear();
        int offset = 0;
        for (int i = 0; i < _shapes.Count; i++)
        {
            _offsets.Add(offset);
            int slotSize = 1;
            foreach (int dim in _shapes[i])
                slotSize = checked(slotSize * dim);
            offset = checked(offset + slotSize);
        }

        TotalElements = offset;

        // Single allocation — no fragmentation, no GC pressure during forward passes
#if NET5_0_OR_GREATER
        if (TotalElements >= TensorAllocator.ArrayPoolThresholdValue)
        {
            _buffer = ArrayPool<T>.Shared.Rent(TotalElements);
        }
        else
        {
            _buffer = GC.AllocateUninitializedArray<T>(TotalElements);
        }
#else
        _buffer = new T[TotalElements];
#endif
        _isAllocated = true;
    }

    /// <summary>
    /// Gets a pre-allocated tensor view for the given slot ID.
    /// Zero allocation — returns a Memory view into the workspace buffer.
    /// </summary>
    /// <param name="slotId">The slot ID returned by <see cref="Register"/>.</param>
    /// <returns>A tensor backed by the workspace buffer.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Get(int slotId)
    {
        if (!_isAllocated)
            throw new InvalidOperationException("Workspace not allocated. Call Allocate() first.");
        if (_buffer is null)
            throw new ObjectDisposedException(nameof(TensorWorkspace<T>));

        int offset = _offsets[slotId];
        int[] shape = _shapes[slotId];
        int length = 1;
        foreach (int dim in shape)
            length *= dim;

        var memory = new Memory<T>(_buffer, offset, length);
        return Tensor<T>.FromMemory(memory, shape);
    }

    /// <summary>
    /// Gets the raw Span for a slot — zero allocation, direct memory access.
    /// Use when you need to pass data to Engine operations that accept Span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> GetSpan(int slotId)
    {
        if (!_isAllocated || _buffer is null)
            throw new InvalidOperationException("Workspace not allocated.");

        int offset = _offsets[slotId];
        int length = GetSlotLength(slotId);
        return _buffer.AsSpan(offset, length);
    }

    /// <summary>
    /// Gets the raw Memory for a slot — zero allocation, can be stored in fields/async.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Memory<T> GetMemory(int slotId)
    {
        if (!_isAllocated || _buffer is null)
            throw new InvalidOperationException("Workspace not allocated.");

        int offset = _offsets[slotId];
        int length = GetSlotLength(slotId);
        return new Memory<T>(_buffer, offset, length);
    }

    /// <summary>
    /// Gets the shape registered for a slot.
    /// </summary>
    public int[] GetShape(int slotId) => _shapes[slotId];

    /// <summary>
    /// Gets the element count for a slot.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetSlotLength(int slotId)
    {
        int length = 1;
        foreach (int dim in _shapes[slotId])
            length *= dim;
        return length;
    }

    /// <summary>
    /// Clears all slot data to zero. Call between independent forward passes
    /// if you need deterministic behavior (otherwise stale data from previous
    /// passes remains, which is fine for overwrite-style operations like Conv2DInto).
    /// </summary>
    public void Clear()
    {
        if (_buffer is not null)
            Array.Clear(_buffer, 0, TotalElements);
    }

    /// <summary>
    /// Resets the workspace so new shapes can be registered.
    /// Does not free the buffer — call <see cref="Dispose"/> for that.
    /// </summary>
    public void Reset()
    {
        _shapes.Clear();
        _offsets.Clear();
        _isAllocated = false;
        // Keep the buffer for potential reuse
    }

    /// <summary>
    /// Returns the workspace buffer to the pool and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_buffer is not null)
        {
#if NET5_0_OR_GREATER
            if (_buffer.Length >= TensorAllocator.ArrayPoolThresholdValue)
            {
                ArrayPool<T>.Shared.Return(_buffer,
                    clearArray: RuntimeHelpers.IsReferenceOrContainsReferences<T>());
            }
#endif
            _buffer = null;
        }
    }
}
