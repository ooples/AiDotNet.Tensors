using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Shared storage for tensor data. Multiple tensors (views) can reference the same storage
/// with different shapes, strides, and offsets — enabling O(1) Transpose, Reshape, and Slice.
/// </summary>
/// <remarks>
/// <para>This follows the PyTorch Storage model where the actual data lives in a Storage object
/// and Tensor is a view into that storage with shape/stride metadata.</para>
/// <para>Thread-safe reference counting ensures storage is kept alive while any view exists
/// and returned to the pool when the last view is released.</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the stored elements.</typeparam>
internal sealed class TensorStorage<T>
{
    private readonly Vector<T> _data;
    private int _refCount;

    /// <summary>
    /// Creates a new storage wrapping an existing Vector (zero-copy).
    /// </summary>
    internal TensorStorage(Vector<T> data)
    {
        _data = data;
        _refCount = 1;
    }

    /// <summary>
    /// Creates a new storage with the specified size.
    /// </summary>
    internal TensorStorage(int size)
    {
        _data = new Vector<T>(size);
        _refCount = 1;
    }

    /// <summary>
    /// Gets the total number of elements in this storage.
    /// </summary>
    internal int Length => _data.Length;

    /// <summary>
    /// Gets the current reference count (number of tensor views sharing this storage).
    /// </summary>
    internal int RefCount => Volatile.Read(ref _refCount);

    /// <summary>
    /// Increments the reference count. Called when a new view is created from this storage.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void AddRef()
    {
        Interlocked.Increment(ref _refCount);
    }

    /// <summary>
    /// Decrements the reference count. When it reaches zero, the storage can be reclaimed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void Release()
    {
        if (Interlocked.Decrement(ref _refCount) == 0)
        {
            // Storage is no longer referenced by any tensor view.
            // Future: return to TensorAllocator pool here.
        }
    }

    /// <summary>
    /// Gets the underlying data as a read-only span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal ReadOnlySpan<T> AsSpan() => _data.AsSpan();

    /// <summary>
    /// Gets the underlying data as a writable span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Span<T> AsWritableSpan() => _data.AsWritableSpan();

    /// <summary>
    /// Gets the underlying data as Memory&lt;T&gt; for pinning/GPU transfer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Memory<T> AsMemory() => _data.AsWritableMemory();

    /// <summary>
    /// Gets the raw backing array. Use with care — bypasses safety checks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal T[] GetDataArray() => _data.GetDataArray();

    /// <summary>
    /// Gets the underlying Vector for compatibility with existing code.
    /// </summary>
    internal Vector<T> GetVector() => _data;
}
