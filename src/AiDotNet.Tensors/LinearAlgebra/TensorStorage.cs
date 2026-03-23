using System.Runtime.CompilerServices;
using System.Threading;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Shared storage for tensor data. Multiple tensors (views) can reference the same storage
/// with different shapes, strides, and offsets — enabling O(1) Transpose, Reshape, and Slice.
/// </summary>
/// <remarks>
/// <para>This follows the PyTorch Storage model where the actual data lives in a Storage object
/// and Tensor is a view into that storage with shape/stride metadata.</para>
/// <para>Thread-safe reference counting tracks how many tensors/views share this storage.
/// When a tensor is disposed, it calls Release() to decrement the count. When the count
/// reaches zero the storage is eligible for GC (future: pool return).</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the stored elements.</typeparam>
internal sealed class TensorStorage<T>
{
    private readonly Vector<T> _data;
    private int _refCount;

    /// <summary>
    /// Creates a new storage wrapping an existing Vector (zero-copy).
    /// </summary>
    /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
    internal TensorStorage(Vector<T> data)
    {
        _data = data ?? throw new ArgumentNullException(nameof(data));
        _refCount = 1;
    }

    /// <summary>
    /// Creates a new storage with the specified size.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when size is not positive.</exception>
    internal TensorStorage(int size)
    {
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Storage size must be positive.");
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
    /// Increments the reference count atomically. Called when a new view is created from this storage.
    /// Uses compare-and-swap loop to prevent TOCTOU race between disposal check and increment.
    /// </summary>
    /// <exception cref="ObjectDisposedException">Thrown when storage has been fully released.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void AddRef()
    {
        while (true)
        {
            int current = Volatile.Read(ref _refCount);
            if (current <= 0)
                throw new ObjectDisposedException(nameof(TensorStorage<T>),
                    "Cannot acquire reference to released storage.");
            if (Interlocked.CompareExchange(ref _refCount, current + 1, current) == current)
                return;
            // CAS failed — another thread modified refCount, retry
        }
    }

    /// <summary>
    /// Decrements the reference count. When it reaches zero, the storage is marked as disposed
    /// and the pooled array (if any) can be reclaimed by TensorAllocator.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown on double-release (refCount already 0).</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void Release()
    {
        int newCount = Interlocked.Decrement(ref _refCount);
        if (newCount < 0)
        {
            // Restore and throw — this is a bug in the caller.
            Interlocked.Increment(ref _refCount);
            throw new InvalidOperationException("TensorStorage released more times than it was acquired.");
        }
        // When refCount reaches 0, storage can be reclaimed.
        // Future: integrate with TensorAllocator pool return.
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
