using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Arena-style allocator for tensor training loops. Pre-allocates arrays during the first
/// iteration, then reuses them for subsequent iterations — zero allocation, zero GC after warmup.
///
/// <para><b>Why this beats PyTorch:</b> PyTorch CPU tensors use malloc/free per tensor, causing
/// fragmentation and GC pauses. TensorArena eliminates per-tensor allocation after the first
/// iteration by reusing the same arrays via a ring buffer of pooled arrays.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// using var arena = TensorArena.Create();
/// for (int epoch = 0; epoch &lt; epochs; epoch++)
/// {
///     arena.Reset(); // Reuse arrays from previous iteration
///     var output = model.Forward(input);  // Allocations come from arena
///     var grads = model.Backward(loss);   // Zero new GC allocations after warmup
/// }
/// </code>
///
/// <para><b>How it works:</b> On first use (warmup), arrays are allocated normally and tracked.
/// On Reset(), the cursor rewinds to 0 — subsequent Rent calls return the same arrays.
/// Arrays are only allocated if the arena doesn't have one of the right size.</para>
///
/// <para><b>Thread safety:</b> Each thread gets its own arena via [ThreadStatic]. No locks.</para>
/// </summary>
public sealed class TensorArena : IDisposable
{
    [ThreadStatic]
    private static TensorArena? _current;

    /// <summary>
    /// Pool of arrays allocated during warmup, keyed by (element type, element count).
    /// Each bucket holds a list of arrays available for reuse.
    /// </summary>
    private readonly Dictionary<(Type, int), List<Array>> _pool = new();

    /// <summary>
    /// Tracks the reuse cursor per bucket — how many arrays of each (type, size)
    /// have been handed out since the last Reset().
    /// </summary>
    private readonly Dictionary<(Type, int), int> _cursor = new();

    /// <summary>
    /// Pool of Tensor objects keyed by (element type, total element count).
    /// On Reset(), cursors rewind so the same Tensor<T> objects (with their backing arrays) are reused.
    /// This eliminates both the array allocation AND the Tensor<T>/Vector<T>/int[] construction.
    /// </summary>
    private readonly Dictionary<(Type, int), List<object>> _tensorPool = new();
    private readonly Dictionary<(Type, int), int> _tensorCursor = new();

    private bool _disposed;
    private readonly TensorArena? _previous;

    /// <summary>
    /// Gets the currently active arena for this thread, or null if none.
    /// </summary>
    internal static TensorArena? Current => _current;

    private TensorArena(TensorArena? previous)
    {
        _previous = previous;
    }

    /// <summary>
    /// Creates and activates a new arena for this thread.
    /// All <see cref="TensorAllocator.Rent{T}"/> calls on this thread will allocate from the arena
    /// until it is disposed.
    /// </summary>
    /// <returns>An arena that must be disposed to deactivate.</returns>
    public static TensorArena Create()
    {
        var arena = new TensorArena(_current);
        _current = arena;
        return arena;
    }

    /// <summary>
    /// Tries to rent an array of the given size from the arena.
    /// During warmup (first iteration): allocates a new array and tracks it.
    /// After Reset() (subsequent iterations): returns a previously-allocated array.
    /// Returns null only if disposed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal T[]? TryAllocate<T>(int elementCount)
    {
        return TryAllocateCore<T>(elementCount, clear: true);
    }

    /// <summary>
    /// Like TryAllocate but skips Array.Clear. Use ONLY when the caller guarantees it will
    /// write every element before reading (e.g., backward kernels that overwrite the entire buffer).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal T[]? TryAllocateUninitialized<T>(int elementCount)
    {
        return TryAllocateCore<T>(elementCount, clear: false);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private T[]? TryAllocateCore<T>(int elementCount, bool clear)
    {
        if (_disposed) return null;

        var key = (typeof(T), elementCount);
        if (!_pool.TryGetValue(key, out var bucket))
        {
            bucket = new List<Array>(4);
            _pool[key] = bucket;
            _cursor[key] = 0;
        }

        int cursor = _cursor[key];

        if (cursor < bucket.Count)
        {
            // Reuse path: return existing array, advance cursor
            var existing = (T[])bucket[cursor];
            _cursor[key] = cursor + 1;
            if (clear) Array.Clear(existing, 0, elementCount);
            return existing;
        }

        // Warmup path: allocate new array and track it
        var arr = new T[elementCount];
        bucket.Add(arr);
        _cursor[key] = cursor + 1;
        return arr;
    }

    /// <summary>
    /// Rents a Tensor object from the arena. Returns a cached Tensor<T> with its
    /// backing array already pooled — zero allocation after warmup. The Tensor object
    /// itself is reused (same strides, same Vector wrapper, same backing array).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal LinearAlgebra.Tensor<T>? TryRentTensor<T>(int totalSize, int[] shape)
    {
        if (_disposed) return null;

        var key = (typeof(T), totalSize);
        if (!_tensorPool.TryGetValue(key, out var bucket))
        {
            bucket = new List<object>(4);
            _tensorPool[key] = bucket;
            _tensorCursor[key] = 0;
        }

        int cursor = _tensorCursor.TryGetValue(key, out var c) ? c : 0;

        if (cursor < bucket.Count)
        {
            _tensorCursor[key] = cursor + 1;
            return (LinearAlgebra.Tensor<T>)bucket[cursor];
        }

        // Warmup: allocate new tensor, track it
        var arr = new T[totalSize];
        var memory = new Memory<T>(arr, 0, totalSize);
        var tensor = LinearAlgebra.Tensor<T>.FromMemory(memory, shape);
        bucket.Add(tensor);
        _tensorCursor[key] = cursor + 1;
        return tensor;
    }

    /// <summary>
    /// Resets the arena for the next iteration. After this, subsequent Rent calls
    /// will reuse arrays from the previous iteration — zero allocation.
    /// </summary>
    public void Reset()
    {
        // Rewind all cursors to 0 — arrays and tensors stay pooled
        foreach (var key in _cursor.Keys)
            _cursor[key] = 0;
        foreach (var key in _tensorCursor.Keys)
            _tensorCursor[key] = 0;
    }

    /// <summary>
    /// Gets the total number of arrays tracked by this arena.
    /// </summary>
    public int TrackedArrayCount
    {
        get
        {
            int count = 0;
            foreach (var bucket in _pool.Values)
                count += bucket.Count;
            return count;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_current == this)
            _current = _previous; // restore outer arena if nested

        _pool.Clear();
        _cursor.Clear();
    }
}
