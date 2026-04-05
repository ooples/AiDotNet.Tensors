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

    // Flat tensor ring buffer — sequential scan is faster than dictionary hash for <10 sizes
    private object[]? _tensorRing;
    private int[]? _tensorRingSizes;
    private int[]? _tensorRingCursors;
    private int _tensorRingCount;
    private const int MaxTensorRingSlots = 32;

    /// <summary>
    /// Rents a Tensor object from the arena. Returns a cached Tensor with its
    /// backing array already pooled. Uses a flat array with linear scan instead of
    /// Dictionary — eliminates hash computation overhead for the 3-5 unique sizes
    /// in a typical training step.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal LinearAlgebra.Tensor<T>? TryRentTensor<T>(int totalSize, int[] shape)
    {
        if (_disposed) return null;

        // Lazy init
        if (_tensorRing == null)
        {
            _tensorRing = new object[MaxTensorRingSlots];
            _tensorRingSizes = new int[MaxTensorRingSlots];
            _tensorRingCursors = new int[MaxTensorRingSlots];
        }

        // Linear scan for matching size (fast for <10 entries)
        for (int i = 0; i < _tensorRingCount; i++)
        {
            if (_tensorRingSizes![i] == totalSize && _tensorRing[i] is List<object> bucket)
            {
                int cursor = _tensorRingCursors![i];
                if (cursor < bucket.Count)
                {
                    _tensorRingCursors[i] = cursor + 1;
                    return (LinearAlgebra.Tensor<T>)bucket[cursor];
                }

                // Need one more tensor of this size
                var newArr = new T[totalSize];
                var newTensor = LinearAlgebra.Tensor<T>.FromMemory(new Memory<T>(newArr, 0, totalSize), shape);
                bucket.Add(newTensor);
                _tensorRingCursors[i] = cursor + 1;
                return newTensor;
            }
        }

        // New size — add slot
        if (_tensorRingCount < MaxTensorRingSlots)
        {
            var arr = new T[totalSize];
            var tensor = LinearAlgebra.Tensor<T>.FromMemory(new Memory<T>(arr, 0, totalSize), shape);
            var newBucket = new List<object>(4) { tensor };
            int idx = _tensorRingCount++;
            _tensorRing[idx] = newBucket;
            _tensorRingSizes![idx] = totalSize;
            _tensorRingCursors![idx] = 1;
            return tensor;
        }

        return null; // Arena full — caller falls through to other allocators
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
        // Reset flat tensor ring cursors
        if (_tensorRingCursors != null)
        {
            for (int i = 0; i < _tensorRingCount; i++)
                _tensorRingCursors[i] = 0;
        }
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

        // Clear tensor ring pools
        if (_tensorRing != null)
        {
            Array.Clear(_tensorRing, 0, _tensorRingCount);
            _tensorRingCount = 0;
        }
    }
}
