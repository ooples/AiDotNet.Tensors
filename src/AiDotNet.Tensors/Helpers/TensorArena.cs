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
    /// Pool of arrays allocated during warmup, keyed by element count.
    /// Each size bucket holds a list of arrays available for reuse.
    /// </summary>
    private readonly Dictionary<int, List<Array>> _pool = new();

    /// <summary>
    /// Tracks the reuse cursor per size bucket — how many arrays of each size
    /// have been handed out since the last Reset().
    /// </summary>
    private readonly Dictionary<int, int> _cursor = new();

    private bool _disposed;

    /// <summary>
    /// Gets the currently active arena for this thread, or null if none.
    /// </summary>
    internal static TensorArena? Current => _current;

    private TensorArena()
    {
    }

    /// <summary>
    /// Creates and activates a new arena for this thread.
    /// All <see cref="TensorAllocator.Rent{T}"/> calls on this thread will allocate from the arena
    /// until it is disposed.
    /// </summary>
    /// <returns>An arena that must be disposed to deactivate.</returns>
    public static TensorArena Create()
    {
        var arena = new TensorArena();
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
        if (_disposed) return null;

        if (!_pool.TryGetValue(elementCount, out var bucket))
        {
            bucket = new List<Array>(4);
            _pool[elementCount] = bucket;
            _cursor[elementCount] = 0;
        }

        int cursor = _cursor[elementCount];

        if (cursor < bucket.Count)
        {
            // Reuse path: return existing array, advance cursor
            var existing = (T[])bucket[cursor];
            _cursor[elementCount] = cursor + 1;
            // Zero the reused array for correctness
            Array.Clear(existing, 0, elementCount);
            return existing;
        }

        // Warmup path: allocate new array and track it
        var arr = new T[elementCount];
        bucket.Add(arr);
        _cursor[elementCount] = cursor + 1;
        return arr;
    }

    /// <summary>
    /// Resets the arena for the next iteration. After this, subsequent Rent calls
    /// will reuse arrays from the previous iteration — zero allocation.
    /// </summary>
    public void Reset()
    {
        // Rewind all cursors to 0 — arrays stay pooled
        foreach (var key in _cursor.Keys)
            _cursor[key] = 0;
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
            _current = null;

        _pool.Clear();
        _cursor.Clear();
    }
}
