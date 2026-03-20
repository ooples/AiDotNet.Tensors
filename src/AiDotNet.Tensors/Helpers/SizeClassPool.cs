using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Size-exact memory pool for tensor buffers with zero power-of-2 rounding waste.
/// Unlike ArrayPool which rounds up to the next power of 2 (wasting up to 50% memory),
/// this pool maintains exact-size buckets for commonly used tensor dimensions.
/// </summary>
/// <remarks>
/// <para>
/// PyTorch's caching allocator uses power-of-2 size classes, wasting memory when
/// tensor sizes don't align. Our approach: maintain a pool per unique size, returning
/// exact-fit buffers. For dynamic batch sizes, the pool naturally adapts by caching
/// buffers at each observed size.
/// </para>
/// <para>
/// Thread-safe via ConcurrentDictionary + ConcurrentBag per size class.
/// Buffers are not cleared on return (callers are expected to overwrite all elements).
/// </para>
/// </remarks>
/// <typeparam name="T">Element type.</typeparam>
public sealed class SizeClassPool<T>
{
    private readonly ConcurrentDictionary<int, ConcurrentBag<T[]>> _pools = new();
    private readonly int _maxBuffersPerSize;
    private int _totalBuffers;
    private long _hits;
    private long _misses;

    /// <summary>Total buffers currently cached across all size classes.</summary>
    public int TotalCachedBuffers => _totalBuffers;

    /// <summary>Number of size classes with at least one cached buffer.</summary>
    public int ActiveSizeClasses => _pools.Count;

    /// <summary>Cache hit count (buffer returned from pool).</summary>
    public long Hits => _hits;

    /// <summary>Cache miss count (new buffer allocated).</summary>
    public long Misses => _misses;

    /// <summary>Hit ratio (0.0 to 1.0).</summary>
    public double HitRatio => _hits + _misses > 0 ? (double)_hits / (_hits + _misses) : 0;

    /// <summary>
    /// Creates a new size-class pool.
    /// </summary>
    /// <param name="maxBuffersPerSize">Maximum buffers to cache per size class.
    /// Prevents unbounded memory growth for many distinct sizes.</param>
    public SizeClassPool(int maxBuffersPerSize = 8)
    {
        _maxBuffersPerSize = maxBuffersPerSize;
    }

    /// <summary>
    /// Rents a buffer of exactly the specified size.
    /// Returns a cached buffer if available, otherwise allocates a new one.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T[] Rent(int size)
    {
        if (_pools.TryGetValue(size, out var bag) && bag.TryTake(out var buffer))
        {
            Interlocked.Increment(ref _hits);
            Interlocked.Decrement(ref _totalBuffers);
            return buffer;
        }

        Interlocked.Increment(ref _misses);
#if NET5_0_OR_GREATER
        return GC.AllocateUninitializedArray<T>(size);
#else
        return new T[size];
#endif
    }

    /// <summary>
    /// Returns a buffer to the pool for reuse. The buffer is NOT cleared.
    /// Callers must not access the buffer after returning it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Return(T[] buffer)
    {
        if (buffer == null) return;

        int size = buffer.Length;
        var bag = _pools.GetOrAdd(size, _ => new ConcurrentBag<T[]>());

        // Limit cached buffers per size to prevent unbounded growth
        if (bag.Count < _maxBuffersPerSize)
        {
#if NET5_0_OR_GREATER
            // Clear reference types to avoid retaining stale objects
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(buffer, 0, buffer.Length);
#endif
            bag.Add(buffer);
            Interlocked.Increment(ref _totalBuffers);
        }
        // If over limit, let GC collect the buffer
    }

    /// <summary>
    /// Clears all cached buffers. Call during memory-constrained scenarios
    /// or when the model changes shape (e.g., different batch size range).
    /// </summary>
    public void Clear()
    {
        _pools.Clear();
        Interlocked.Exchange(ref _totalBuffers, 0);
    }

    /// <summary>
    /// Pre-populates the pool with buffers at the specified sizes.
    /// Call at startup with known tensor sizes to avoid cold-start misses.
    /// </summary>
    /// <param name="sizes">Array of buffer sizes to pre-allocate.</param>
    /// <param name="count">Number of buffers per size.</param>
    public void Warmup(int[] sizes, int count = 2)
    {
        foreach (int size in sizes)
        {
            for (int i = 0; i < Math.Min(count, _maxBuffersPerSize); i++)
            {
#if NET5_0_OR_GREATER
                var buf = GC.AllocateUninitializedArray<T>(size);
#else
                var buf = new T[size];
#endif
                Return(buf);
            }
        }
    }
}
