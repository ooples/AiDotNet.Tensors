using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Thread-local zero-contention tensor buffer cache. After the first forward pass warms
/// up the cache, all subsequent operations get buffer reuse with zero allocation, zero GC,
/// and zero lock contention — completely invisible to callers.
///
/// This is the facade that makes <see cref="TensorAllocator.Rent{T}"/> zero-alloc without
/// any caller changes. The cache is per-thread and per-type, so float and double operations
/// maintain separate pools.
///
/// Flow:
/// 1. TensorAllocator.Rent(shape) → ThreadLocalTensorCache.TryRent(totalSize) → reuse or allocate
/// 2. TensorAllocator.Return(tensor) → ThreadLocalTensorCache.Return(array) → cache for reuse
/// </summary>
/// <remarks>
/// <para><b>Retention is bounded by bytes, not just by bucket depth.</b> This used to keep up to
/// <see cref="MaxBuffersPerSize"/> buffers per size and describe that as preventing "unbounded
/// memory growth". It did not: the key is an EXACT element count, nothing capped how many distinct
/// sizes a thread accumulated, and unlike the arena's cross-lifetime pool this one caches at ANY
/// size. A process that keeps meeting new shapes — a model-family test class, a service hosting
/// several models, a shape-polymorphic pipeline — therefore grew one bucket per size it had ever
/// seen and released none of them until someone called <see cref="Clear"/> by hand.
/// </para>
/// <para>
/// The cache now tracks retained bytes and trims least-recently-touched buckets when a return would
/// push it past <see cref="MaxRetainedBytes"/>. Recency is a cheap monotonic counter stamped on the
/// bucket, so the hot path stays two dictionary operations and the O(buckets) trim runs only on the
/// rare return that actually hits the budget.
/// </para>
/// </remarks>
/// <typeparam name="T">The element type (float, double, etc.)</typeparam>
internal static class ThreadLocalTensorCache<T>
{
    /// <summary>
    /// Maximum buffers cached per size bucket per thread.
    /// 4 is sufficient for most forward passes (conv output, norm output, activation output, residual).
    /// </summary>
    private const int MaxBuffersPerSize = 4;

    /// <summary>
    /// Total bytes this thread may retain for this element type.
    /// </summary>
    /// <remarks>
    /// The cache exists to make the buffers of a single forward/backward reusable, so the budget only
    /// has to cover one pass's working set — 64 MiB is generous for that and small enough that a
    /// process cycling through many models cannot accumulate their combined working sets in Gen2.
    /// It is per (element type, thread), matching the <c>[ThreadStatic]</c> generic-static contract.
    /// </remarks>
    private const long MaxRetainedBytes = 64L * 1024 * 1024;

    /// <summary>
    /// How far below <see cref="MaxRetainedBytes"/> a trim reclaims to, so that hitting the budget
    /// does not make every subsequent return pay for another trim.
    /// </summary>
    private const double TrimTargetFraction = 0.75;

    private static readonly int ElementSize = Unsafe.SizeOf<T>();

    private sealed class Bucket
    {
        internal readonly Stack<T[]> Buffers = new Stack<T[]>(MaxBuffersPerSize);

        internal long LastTouch;
    }

    /// <summary>
    /// Thread-local cache: size → bucket of reusable buffers.
    /// [ThreadStatic] ensures zero lock contention — each thread has its own cache.
    /// Generic type parameter ensures float[] and double[] caches are separate.
    /// </summary>
    [ThreadStatic]
    private static Dictionary<int, Bucket>? _cache;

    [ThreadStatic]
    private static long _retainedBytes;

    [ThreadStatic]
    private static long _touchCounter;

    /// <summary>Bytes currently retained on this thread for this element type. Diagnostic.</summary>
    internal static long RetainedBytes => _retainedBytes;

    /// <summary>Size buckets currently held on this thread for this element type. Diagnostic.</summary>
    internal static int BucketCount => _cache?.Count ?? 0;

    /// <summary>
    /// Tries to get a reusable buffer of exactly <paramref name="minSize"/> elements.
    /// Returns null if no cached buffer of that exact size is available.
    /// Zero contention — thread-local access only.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T[]? TryRent(int minSize)
    {
        if (_cache is null) return null;

        if (_cache.TryGetValue(minSize, out var bucket) && bucket.Buffers.Count > 0)
        {
            var array = bucket.Buffers.Pop();
            _retainedBytes -= (long)array.Length * ElementSize;
            bucket.LastTouch = ++_touchCounter;
            return array;
        }

        return null;
    }

    /// <summary>
    /// Returns a buffer to the thread-local cache for reuse.
    /// If the bucket for this size is full, or caching the buffer would exceed
    /// <see cref="MaxRetainedBytes"/> even after trimming, the buffer is not cached (let GC collect
    /// it or caller can fall through to ArrayPool.Return).
    /// </summary>
    /// <returns>true if cached for reuse, false if not cached (caller should dispose otherwise).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TryReturn(T[] array)
    {
        _cache ??= new Dictionary<int, Bucket>();

        int size = array.Length;
        long bytes = (long)size * ElementSize;

        // A single buffer larger than the whole budget is never worth evicting everything for.
        if (bytes > MaxRetainedBytes) return false;

        // Trim BEFORE get-or-add so a bucket this call creates cannot be dropped underneath it.
        if (_retainedBytes + bytes > MaxRetainedBytes) Trim(bytes);

        if (!_cache.TryGetValue(size, out var bucket))
        {
            bucket = new Bucket();
            _cache[size] = bucket;
        }

        bucket.LastTouch = ++_touchCounter;

        if (bucket.Buffers.Count >= MaxBuffersPerSize)
            return false; // Bucket full, don't hoard

        if (_retainedBytes + bytes > MaxRetainedBytes)
            return false; // Still over budget after trimming — this thread is saturated

        bucket.Buffers.Push(array);
        _retainedBytes += bytes;
        return true;
    }

    /// <summary>
    /// Drops least-recently-touched buckets until retention leaves room for
    /// <paramref name="incomingBytes"/> with headroom to spare.
    /// </summary>
    private static void Trim(long incomingBytes)
    {
        var cache = _cache;
        if (cache is null || cache.Count == 0) return;

        long target = (long)(MaxRetainedBytes * TrimTargetFraction) - incomingBytes;
        if (target < 0) target = 0;

        var byAge = new List<KeyValuePair<int, Bucket>>(cache);
        byAge.Sort(static (left, right) => left.Value.LastTouch.CompareTo(right.Value.LastTouch));

        for (int i = 0; i < byAge.Count && _retainedBytes > target; i++)
        {
            var bucket = byAge[i].Value;
            while (bucket.Buffers.Count > 0 && _retainedBytes > target)
            {
                var array = bucket.Buffers.Pop();
                _retainedBytes -= (long)array.Length * ElementSize;
            }

            if (bucket.Buffers.Count == 0) cache.Remove(byAge[i].Key);
        }
    }

    /// <summary>
    /// Clears all cached buffers for this thread and type.
    /// Call during shutdown or when memory pressure is detected.
    /// </summary>
    public static void Clear()
    {
        _cache?.Clear();
        _retainedBytes = 0;
    }
}
