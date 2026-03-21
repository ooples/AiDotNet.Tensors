using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Thread-local zero-contention tensor buffer cache. After the first forward pass warms
/// up the cache, all subsequent operations get buffer reuse with zero allocation, zero GC,
/// and zero lock contention — completely invisible to callers.
///
/// This is the facade that makes <see cref="TensorAllocator.Rent{T}"/> zero-alloc without
/// any caller changes. The cache is per-thread and per-type, so float and double operations
/// maintain separate pools. Each size bucket holds up to <see cref="MaxBuffersPerSize"/>
/// buffers to prevent unbounded memory growth.
///
/// Flow:
/// 1. TensorAllocator.Rent(shape) → ThreadLocalTensorCache.TryRent(totalSize) → reuse or allocate
/// 2. TensorAllocator.Return(tensor) → ThreadLocalTensorCache.Return(array) → cache for reuse
/// </summary>
/// <typeparam name="T">The element type (float, double, etc.)</typeparam>
internal static class ThreadLocalTensorCache<T>
{
    /// <summary>
    /// Maximum buffers cached per size bucket per thread. Prevents unbounded memory growth.
    /// 4 is sufficient for most forward passes (conv output, norm output, activation output, residual).
    /// </summary>
    private const int MaxBuffersPerSize = 4;

    /// <summary>
    /// Thread-local cache: size → stack of reusable buffers.
    /// [ThreadStatic] ensures zero lock contention — each thread has its own cache.
    /// Generic type parameter ensures float[] and double[] caches are separate.
    /// </summary>
    [ThreadStatic]
    private static Dictionary<int, Stack<T[]>>? _cache;

    /// <summary>
    /// Tries to get a reusable buffer of exactly <paramref name="minSize"/> elements.
    /// Returns null if no cached buffer of that exact size is available.
    /// Zero contention — thread-local access only.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T[]? TryRent(int minSize)
    {
        if (_cache is null) return null;

        if (_cache.TryGetValue(minSize, out var stack) && stack.Count > 0)
        {
            return stack.Pop();
        }

        return null;
    }

    /// <summary>
    /// Returns a buffer to the thread-local cache for reuse.
    /// If the cache is full for this size bucket, the buffer is not cached (let GC collect it
    /// or caller can fall through to ArrayPool.Return).
    /// </summary>
    /// <returns>true if cached for reuse, false if cache was full (caller should dispose otherwise).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TryReturn(T[] array)
    {
        _cache ??= new Dictionary<int, Stack<T[]>>();

        int size = array.Length;

        if (!_cache.TryGetValue(size, out var stack))
        {
            stack = new Stack<T[]>(MaxBuffersPerSize);
            _cache[size] = stack;
        }

        if (stack.Count >= MaxBuffersPerSize)
            return false; // Cache full, don't hoard

        stack.Push(array);
        return true;
    }

    /// <summary>
    /// Clears all cached buffers for this thread and type.
    /// Call during shutdown or when memory pressure is detected.
    /// </summary>
    public static void Clear()
    {
        _cache?.Clear();
    }
}
