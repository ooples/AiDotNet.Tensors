using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Vector allocation with transparent zero-alloc caching via <see cref="ThreadLocalTensorCache{T}"/>.
/// After the first forward pass warms the cache, <see cref="Rent{T}(int)"/> reuses thread-local
/// buffers — zero allocation, zero GC, zero lock contention, completely invisible to callers.
/// Falls back to ArrayPool for cache misses, then standard allocation when pooling is disabled.
/// Return pooled vectors via <see cref="Return{T}"/> to enable buffer reuse.
/// </summary>
public static class VectorAllocator
{
    /// <summary>
    /// Threshold above which ArrayPool is used instead of standard allocation.
    /// ArrayPool avoids GC pressure for repeated large allocations.
    /// 256K elements = 1MB for float, 2MB for double.
    /// </summary>
    private const int ArrayPoolThreshold = 256 * 1024;

    /// <summary>
    /// Public accessor for the ArrayPool threshold used by workspace and other helpers.
    /// </summary>
    public const int ArrayPoolThresholdValue = ArrayPoolThreshold;

    /// <summary>
    /// Creates a zero-initialized vector with the given length.
    /// Large vectors use ArrayPool to reduce GC pressure; small-medium vectors
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Vector<T> Rent<T>(int length)
    {
        if (!TensorPool.Enabled || length == 0)
        {
            return new Vector<T>(length);
        }

        // Tier 0: Arena allocation — zero GC during training loops.
        var arena = TensorArena.Current;
        if (arena != null)
        {
            T[]? arenaArray = arena.TryAllocate<T>(length);
            if (arenaArray != null)
            {
                var memory = new Memory<T>(arenaArray, 0, length);
                return Vector<T>.FromMemory(memory);
            }
            // Arena full — fall through to other tiers
        }

#if NET5_0_OR_GREATER
        // Tier 1: Thread-local cache — zero allocation after warmup.
        T[]? cached = ThreadLocalTensorCache<T>.TryRent(length);
        if (cached is null && length >= ArrayPoolThreshold)
            cached = ThreadLocalTensorCache<T>.TryRent(ArrayPoolBucketSize(length));
        if (cached is not null)
        {
            Array.Clear(cached, 0,
                RuntimeHelpers.IsReferenceOrContainsReferences<T>() ? cached.Length : length);
            var memory = new Memory<T>(cached, 0, length);
            return Vector<T>.FromPooledMemory(memory, cached);
        }

        // Tier 2: ArrayPool for large allocations.
        if (length >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(length);
            Array.Clear(pooled, 0,
                RuntimeHelpers.IsReferenceOrContainsReferences<T>() ? pooled.Length : length);
            var memory = new Memory<T>(pooled, 0, length);
            return Vector<T>.FromPooledMemory(memory, pooled);
        }

        // Tier 3: Standard managed allocation for small vectors.
        T[] arr = new T[length];
        var mem = new Memory<T>(arr);
        return Vector<T>.FromMemory(mem);
#else
        return new Vector<T>(length);
#endif
    }

    /// <summary>
    /// Creates a vector backed by NativeMemory (64-byte aligned, zero GC overhead).
    /// Use for vectors that will ONLY be accessed via Span — never GetDataArray().
    /// oneDNN and VML operations should use this for optimal performance.
    /// </summary>
    public static Vector<T> RentNative<T>(int length) where T : unmanaged
    {
        if (length == 0)
            return new Vector<T>(length);

#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(float))
        {
            var owner = new NativeMemoryOwner<float>(length, zeroed: true);
            return (Vector<T>)(object)Vector<float>.FromMemory(owner.Memory);
        }
        if (typeof(T) == typeof(double))
        {
            var owner = new NativeMemoryOwner<double>(length, zeroed: true);
            return (Vector<T>)(object)Vector<double>.FromMemory(owner.Memory);
        }
#endif
        // Fallback for other unmanaged types
        return Rent<T>(length);
    }

    /// <summary>
    /// Creates a vector with the given length WITHOUT zero-initialization.
    /// Use ONLY for vectors that will be immediately and fully overwritten
    /// (e.g., weight initialization with random values, copy targets).
    /// WARNING: Contains stale/garbage data until overwritten.
    /// </summary>
    public static Vector<T> RentUninitialized<T>(int length)
    {
        if (!TensorPool.Enabled || length == 0)
        {
            return new Vector<T>(length);
        }

#if NET5_0_OR_GREATER
        if (length >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(length);
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(pooled, 0, pooled.Length);
            var memory = new Memory<T>(pooled, 0, length);
            return Vector<T>.FromPooledMemory(memory, pooled);
        }

        // Skip zero-initialization for immediate-overwrite vectors
        T[] array = GC.AllocateUninitializedArray<T>(length);
        var mem = new Memory<T>(array);
        return Vector<T>.FromMemory(mem);
#else
        return new Vector<T>(length);
#endif
    }

    /// <summary>
    /// Creates a vector with data from an existing array.
    /// Zero-copy when the array is exactly the right size.
    /// Falls back to copy when pooling is needed for large arrays.
    /// </summary>
    public static Vector<T> Rent<T>(T[] data)
    {
        int length = data.Length;

#if NET5_0_OR_GREATER
        // Clone to avoid aliasing — caller retains original, vector gets its own copy.
        if (!TensorPool.Enabled || length == 0)
        {
            var clone = new Memory<T>((T[])data.Clone());
            return Vector<T>.FromMemory(clone);
        }

        if (length >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(length);
            data.AsSpan().CopyTo(pooled.AsSpan(0, length));
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>() && pooled.Length > length)
                Array.Clear(pooled, length, pooled.Length - length);
            var memory = new Memory<T>(pooled, 0, length);
            return Vector<T>.FromPooledMemory(memory, pooled);
        }

        T[] array = GC.AllocateUninitializedArray<T>(length);
        data.AsSpan().CopyTo(array);
        var mem = new Memory<T>(array);
        return Vector<T>.FromMemory(mem);
#else
        return new Vector<T>(data);
#endif
    }

    /// <summary>
    /// Returns a vector's backing array to the pool if it was pooled.
    /// SAFETY: The caller MUST ensure the vector is never accessed after this call.
    /// </summary>
    internal static void Return<T>(Vector<T>? vector)
    {
        if (vector == null) return;

        T[]? pooledArray = vector.PooledArray;
        if (pooledArray != null)
        {
            vector.DetachPooledArray();
#if NET5_0_OR_GREATER
            // Tier 1: Try thread-local cache first — zero contention, instant reuse.
            if (ThreadLocalTensorCache<T>.TryReturn(pooledArray))
                return;

            // Tier 2: Cache full — fall through to ArrayPool.
            ArrayPool<T>.Shared.Return(pooledArray,
                clearArray: RuntimeHelpers.IsReferenceOrContainsReferences<T>());
#else
            ArrayPool<T>.Shared.Return(pooledArray, clearArray: true);
#endif
        }
    }

    /// <summary>
    /// Computes the ArrayPool bucket size for a given request.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ArrayPoolBucketSize(int requestedSize)
    {
        if (requestedSize <= 16) return 16;
        int v = requestedSize - 1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }
}
