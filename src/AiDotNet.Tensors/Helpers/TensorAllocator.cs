using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Tensor allocation with transparent zero-alloc caching via <see cref="ThreadLocalTensorCache{T}"/>.
/// After the first forward pass warms the cache, <see cref="Rent{T}(int[])"/> reuses thread-local
/// buffers — zero allocation, zero GC, zero lock contention, completely invisible to callers.
/// Falls back to ArrayPool for cache misses, then standard allocation when pooling is disabled.
/// Return pooled tensors via <see cref="TensorPool.Return{T}"/> to enable buffer reuse.
/// </summary>
public static class TensorAllocator
{
    /// <summary>
    /// Threshold above which ArrayPool is used instead of standard allocation.
    /// ArrayPool avoids GC pressure for repeated large allocations (e.g., GEMM temporaries).
    /// 256K elements = 1MB for float, 2MB for double.
    /// </summary>
    private const int ArrayPoolThreshold = 256 * 1024;

    /// <summary>
    /// Public accessor for the ArrayPool threshold used by TensorWorkspace and other helpers.
    /// </summary>
    public const int ArrayPoolThresholdValue = ArrayPoolThreshold;

    /// <summary>
    /// Creates a zero-initialized tensor with the given shape.
    /// Large tensors use ArrayPool to reduce GC pressure; small-medium tensors
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        if (!TensorPool.Enabled || totalSize == 0)
        {
            return new Tensor<T>(shape);
        }

#if NET5_0_OR_GREATER
        // Tier 1: Thread-local cache — zero allocation, zero contention.
        // After warmup, this path hits on every call (same shapes used repeatedly).
        T[]? cached = ThreadLocalTensorCache<T>.TryRent(
            totalSize >= ArrayPoolThreshold ? 0 : totalSize); // only cache small; large go to ArrayPool
        if (cached is null && totalSize >= ArrayPoolThreshold)
        {
            // For large tensors, ArrayPool.Rent returns power-of-2 sizes.
            // Cache by ArrayPool bucket size so we match on return.
            cached = ThreadLocalTensorCache<T>.TryRent(ArrayPoolBucketSize(totalSize));
        }
        if (cached is not null)
        {
            Array.Clear(cached, 0,
                RuntimeHelpers.IsReferenceOrContainsReferences<T>() ? cached.Length : totalSize);
            var memory = new Memory<T>(cached, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, cached);
        }

        // Tier 2: ArrayPool — O(1) allocation with buffer reuse across threads.
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            Array.Clear(pooled, 0,
                RuntimeHelpers.IsReferenceOrContainsReferences<T>() ? pooled.Length : totalSize);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Tier 3: Standard managed allocation for small tensors.
        T[] arr = new T[totalSize];
        var mem = new Memory<T>(arr);
        return Tensor<T>.FromMemory(mem, shape);
#else
        return new Tensor<T>(shape);
#endif
    }

    /// <summary>
    /// Computes the ArrayPool bucket size for a given request.
    /// ArrayPool returns power-of-2 sizes, so we need to match on return.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ArrayPoolBucketSize(int requestedSize)
    {
        // ArrayPool.Shared uses power-of-2 buckets starting at 16
        if (requestedSize <= 16) return 16;
        // Round up to next power of 2
        int v = requestedSize - 1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    /// <summary>
    /// Creates a tensor with the given shape WITHOUT zero-initialization.
    /// Use ONLY for tensors that will be immediately and fully overwritten
    /// (e.g., weight initialization with random values, copy targets).
    /// WARNING: Contains stale/garbage data until overwritten.
    /// </summary>
    /// <remarks>
    /// On .NET 5+, uses <c>GC.AllocateUninitializedArray</c> or <c>ArrayPool</c> to skip zeroing.
    /// On pre-.NET 5 targets (e.g., net471), <c>GC.AllocateUninitializedArray</c> is unavailable,
    /// so the fallback returns zero-initialized memory via <c>new T[]</c>. This is safe (callers
    /// overwrite all elements anyway) but does not provide the performance benefit of skipping
    /// zero-initialization on older frameworks.
    /// </remarks>
    public static Tensor<T> RentUninitialized<T>(int[] shape)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        if (!TensorPool.Enabled || totalSize == 0)
        {
            return new Tensor<T>(shape);
        }

#if NET5_0_OR_GREATER
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            // Clear the full array when T holds references to avoid keeping stale
            // objects alive — including the tail [totalSize, pooled.Length)
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(pooled, 0, pooled.Length);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Skip zero-initialization for immediate-overwrite tensors
        T[] array = GC.AllocateUninitializedArray<T>(totalSize);
        var mem = new Memory<T>(array);
        return Tensor<T>.FromMemory(mem, shape);
#else
        return new Tensor<T>(shape);
#endif
    }

    /// <summary>
    /// Creates a tensor with the given shape and copies data from a Vector.
    /// Uses pooled memory for large tensors to reduce GC pressure.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape, Vector<T> data)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        if (totalSize != data.Length)
            throw new ArgumentException(
                $"Data length ({data.Length}) must match shape total ({totalSize}).",
                nameof(data));

        if (!TensorPool.Enabled || totalSize == 0)
        {
            return new Tensor<T>(shape, data);
        }

#if NET5_0_OR_GREATER
        ReadOnlySpan<T> src = data.AsSpan();
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            src.CopyTo(pooled.AsSpan(0, totalSize));
            // Clear tail [totalSize, pooled.Length) when T contains references so
            // stale objects from previous tenants aren't kept alive via _pooledArray
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>() && pooled.Length > totalSize)
                Array.Clear(pooled, totalSize, pooled.Length - totalSize);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Small-medium: allocate uninitialized then span-copy (avoid double-write from new T[] + copy)
        T[] array = GC.AllocateUninitializedArray<T>(totalSize);
        src.CopyTo(array);
        var mem = new Memory<T>(array);
        return Tensor<T>.FromMemory(mem, shape);
#else
        return new Tensor<T>(shape, data);
#endif
    }

    /// <summary>
    /// Returns a tensor's backing array to the pool if it was pooled.
    /// SAFETY: The caller MUST ensure the tensor is never accessed after this call.
    /// The tensor's Memory still references the returned array — any access after
    /// Return is undefined behavior (data corruption from reuse).
    /// Only call this for internal temporaries that immediately go out of scope.
    /// External callers should use <see cref="TensorPool.Return{T}"/> instead.
    /// </summary>
    internal static void Return<T>(Tensor<T>? tensor)
    {
        if (tensor == null) return;

        T[]? pooledArray = tensor.PooledArray;
        if (pooledArray != null)
        {
            tensor.DetachPooledArray();
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
}
