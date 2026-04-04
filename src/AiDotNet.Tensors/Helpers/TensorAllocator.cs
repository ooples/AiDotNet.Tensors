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

        // Tier 0: Arena allocation — zero GC during training loops.
        var arena = TensorArena.Current;
        if (arena != null)
        {
            T[]? arenaArray = arena.TryAllocate<T>(totalSize);
            if (arenaArray != null)
            {
                var memory = new Memory<T>(arenaArray, 0, totalSize);
                return Tensor<T>.FromMemory(memory, shape);
            }
            // Arena full — fall through to other tiers
        }

#if NET5_0_OR_GREATER
        // Tier 1: Thread-local cache — zero allocation after warmup.
        T[]? cached = ThreadLocalTensorCache<T>.TryRent(totalSize);
        if (cached is null && totalSize >= ArrayPoolThreshold)
            cached = ThreadLocalTensorCache<T>.TryRent(ArrayPoolBucketSize(totalSize));
        if (cached is not null)
        {
            Array.Clear(cached, 0,
                RuntimeHelpers.IsReferenceOrContainsReferences<T>() ? cached.Length : totalSize);
            var memory = new Memory<T>(cached, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, cached);
        }

        // Tier 3: ArrayPool for large reference types.
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            Array.Clear(pooled, 0,
                RuntimeHelpers.IsReferenceOrContainsReferences<T>() ? pooled.Length : totalSize);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Tier 4: Standard managed allocation for small tensors.
        T[] arr = new T[totalSize];
        var mem = new Memory<T>(arr);
        return Tensor<T>.FromMemory(mem, shape);
#else
        return new Tensor<T>(shape);
#endif
    }

    /// <summary>
    /// Rents a tensor without zeroing its backing array. The caller MUST write every element
    /// before reading. Use for backward kernels and operations that overwrite the full buffer.
    /// When a TensorArena is active, reuses arena-allocated arrays without clearing — saving
    /// ~0.3ms per 1M-element tensor (the Array.Clear cost).
    /// </summary>
    public static Tensor<T> RentUninitialized<T>(int[] shape)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        // Arena path: skip Array.Clear entirely
        var arena = TensorArena.Current;
        if (arena != null)
        {
            T[]? arenaArray = arena.TryAllocateUninitialized<T>(totalSize);
            if (arenaArray != null)
            {
                var memory = new Memory<T>(arenaArray, 0, totalSize);
                return Tensor<T>.FromMemory(memory, shape);
            }
        }

#if NET5_0_OR_GREATER
        // Thread-local cache: skip Array.Clear
        T[]? cached = ThreadLocalTensorCache<T>.TryRent(totalSize);
        if (cached is null && totalSize >= ArrayPoolThreshold)
            cached = ThreadLocalTensorCache<T>.TryRent(ArrayPoolBucketSize(totalSize));
        if (cached is not null)
        {
            // Must clear reference types to avoid retaining stale objects
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(cached, 0, cached.Length);
            var memory = new Memory<T>(cached, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, cached);
        }

        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            // Must clear reference types to avoid retaining stale objects
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(pooled, 0, pooled.Length);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Small allocation: new T[] is zeroed by CLR but that's unavoidable
        T[] arr = new T[totalSize];
        return Tensor<T>.FromMemory(new Memory<T>(arr), shape);
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
    /// Creates a tensor backed by NativeMemory (64-byte aligned, zero GC overhead).
    /// Use for tensors that will ONLY be accessed via Pin()/Memory.Span — never GetDataArray().
    /// oneDNN and VML operations should use this for optimal performance.
    /// GetDataArray() triggers lazy demotion (one-time copy to managed array).
    /// </summary>
    public static Tensor<T> RentNative<T>(int[] shape) where T : unmanaged
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        if (totalSize == 0)
            return new Tensor<T>(shape);

#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(float))
        {
            var owner = new NativeMemoryOwner<float>(totalSize, zeroed: true);
            return (Tensor<T>)(object)Tensor<float>.FromMemory(owner.Memory, shape);
        }
        if (typeof(T) == typeof(double))
        {
            var owner = new NativeMemoryOwner<double>(totalSize, zeroed: true);
            return (Tensor<T>)(object)Tensor<double>.FromMemory(owner.Memory, shape);
        }
#endif
        // Fallback for other unmanaged types
        return Rent<T>(shape);
    }

    // RentUninitialized is defined above (line ~95) with arena support

    /// <summary>
    /// Creates a tensor with the given shape and data from a Vector.
    /// Zero-copy when the Vector's backing array is exactly the right size
    /// (common pattern: caller does new T[n], computes into it, wraps in Vector).
    /// Falls back to copy when the backing array can't be extracted.
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

#if NET5_0_OR_GREATER
        // Zero-copy path: if the Vector's backing array is exactly totalSize,
        // wrap it directly — no allocation, no copy. This is the common case when
        // callers do: var arr = new T[n]; compute(arr); Rent(shape, new Vector<T>(arr))
        T[]? backingArray = data._cachedArray;
        if (backingArray is not null && backingArray.Length == totalSize)
        {
            var memory = new Memory<T>(backingArray);
            return Tensor<T>.FromMemory(memory, shape);
        }
#endif

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
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>() && pooled.Length > totalSize)
                Array.Clear(pooled, totalSize, pooled.Length - totalSize);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

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
