using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Matrix allocation with transparent zero-alloc caching via <see cref="ThreadLocalTensorCache{T}"/>.
/// After the first forward pass warms the cache, <see cref="Rent{T}(int, int)"/> reuses thread-local
/// buffers — zero allocation, zero GC, zero lock contention, completely invisible to callers.
/// Falls back to ArrayPool for cache misses, then standard allocation when pooling is disabled.
/// Return pooled matrices via <see cref="Return{T}"/> to enable buffer reuse.
/// </summary>
public static class MatrixAllocator
{
    /// <summary>
    /// Threshold above which ArrayPool is used instead of standard allocation.
    /// ArrayPool avoids GC pressure for repeated large allocations.
    /// 256K elements = 1MB for float, 2MB for double.
    /// </summary>
    private const int ArrayPoolThreshold = 256 * 1024;

    /// <summary>
    /// Creates a zero-initialized matrix with the given dimensions.
    /// Large matrices use ArrayPool to reduce GC pressure; small-medium matrices
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Matrix<T> Rent<T>(int rows, int cols)
    {
        int totalSize = checked(rows * cols);

        if (!TensorPool.Enabled || totalSize == 0)
        {
            return new Matrix<T>(rows, cols);
        }

        // Tier 0: Arena allocation — zero GC during training loops.
        var arena = TensorArena.Current;
        if (arena != null)
        {
            T[]? arenaArray = arena.TryAllocate<T>(totalSize);
            if (arenaArray != null)
            {
                var memory = new Memory<T>(arenaArray, 0, totalSize);
                return Matrix<T>.FromMemory(memory, rows, cols);
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
            return Matrix<T>.FromPooledMemory(memory, rows, cols, cached);
        }

        // Tier 2: ArrayPool for large allocations.
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            Array.Clear(pooled, 0,
                RuntimeHelpers.IsReferenceOrContainsReferences<T>() ? pooled.Length : totalSize);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Matrix<T>.FromPooledMemory(memory, rows, cols, pooled);
        }

        // Tier 3: Standard managed allocation for small matrices.
        T[] arr = new T[totalSize];
        var mem = new Memory<T>(arr);
        return Matrix<T>.FromMemory(mem, rows, cols);
#else
        return new Matrix<T>(rows, cols);
#endif
    }

    /// <summary>
    /// Creates a matrix with the given dimensions WITHOUT zero-initialization.
    /// Use ONLY for matrices that will be immediately and fully overwritten
    /// (e.g., matmul results, copy targets).
    /// WARNING: Contains stale/garbage data until overwritten.
    /// </summary>
    public static Matrix<T> RentUninitialized<T>(int rows, int cols)
    {
        int totalSize = checked(rows * cols);

        if (!TensorPool.Enabled || totalSize == 0)
        {
            return new Matrix<T>(rows, cols);
        }

#if NET5_0_OR_GREATER
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(pooled, 0, pooled.Length);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Matrix<T>.FromPooledMemory(memory, rows, cols, pooled);
        }

        // Skip zero-initialization for immediate-overwrite matrices
        T[] array = GC.AllocateUninitializedArray<T>(totalSize);
        var mem = new Memory<T>(array);
        return Matrix<T>.FromMemory(mem, rows, cols);
#else
        return new Matrix<T>(rows, cols);
#endif
    }

    /// <summary>
    /// Returns a matrix's backing array to the pool if it was pooled.
    /// SAFETY: The caller MUST ensure the matrix is never accessed after this call.
    /// </summary>
    internal static void Return<T>(Matrix<T>? matrix)
    {
        if (matrix == null) return;

        T[]? pooledArray = matrix.PooledArray;
        if (pooledArray != null)
        {
            matrix.DetachPooledArray();
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
