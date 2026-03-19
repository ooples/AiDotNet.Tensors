using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Low-level tensor allocation helper that uses ArrayPool for large tensors to reduce GC pressure.
/// <see cref="Rent{T}(int[])"/> returns zero-initialized memory for correctness under concurrent access.
/// <see cref="RentUninitialized{T}"/> returns uninitialized memory for callers that will
/// immediately overwrite all elements (e.g., weight initialization, copy targets).
/// Callers should prefer <see cref="TensorPool"/> which gates all paths through
/// <see cref="TensorPool.Enabled"/> before delegating here.
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
    /// Creates a zero-initialized tensor with the given shape.
    /// Large tensors use ArrayPool to reduce GC pressure; small-medium tensors
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        if (totalSize == 0)
        {
            return new Tensor<T>(shape);
        }

#if NET5_0_OR_GREATER
        // Large tensors: use ArrayPool to avoid GC pressure from repeated allocations
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            // Zero-fill the rented region — ArrayPool returns arrays with stale data
            // from previous tenants, causing data corruption under concurrent access
            Array.Clear(pooled, 0, totalSize);
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Small-medium tensors: use zero-initialized allocation for correctness.
        // GC.AllocateUninitializedArray skips zeroing for performance, but this
        // causes stale-data races when tensors are allocated concurrently (e.g.,
        // parallel test execution or multi-threaded model training).
        T[] array = new T[totalSize];
        var mem = new Memory<T>(array);
        return Tensor<T>.FromMemory(mem, shape);
#else
        return new Tensor<T>(shape);
#endif
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

        if (totalSize == 0)
        {
            return new Tensor<T>(shape);
        }

#if NET5_0_OR_GREATER
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            // Clear only when T holds references to avoid keeping stale objects alive
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                Array.Clear(pooled, 0, totalSize);
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

        if (totalSize == 0)
        {
            return new Tensor<T>(shape, data);
        }

#if NET5_0_OR_GREATER
        ReadOnlySpan<T> src = data.AsWritableSpan();
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            src.CopyTo(pooled.AsSpan(0, totalSize));
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
    /// </summary>
    public static void Return<T>(Tensor<T>? tensor)
    {
        if (tensor == null) return;

        T[]? pooledArray = tensor.PooledArray;
        if (pooledArray != null)
        {
            tensor.DetachPooledArray();
#if NET5_0_OR_GREATER
            ArrayPool<T>.Shared.Return(pooledArray,
                clearArray: RuntimeHelpers.IsReferenceOrContainsReferences<T>());
#else
            ArrayPool<T>.Shared.Return(pooledArray, clearArray: true);
#endif
        }
    }
}
