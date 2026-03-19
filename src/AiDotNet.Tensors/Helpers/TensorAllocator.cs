using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Tensor allocation helper that uses ArrayPool for large tensors to reduce GC pressure.
/// All returned tensors are zero-initialized for correctness under concurrent access.
/// On .NET 5+, large tensors use ArrayPool (with explicit clearing), and small-medium
/// tensors use standard <c>new T[]</c> allocation which is zero-initialized by the CLR.
/// </summary>
internal static class TensorAllocator
{
    /// <summary>
    /// Whether pooled tensor allocation is enabled. Defaults to true.
    /// Can be disabled via AIDOTNET_DISABLE_TENSOR_POOL=1 environment variable.
    /// </summary>
    public static bool Enabled { get; set; } = !IsEnvTrue("AIDOTNET_DISABLE_TENSOR_POOL");

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

        if (!Enabled || totalSize == 0)
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
    public static Tensor<T> RentUninitialized<T>(int[] shape)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize = checked(totalSize * shape[i]);

        if (!Enabled || totalSize == 0)
        {
            return new Tensor<T>(shape);
        }

#if NET5_0_OR_GREATER
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            // NO Array.Clear — caller will overwrite all elements
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

        if (!Enabled || totalSize == 0)
        {
            return new Tensor<T>(shape, data);
        }

#if NET5_0_OR_GREATER
        if (totalSize >= ArrayPoolThreshold)
        {
            T[] pooled = ArrayPool<T>.Shared.Rent(totalSize);
            // Copy data into pooled array
            for (int i = 0; i < totalSize; i++)
                pooled[i] = data[i];
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Small-medium: allocate uninitialized then copy (avoid double-write from new T[] + copy)
        T[] array = GC.AllocateUninitializedArray<T>(totalSize);
        for (int i = 0; i < totalSize; i++)
            array[i] = data[i];
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
    internal static void Return<T>(Tensor<T>? tensor)
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

    private static bool IsEnvTrue(string name)
    {
        string? val = Environment.GetEnvironmentVariable(name);
        return string.Equals(val, "1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(val, "true", StringComparison.OrdinalIgnoreCase);
    }
}
