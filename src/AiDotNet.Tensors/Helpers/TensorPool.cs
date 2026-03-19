using System.Buffers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// High-performance tensor allocation helper that eliminates zero-initialization overhead.
/// Uses GC.AllocateUninitializedArray on .NET 5+ for small-medium tensors, and
/// ArrayPool for large tensors to reduce GC pressure from frequent allocations.
/// </summary>
internal static class TensorPool
{
    /// <summary>
    /// Whether fast tensor allocation is enabled. Defaults to true.
    /// Can be disabled via AIDOTNET_DISABLE_TENSOR_POOL=1 environment variable.
    /// </summary>
    public static bool Enabled { get; set; } = !IsEnvTrue("AIDOTNET_DISABLE_TENSOR_POOL");

    /// <summary>
    /// Threshold above which ArrayPool is used instead of GC.AllocateUninitializedArray.
    /// ArrayPool avoids GC pressure for repeated large allocations (e.g., GEMM temporaries).
    /// 256K elements = 1MB for float, 2MB for double.
    /// </summary>
    private const int ArrayPoolThreshold = 256 * 1024;

    /// <summary>
    /// Creates a tensor with the given shape using uninitialized or pooled memory.
    /// The tensor's data is NOT zero-initialized for performance.
    /// Caller MUST overwrite all elements before exposing to consumers.
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
        // causes stale-data races when tensors are allocated concurrently.
        T[] array = new T[totalSize];
        var mem = new Memory<T>(array);
        return Tensor<T>.FromMemory(mem, shape);
#else
        return new Tensor<T>(shape);
#endif
    }

    /// <summary>
    /// Returns a tensor's backing array to the pool if it was pooled.
    /// Call this when a tensor from Rent() is no longer needed.
    /// </summary>
    public static void Return<T>(Tensor<T>? tensor)
    {
        if (tensor == null) return;

        T[]? pooledArray = tensor.PooledArray;
        if (pooledArray != null)
        {
            tensor.DetachPooledArray();
            ArrayPool<T>.Shared.Return(pooledArray);
        }
    }

    private static bool IsEnvTrue(string name)
    {
        string? val = Environment.GetEnvironmentVariable(name);
        return string.Equals(val, "1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(val, "true", StringComparison.OrdinalIgnoreCase);
    }
}
