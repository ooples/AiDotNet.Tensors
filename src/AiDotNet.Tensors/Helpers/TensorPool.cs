using System.Buffers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// High-performance tensor allocation helper that eliminates zero-initialization overhead.
/// Uses GC.AllocateUninitializedArray on .NET 5+ to skip zeroing memory that will be
/// immediately overwritten by SIMD operations.
/// </summary>
internal static class TensorPool
{
    /// <summary>
    /// Whether fast tensor allocation is enabled. Defaults to true.
    /// Can be disabled via AIDOTNET_DISABLE_TENSOR_POOL=1 environment variable.
    /// </summary>
    public static bool Enabled { get; set; } = !IsEnvTrue("AIDOTNET_DISABLE_TENSOR_POOL");

    /// <summary>
    /// Creates a tensor with the given shape using uninitialized memory.
    /// The tensor's data is NOT zero-initialized for performance.
    /// Caller MUST overwrite all elements before exposing to consumers.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape)
    {
        int totalSize = 1;
        for (int i = 0; i < shape.Length; i++)
            totalSize *= shape[i];

        if (!Enabled || totalSize == 0)
        {
            return new Tensor<T>(shape);
        }

#if NET5_0_OR_GREATER
        T[] array = GC.AllocateUninitializedArray<T>(totalSize);
        var memory = new Memory<T>(array);
        return Tensor<T>.FromMemory(memory, shape);
#else
        return new Tensor<T>(shape);
#endif
    }

    /// <summary>
    /// Returns a tensor's backing array to the pool if it was pooled.
    /// Currently a no-op since we use GC.AllocateUninitializedArray instead of ArrayPool.
    /// Kept for future ArrayPool integration.
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
