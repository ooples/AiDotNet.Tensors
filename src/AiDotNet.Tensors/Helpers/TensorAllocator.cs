using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// High-performance tensor allocation helper that eliminates zero-initialization overhead.
/// Uses GC.AllocateUninitializedArray on .NET 5+ for small-medium tensors, and
/// ArrayPool for large tensors to reduce GC pressure from frequent allocations.
/// </summary>
internal static class TensorAllocator
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
            // Slice to exact size (pooled array may be larger)
            var memory = new Memory<T>(pooled, 0, totalSize);
            return Tensor<T>.FromPooledMemory(memory, shape, pooled);
        }

        // Small-medium tensors: skip zero-initialization
        T[] array = GC.AllocateUninitializedArray<T>(totalSize);
        var mem = new Memory<T>(array);
        return Tensor<T>.FromMemory(mem, shape);
#else
        return new Tensor<T>(shape);
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
