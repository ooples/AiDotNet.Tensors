using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Public facade for zero-alloc tensor allocation with transparent caching.
/// Behind the scenes, <see cref="Rent{T}(int[])"/> uses a thread-local buffer cache
/// (zero contention, zero allocation after warmup) backed by ArrayPool for cache misses.
/// Call <see cref="Return{T}"/> when tensors are no longer needed to enable buffer reuse.
/// All overloads respect <see cref="Enabled"/>; when disabled, plain non-pooled
/// tensors are returned instead.
/// </summary>
public static class TensorPool
{
    /// <summary>
    /// Single source of truth for whether pooled/optimized tensor allocation is enabled.
    /// Defaults to true. Can be disabled via AIDOTNET_DISABLE_TENSOR_POOL=1 environment variable.
    /// When disabled, all Rent overloads fall back to standard <c>new Tensor&lt;T&gt;(shape)</c>.
    /// </summary>
    public static bool Enabled { get; set; } = !IsEnvTrue("AIDOTNET_DISABLE_TENSOR_POOL");

    /// <summary>
    /// Creates a zero-initialized tensor with the given shape.
    /// Large tensors use ArrayPool to reduce GC pressure; small-medium tensors
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape)
    {
        if (!Enabled)
            return new Tensor<T>(shape);

        return TensorAllocator.Rent<T>(shape);
    }

    /// <summary>
    /// Creates a tensor for immediate-overwrite scenarios, skipping zero-initialization
    /// where possible. On .NET 5+ with pooling enabled, memory is truly uninitialized
    /// (except for reference-containing types which are always cleared). When pooling is
    /// disabled or on older targets, the returned memory may be zero-initialized.
    /// Callers MUST overwrite all elements before reading regardless of initialization state.
    /// </summary>
    public static Tensor<T> RentUninitialized<T>(int[] shape)
    {
        if (!Enabled)
            return new Tensor<T>(shape);

        return TensorAllocator.RentUninitialized<T>(shape);
    }

    /// <summary>
    /// Creates a tensor with data from a Vector, using pooled memory for large tensors.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape, Vector<T> data)
    {
        if (!Enabled)
            return new Tensor<T>(shape, data);

        return TensorAllocator.Rent(shape, data);
    }

    /// <summary>
    /// Returns a tensor's backing array to the pool if it was pooled.
    /// Call this when a tensor from Rent() is no longer needed.
    /// </summary>
    public static void Return<T>(Tensor<T>? tensor)
    {
        TensorAllocator.Return(tensor);
    }

    private static bool IsEnvTrue(string name)
    {
        string? val = Environment.GetEnvironmentVariable(name);
        return string.Equals(val, "1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(val, "true", StringComparison.OrdinalIgnoreCase);
    }
}
