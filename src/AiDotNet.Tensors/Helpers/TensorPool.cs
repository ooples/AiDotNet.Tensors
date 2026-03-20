using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Public facade for tensor pooling that delegates to <see cref="TensorAllocator"/>.
/// <see cref="Rent{T}(int[])"/> returns zero-initialized memory for safe concurrent access.
/// <see cref="RentUninitialized{T}"/> skips zero-initialization on .NET 5+ when
/// <see cref="Enabled"/> is true; on older targets or when disabled, the returned
/// memory may be zero-initialized (callers must not rely on uninitialized content).
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
