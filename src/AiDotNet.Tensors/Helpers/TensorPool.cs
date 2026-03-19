using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Public facade for tensor pooling that delegates to <see cref="TensorAllocator"/>.
/// <see cref="Rent{T}(int[])"/> returns zero-initialized memory for safe concurrent access.
/// <see cref="RentUninitialized{T}"/> returns uninitialized memory for callers that will
/// immediately overwrite all elements (e.g., weight initialization, copy targets).
/// </summary>
public static class TensorPool
{
    /// <summary>
    /// Creates a zero-initialized tensor with the given shape.
    /// Large tensors use ArrayPool to reduce GC pressure; small-medium tensors
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape)
    {
        return TensorAllocator.Rent<T>(shape);
    }

    /// <summary>
    /// Creates a tensor WITHOUT zero-initialization for immediate-overwrite scenarios.
    /// Use ONLY when caller will fully overwrite all elements before reading.
    /// WARNING: Contains stale/garbage data until overwritten (except for reference-containing
    /// types, which are always cleared to prevent keeping stale objects alive).
    /// </summary>
    public static Tensor<T> RentUninitialized<T>(int[] shape)
    {
        return TensorAllocator.RentUninitialized<T>(shape);
    }

    /// <summary>
    /// Creates a tensor with data from a Vector, using pooled memory for large tensors.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape, Vector<T> data)
    {
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
}
