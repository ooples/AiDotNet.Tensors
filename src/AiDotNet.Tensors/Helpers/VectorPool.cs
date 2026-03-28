using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Public facade for zero-alloc vector allocation with transparent caching.
/// Behind the scenes, <see cref="Rent{T}(int)"/> uses a thread-local buffer cache
/// (zero contention, zero allocation after warmup) backed by ArrayPool for cache misses.
/// Call <see cref="Return{T}"/> when vectors are no longer needed to enable buffer reuse.
/// All overloads respect <see cref="TensorPool.Enabled"/>; when disabled, plain non-pooled
/// vectors are returned instead.
/// </summary>
public static class VectorPool
{
    /// <summary>
    /// Creates a zero-initialized vector with the given length.
    /// Large vectors use ArrayPool to reduce GC pressure; small-medium vectors
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Vector<T> Rent<T>(int length)
    {
        if (!TensorPool.Enabled)
            return new Vector<T>(length);

        return VectorAllocator.Rent<T>(length);
    }

    /// <summary>
    /// Creates a vector backed by NativeMemory (64-byte aligned, zero GC overhead).
    /// Use for vectors accessed via Span in native interop (oneDNN, VML).
    /// </summary>
    public static Vector<T> RentNative<T>(int length) where T : unmanaged
    {
        return VectorAllocator.RentNative<T>(length);
    }

    /// <summary>
    /// Creates a vector for immediate-overwrite scenarios, skipping zero-initialization
    /// where possible. Callers MUST overwrite all elements before reading.
    /// </summary>
    public static Vector<T> RentUninitialized<T>(int length)
    {
        if (!TensorPool.Enabled)
            return new Vector<T>(length);

        return VectorAllocator.RentUninitialized<T>(length);
    }

    /// <summary>
    /// Creates a vector with data from an existing array, using pooled memory for large vectors.
    /// </summary>
    public static Vector<T> Rent<T>(T[] data)
    {
        if (!TensorPool.Enabled)
            return new Vector<T>(data);

        return VectorAllocator.Rent(data);
    }

    /// <summary>
    /// Returns a vector's backing array to the pool if it was pooled.
    /// Call this when a vector from Rent() is no longer needed.
    /// </summary>
    public static void Return<T>(Vector<T>? vector)
    {
        VectorAllocator.Return(vector);
    }
}
