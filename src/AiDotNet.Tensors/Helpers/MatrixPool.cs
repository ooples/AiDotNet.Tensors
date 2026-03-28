using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Public facade for zero-alloc matrix allocation with transparent caching.
/// Behind the scenes, <see cref="Rent{T}(int, int)"/> uses a thread-local buffer cache
/// (zero contention, zero allocation after warmup) backed by ArrayPool for cache misses.
/// Call <see cref="Return{T}"/> when matrices are no longer needed to enable buffer reuse.
/// All overloads respect <see cref="TensorPool.Enabled"/>; when disabled, plain non-pooled
/// matrices are returned instead.
/// </summary>
public static class MatrixPool
{
    /// <summary>
    /// Creates a zero-initialized matrix with the given dimensions.
    /// Large matrices use ArrayPool to reduce GC pressure; small-medium matrices
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Matrix<T> Rent<T>(int rows, int cols)
    {
        if (!TensorPool.Enabled)
            return new Matrix<T>(rows, cols);

        return MatrixAllocator.Rent<T>(rows, cols);
    }

    /// <summary>
    /// Creates a matrix backed by NativeMemory (64-byte aligned, zero GC overhead).
    /// Use for matrices accessed via Span in native interop (oneDNN, VML).
    /// </summary>
    public static Matrix<T> RentNative<T>(int rows, int cols) where T : unmanaged
    {
        return MatrixAllocator.RentNative<T>(rows, cols);
    }

    /// <summary>
    /// Creates a matrix for immediate-overwrite scenarios, skipping zero-initialization
    /// where possible. Callers MUST overwrite all elements before reading.
    /// </summary>
    public static Matrix<T> RentUninitialized<T>(int rows, int cols)
    {
        if (!TensorPool.Enabled)
            return new Matrix<T>(rows, cols);

        return MatrixAllocator.RentUninitialized<T>(rows, cols);
    }

    /// <summary>
    /// Creates a matrix with data from an existing flat array, using pooled memory for large matrices.
    /// </summary>
    public static Matrix<T> Rent<T>(int rows, int cols, T[] data)
    {
        if (!TensorPool.Enabled)
        {
            var memory = new Memory<T>((T[])data.Clone());
            return Matrix<T>.FromMemory(memory, rows, cols);
        }

        return MatrixAllocator.Rent(rows, cols, data);
    }

    /// <summary>
    /// Returns a matrix's backing array to the pool if it was pooled.
    /// Call this when a matrix from Rent() is no longer needed.
    /// </summary>
    public static void Return<T>(Matrix<T>? matrix)
    {
        MatrixAllocator.Return(matrix);
    }
}
