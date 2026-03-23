using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a base class for multi-dimensional arrays of numeric values used in machine learning and AI computations.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double, int).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> TensorBase is an abstract class that provides the foundation for working with tensors.
/// It defines common properties and methods that all tensor implementations should have, regardless of their specific type or dimensionality.
/// </para>
/// </remarks>
public abstract class TensorBase<T>
{
    /// <summary>
    /// The underlying data storage for the tensor elements.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This field stores all the values in the tensor in a one-dimensional array.
    /// Even though a tensor can have multiple dimensions, we store its data in a flat structure for efficiency.
    /// The class provides methods to convert between multi-dimensional indices and this flat storage.</para>
    /// </remarks>
    protected readonly Vector<T> _data;

    /// <summary>
    /// Pre-computed strides for each dimension, following PyTorch's stride convention.
    /// For row-major order: strides[i] = product of Shape[i+1..end].
    /// For transposed views: strides are permuted without copying data.
    /// </summary>
    protected readonly int[] _strides;

    /// <summary>
    /// Offset into the underlying storage where this tensor's data begins.
    /// Zero for non-view tensors. Non-zero for sliced views.
    /// </summary>
    protected readonly int _storageOffset;

    /// <summary>
    /// Whether this tensor's data is contiguous in memory (row-major with no gaps).
    /// When true, raw span/array access is safe. When false, Contiguous() must be called
    /// before passing to BLAS/SIMD operations.
    /// </summary>
    public bool IsContiguous { get; }

    /// <summary>
    /// Whether this tensor is a view into another tensor's storage.
    /// Views share memory — mutations through one view are visible in others.
    /// </summary>
    public bool IsView { get; }

    /// <summary>
    /// Gets the pre-computed strides for each dimension.
    /// </summary>
    public ReadOnlySpan<int> Strides => _strides;



    /// <summary>
    /// Provides numeric operations for the tensor's element type.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This field holds a set of mathematical operations (like addition, multiplication, etc.)
    /// that work with the specific numeric type of this tensor. It allows the tensor to perform calculations
    /// regardless of whether it contains integers, floating-point numbers, or other numeric types.</para>
    /// </remarks>
    protected static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the shape (dimensions) of the tensor.
    /// </summary>
    public int[] Shape { get; }

    /// <summary>
    /// Gets the total number of logical elements in this tensor (product of all shape dimensions).
    /// For views, this is the view's element count, not the underlying storage size.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets the rank (number of dimensions) of the tensor.
    /// </summary>
    public int Rank => Shape.Length;

    /// <summary>
    /// Gets the underlying data as a Memory&lt;T&gt; for zero-copy access.
    /// Use .Pin() for unsafe pointer access compatible with both managed and POH-backed tensors.
    /// Use .Span for indexed access or iteration.
    /// </summary>
    /// <remarks>
    /// <para>Internal to prevent external consumers from directly accessing model weights/tensor data.
    /// AiDotNet libraries access this via InternalsVisibleTo.</para>
    /// <para>For large tensors (>256K elements), the memory is POH-pinned, so Pin() is essentially free.</para>
    /// </remarks>
    internal Memory<T> Memory
    {
        get
        {
            if (!IsContiguous)
                throw new InvalidOperationException(
                    "Cannot get contiguous Memory from a non-contiguous tensor view. Call Contiguous() first.");
            return _storageOffset == 0 && _data.Length == Length
                ? _data.AsWritableMemory()
                : _data.AsWritableMemory().Slice(_storageOffset, Length);
        }
    }

    /// <summary>
    /// Shorthand alias for <see cref="Memory"/> — used by engine code.
    /// </summary>
    internal Memory<T> Data => Memory;

    /// <summary>
    /// Creates a new array containing a copy of the tensor's elements in flattened order.
    /// </summary>
    /// <returns>A new array containing the tensor's elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts the tensor into a regular one-dimensional array.
    /// The data is returned in flattened (row-major) order. The returned array is a copy, so
    /// changes to it will not affect the original tensor.</para>
    /// </remarks>
    public virtual T[] ToArray()
    {
        if (IsContiguous && _storageOffset == 0 && _data.Length == Length)
        {
            return _data.ToArray();
        }
        // For views or offset tensors: copy logical elements in row-major order
        var result = new T[Length];
        if (IsContiguous)
        {
            // Contiguous with offset — bulk copy from offset
            _data.AsSpan().Slice(_storageOffset, Length).CopyTo(result);
        }
        else
        {
            // Non-contiguous view — use FlatIndexToStorageIndex (O(Rank) per element)
            var srcData = _data.AsSpan();
            for (int i = 0; i < Length; i++)
            {
                result[i] = srcData[FlatIndexToStorageIndex(i)];
            }
        }
        return result;
    }

    /// <summary>
    /// Copies data from a source array into this tensor's internal storage.
    /// </summary>
    /// <param name="source">The source array to copy from. Must have the same length as the tensor.</param>
    /// <exception cref="ArgumentException">Thrown when source array length doesn't match tensor length.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method copies values from a regular array into the tensor.
    /// The array must have exactly the same number of elements as the tensor (the product of all dimensions).
    /// This is useful for deserialization and bulk data loading.</para>
    /// </remarks>
    public virtual void CopyFromArray(T[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }
        if (source.Length != Length)
        {
            throw new ArgumentException($"Source array length ({source.Length}) must match tensor length ({Length}).");
        }
        if (IsContiguous && _storageOffset == 0 && _data.Length == Length)
        {
            source.AsSpan().CopyTo(_data.AsWritableSpan());
        }
        else if (IsContiguous)
        {
            source.AsSpan().CopyTo(_data.AsWritableSpan().Slice(_storageOffset, Length));
        }
        else
        {
            // Non-contiguous view — use FlatIndexToStorageIndex (O(Rank) per element)
            var dstData = _data.AsWritableSpan();
            for (int i = 0; i < Length; i++)
            {
                dstData[FlatIndexToStorageIndex(i)] = source[i];
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the TensorBase class with the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    protected TensorBase(params int[] shape)
    {
        Shape = shape;
        _strides = ComputeRowMajorStrides(shape);
        _storageOffset = 0;
        IsContiguous = true;
        IsView = false;
        int totalSize = shape.Aggregate(1, (acc, dim) => acc * dim);
        Length = totalSize;
        _data = new Vector<T>(totalSize);
    }

    /// <summary>
    /// Initializes a new instance of the TensorBase class with the specified data and shape.
    /// </summary>
    /// <param name="data">The data to populate the tensor with.</param>
    /// <param name="shape">The shape of the tensor.</param>
    protected TensorBase(IEnumerable<T> data, params int[] shape)
    {
        Shape = shape;
        _strides = ComputeRowMajorStrides(shape);
        _storageOffset = 0;
        IsContiguous = true;
        IsView = false;
        // When data is already a T[], use zero-copy Memory<T> path to preserve identity
        // (critical for GPU deferred materialization — the array reference must match)
        if (data is T[] array)
            _data = Vector<T>.WrapMemory(array);
        else
            _data = new Vector<T>(data);
        int expectedSize = shape.Aggregate(1, (acc, dim) => acc * dim);
        Length = expectedSize;
        if (_data.Length != expectedSize)
        {
            throw new ArgumentException("The number of values does not match the specified shape.");
        }
    }

    /// <summary>
    /// Initializes a new instance of the TensorBase class with an existing Vector (zero-copy).
    /// </summary>
    /// <param name="data">The vector to use as backing storage (not copied).</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <remarks>
    /// <para><b>Performance:</b> This constructor does NOT copy data. The tensor directly uses
    /// the provided vector's memory. This is useful for high-performance scenarios where
    /// memory pooling or external memory management is used.</para>
    /// </remarks>
    protected TensorBase(Vector<T> data, int[] shape)
    {
        Shape = shape;
        _strides = ComputeRowMajorStrides(shape);
        _storageOffset = 0;
        IsContiguous = true;
        IsView = false;
        _data = data;
        int expectedSize = shape.Aggregate(1, (acc, dim) => acc * dim);
        Length = expectedSize;
        if (_data.Length != expectedSize)
        {
            throw new ArgumentException("The number of values does not match the specified shape.");
        }
    }

    /// <summary>
    /// Internal constructor for creating views with custom strides and offset.
    /// No data is copied — the view shares the same underlying storage.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when strides length doesn't match shape, or when the view exceeds storage bounds.</exception>
    protected TensorBase(Vector<T> data, int[] shape, int[] strides, int storageOffset, bool isView)
    {
        if (strides.Length != shape.Length)
            throw new ArgumentException($"Strides length ({strides.Length}) must match shape length ({shape.Length}).");
        if (storageOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(storageOffset), "Storage offset must be non-negative.");

        // Validate that the maximum addressable index doesn't exceed storage
        int maxIndex = storageOffset;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] > 1)
                maxIndex += (shape[i] - 1) * Math.Abs(strides[i]);
        }
        if (maxIndex >= data.Length)
            throw new ArgumentException(
                $"View exceeds storage bounds: max index {maxIndex} >= storage length {data.Length}.");

        Shape = shape;
        _strides = strides;
        _storageOffset = storageOffset;
        IsView = isView;
        _data = data;
        Length = shape.Aggregate(1, (acc, dim) => acc * dim);
        IsContiguous = CheckContiguous(shape, strides);
    }

    /// <summary>
    /// Gets or sets the value at the specified indices.
    /// </summary>
    /// <param name="indices">The indices of the element.</param>
    /// <returns>The value at the specified indices.</returns>
    public virtual T this[params int[] indices]
    {
        get
        {
            ValidateIndices(indices);
            return _data[GetFlatIndex(indices)];
        }
        set
        {
            ValidateIndices(indices);
            _data[GetFlatIndex(indices)] = value;
        }
    }

    /// <summary>
    /// Gets or sets the value at the specified flat index.
    /// </summary>
    /// <param name="flatIndex">The flat (linear) index of the element.</param>
    /// <returns>The value at the specified flat index.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This indexer allows accessing tensor elements using a single
    /// index that treats the tensor as a 1D array. The flat index corresponds to
    /// row-major ordering where the last dimension varies fastest.</para>
    /// </remarks>
    public virtual T this[int flatIndex]
    {
        get => GetFlat(flatIndex);
        set => SetFlat(flatIndex, value);
    }

    /// <summary>
    /// Validates the provided indices against the tensor's shape.
    /// </summary>
    /// <param name="indices">The indices to validate.</param>
    protected void ValidateIndices(int[] indices)
    {
        if (indices.Length != Shape.Length)
            throw new ArgumentException("Number of indices must match the tensor's rank.");

        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= Shape[i])
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {i} is out of range.");
        }
    }

    /// <summary>
    /// Converts multi-dimensional indices to a flat index.
    /// </summary>
    /// <param name="indices">The multi-dimensional indices.</param>
    /// <returns>The corresponding flat index.</returns>
    protected int GetFlatIndex(int[] indices)
    {
        int flatIndex = _storageOffset;
        for (int i = 0; i < indices.Length; i++)
        {
            flatIndex += indices[i] * _strides[i];
        }
        return flatIndex;
    }

    /// <summary>
    /// Pre-computed row-major strides for logical flat index decomposition.
    /// Cached to avoid recomputing in hot paths (FlatIndexToStorageIndex, ToArray, etc.).
    /// For contiguous tensors, this equals _strides. For views, these are the OUTPUT strides.
    /// </summary>
    private int[]? _rowMajorStridesCache;
    private int[] RowMajorStrides => _rowMajorStridesCache ??= ComputeRowMajorStrides(Shape);

    /// <summary>
    /// Converts a logical flat index (row-major) to a storage index using strides and offset.
    /// Used by GetFlat/SetFlat for non-contiguous views.
    /// O(Rank) per call using pre-computed row-major strides.
    /// </summary>
    private int FlatIndexToStorageIndex(int flatIndex)
    {
        int storageIndex = _storageOffset;
        int remaining = flatIndex;
        var rmStrides = RowMajorStrides;
        for (int d = 0; d < Rank; d++)
        {
            int dimIndex = remaining / rmStrides[d];
            remaining -= dimIndex * rmStrides[d];
            storageIndex += dimIndex * _strides[d];
        }
        return storageIndex;
    }

    /// <summary>
    /// Computes row-major strides for the given shape.
    /// strides[i] = product of shape[i+1..end]
    /// Example: shape [3,4,5] → strides [20, 5, 1]
    /// </summary>
    protected static int[] ComputeRowMajorStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        if (shape.Length == 0) return strides;
        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    /// <summary>
    /// Checks whether the given shape+strides represent a contiguous row-major layout.
    /// </summary>
    private static bool CheckContiguous(int[] shape, int[] strides)
    {
        if (shape.Length == 0) return true;
        int expected = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            if (shape[i] != 1 && strides[i] != expected)
                return false;
            expected *= shape[i];
        }
        return true;
    }

    /// <summary>
    /// Creates a deep copy of this tensor.
    /// </summary>
    /// <returns>A new tensor with the same shape and values as this tensor.</returns>
    public virtual TensorBase<T> Clone()
    {
        var result = CreateInstance(Shape);

        if (IsContiguous && _storageOffset == 0 && _data.Length == Length)
        {
            // Fast path: SIMD bulk copy for non-view tensors
            _numOps.Copy(_data.AsSpan(), result._data.AsWritableSpan());
        }
        else
        {
            // View-safe: copy logical elements in row-major order
            var srcArray = ToArray();
            srcArray.AsSpan().CopyTo(result._data.AsWritableSpan());
        }

        return result;
    }

    /// <summary>
    /// Creates a new instance of the tensor with the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified shape.</returns>
    protected abstract TensorBase<T> CreateInstance(int[] shape);

    /// <summary>
    /// Creates a new instance of the tensor with the specified data and shape.
    /// </summary>
    /// <param name="data">The data to populate the new tensor with.</param>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified data and shape.</returns>
    protected abstract TensorBase<T> CreateInstance(T[] data, int[] shape);

    /// <summary>
    /// Creates a new instance of the tensor with the specified shape and a different element type.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the new tensor.</typeparam>
    /// <param name="shape">The shape of the new tensor.</param>
    /// <returns>A new tensor with the specified shape and element type.</returns>
    protected abstract TensorBase<TResult> CreateInstance<TResult>(params int[] shape);

    /// <summary>
    /// Applies a function to each element of the tensor.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the resulting tensor.</typeparam>
    /// <param name="func">The function to apply to each element.</param>
    /// <returns>A new tensor with the function applied to each element.</returns>
    public TensorBase<TResult> Transform<TResult>(Func<T, TResult> func)
    {
        var result = CreateInstance<TResult>(Shape);
        for (int i = 0; i < Length; i++)
        {
            // Use GetFlat which handles views correctly (offset + strides)
            result._data[i] = func(GetFlat(i));
        }

        return result;
    }

    /// <summary>
    /// Applies a function to each element of the tensor, providing the element's indices.
    /// </summary>
    /// <typeparam name="TResult">The type of elements in the resulting tensor.</typeparam>
    /// <param name="func">The function to apply to each element, which takes the element value and its indices as parameters.</param>
    /// <returns>A new tensor with the function applied to each element.</returns>
    public TensorBase<TResult> Transform<TResult>(Func<T, int[], TResult> func)
    {
        var result = CreateInstance<TResult>(Shape);
        var indices = new int[Rank];
        for (int i = 0; i < Length; i++)
        {
            GetIndices(i, indices);
            // Use GetFlat which handles views correctly (offset + strides)
            result._data[i] = func(GetFlat(i), indices);
        }

        return result;
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional indices.
    /// </summary>
    /// <param name="flatIndex">The flat index to convert.</param>
    /// <param name="indices">An array to store the resulting indices.</param>
    protected void GetIndices(int flatIndex, int[] indices)
    {
        int remainder = flatIndex;
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = remainder % Shape[i];
            remainder /= Shape[i];
        }
    }

    /// <summary>
    /// Gets a read-only span over the internal tensor data.
    /// </summary>
    /// <returns>A read-only span view of the tensor data (row-major order).</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-003 - Zero-Copy Operations</b></para>
    /// <para>
    /// This method provides direct access to the underlying storage without copying.
    /// The tensor is stored in row-major order (last dimension varies fastest).
    /// </para>
    /// <para><b>For Beginners:</b> A span is a view over memory that doesn't copy the data.
    /// This is much faster than copying the entire tensor into a new array, especially for large tensors.
    /// Use this when you need to pass tensor data to GPU or other operations that can work with spans.</para>
    /// </remarks>
    public ReadOnlySpan<T> AsSpan()
    {
        if (IsContiguous && _storageOffset == 0 && _data.Length == Length)
            return _data.AsSpan();
        if (IsContiguous)
            return _data.AsSpan().Slice(_storageOffset, Length);
        // Non-contiguous: caller must call Contiguous() first
        throw new InvalidOperationException(
            "Cannot get a contiguous span from a non-contiguous tensor view. Call Contiguous() first.");
    }

    /// <summary>
    /// Gets a writable span over the internal tensor data.
    /// </summary>
    /// <returns>A writable span view of the tensor data (row-major order).</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-003 - Zero-Copy Operations</b></para>
    /// <para>
    /// Internal use only. Provides direct write access to underlying storage.
    /// Used by GpuEngine to write results directly without intermediate copying.
    /// </para>
    /// </remarks>
    internal Span<T> AsWritableSpan()
    {
        if (IsContiguous && _storageOffset == 0 && _data.Length == Length)
            return _data.AsWritableSpan();
        if (IsContiguous)
            return _data.AsWritableSpan().Slice(_storageOffset, Length);
        throw new InvalidOperationException(
            "Cannot get a contiguous writable span from a non-contiguous tensor view. Call Contiguous() first.");
    }

    /// <summary>
    /// Gets a reference to the underlying array without copying. This is safe because
    /// VectorBase always allocates _memory from a T[].
    /// </summary>
    internal T[] GetDataArray()
    {
        if (!IsContiguous || _storageOffset != 0 || _data.Length != Length)
        {
            // For views: return a fresh contiguous array of just this view's data
            return ToArray();
        }
        return _data.GetDataArray();
    }

    /// <summary>
    /// Gets the value at a flat (linear) index in the underlying data.
    /// </summary>
    /// <param name="flatIndex">The flat index (0 to Length-1).</param>
    /// <returns>The value at the specified flat index.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This allows accessing tensor elements using a single
    /// index that treats the tensor as a 1D array. The flat index corresponds to
    /// row-major ordering where the last dimension varies fastest.</para>
    /// </remarks>
    public T GetFlat(int flatIndex)
    {
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        if (IsContiguous && _storageOffset == 0)
        {
            return _data[flatIndex];
        }
        // For views: convert flat index to multi-dim indices, then use strides
        return _data[FlatIndexToStorageIndex(flatIndex)];
    }

    /// <summary>
    /// Sets the value at a flat (linear) index in the underlying data.
    /// </summary>
    /// <param name="flatIndex">The flat index (0 to Length-1).</param>
    /// <param name="value">The value to set.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This allows setting tensor elements using a single
    /// index that treats the tensor as a 1D array. The flat index corresponds to
    /// row-major ordering where the last dimension varies fastest.</para>
    /// </remarks>
    public void SetFlat(int flatIndex, T value)
    {
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        if (IsContiguous && _storageOffset == 0)
        {
            _data[flatIndex] = value;
        }
        else
        {
            _data[FlatIndexToStorageIndex(flatIndex)] = value;
        }
    }

    /// <summary>
    /// Returns a string representation of the tensor.
    /// </summary>
    /// <returns>A string representation of the tensor.</returns>
    public override string ToString()
    {
        return $"Tensor<{typeof(T).Name}> with shape [{string.Join(", ", Shape)}]";
    }
}
