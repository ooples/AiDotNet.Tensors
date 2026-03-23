using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a base class for multi-dimensional arrays of numeric values used in machine learning and AI computations.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double, int).</typeparam>
public abstract class TensorBase<T>
{
    // ================================================================
    // Core storage and metadata
    // Non-readonly to support Copy-on-Write materialization.
    // Thread-safe: EnsureWritable() uses lock(_cowLock) to prevent
    // double-materialization. Read paths never modify these fields.
    // ================================================================

    /// <summary>
    /// Shared storage for tensor data. Multiple views/clones can reference the same storage.
    /// Swapped during COW materialization.
    /// </summary>
    internal TensorStorage<T> _storage;

    /// <summary>
    /// Direct reference to underlying Vector for backward compatibility with existing engine code.
    /// Swapped during COW materialization.
    /// </summary>
    protected Vector<T> _data;

    /// <summary>
    /// Internal shape array. Direct access for same-assembly code (CpuEngine, etc.) — zero overhead.
    /// External consumers use the Shape property which returns an immutable TensorShape wrapper.
    /// Never changes — shape is the tensor's identity.
    /// </summary>
    internal readonly int[] _shape;

    /// <summary>
    /// Pre-computed strides for each dimension, following PyTorch's stride convention.
    /// For row-major order: strides[i] = product of shape[i+1..end].
    /// For transposed views: strides are permuted without copying data.
    /// Reset to row-major during COW materialization.
    /// </summary>
    internal int[] _strides;

    /// <summary>
    /// Offset into the underlying storage where this tensor's data begins.
    /// Zero for non-view tensors. Non-zero for sliced views.
    /// Reset to 0 during COW materialization.
    /// </summary>
    internal int _storageOffset;

    /// <summary>
    /// Whether this tensor's data is contiguous in memory (row-major with no gaps).
    /// When true, raw span/array access is safe. When false, Contiguous() must be called
    /// before passing to BLAS/SIMD operations.
    /// Set to true during COW materialization.
    /// </summary>
    public bool IsContiguous { get; private set; }

    /// <summary>
    /// Whether this tensor is a view into another tensor's storage.
    /// Views share memory — mutations through one view are visible in others.
    /// Views do NOT trigger COW on write (shared mutation is intentional for views).
    /// </summary>
    public bool IsView { get; }

    // ================================================================
    // Copy-on-Write (COW) support
    // ================================================================

    /// <summary>
    /// True if this tensor shares storage via Clone() and needs to materialize
    /// a private copy before the first write. Views (IsView=true) never set this.
    /// </summary>
    private volatile bool _cowShared;

    /// <summary>
    /// Lock object for thread-safe COW materialization.
    /// Allocated lazily (only when COW is actually used).
    /// </summary>
    private object? _cowLock;

    /// <summary>
    /// Gets the COW lock, creating it on first access via double-check locking.
    /// </summary>
    private object CowLock
    {
        get
        {
            if (_cowLock != null) return _cowLock;
            Interlocked.CompareExchange(ref _cowLock, new object(), null);
            return _cowLock;
        }
    }

    /// <summary>
    /// Ensures this tensor owns its storage exclusively before mutation.
    /// If storage is shared via COW Clone(), materializes a private copy.
    /// No-op for views (shared mutation is intentional) and non-COW tensors.
    /// Thread-safe: uses lock to prevent double-materialization.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void EnsureWritable()
    {
        if (!_cowShared) return; // fast path — no volatile read needed; worst case is one extra check

        // Slow path: materialize
        lock (CowLock)
        {
            if (!_cowShared) return; // double-check after lock

            if (_storage.RefCount <= 1)
            {
                // The other side already materialized — we own storage exclusively now
                _cowShared = false;
                return;
            }

            // Materialize private copy
            var newData = new Vector<T>(Length);
            if (Length > 0)
            {
                if (IsContiguous)
                {
                    _data.AsSpan().Slice(_storageOffset, Length).CopyTo(newData.AsWritableSpan());
                }
                else
                {
                    var srcData = _data.AsSpan();
                    var dstData = newData.AsWritableSpan();
                    var rmStrides = RowMajorStrides;
                    for (int i = 0; i < Length; i++)
                    {
                        int srcIdx = _storageOffset;
                        int remaining = i;
                        for (int d = 0; d < _shape.Length; d++)
                        {
                            int dimIndex = remaining / rmStrides[d];
                            remaining -= dimIndex * rmStrides[d];
                            srcIdx += dimIndex * _strides[d];
                        }
                        dstData[i] = srcData[srcIdx];
                    }
                }
            }

            var oldStorage = _storage;
            _data = newData;
            _storage = new TensorStorage<T>(newData);
            _strides = ComputeRowMajorStrides(_shape);
            _storageOffset = 0;
            IsContiguous = true;
            _rowMajorStridesCache = null; // invalidate cache
            _cowShared = false;

            // Release our reference to the shared storage
            oldStorage.Release();
        }
    }

    /// <summary>
    /// Marks this tensor as COW-shared. Called by Clone() on both the original and the clone.
    /// </summary>
    internal void MarkCowShared()
    {
        _cowShared = true;
    }

    /// <summary>
    /// Whether this tensor is in a COW-shared state (needs materialization before write).
    /// </summary>
    internal bool IsCowShared => _cowShared;

    // ================================================================
    // Cached derived values
    // ================================================================

    /// <summary>
    /// Pre-computed row-major strides for logical flat index decomposition.
    /// Cached to avoid recomputing in hot paths (FlatIndexToStorageIndex, ToArray, etc.).
    /// </summary>
    private int[]? _rowMajorStridesCache;
    internal int[] RowMajorStrides => _rowMajorStridesCache ??= ComputeRowMajorStrides(_shape);

    /// <summary>
    /// Provides numeric operations for the tensor's element type.
    /// </summary>
    protected static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    // ================================================================
    // Public properties
    // ================================================================

    /// <summary>
    /// Gets the shape (dimensions) of the tensor as an immutable wrapper.
    /// Use Shape[i] for element access, Shape.Span for zero-copy iteration.
    /// </summary>
    public TensorShape Shape { get; }

    /// <summary>
    /// Gets the total number of logical elements in this tensor (product of all shape dimensions).
    /// For views, this is the view's element count, not the underlying storage size.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets the rank (number of dimensions) of the tensor.
    /// </summary>
    public int Rank => _shape.Length;

    /// <summary>
    /// Gets the pre-computed strides for each dimension as a read-only span.
    /// </summary>
    public ReadOnlySpan<int> Strides => _strides;

    // ================================================================
    // Internal data access (same-assembly only)
    // ================================================================

    /// <summary>
    /// Gets the underlying data as a Memory&lt;T&gt; for GPU transfer and pinning.
    /// Throws for non-contiguous views — call Contiguous() first.
    /// Triggers COW materialization if needed (write access assumed).
    /// </summary>
    internal Memory<T> Memory
    {
        get
        {
            EnsureWritable();
            if (!IsContiguous)
                throw new InvalidOperationException(
                    "Cannot get contiguous Memory from a non-contiguous tensor view. Call Contiguous() first.");
            if (_storageOffset == 0 && _storage.Length == Length)
                return _storage.AsMemory();
            return _storage.AsMemory().Slice(_storageOffset, Length);
        }
    }

    /// <summary>
    /// Shorthand alias for Memory — used by engine code.
    /// </summary>
    internal Memory<T> Data => Memory;

    // ================================================================
    // Constructors
    // ================================================================

    /// <summary>
    /// Initializes a new tensor with the specified shape (all elements zero-initialized).
    /// </summary>
    protected TensorBase(params int[] shape)
    {
        ValidateShape(shape);
        _shape = shape;
        Shape = new TensorShape(shape);
        _strides = ComputeRowMajorStrides(shape);
        _storageOffset = 0;
        IsContiguous = true;
        IsView = false;
        int totalSize = ComputeProduct(shape);
        Length = totalSize;
        _data = new Vector<T>(totalSize);
        _storage = new TensorStorage<T>(_data);
    }

    /// <summary>
    /// Initializes a new tensor with the specified data and shape.
    /// </summary>
    protected TensorBase(IEnumerable<T> data, params int[] shape)
    {
        ValidateShape(shape);
        _shape = shape;
        Shape = new TensorShape(shape);
        _strides = ComputeRowMajorStrides(shape);
        _storageOffset = 0;
        IsContiguous = true;
        IsView = false;
        if (data is T[] array)
            _data = Vector<T>.WrapMemory(array);
        else
            _data = new Vector<T>(data);
        int expectedSize = ComputeProduct(shape);
        Length = expectedSize;
        if (_data.Length != expectedSize)
            throw new ArgumentException("The number of values does not match the specified shape.");
        _storage = new TensorStorage<T>(_data);
    }

    /// <summary>
    /// Initializes a new tensor with an existing Vector (zero-copy).
    /// </summary>
    protected TensorBase(Vector<T> data, int[] shape)
    {
        ValidateShape(shape);
        _shape = shape;
        Shape = new TensorShape(shape);
        _strides = ComputeRowMajorStrides(shape);
        _storageOffset = 0;
        IsContiguous = true;
        IsView = false;
        _data = data;
        int expectedSize = ComputeProduct(shape);
        Length = expectedSize;
        if (_data.Length != expectedSize)
            throw new ArgumentException("The number of values does not match the specified shape.");
        _storage = new TensorStorage<T>(_data);
    }

    /// <summary>
    /// Internal constructor for creating views with custom strides and offset.
    /// No data is copied — the view shares the same underlying storage via reference counting.
    /// </summary>
    protected TensorBase(Vector<T> data, int[] shape, int[] strides, int storageOffset, bool isView)
        : this(data, shape, strides, storageOffset, isView, storage: null)
    {
    }

    /// <summary>
    /// Internal constructor for creating views that share the parent's TensorStorage.
    /// When storage is provided, AddRef is called on the shared storage (no new allocation).
    /// When storage is null (backward compat), a new TensorStorage is created.
    /// </summary>
    internal TensorBase(Vector<T> data, int[] shape, int[] strides, int storageOffset, bool isView, TensorStorage<T>? storage)
    {
        if (strides.Length != shape.Length)
            throw new ArgumentException($"Strides length ({strides.Length}) must match shape length ({shape.Length}).");
        if (storageOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(storageOffset), "Storage offset must be non-negative.");

        int totalElements = ComputeProduct(shape);

        // Validate bounds — skip for zero-size tensors (no elements to address)
        if (totalElements > 0)
        {
            int maxIndex = storageOffset;
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] > 1)
                    maxIndex += (shape[i] - 1) * Math.Abs(strides[i]);
            }
            if (maxIndex >= data.Length)
                throw new ArgumentException(
                    $"View exceeds storage bounds: max index {maxIndex} >= storage length {data.Length}.");
        }

        _shape = shape;
        Shape = new TensorShape(shape);
        _strides = strides;
        _storageOffset = storageOffset;
        IsView = isView;
        _data = data;

        if (storage != null)
        {
            // Share parent's storage — single AddRef on the shared instance
            _storage = storage;
            _storage.AddRef();
        }
        else
        {
            // Backward compat: create new storage (non-view or legacy path)
            _storage = new TensorStorage<T>(_data);
            if (isView) _storage.AddRef();
        }

        Length = totalElements;
        IsContiguous = CheckContiguous(shape, strides);
    }

    // ================================================================
    // Indexers
    // ================================================================

    /// <summary>
    /// Gets or sets the value at the specified multi-dimensional indices.
    /// </summary>
    public virtual T this[params int[] indices]
    {
        get
        {
            ValidateIndices(indices);
            return _data[GetFlatIndex(indices)];
        }
        set
        {
            EnsureWritable();
            ValidateIndices(indices);
            _data[GetFlatIndex(indices)] = value;
        }
    }

    /// <summary>
    /// Gets or sets the value at the specified flat (logical) index.
    /// </summary>
    public virtual T this[int flatIndex]
    {
        get => GetFlat(flatIndex);
        set => SetFlat(flatIndex, value);
    }

    // ================================================================
    // Typed indexer overloads — eliminate params int[] heap allocation
    // Each call to this[i,j] with the params overload allocates a new int[].
    // These overloads compile to direct arithmetic — zero allocation, JIT-inlined.
    // ================================================================

    /// <summary>
    /// Gets or sets the value at the specified 2D indices. Zero-allocation.
    /// </summary>
    public T this[int i0, int i1]
    {
        get => _data[_storageOffset + i0 * _strides[0] + i1 * _strides[1]];
        set
        {
            EnsureWritable();
            _data[_storageOffset + i0 * _strides[0] + i1 * _strides[1]] = value;
        }
    }

    /// <summary>
    /// Gets or sets the value at the specified 3D indices. Zero-allocation.
    /// </summary>
    public T this[int i0, int i1, int i2]
    {
        get => _data[_storageOffset + i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2]];
        set
        {
            EnsureWritable();
            _data[_storageOffset + i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2]] = value;
        }
    }

    /// <summary>
    /// Gets or sets the value at the specified 4D indices. Zero-allocation.
    /// </summary>
    public T this[int i0, int i1, int i2, int i3]
    {
        get => _data[_storageOffset + i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2] + i3 * _strides[3]];
        set
        {
            EnsureWritable();
            _data[_storageOffset + i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2] + i3 * _strides[3]] = value;
        }
    }

    // ================================================================
    // Data access methods
    // ================================================================

    /// <summary>
    /// Creates a new array containing a copy of the tensor's elements in row-major order.
    /// </summary>
    public virtual T[] ToArray()
    {
        if (Length == 0) return Array.Empty<T>();
        if (IsContiguous && _storageOffset == 0 && _storage.Length == Length)
            return _data.ToArray();
        var result = new T[Length];
        if (IsContiguous)
        {
            _data.AsSpan().Slice(_storageOffset, Length).CopyTo(result);
        }
        else
        {
            var srcData = _data.AsSpan();
            for (int i = 0; i < Length; i++)
                result[i] = srcData[FlatIndexToStorageIndex(i)];
        }
        return result;
    }

    /// <summary>
    /// Copies data from a source array into this tensor's storage.
    /// </summary>
    public virtual void CopyFromArray(T[] source)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (source.Length != Length)
            throw new ArgumentException($"Source array length ({source.Length}) must match tensor length ({Length}).");
        if (Length == 0) return;
        EnsureWritable();
        if (IsContiguous && _storageOffset == 0 && _storage.Length == Length)
        {
            source.AsSpan().CopyTo(_data.AsWritableSpan());
        }
        else if (IsContiguous)
        {
            source.AsSpan().CopyTo(_data.AsWritableSpan().Slice(_storageOffset, Length));
        }
        else
        {
            var dstData = _data.AsWritableSpan();
            for (int i = 0; i < Length; i++)
                dstData[FlatIndexToStorageIndex(i)] = source[i];
        }
    }

    /// <summary>
    /// Gets the value at a flat (logical) index. Handles views correctly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T GetFlat(int flatIndex)
    {
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        if (IsContiguous)
            return _data[flatIndex + _storageOffset];
        return _data[FlatIndexToStorageIndex(flatIndex)];
    }

    /// <summary>
    /// Sets the value at a flat (logical) index. Handles views correctly.
    /// Triggers COW materialization if storage is shared.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetFlat(int flatIndex, T value)
    {
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        EnsureWritable();
        if (IsContiguous)
            _data[flatIndex + _storageOffset] = value;
        else
            _data[FlatIndexToStorageIndex(flatIndex)] = value;
    }

    /// <summary>
    /// Gets a read-only span over the tensor data. Throws for non-contiguous views.
    /// Does NOT trigger COW (read-only access).
    /// </summary>
    public ReadOnlySpan<T> AsSpan()
    {
        if (Length == 0) return ReadOnlySpan<T>.Empty;
        if (!IsContiguous)
            throw new InvalidOperationException(
                "Cannot get a contiguous span from a non-contiguous tensor view. Call Contiguous() first.");
        if (_storageOffset == 0 && _storage.Length == Length)
            return _data.AsSpan();
        return _data.AsSpan().Slice(_storageOffset, Length);
    }

    /// <summary>
    /// Gets a writable span over the tensor data. Throws for non-contiguous views.
    /// Triggers COW materialization if storage is shared.
    /// </summary>
    internal Span<T> AsWritableSpan()
    {
        if (Length == 0) return Span<T>.Empty;
        EnsureWritable();
        if (!IsContiguous)
            throw new InvalidOperationException(
                "Cannot get a contiguous writable span from a non-contiguous tensor view. Call Contiguous() first.");
        if (_storageOffset == 0 && _storage.Length == Length)
            return _data.AsWritableSpan();
        return _data.AsWritableSpan().Slice(_storageOffset, Length);
    }

    /// <summary>
    /// Gets the underlying array. For views, returns a fresh contiguous copy.
    /// </summary>
    internal T[] GetDataArray()
    {
        if (!IsContiguous || _storageOffset != 0 || _storage.Length != Length)
            return ToArray();
        return _storage.GetDataArray();
    }

    // ================================================================
    // Clone and Transform
    // ================================================================

    /// <summary>
    /// Creates a copy of this tensor. Uses Copy-on-Write: no data is copied until
    /// either the original or the clone is written to. This is O(1) for the common
    /// case where clones are read-only (gradient checkpointing, model snapshots).
    /// PyTorch's clone() always copies immediately — COW is strictly better.
    /// </summary>
    public virtual TensorBase<T> Clone()
    {
        if (Length == 0)
            return CreateInstance(_shape);

        // COW path: share storage, defer copy until first write
        _storage.AddRef();
        var clone = CreateInstance(_shape);

        // Point clone at our storage
        clone._data = _data;
        clone._storage.Release(); // release the fresh storage CreateInstance made
        clone._storage = _storage;
        clone._strides = (int[])_strides.Clone();
        clone._storageOffset = _storageOffset;
        clone.IsContiguous = IsContiguous;
        clone._rowMajorStridesCache = null;

        // Mark both sides as COW — whichever writes first materializes
        this.MarkCowShared();
        clone.MarkCowShared();

        return clone;
    }

    protected abstract TensorBase<T> CreateInstance(int[] shape);
    protected abstract TensorBase<T> CreateInstance(T[] data, int[] shape);
    protected abstract TensorBase<TResult> CreateInstance<TResult>(params int[] shape);

    /// <summary>
    /// Applies a function to each element. View-safe.
    /// </summary>
    public TensorBase<TResult> Transform<TResult>(Func<T, TResult> func)
    {
        var result = CreateInstance<TResult>(_shape);
        for (int i = 0; i < Length; i++)
            result._data[i] = func(GetFlat(i));
        return result;
    }

    /// <summary>
    /// Applies a function to each element with indices. View-safe.
    /// </summary>
    public TensorBase<TResult> Transform<TResult>(Func<T, int[], TResult> func)
    {
        var result = CreateInstance<TResult>(_shape);
        var indices = new int[Rank];
        for (int i = 0; i < Length; i++)
        {
            GetIndices(i, indices);
            result._data[i] = func(GetFlat(i), indices);
        }
        return result;
    }

    // ================================================================
    // Index computation
    // ================================================================

    protected void ValidateIndices(int[] indices)
    {
        if (indices.Length != _shape.Length)
            throw new ArgumentException("Number of indices must match the tensor's rank.");
        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= _shape[i])
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {i} is out of range.");
        }
    }

    /// <summary>
    /// Converts multi-dimensional indices to a storage index using strides and offset.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected int GetFlatIndex(int[] indices)
    {
        int flatIndex = _storageOffset;
        for (int i = 0; i < indices.Length; i++)
            flatIndex += indices[i] * _strides[i];
        return flatIndex;
    }

    /// <summary>
    /// Converts a logical flat index (row-major) to a storage index for views.
    /// O(Rank) per call using cached row-major strides.
    /// </summary>
    private int FlatIndexToStorageIndex(int flatIndex)
    {
        int storageIndex = _storageOffset;
        int remaining = flatIndex;
        var rmStrides = RowMajorStrides;
        for (int d = 0; d < _shape.Length; d++)
        {
            int dimIndex = remaining / rmStrides[d];
            remaining -= dimIndex * rmStrides[d];
            storageIndex += dimIndex * _strides[d];
        }
        return storageIndex;
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional indices using shape.
    /// </summary>
    protected void GetIndices(int flatIndex, int[] indices)
    {
        int remainder = flatIndex;
        for (int i = _shape.Length - 1; i >= 0; i--)
        {
            indices[i] = remainder % _shape[i];
            remainder /= _shape[i];
        }
    }

    // ================================================================
    // Static helpers
    // ================================================================

    /// <summary>
    /// Computes row-major strides: strides[i] = product of shape[i+1..end].
    /// Example: shape [3,4,5] → strides [20, 5, 1].
    /// </summary>
    protected static int[] ComputeRowMajorStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        if (shape.Length == 0) return strides;
        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }

    /// <summary>
    /// Checks whether shape+strides represent contiguous row-major layout.
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
    /// Computes the product of all dimensions. Returns 0 for zero-size tensors.
    /// </summary>
    private static int ComputeProduct(int[] shape)
    {
        if (shape.Length == 0) return 1; // Scalar
        int product = 1;
        for (int i = 0; i < shape.Length; i++)
            product *= shape[i];
        return product;
    }

    /// <summary>
    /// Compares two shape arrays for equality without LINQ allocation.
    /// Replaces SequenceEqual which allocates an enumerator per call.
    /// </summary>
    internal static bool ShapeEquals(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    /// <summary>
    /// Validates that shape dimensions are non-negative.
    /// Zero-size dimensions are allowed (empty tensors for empty batches, masks, etc.).
    /// Negative dimensions are rejected.
    /// </summary>
    private static void ValidateShape(int[] shape)
    {
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0)
                throw new ArgumentException($"Shape dimension {i} must be non-negative, got {shape[i]}.");
        }
    }

    public override string ToString()
    {
        return $"Tensor<{typeof(T).Name}> with shape {Shape}";
    }
}
