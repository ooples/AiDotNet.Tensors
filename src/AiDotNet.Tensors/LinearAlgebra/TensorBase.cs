using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a base class for multi-dimensional arrays of numeric values used in machine learning and AI computations.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double, int).</typeparam>
public abstract class TensorBase<T> : IDisposable
{
    private bool _disposed;
    // ================================================================
    // Core storage and metadata
    // ================================================================

    /// <summary>
    /// Shared storage for tensor data. Multiple views can reference the same storage.
    /// </summary>
    internal readonly TensorStorage<T> _storage;

    /// <summary>
    /// Direct reference to underlying Vector for backward compatibility with existing engine code.
    /// All new code should prefer _storage methods.
    /// </summary>
    protected readonly Vector<T> _data;

    /// <summary>
    /// Internal accessor for the backing vector. Used by DirectGpuTensorEngine to check
    /// activation cache without triggering CPU materialization.
    /// </summary>
    internal Vector<T> DataVector => _data;

    /// <summary>
    /// Internal shape array. Direct access for same-assembly code (CpuEngine, etc.) — zero overhead.
    /// External consumers use the Shape property which returns an immutable TensorShape wrapper.
    /// </summary>
    internal readonly int[] _shape;

    /// <summary>
    /// Pre-computed strides for each dimension, following PyTorch's stride convention.
    /// For row-major order: strides[i] = product of shape[i+1..end].
    /// For transposed views: strides are permuted without copying data.
    /// </summary>
    internal readonly int[] _strides;

    /// <summary>
    /// Offset into the underlying storage where this tensor's data begins.
    /// Zero for non-view tensors. Non-zero for sliced views.
    /// </summary>
    internal readonly int _storageOffset;

    /// <summary>
    /// Tracks the device where this tensor's data currently resides.
    /// Set to GPU by DirectGpuTensorEngine when a GPU op defers its download.
    /// Reset to CPU when the data is materialized to the CPU-side array.
    /// </summary>
    internal TensorDevice _device = TensorDevice.CPU;

    /// <summary>
    /// Optional GPU buffer reference for GPU-resident tensors.
    /// When non-null, this tensor's authoritative data is on the GPU — the CPU-side
    /// _data array may be empty/stale until explicitly synchronized.
    /// This is the PyTorch-equivalent of tensor.data_ptr() on a CUDA tensor.
    /// </summary>
    internal Engines.DirectGpu.IGpuBuffer? _gpuBuffer;

    /// <summary>
    /// Backend that owns the GPU buffer. Required for downloading data to CPU.
    /// </summary>
    internal Engines.DirectGpu.IDirectGpuBackend? _gpuBackend;

    /// <summary>
    /// The GPU memory management role for this tensor (weight, activation, gradient, etc.).
    /// Used by the GPU memory planner for allocation and eviction decisions.
    /// </summary>
    internal Engines.Gpu.GpuTensorRole _gpuRole;

    /// <summary>
    /// Whether this tensor owns the GPU buffer and should dispose it when the tensor is disposed.
    /// </summary>
    internal bool _ownsGpuBuffer;

    /// <summary>
    /// Sync point for the last GPU write operation on this tensor.
    /// Used by the GPU graph executor to enforce correct ordering.
    /// </summary>
    internal Engines.Gpu.GpuSyncPoint? _lastWriteSync;

    /// <summary>
    /// Cached deferred materializer callback for re-registration after GPU writes.
    /// Stored so MarkModified can invalidate CPU data and set up fresh download.
    /// </summary>
    internal Action<object>? _gpuMaterializerCallback;
    internal object? _gpuMaterializerKey;

    /// <summary>
    /// Gets the GPU buffer for this tensor.
    /// Throws if the tensor is CPU-resident — call <see cref="IsGpuResident"/> first to check,
    /// or use <see cref="Tensor{T}.Gpu()"/> / <see cref="Tensor{T}.To(DeviceInfo)"/> to move to GPU.
    /// </summary>
    public Engines.DirectGpu.IGpuBuffer Buffer =>
        _device != TensorDevice.CPU && _gpuBuffer is not null
            ? _gpuBuffer
            : throw new InvalidOperationException("Tensor is not GPU-resident. Call .Gpu() or .To(device) first.");

    /// <summary>
    /// Gets the GPU memory management role for this tensor.
    /// </summary>
    public Engines.Gpu.GpuTensorRole Role => _gpuRole;

    /// <summary>
    /// Gets or sets the device where this tensor's data resides.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you whether the tensor's data is on the CPU
    /// (regular computer memory) or on the GPU (graphics card memory). When a GPU operation
    /// produces a result, the tensor is marked as GPU-resident. Reading the data from CPU
    /// code triggers a lazy download, and the device changes back to CPU.</para>
    /// </remarks>
    public TensorDevice Device
    {
        get => _device;
        internal set => _device = value;
    }

    /// <summary>
    /// Returns true if this tensor's data is believed to reside on a GPU device.
    /// After CPU materialization (e.g., via AsSpan/GetDataArray), call Cpu()
    /// to update the device state. The tensor does not auto-detect materialization.
    /// </summary>
    public bool IsGpuResident => _device != TensorDevice.CPU;

    /// <summary>
    /// Gets the full device info including device index for multi-GPU scenarios.
    /// Equivalent to PyTorch's <c>tensor.device</c> which returns e.g. <c>device(type='cuda', index=0)</c>.
    /// </summary>
    public DeviceInfo DeviceInfo => new DeviceInfo(Device, _gpuDeviceIndex);

    /// <summary>
    /// The device index for multi-GPU scenarios (0 = first GPU, 1 = second, etc.).
    /// </summary>
    internal int _gpuDeviceIndex;

    /// <summary>
    /// Monotonically increasing version counter. Incremented by in-place mutation operations.
    /// Used by GradientTape to detect when a recorded tensor has been mutated after recording,
    /// which would produce incorrect gradients during the backward pass.
    /// </summary>
    internal int _version;

    /// <summary>
    /// Gets the current mutation version of this tensor.
    /// </summary>
    public int Version => _version;

    /// <summary>
    /// Increments the version counter. Called by in-place operations to signal mutation.
    /// </summary>
    internal void IncrementVersion() => System.Threading.Interlocked.Increment(ref _version);

    /// <summary>
    /// Gets the sync point for the last GPU write operation on this tensor.
    /// </summary>
    public Engines.Gpu.GpuSyncPoint? LastWriteSync => _lastWriteSync;

    /// <summary>
    /// Whether the GPU data has been modified since the last CPU synchronization.
    /// True when a GPU operation writes to this tensor; false after CPU data is downloaded.
    /// </summary>
    public bool IsDirty { get; internal set; }

    /// <summary>
    /// When non-null, indicates this tensor was created by a uniform Fill operation and
    /// every element has this value. Backward kernels can use the scalar directly instead
    /// of reading N identical elements from memory (saves bandwidth for fused backward paths).
    /// Reset to null by any non-fill mutation.
    /// </summary>
    internal double? UniformFillValue { get; set; }

    /// <summary>
    /// When true, indicates this gradient tensor's buffer can be overwritten in-place by
    /// a backward op because no other backward op will read it (refcount == 1).
    /// Set by ComputeGradients; consumed by backward functions like ReluBackward.
    /// </summary>
    internal bool _canReuseBuffer;

    /// <summary>
    /// Waits for all pending GPU operations on this tensor to complete.
    /// Call this before reading GPU results to ensure correctness.
    /// </summary>
    public void Synchronize()
    {
        if (_lastWriteSync is { IsComplete: false })
            _lastWriteSync.Wait();
    }

    /// <summary>
    /// Marks this tensor as modified by a GPU operation.
    /// Increments the version counter, stores the sync point for the write fence,
    /// and sets IsDirty so the next CPU read triggers a fresh download.
    /// </summary>
    internal void MarkModified(Engines.Gpu.GpuSyncPoint? syncPoint)
    {
        IncrementVersion();
        IsDirty = true;
        var previous = _lastWriteSync;
        _lastWriteSync = syncPoint;
        if (previous is not null && !ReferenceEquals(previous, syncPoint))
            previous.Dispose();

        // Re-register deferred materializer so next CPU read downloads fresh GPU data
        if (_gpuMaterializerCallback is not null && _gpuMaterializerKey is not null)
            Helpers.DeferredArrayMaterializer.Register(_gpuMaterializerKey, _gpuMaterializerCallback);
    }

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
    /// Whether this tensor uses sparse storage (COO, CSR, or CSC format).
    /// When true, the backing data contains only non-zero values — its length
    /// is less than the logical element count (product of shape dimensions).
    /// Dense operations must check this flag and dispatch to sparse kernels.
    /// </summary>
    /// <remarks>
    /// <para>This follows the PyTorch model where <c>tensor.is_sparse</c> indicates
    /// the storage layout. Sparse tensors inherit the full Tensor interface but
    /// store data efficiently for matrices with many zero elements.</para>
    /// </remarks>
    public bool IsSparse { get; }

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
    /// </summary>
    internal Memory<T> Memory
    {
        get
        {
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
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        ValidateShape(shape);
        _shape = (int[])shape.Clone();
        Shape = TensorShape.WrapUnsafe(_shape);
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
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        ValidateShape(shape);
        _shape = (int[])shape.Clone();
        Shape = TensorShape.WrapUnsafe(_shape);
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
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        ValidateShape(shape);
        _shape = (int[])shape.Clone();
        Shape = TensorShape.WrapUnsafe(_shape);
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
    /// Creates a GPU-resident tensor with zero CPU memory allocation.
    /// The backing array is allocated lazily when CPU code first accesses the data.
    /// Used by DirectGpuTensorEngine for GPU-only intermediate results.
    /// </summary>
    internal TensorBase(int[] shape, TensorDevice gpuDevice)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        ValidateShape(shape);
        _shape = (int[])shape.Clone();
        Shape = TensorShape.WrapUnsafe(_shape);
        _strides = ComputeRowMajorStrides(shape);
        _storageOffset = 0;
        IsContiguous = true;
        IsView = false;
        int expectedSize = ComputeProduct(shape);
        Length = expectedSize;
        _data = Vector<T>.CreateGpuResident(expectedSize);
        _storage = new TensorStorage<T>(_data);
        _device = gpuDevice;
    }

    /// <summary>
    /// Constructor for sparse tensors where the backing data contains only non-zero values.
    /// The logical shape may be larger than the data length (e.g., [1000, 1000] shape
    /// but only 500 non-zero values stored).
    /// </summary>
    /// <param name="values">The non-zero values vector.</param>
    /// <param name="logicalShape">The full logical shape (e.g., [rows, columns]).</param>
    /// <remarks>
    /// <para>This constructor skips the data.Length == product(shape) validation that
    /// the dense constructors enforce, since sparse tensors intentionally store fewer
    /// elements than the logical element count.</para>
    /// </remarks>
    internal TensorBase(Vector<T> values, int[] logicalShape, bool isSparse)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (logicalShape is null) throw new ArgumentNullException(nameof(logicalShape));
        if (!isSparse) throw new ArgumentException("Use the dense constructor for non-sparse tensors.", nameof(isSparse));
        ValidateShape(logicalShape);

        _shape = (int[])logicalShape.Clone();
        Shape = TensorShape.WrapUnsafe(_shape);
        // Strides describe the logical dense layout for shape queries.
        // Sparse tensors must NOT use strides for data access — use sparse indices instead.
        _strides = ComputeRowMajorStrides(logicalShape);
        _storageOffset = 0;
        IsContiguous = false; // Sparse tensors are not contiguous in the dense sense
        IsView = false;
        IsSparse = true;
        // Length is the logical element count (product of shape), NOT _data.Length.
        // Dense APIs (indexers, GetFlat, AsSpan, etc.) check IsSparse and throw
        // if called on sparse tensors — use SparseTensor-specific APIs instead.
        Length = ComputeProduct(logicalShape);
        _data = values;
        _storage = new TensorStorage<T>(_data);
    }

    /// <summary>
    /// Internal constructor for creating views with custom strides and offset.
    /// No data is copied — the view shares the same underlying storage via reference counting.
    /// </summary>
    protected TensorBase(Vector<T> data, int[] shape, int[] strides, int storageOffset, bool isView)
        : this(data, shape, strides, storageOffset, isView, null) { }

    /// <summary>
    /// View constructor that shares an existing TensorStorage (PyTorch model).
    /// When parentStorage is provided, it is shared via AddRef instead of creating a new one.
    /// </summary>
    internal TensorBase(Vector<T> data, int[] shape, int[] strides, int storageOffset, bool isView, TensorStorage<T>? parentStorage)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        if (strides == null) throw new ArgumentNullException(nameof(strides));
        if (strides.Length != shape.Length)
            throw new ArgumentException($"Strides length ({strides.Length}) must match shape length ({shape.Length}).");
        if (storageOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(storageOffset), "Storage offset must be non-negative.");

        // Defensive copy of metadata
        var shapeCopy = new int[shape.Length];
        var stridesCopy = new int[strides.Length];
        Array.Copy(shape, shapeCopy, shape.Length);
        Array.Copy(strides, stridesCopy, strides.Length);

        int totalElements = ComputeProduct(shapeCopy);

        // Validate bounds with correct negative stride handling
        if (totalElements > 0)
        {
            int minIndex = storageOffset;
            int maxIndex = storageOffset;
            for (int i = 0; i < shapeCopy.Length; i++)
            {
                if (shapeCopy[i] > 1)
                {
                    int extent = (shapeCopy[i] - 1) * stridesCopy[i];
                    if (extent >= 0)
                        maxIndex += extent;
                    else
                        minIndex += extent;
                }
            }
            if (minIndex < 0 || maxIndex >= data.Length)
                throw new ArgumentException(
                    $"View exceeds storage bounds: index range [{minIndex}, {maxIndex}] outside storage [0, {data.Length - 1}].");
        }

        _shape = shapeCopy;
        Shape = TensorShape.WrapUnsafe(shapeCopy);
        _strides = stridesCopy;
        _storageOffset = storageOffset;
        IsView = isView;
        _data = data;

        // Share parent's storage if provided, otherwise create new
        if (parentStorage != null)
        {
            _storage = parentStorage;
            _storage.AddRef();
        }
        else
        {
            _storage = new TensorStorage<T>(_data);
        }

        Length = totalElements;
        IsContiguous = CheckContiguous(shapeCopy, stridesCopy);
    }

    // ================================================================
    // Sparse safety
    // ================================================================

    /// <summary>
    /// Throws if this tensor is sparse. Call at the top of any method that assumes
    /// dense storage layout (_data.Length == Length, stride-based indexing is valid).
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    protected void ThrowIfSparse(
        [System.Runtime.CompilerServices.CallerMemberName] string caller = "")
    {
        if (IsSparse)
            throw new InvalidOperationException(
                $"{caller} is not supported on sparse tensors. " +
                "Use SparseTensor-specific APIs or call ToDense() first.");
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
            ThrowIfSparse();
            ValidateIndices(indices);
            return _data[GetFlatIndex(indices)];
        }
        set
        {
            ThrowIfSparse();
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
    // Data access methods
    // ================================================================

    /// <summary>
    /// Creates a new array containing a copy of the tensor's elements in row-major order.
    /// </summary>
    public virtual T[] ToArray()
    {
        ThrowIfSparse();
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
        ThrowIfSparse();
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (source.Length != Length)
            throw new ArgumentException($"Source array length ({source.Length}) must match tensor length ({Length}).");
        if (Length == 0) return;
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
    public T GetFlat(int flatIndex)
    {
        ThrowIfSparse();
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        if (IsContiguous)
            return _data[flatIndex + _storageOffset];
        return _data[FlatIndexToStorageIndex(flatIndex)];
    }

    /// <summary>
    /// Sets the value at a flat (logical) index. Handles views correctly.
    /// </summary>
    public void SetFlat(int flatIndex, T value)
    {
        ThrowIfSparse();
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        if (IsContiguous)
            _data[flatIndex + _storageOffset] = value;
        else
            _data[FlatIndexToStorageIndex(flatIndex)] = value;
    }

    /// <summary>
    /// Gets a read-only span over the tensor data. Throws for non-contiguous views.
    /// </summary>
    public ReadOnlySpan<T> AsSpan()
    {
        ThrowIfSparse();
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
    /// </summary>
    internal Span<T> AsWritableSpan()
    {
        if (Length == 0) return Span<T>.Empty;
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
    /// Creates a deep copy of this tensor (always contiguous, never a view).
    /// </summary>
    public virtual TensorBase<T> Clone()
    {
        ThrowIfSparse();
        var result = CreateInstance(_shape);
        if (Length == 0) return result;
        if (IsContiguous && _storageOffset == 0 && _storage.Length == Length)
        {
            _numOps.Copy(_data.AsSpan(), result._data.AsWritableSpan());
        }
        else
        {
            var srcArray = ToArray();
            srcArray.AsSpan().CopyTo(result._data.AsWritableSpan());
        }
        return result;
    }

    protected abstract TensorBase<T> CreateInstance(int[] shape);
    protected abstract TensorBase<T> CreateInstance(T[] data, int[] shape);
    protected abstract TensorBase<TResult> CreateInstance<TResult>(params int[] shape);

    /// <summary>
    /// Applies a function to each element. View-safe.
    /// </summary>
    public TensorBase<TResult> Transform<TResult>(Func<T, TResult> func)
    {
        ThrowIfSparse();
        var result = CreateInstance<TResult>(_shape);
        if (IsContiguous && _storageOffset == 0 && _storage.Length == Length)
        {
            var src = _data.AsSpan();
            for (int i = 0; i < Length; i++)
                result._data[i] = func(src[i]);
        }
        else
        {
            for (int i = 0; i < Length; i++)
                result._data[i] = func(GetFlat(i));
        }
        return result;
    }

    /// <summary>
    /// Applies a function to each element with indices. View-safe.
    /// </summary>
    public TensorBase<TResult> Transform<TResult>(Func<T, int[], TResult> func)
    {
        ThrowIfSparse();
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
        long product = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            product *= shape[i];
            if (product > int.MaxValue)
                throw new ArgumentException($"Shape product overflow: shape [{string.Join(", ", shape)}] exceeds int.MaxValue.");
        }
        return (int)product;
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

    // ================================================================
    // Stride-aware helpers for CpuEngine
    // ================================================================

    /// <summary>
    /// Whether this tensor is a simple 2D transpose (swap of last two dims) of contiguous storage.
    /// When true, GEMM can use transA/transB flags instead of materializing.
    /// </summary>
    internal bool IsSimpleTranspose
    {
        get
        {
            if (IsContiguous || _shape.Length < 2) return false;
            // Check if the strides are a permutation of row-major strides
            // For a 2D transpose: shape=[M,N], strides=[1, M] instead of [N, 1]
            // For an ND transpose of last two dims: strides[rank-2] == 1 && strides[rank-1] == shape[rank-2]
            int rank = _shape.Length;
            var rmStrides = RowMajorStrides; // use cached, avoid allocation

            // Check if only the last two dimensions are swapped
            for (int i = 0; i < rank - 2; i++)
            {
                if (_strides[i] != rmStrides[i]) return false;
            }
            // Last two dims swapped: strides[rank-2] should be 1, strides[rank-1] should be shape[rank-2]
            return _strides[rank - 1] == _shape[rank - 2] && _strides[rank - 2] == 1;
        }
    }

    /// <summary>
    /// Gets the leading dimension (row stride) for BLAS-style operations.
    /// For row-major: lda = shape[rank-1] (number of columns).
    /// For transposed 2D: lda = shape[rank-2] (original number of rows).
    /// Only valid for contiguous or simple-transposed tensors.
    /// </summary>
    internal int LeadingDimension
    {
        get
        {
            if (_shape.Length < 2) return 1;
            // Leading dimension = the stride of the second-to-last dimension
            // For row-major [M,N] with strides [N,1]: ld = N = strides[rank-2] (but that's N, not stride)
            // Actually for BLAS: lda = the distance between consecutive rows in memory
            // For row-major: lda = N (columns)
            // For column-major (transposed): lda = M (rows of original)
            if (IsContiguous)
                return _shape[_shape.Length - 1];
            if (!IsSimpleTranspose)
                throw new InvalidOperationException("LeadingDimension is only valid for contiguous or simple-transposed tensors.");
            return Math.Max(_strides[_shape.Length - 2], _strides[_shape.Length - 1]);
        }
    }

    /// <summary>
    /// Returns true if the tensor's data can be accessed as a contiguous span
    /// (contiguous layout, possibly with a storage offset).
    /// Out parameter provides the span when true.
    /// </summary>
    internal bool TryGetContiguousSpan(out ReadOnlySpan<T> span)
    {
        if (!IsContiguous)
        {
            span = default;
            return false;
        }
        if (_storageOffset == 0 && _storage.Length == Length)
            span = _data.AsSpan();
        else
            span = _data.AsSpan().Slice(_storageOffset, Length);
        return true;
    }

    /// <summary>
    /// Converts a logical flat index (row-major) to a storage index using strides.
    /// For contiguous tensors, this is offset + flatIndex.
    /// For strided views, decomposes into multi-dim indices then applies strides.
    /// </summary>
    internal int LogicalToStorageIndex(int flatIndex)
    {
        if (IsContiguous)
            return _storageOffset + flatIndex;
        return FlatIndexToStorageIndex(flatIndex);
    }

    /// <summary>
    /// Iterates over all logical elements applying an action with the storage index.
    /// Provides efficient stride-aware iteration for any view layout.
    /// For contiguous tensors, the callback receives sequential indices.
    /// For strided views, indices follow the stride pattern.
    /// </summary>
    internal void ForEachStorageIndex(Action<int> action)
    {
        if (IsContiguous)
        {
            int start = _storageOffset;
            for (int i = 0; i < Length; i++)
                action(start + i);
        }
        else
        {
            for (int i = 0; i < Length; i++)
                action(FlatIndexToStorageIndex(i));
        }
    }

    /// <summary>
    /// Returns the tensor's elements as a flat row-major T[] array.
    /// For contiguous tensors with zero offset: returns the actual backing array (zero-copy).
    /// For contiguous with offset: returns a slice copy.
    /// For non-contiguous views: creates a new array via stride iteration.
    /// This is the stride-aware replacement for GetDataArray() in engine code.
    /// Unlike Contiguous(), this returns T[] not Tensor&lt;T&gt;, avoiding tensor allocation.
    /// </summary>
    internal T[] GetFlattenedData()
    {
        if (IsContiguous && _storageOffset == 0 && _storage.Length == Length)
            return _storage.GetDataArray();

        // Must create a flat copy — either offset or non-contiguous
        var result = new T[Length];
        if (IsContiguous)
        {
            // Contiguous with offset — slice copy
            Array.Copy(_storage.GetDataArray(), _storageOffset, result, 0, Length);
        }
        else
        {
            // Non-contiguous — stride iteration
            var src = _storage.GetDataArray();
            var indices = new int[Length];
            FillStorageIndices(indices);
            for (int i = 0; i < Length; i++)
                result[i] = src[indices[i]];
        }
        return result;
    }

    /// <summary>
    /// Gets the raw underlying data span (full storage, no offset applied).
    /// Use with LogicalToStorageIndex for stride-aware element access.
    /// This never throws — returns the full backing storage.
    /// </summary>
    internal ReadOnlySpan<T> RawStorageSpan => _data.AsSpan();

    /// <summary>
    /// Gets the raw underlying writable data span (full storage, no offset applied).
    /// Use with LogicalToStorageIndex for stride-aware element writes.
    /// </summary>
    internal Span<T> RawWritableStorageSpan => _data.AsWritableSpan();

    /// <summary>
    /// Fills a pre-allocated array with storage indices for every logical element.
    /// For sequential iteration this is O(n) total — amortized O(1) per element via
    /// odometer-style coordinate increment (no division/modulo per element).
    /// </summary>
    internal void FillStorageIndices(int[] indices)
    {
        int rank = _shape.Length;
        if (rank == 0) { if (indices.Length > 0) indices[0] = _storageOffset; return; }

        var coords = new int[rank];
        int storageIdx = _storageOffset;

        for (int i = 0; i < Length; i++)
        {
            indices[i] = storageIdx;

            // Odometer increment: advance last dimension, carry into earlier dimensions
            for (int d = rank - 1; d >= 0; d--)
            {
                coords[d]++;
                storageIdx += _strides[d];
                if (coords[d] < _shape[d])
                    break;
                // Carry: reset this dimension, subtract its full contribution
                storageIdx -= coords[d] * _strides[d];
                coords[d] = 0;
            }
        }
    }

    /// <summary>
    /// Computes the storage index for a reduction along a specific axis.
    /// Returns (outerSize, axisSize, innerSize) for the reduction loop structure.
    /// outerSize = product of dims before axis, axisSize = shape[axis], innerSize = product of dims after axis.
    /// </summary>
    internal (int outerSize, int axisSize, int innerSize) GetReductionDims(int axis)
    {
        int outerSize = 1, innerSize = 1;
        for (int d = 0; d < axis; d++) outerSize *= _shape[d];
        for (int d = axis + 1; d < _shape.Length; d++) innerSize *= _shape[d];
        return (outerSize, _shape[axis], innerSize);
    }

    /// <summary>
    /// Computes the storage index for element (outer, axisIdx, inner) in a reduction.
    /// Uses strides directly — no coordinate decomposition needed.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    internal int ReductionStorageIndex(int outer, int axisIdx, int inner, int axis)
    {
        int idx = _storageOffset + axisIdx * _strides[axis];
        // Decompose outer into dims before axis
        int remaining = outer;
        for (int d = axis - 1; d >= 0; d--)
        {
            int dimIdx = remaining % _shape[d];
            remaining /= _shape[d];
            idx += dimIdx * _strides[d];
        }
        // Decompose inner into dims after axis
        remaining = inner;
        for (int d = _shape.Length - 1; d > axis; d--)
        {
            int dimIdx = remaining % _shape[d];
            remaining /= _shape[d];
            idx += dimIdx * _strides[d];
        }
        return idx;
    }

    public override string ToString()
    {
        return $"Tensor<{typeof(T).Name}> with shape {Shape}";
    }

    /// <summary>
    /// Releases the tensor's reference to shared storage.
    /// When the last tensor/view sharing this storage is disposed, the storage can be reclaimed.
    /// </summary>
    public virtual void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Wait for any in-flight GPU operations before releasing the buffer
        if (_lastWriteSync is not null)
        {
            if (!_lastWriteSync.IsComplete)
                _lastWriteSync.Wait();
            _lastWriteSync.Dispose();
            _lastWriteSync = null;
        }

        // Remove pending deferred materializer to prevent callback on disposed tensor
        if (_gpuMaterializerKey is not null)
        {
            Helpers.DeferredArrayMaterializer.Remove(_gpuMaterializerKey);
            _gpuMaterializerKey = null;
            _gpuMaterializerCallback = null;
        }

        _storage.Release();
        if (_ownsGpuBuffer && _gpuBuffer is IDisposable disposableBuffer)
        {
            disposableBuffer.Dispose();
        }
        _gpuBuffer = null;
        GC.SuppressFinalize(this);
    }
}
