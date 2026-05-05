using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
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
    /// <remarks>
    /// Was <c>readonly</c> historically. The <see langword="readonly"/> was removed to
    /// support <see cref="RebindStorageFrom"/>, which is the ONLY path that should
    /// reassign this field. Under all other circumstances this field is effectively
    /// readonly — the rebind operation is narrow (plan stitching per issue #170 and
    /// memory-planning buffer reuse per issue #182) and intentionally bypasses the
    /// usual "views share storage, storage itself is immutable" invariant.
    /// </remarks>
    internal TensorStorage<T> _storage;

    /// <summary>
    /// Direct reference to underlying Vector for backward compatibility with existing engine code.
    /// All new code should prefer _storage methods.
    /// </summary>
    /// <remarks>
    /// Was <c>readonly</c> historically — see <see cref="_storage"/>'s remarks for why
    /// the modifier was dropped. Same narrow intent: only <see cref="RebindStorageFrom"/>
    /// should ever reassign this field.
    /// </remarks>
    protected Vector<T> _data;

    /// <summary>
    /// Internal accessor for the backing vector. Used by DirectGpuTensorEngine to check
    /// activation cache without triggering CPU materialization.
    /// </summary>
    internal Vector<T> DataVector => _data;

    /// <summary>
    /// Physical memory layout of this tensor's data. Default
    /// <see cref="TensorLayout.Nchw"/> (standard row-major). The
    /// channel-packed variants (<see cref="TensorLayout.Nchwc8"/>,
    /// <see cref="TensorLayout.Nchwc16"/>) are produced by the engine's
    /// <c>ReorderToNchwc</c> primitive and consumed by the matching
    /// NCHWc op fast paths. The layout is advisory metadata — storage is
    /// still a flat <see cref="Vector{T}"/>; the layout field tells the
    /// dispatcher how to interpret the flat elements.
    /// </summary>
    public TensorLayout Layout { get; internal set; } = TensorLayout.Nchw;

    /// <summary>
    /// Rebinds this tensor's backing storage to alias <paramref name="source"/>'s storage.
    /// After a successful rebind, both tensors read from and write to the <i>same</i>
    /// underlying <see cref="Vector{T}"/> and <see cref="TensorStorage{T}"/> — no data
    /// copy, no new allocation. Narrow use cases:
    /// plan stitching (<see cref="AiDotNet.Tensors.Engines.Compilation.ICompiledPlan{T}.ThenAsync"/>
    /// needs the downstream plan's captured input to point at the upstream plan's
    /// captured output so execute-time operations see each other's results without
    /// going through a boundary memcpy), and memory-planning buffer reuse
    /// (issue #182 — intermediate activations whose live ranges don't overlap
    /// are aliased to a shared backing buffer to cut peak memory).
    /// </summary>
    /// <param name="source">The tensor whose storage this tensor should alias.</param>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="source"/> has a different
    /// shape, is non-contiguous, or has a non-zero storage offset. Rebind is only
    /// well-defined when the two buffers are flat-equivalent.</exception>
    /// <remarks>
    /// <para>
    /// <b>Side effect — identity transfer.</b> After this call, mutating the data of
    /// either tensor is visible through the other. Callers who previously treated
    /// <c>this</c> as an independent buffer must stop writing to it (any writes now
    /// clobber <paramref name="source"/>'s contents as well).
    /// </para>
    /// <para>
    /// <b>View caveat.</b> <see cref="Tensor{T}.Reshape"/>, <see cref="Tensor{T}.Transpose"/>
    /// and other view-producing operations create new Tensor objects that capture the
    /// <b>current</b> <see cref="_data"/> reference at the time of creation. Views
    /// constructed before a rebind will continue to see the <i>old</i> storage; only
    /// direct operations on <c>this</c> (and views constructed <i>after</i> the rebind)
    /// see the new storage. This is intentional — the common stitching case passes the
    /// input tensor directly to its first op (no pre-rebind views), so views are rare
    /// in practice.
    /// </para>
    /// <para>
    /// <b>Refcount invariant.</b> Each <see cref="TensorBase{T}"/> instance holds
    /// exactly one reference on its <see cref="_storage"/> (released in Dispose).
    /// Rebinding must AddRef the new storage and Release the old one, in that order,
    /// so a concurrent dispose of <paramref name="source"/> cannot drop the new storage
    /// to zero between the two operations. Skipping refcount adjustment (as the earlier
    /// implementation did) left the old storage leaked and the new storage one ref short,
    /// leading to use-after-free when <paramref name="source"/> was disposed.
    /// </para>
    /// </remarks>
    internal void RebindStorageFrom(TensorBase<T> source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));

        // Shape equality: the user-facing `Length` of both tensors must match
        // element-for-element or the rebind would leave our `_shape`/`_strides`
        // out of sync with the new storage. We accept any shape permutation
        // the caller claims — strides/shape on `this` are untouched.
        if (source._shape.Length != _shape.Length)
            throw new ArgumentException(
                $"Rebind requires same rank. This tensor has rank {_shape.Length}, " +
                $"source has rank {source._shape.Length}.",
                nameof(source));
        for (int i = 0; i < _shape.Length; i++)
        {
            if (_shape[i] != source._shape[i])
                throw new ArgumentException(
                    $"Rebind requires same shape. This [{string.Join(", ", _shape)}] vs " +
                    $"source [{string.Join(", ", source._shape)}].",
                    nameof(source));
        }

        // Contiguity + zero-offset: the layout in `this` must describe a flat
        // element range over the new storage. View tensors (non-contiguous
        // strides, non-zero storage offset) would silently misread the aliased
        // buffer. Fail loud instead of corrupt-quiet.
        if (!IsContiguous || _storageOffset != 0)
            throw new ArgumentException(
                "Rebind target must be contiguous with zero storage offset; " +
                "views are not supported.", nameof(source));
        if (!source.IsContiguous || source._storageOffset != 0)
            throw new ArgumentException(
                "Rebind source must be contiguous with zero storage offset; " +
                "views are not supported.", nameof(source));

        // Fast path: already aliasing the same storage. Still refresh _data in
        // case the source's _data field was swapped to a different Vector view
        // (shouldn't happen with current code, but preserves source-of-truth).
        if (ReferenceEquals(_storage, source._storage))
        {
            _data = source._data;
            return;
        }

        // Acquire the new reference BEFORE releasing the old one. If we released
        // first and source was the only live referent of its storage, it could
        // be disposed concurrently and AddRef would throw ObjectDisposedException,
        // leaving us with neither storage. Both fields update together so an
        // intervening read couldn't observe a half-rebound state.
        source._storage.AddRef();
        var oldStorage = _storage;
        _storage = source._storage;
        _data = source._data;
        oldStorage.Release();
    }

    /// <summary>
    /// Drops this tensor's in-memory data, replacing it with an empty
    /// <see cref="Vector{T}"/> placeholder. Used by
    /// <c>WeightRegistry.ReleaseToPool</c> when streaming weights have
    /// been registered with <see cref="StreamingTensorPool"/> — the pool
    /// becomes the canonical owner of the bytes, and this tensor's
    /// resident memory is freed until <see cref="RestoreStorageFromBytes"/>
    /// is called.
    /// </summary>
    /// <remarks>
    /// Must only be called on contiguous, zero-offset, non-view tensors.
    /// Trainable weight tensors are exactly this — they're owned by one
    /// layer, never sliced into views, and their <see cref="Length"/>
    /// stays unchanged across drop/restore cycles.
    /// </remarks>
    internal void DropStorageForStreaming()
    {
        if (!IsContiguous || _storageOffset != 0 || IsView)
            throw new InvalidOperationException(
                "Streaming drop requires contiguous, non-view, zero-offset tensors. " +
                "Weight tensors satisfy this; views and sliced tensors do not.");

        // Atomically claim sole ownership of the current storage.
        // RebindStorageFrom (used by CompiledInferencePlan / MemoryPlanningPass)
        // creates peer tensors that share storage refcounts; dropping
        // shared storage would leak the bytes the rebound peer is still
        // reading. The previous implementation read RefCount then swapped,
        // which is racy: a sibling AddRef between the read and the swap
        // could leave us replacing storage that's been re-shared. CAS-based
        // TryClaimExclusive closes that window — it succeeds only when
        // refcount was exactly 1 at the moment of the claim, and after
        // success any concurrent AddRef throws ObjectDisposedException.
        if (!_storage.TryClaimExclusive())
            throw new InvalidOperationException(
                $"Streaming drop requires sole storage ownership; storage refcount is {_storage.RefCount}. " +
                "Register the weight via WeightRegistry before any RebindStorageFrom / view operation that " +
                "shares its storage.");

        // Successful claim drove refcount 1 → 0. The old storage is now
        // owned exclusively by this method and no other thread can take
        // a fresh reference (AddRef would observe refcount == 0 and
        // throw). Abandon it for GC and swap in fresh empty storage.
        // Note: do NOT call Release on the claimed storage — refcount
        // is already 0, Release would underflow.
        _data = Vector<T>.Empty();
        _storage = new TensorStorage<T>(_data);
    }

    /// <summary>
    /// Restores this tensor's data from a serialized byte buffer (produced
    /// by <c>WeightRegistry</c>'s SerializeToBytes path). Used by
    /// <c>WeightRegistry.Materialize</c> after the streaming pool returned
    /// the bytes from its resident set or paged them in from the backing
    /// store. Buffer length must equal <see cref="Length"/> ×
    /// element size; mismatch indicates a serialization bug.
    /// </summary>
    /// <param name="bytes">Serialized element data in little-endian
    /// row-major order. Format must match <c>WeightRegistry</c>'s
    /// SerializeToBytes (float / double / int / long / Half / BFloat16
    /// fast paths).</param>
    internal void RestoreStorageFromBytes(ReadOnlySpan<byte> bytes)
    {
        if (!IsContiguous || _storageOffset != 0 || IsView)
            throw new InvalidOperationException(
                "Streaming restore requires contiguous, non-view, zero-offset tensors.");

        // Use the typed-fast-path size table rather than Marshal.SizeOf<T>:
        // the latter throws ArgumentException for non-blittable types
        // (Complex, Multivector) with a confusing native-interop message.
        // ElementSizeForStreaming throws NotSupportedException with a
        // clear "use a supported element type" message instead, matching
        // WeightRegistry.SerializeToBytes' error contract.
        int elementSize = ElementSizeForStreaming();
        long expectedBytes = (long)Length * elementSize;
        if (bytes.Length != expectedBytes)
            throw new ArgumentException(
                $"Streaming restore: buffer length {bytes.Length} does not match " +
                $"expected {expectedBytes} bytes (Length={Length} × element size={elementSize}).");

        // Construct a typed backing array, deserialize bytes into it via
        // typed fast paths — same set WeightRegistry.SerializeToBytes
        // covers — then wrap it back into a Vector<T>. We can't use
        // MemoryMarshal.Cast on Span<T> directly because T isn't
        // constrained to struct on TensorBase, so we go through typed
        // arrays via the (T[])(object)typed[] cast that's safe at runtime
        // when typeof(T) matches.
        var fresh = DeserializeToVector(bytes, Length);

        var oldStorage = _storage;
        _data = fresh;
        _storage = new TensorStorage<T>(_data);
        oldStorage.Release();
    }

    private static int ElementSizeForStreaming()
    {
        if (typeof(T) == typeof(float)) return sizeof(float);
        if (typeof(T) == typeof(double)) return sizeof(double);
        if (typeof(T) == typeof(int)) return sizeof(int);
        if (typeof(T) == typeof(long)) return sizeof(long);
        if (typeof(T) == typeof(Half)) return 2;
        if (typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16)) return 2;
        throw new NotSupportedException(
            $"Streaming requires element type T to be one of: float, double, int, long, Half, BFloat16. " +
            $"Got {typeof(T).Name}.");
    }

    private static Vector<T> DeserializeToVector(ReadOnlySpan<byte> src, int length)
    {
        if (typeof(T) == typeof(float))
        {
            var arr = new float[length];
            src.CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(arr.AsSpan()));
            return Vector<T>.WrapMemory((T[])(object)arr);
        }
        if (typeof(T) == typeof(double))
        {
            var arr = new double[length];
            src.CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(arr.AsSpan()));
            return Vector<T>.WrapMemory((T[])(object)arr);
        }
        if (typeof(T) == typeof(int))
        {
            var arr = new int[length];
            src.CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(arr.AsSpan()));
            return Vector<T>.WrapMemory((T[])(object)arr);
        }
        if (typeof(T) == typeof(long))
        {
            var arr = new long[length];
            src.CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(arr.AsSpan()));
            return Vector<T>.WrapMemory((T[])(object)arr);
        }
        if (typeof(T) == typeof(Half))
        {
            var arr = new Half[length];
            for (int i = 0; i < length; i++)
            {
                ushort raw = (ushort)(src[i * 2] | (src[i * 2 + 1] << 8));
                arr[i] = AiDotNet.Tensors.NumericOperations.HalfBits.FromBits(raw);
            }
            return Vector<T>.WrapMemory((T[])(object)arr);
        }
        if (typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16))
        {
            var arr = new AiDotNet.Tensors.NumericOperations.BFloat16[length];
            for (int i = 0; i < length; i++)
            {
                ushort raw = (ushort)(src[i * 2] | (src[i * 2 + 1] << 8));
                arr[i] = AiDotNet.Tensors.NumericOperations.BFloat16.FromRawBits(raw);
            }
            return Vector<T>.WrapMemory((T[])(object)arr);
        }
        throw new NotSupportedException(
            $"Streaming restore: no exact deserializer for element type {typeof(T).Name}. " +
            "Supported: float, double, int, long, Half, BFloat16.");
    }

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
    /// Lazy computation graph node that will produce this tensor's data when realized.
    /// When non-null, this tensor's data is not yet computed — accessing it via AsSpan()
    /// or GetDataArray() will auto-materialize by calling Realize() on the lazy node.
    /// </summary>
    internal ILazyNode? LazySource;

    /// <summary>
    /// Index into the flat gradient array during backward pass. Assigned by the tape
    /// during recording, used by ComputeGradients for O(1) gradient lookup instead of
    /// O(1)-amortized dictionary hash. -1 means not assigned.
    /// </summary>
    internal int _gradIndex = -1;

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
    /// <remarks>
    /// When an <see cref="Engines.Autodiff.InferenceModeScope{T}"/> is
    /// active on the calling thread, the bump is skipped — mutation
    /// is legal in inference mode and skipping the bump avoids
    /// false-positive version-mismatch errors in code paths that
    /// captured the version before entering the scope. The
    /// type-erased <c>InferenceModeFlag.IsActive</c> check is a
    /// single boolean read so the cost on the non-inference hot
    /// path is one branch.
    /// </remarks>
    internal void IncrementVersion()
    {
        if (Engines.Autodiff.InferenceModeFlag.IsActive)
        {
            // Inference mode: in-place mutation is legal and the
            // version counter must not advance, otherwise outer code
            // that recorded a Version snapshot before the scope
            // entered would observe a phantom mutation.
            UniformFillValue = null;
            return;
        }
        System.Threading.Interlocked.Increment(ref _version);
        UniformFillValue = null; // Invalidate: tensor data no longer uniform after mutation
    }

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
    /// Lifetime / placement hint for the issue-#276 large-model memory paths.
    /// <see cref="WeightLifetime.Default"/> ⇒ regular allocation, GC-bound;
    /// <see cref="WeightLifetime.Streaming"/> ⇒ register with the streaming pool;
    /// <see cref="WeightLifetime.GpuOffload"/> ⇒ allocate from pinned-host pool;
    /// <see cref="WeightLifetime.GpuManaged"/> ⇒ allocate from unified-memory pool.
    /// Default is <see cref="WeightLifetime.Default"/> — only weights tagged
    /// otherwise hit the offload / streaming dispatch.
    /// </summary>
    public WeightLifetime Lifetime { get; set; } = WeightLifetime.Default;

    /// <summary>
    /// Streaming-pool handle when <see cref="Lifetime"/> is
    /// <see cref="WeightLifetime.Streaming"/>. -1 means "not registered".
    /// Owned by <see cref="WeightRegistry"/>; user code reads but doesn't
    /// mutate (the setter is internal so external code can't unilaterally
    /// invalidate the pool's bookkeeping).
    /// </summary>
    public long StreamingPoolHandle { get; internal set; } = -1;

    /// <summary>
    /// Bytes the streaming pool has pre-reserved against its budget on
    /// behalf of this tensor between <c>WeightRegistry.AllocateStreaming</c>
    /// and the matching <c>WeightRegistry.RegisterWeight</c>. Used to
    /// prevent a TOCTOU race where two concurrent allocate-then-register
    /// flows both pass the pre-allocate eviction gate but together push
    /// resident bytes past budget before either Register lands. Owned by
    /// <see cref="WeightRegistry"/>; setter is internal so external code
    /// can't desync the pool's <c>_reservedBytes</c> bookkeeping.
    /// </summary>
    internal long StreamingReservedBytes { get; set; }

    /// <summary>
    /// GPU offload allocation handle when <see cref="Lifetime"/> is
    /// <see cref="WeightLifetime.GpuOffload"/> or
    /// <see cref="WeightLifetime.GpuManaged"/>. Owned by
    /// <see cref="WeightRegistry"/>; setter is internal.
    /// </summary>
    public IntPtr OffloadDevicePointer { get; internal set; } = IntPtr.Zero;

    /// <summary>Host-visible pointer for the offload allocation. Some
    /// backends (CUDA pinned, HIP HostMalloc) return host==device while
    /// others (Vulkan, OpenCL pinned) return distinct host and device
    /// pointers. The allocator's <c>_live</c> dictionary keys by
    /// HostPointer, so we must persist it separately to reconstruct the
    /// full <see cref="Engines.DirectGpu.GpuOffloadHandle"/> at free time
    /// — using DevicePointer as the host arg silently fails the
    /// allocator's TryRemove and leaks the allocation.</summary>
    public IntPtr OffloadHostPointer { get; internal set; } = IntPtr.Zero;

    /// <summary>Backend-specific opaque handle paired with
    /// <see cref="OffloadDevicePointer"/> (e.g. cl_mem for OpenCL,
    /// VkDeviceMemory for Vulkan). Owned by <see cref="WeightRegistry"/>.</summary>
    public object? OffloadOpaqueHandle { get; internal set; }

    /// <summary>Total bytes of the offload allocation. Used by
    /// <see cref="WeightRegistry.UnregisterWeight"/> when reconstructing
    /// the <see cref="Engines.DirectGpu.GpuOffloadHandle"/> for the free
    /// path — element-size × Length is unsafe when T is a managed type
    /// without a stable Marshal.SizeOf.</summary>
    public long OffloadByteCount { get; internal set; }

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
        EnsureMaterialized();
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
        EnsureMaterialized();
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
        EnsureMaterialized();
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
        EnsureMaterialized();
        ThrowIfSparse();
        if (flatIndex < 0 || flatIndex >= Length)
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        if (IsContiguous)
            _data[flatIndex + _storageOffset] = value;
        else
            _data[FlatIndexToStorageIndex(flatIndex)] = value;
    }

    /// <summary>
    /// Ensures lazy tensor data has been materialized before access.
    /// Centralizes the auto-materialization guard so all data access paths use it.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private void EnsureMaterialized()
    {
        if (LazySource is ILazyNode node && !node.IsRealized)
            node.Realize(node.RecordingEngine);
    }

    /// <summary>
    /// Gets a read-only span over the tensor data. Throws for non-contiguous views.
    /// </summary>
    public ReadOnlySpan<T> AsSpan()
    {
        EnsureMaterialized();

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
        EnsureMaterialized();

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
    /// NOTE: intentionally does NOT call EnsureMaterialized on the simple-
    /// layout path. Callers that pin this array for use at plan-replay time
    /// (specialization compile) must NOT force Realize at compile time —
    /// Realize with placeholder=0 inputs snapshots each upstream node with
    /// IsRealized=true, which then makes certain specializations' pinned
    /// array reads observe pre-user-fill data across subsequent executes
    /// (BERT-SQuAD × 100 sample bug). Callers that NEED live data read a
    /// copy (view case) — ToArray below still materializes in that path,
    /// which is correct because ToArray's very purpose is returning data.
    /// </summary>
    internal T[] GetDataArray()
    {
        // Eager simple-layout CPU tensors: hand back the backing array
        // directly. Specializations (TryBuildSpecializedForward) capture
        // this reference at compile time and need later writes — notably
        // user-supplied graph-input data landing in a placeholder — to
        // show up at replay. Returning a copy here was the missing piece
        // that produced "importer plan outputs all zeros" on ONNX graphs:
        // specializations pinned the placeholder's zero-initialized state
        // and never saw the user's Execute-time input.
        //
        // Lazy or GPU-resident tensors still go through ToArray() so the
        // realized CPU snapshot is what the caller sees. That path
        // preserves the BERT-SQuAD × 100 fix: a lazy tensor whose
        // upstream hasn't run yet would have had its placeholder-filled
        // backing pinned, leaking stale/zero bytes into every replay.
        var live = GetLiveBackingArrayOrNull();
        if (live is not null) return live;
        return ToArray();
    }

    /// <summary>
    /// Gets the BACKING storage array without triggering lazy realization
    /// and without ever returning a copy. Returns <c>null</c> when the
    /// tensor's layout is not a simple contiguous-at-offset-0-whose-storage-
    /// length-matches-logical-length view of its backing storage — in which
    /// case callers that need a live pin must skip their specialization and
    /// fall back to the general AsSpan-based path. Live callers (the
    /// TryBuildSpecializedForward pinners) use this to avoid the Realize-
    /// triggered-during-compile cascade that bakes placeholder=0 data into
    /// every upstream IsRealized node (issue surfaced by BERT-SQuAD × 100
    /// sample replay: first execute correct, subsequent executes returned
    /// the first execute's output verbatim because some specialization
    /// pinned a ToArray snapshot taken during compile-time realize).
    /// </summary>
    internal T[]? GetLiveBackingArrayOrNull()
    {
        if (!IsContiguous || _storageOffset != 0 || _storage.Length != Length)
            return null;
        // GPU-resident tensors keep the authoritative data on-device; the
        // CPU backing array may hold stale/placeholder bytes that haven't
        // been copied back. Pinning that into a specialization would read
        // the wrong values at replay. Force these callers onto the AsSpan
        // path, which goes through EnsureMaterialized and copies GPU→CPU
        // before the caller touches the buffer.
        if (_device != TensorDevice.CPU)
            return null;
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
        // Packed layouts (Nchwc8/Nchwc16) reinterpret the flat buffer via a
        // different stride pattern; a contiguous element-for-element copy
        // that leaves Layout at the default Nchw would send later ops down
        // the wrong dispatch branch. Propagate the source layout so the
        // copy stays semantically identical.
        result.Layout = Layout;
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
        // Transform walks flat index i in source order. A packed source
        // with the default Nchw layout on the result would mislabel the
        // output; propagate so downstream dispatch stays correct.
        result.Layout = Layout;
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
        result.Layout = Layout;
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
        EnsureMaterialized();
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
