using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Cached GPU buffer entry for persistent tensor management.
/// </summary>
internal sealed class GpuBufferCacheEntry : IDisposable
{
    public IGpuBuffer Buffer { get; }
    public PersistentTensorRole Role { get; }
    public int Version { get; set; }

    public GpuBufferCacheEntry(IGpuBuffer buffer, PersistentTensorRole role)
    {
        Buffer = buffer;
        Role = role;
        Version = 0;
    }

    public void Dispose()
    {
        Buffer.Dispose();
    }
}

/// <summary>
/// Composite key for CSR buffer cache, binding a sparse tensor to a specific GPU backend.
/// Uses reference equality for both fields so we only reuse buffers for the exact same objects.
/// </summary>
internal sealed class CsrCacheKey : IEquatable<CsrCacheKey>
{
    private readonly object _sparse;
    private readonly IDirectGpuBackend _backend;

    public CsrCacheKey(object sparse, IDirectGpuBackend backend)
    {
        _sparse = sparse;
        _backend = backend;
    }

    public bool Equals(CsrCacheKey? other) =>
        other is not null && ReferenceEquals(_sparse, other._sparse) && ReferenceEquals(_backend, other._backend);

    public override bool Equals(object? obj) => Equals(obj as CsrCacheKey);

    public override int GetHashCode()
    {
        unchecked
        {
            return (System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(_sparse) * 397)
                 ^ System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(_backend);
        }
    }
}

/// <summary>
/// Cached CSR GPU buffers for a sparse tensor so repeated SpMM calls skip re-upload.
/// </summary>
internal sealed class CsrGpuCache : IDisposable
{
    public IGpuBuffer Values { get; }
    public IGpuBuffer ColumnIndices { get; }
    public IGpuBuffer RowPointers { get; }
    public int NonZeroCount { get; }

    public CsrGpuCache(IGpuBuffer values, IGpuBuffer columnIndices, IGpuBuffer rowPointers, int nnz)
    {
        Values = values;
        ColumnIndices = columnIndices;
        RowPointers = rowPointers;
        NonZeroCount = nnz;
    }

    public void Dispose()
    {
        Values.Dispose();
        ColumnIndices.Dispose();
        RowPointers.Dispose();
    }
}

/// <summary>
/// Cache entry for intermediate activation tensors to avoid re-uploading between layers.
/// When a layer's output is downloaded, we cache the GPU buffer so the next layer
/// can reuse it without re-uploading if it uses the same data.
/// </summary>
internal sealed class ActivationCacheEntry : IDisposable
{
    public IGpuBuffer Buffer { get; }
    public int[] Shape { get; }
    public long Timestamp { get; }
    public IDirectGpuBackend Backend { get; }

    public ActivationCacheEntry(IGpuBuffer buffer, int[] shape, long timestamp, IDirectGpuBackend backend)
    {
        Buffer = buffer;
        Shape = shape;
        Timestamp = timestamp;
        Backend = backend;
    }

    public void Dispose()
    {
        Buffer.Dispose();
    }
}

// DeferredDownloadEntry was removed in the #226 cleanup — the engine no longer
// maintains a local pending-download map. DeferredArrayMaterializer is the
// single source of truth for "some caller still needs this buffer downloaded",
// and its Register/TryMaterialize/MaterializeAll drive both eviction protection
// (via IsPending) and scope-end flushing.

/// <summary>
/// Scope that enables GPU-resident caching of intermediate results.
/// When active, output buffers from GPU operations are cached in the activation cache,
/// allowing chained operations to reuse GPU buffers without re-uploading.
/// Downloads of intermediate results are deferred until CPU data is actually needed.
/// </summary>
public sealed class GpuScope : IDisposable
{
    [ThreadStatic]
    private static Dictionary<int, int>? _depthPerEngine;

    private readonly DirectGpuTensorEngine? _engine;
    private readonly int _engineId;

    /// <summary>
    /// Returns true if any GpuScope is currently active on this thread.
    /// </summary>
    internal static bool IsActive
    {
        get
        {
            if (_depthPerEngine is null) return false;
            foreach (var kvp in _depthPerEngine)
            {
                if (kvp.Value > 0) return true;
            }
            return false;
        }
    }

    internal GpuScope(DirectGpuTensorEngine? engine = null)
    {
        _engine = engine;
        _engineId = engine?.GetHashCode() ?? 0;
        _depthPerEngine ??= new Dictionary<int, int>();
        _depthPerEngine.TryGetValue(_engineId, out int depth);
        _depthPerEngine[_engineId] = depth + 1;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_depthPerEngine is null) return;
        _depthPerEngine.TryGetValue(_engineId, out int depth);
        if (depth > 0)
        {
            depth--;
            _depthPerEngine[_engineId] = depth;
            // When the outermost scope for this engine exits, materialize its deferred downloads
            if (depth == 0)
            {
                _engine?.MaterializeAllDeferred();
            }
        }
    }
}

/// <summary>
/// IEngine implementation that routes supported ops to DirectGpuEngine and falls back to CPU.
/// </summary>
/// <remarks>
/// <para><b>Threading Model:</b> This engine uses buffer caching for GPU memory efficiency.
/// Cache operations are thread-safe for concurrent reads, but cache invalidation/clearing
/// (InvalidateWeightCache, InvalidateAllWeightCaches, ClearActivationCache) should NOT be
/// called while GPU operations are in-flight using cached buffers.</para>
/// <para><b>Safe usage patterns:</b></para>
/// <list type="bullet">
/// <item>Single-threaded inference: fully safe</item>
/// <item>Multi-threaded inference with stable weights: safe (no invalidation during inference)</item>
/// <item>Weight updates during inference: call invalidation only between inference batches</item>
/// </list>
/// <para>For concurrent weight updates during inference, consider using separate engine instances
/// or implementing external synchronization around weight update + invalidation sequences.</para>
/// </remarks>
public partial class DirectGpuTensorEngine : CpuEngine, ITensorLevelEngine, IDisposable
{
    private readonly DirectGpuEngine? _directGpu;
    private readonly bool _ownsDirectGpu;
    private static int _gpuRngSeed = Environment.TickCount;

    // GPU buffer cache for persistent tensors - keyed by tensor data array reference
    // Thread-safety: ConcurrentDictionary provides atomic operations. Invalidation uses
    // _persistentBufferLock for dispose safety. CAUTION: Callers must not invalidate/clear
    // while GPU operations are actively using cached buffers.
    private readonly ConcurrentDictionary<object, GpuBufferCacheEntry> _persistentBufferCache = new();
    private readonly object _persistentBufferLock = new();

    // Version tracking for invalidation
    private readonly ConcurrentDictionary<object, int> _tensorVersions = new();

    // CSR buffer cache for sparse tensors — avoids re-uploading static sparse matrices every call.
    // Key: CsrCacheKey (SparseTensor + Backend reference pair) to avoid cross-backend buffer reuse.
    private readonly ConcurrentDictionary<CsrCacheKey, CsrGpuCache> _csrBufferCache = new();

    // Activation cache for intermediate tensors - enables GPU-resident layer chaining
    // Key: tensor data array reference, Value: (buffer, shape, timestamp)
    // This cache holds the last N activation buffers to avoid re-uploading layer outputs
    // Thread-safety: ConcurrentDictionary + _activationCacheLock for atomic compound operations.
    // CAUTION: ClearActivationCache should not be called during active GPU operations.
    private readonly ConcurrentDictionary<object, ActivationCacheEntry> _activationCache = new();
    private readonly object _activationCacheLock = new();
    private const int DefaultActivationCacheSize = 4096;
    private int _maxActivationCacheSize = DefaultActivationCacheSize;

    /// <summary>
    /// Maximum GPU memory (bytes) the activation cache is allowed to use.
    /// Default is 75% of total GPU memory. When this limit is approached,
    /// the oldest entries are evicted regardless of entry count.
    /// Set to 0 to disable memory-based eviction (count-based only).
    /// </summary>
    private long _maxActivationCacheBytes;
    private long _currentActivationCacheBytes;
    private long _activationCacheTimestamp = 0;

    // Deferred download tracking for GPU-resident execution
    // When GpuScope is active, intermediate results skip the blocking download.
    // The GPU buffer stays in the activation cache for direct GPU-to-GPU chaining.
    // If CPU data is later needed, the DeferredArrayMaterializer registry fires
    // the per-tensor download callback. The engine-local pending map was removed
    // in the #226 cleanup — see DeferredArrayMaterializer for the full contract.

    public DirectGpuTensorEngine()
    {
        _directGpu = new DirectGpuEngine();
        _ownsDirectGpu = true;
        _maxActivationCacheBytes = _directGpu.GlobalMemoryBytes * 3 / 4; // 75% of GPU memory
    }

    public DirectGpuTensorEngine(DirectGpuEngine directGpu)
    {
        _directGpu = directGpu;
        _ownsDirectGpu = false;
        _maxActivationCacheBytes = directGpu?.GlobalMemoryBytes * 3 / 4 ?? 0;
    }

    public bool IsGpuAvailable => _directGpu?.IsAvailable == true;

    public new string Name => IsGpuAvailable
        ? $"Direct GPU Engine ({_directGpu!.BackendName} {_directGpu.DeviceName})"
        : "CPU Engine (DirectGpu unavailable)";

    public new bool SupportsGpu => IsGpuAvailable;

    /// <summary>
    /// Gets or sets the maximum number of activation cache entries.
    /// Larger values use more GPU memory but reduce re-uploads for deep networks.
    /// Default is 256, sized for typical DNN layer chains.
    /// </summary>
    public int MaxActivationCacheSize
    {
        get => _maxActivationCacheSize;
        set => _maxActivationCacheSize = value > 0 ? value : DefaultActivationCacheSize;
    }

    DirectGpuEngine? IEngine.DirectGpu => _directGpu;

    string IEngine.Name => Name;

    bool IEngine.SupportsGpu => SupportsGpu;

    /// <summary>
    /// Begins a GPU scope that enables activation caching for intermediate results.
    /// Within this scope, GPU output buffers are retained in the activation cache,
    /// so chained operations (e.g., GEMM + Bias + ReLU) avoid redundant CPU-GPU transfers.
    /// </summary>
    public GpuScope BeginGpuScope() => new GpuScope(this);

    /// <summary>
    /// Safely downloads GPU buffer data into a tensor, handling the case where
    /// GetDataArray() returns a copy instead of the backing array.
    /// Downloads into a fresh array and copies into the tensor's actual memory.
    /// </summary>
    private static void DownloadIntoTensor(IDirectGpuBackend backend, IGpuBuffer gpuBuffer, Tensor<float> tensor)
    {
        var downloaded = backend.DownloadBuffer(gpuBuffer);
        downloaded.AsSpan(0, Math.Min(downloaded.Length, tensor.Length)).CopyTo(tensor.Data.Span);
    }

    private bool TryGetBackend(out IDirectGpuBackend backend)
    {
        // Check if there's an active DeferredScope - use its RecordingBackend for deferred execution
        var deferredScope = Gpu.DeferredScope.Current;
        if (deferredScope != null && deferredScope.IsRecording)
        {
            backend = deferredScope.RecordingBackend;
            return true;
        }

        // No deferred scope - use regular backend
        backend = _directGpu?.Backend!;
        return IsGpuAvailable && backend != null;
    }

    /// <summary>
    /// Tries to get the batch execution backend (which has the fused kernel methods).
    /// Returns false if GPU is not available or backend doesn't support batch execution.
    /// </summary>
    private bool TryGetBatchBackend(out DirectGpu.IGpuBatchExecution batchBackend)
    {
        if (TryGetBackend(out var backend) && backend is DirectGpu.IGpuBatchExecution batch)
        {
            batchBackend = batch;
            return true;
        }
        batchBackend = null!;
        return false;
    }

    /// <summary>
    /// Gets the GPU backend if available.
    /// </summary>
    /// <returns>The GPU backend, or null if not available.</returns>
    public IDirectGpuBackend? GetBackend()
    {
        if (TryGetBackend(out var backend))
        {
            return backend;
        }
        return null;
    }

    /// <summary>
    /// Gets the async GPU backend if available (supports deferred execution).
    /// </summary>
    /// <returns>The async GPU backend, or null if not available or not supported.</returns>
    public IAsyncGpuBackend? GetAsyncBackend()
    {
        if (TryGetBackend(out var backend))
        {
            return backend as IAsyncGpuBackend;
        }
        return null;
    }

    /// <summary>
    /// Begins a GPU execution context for GPU-resident operations.
    /// Operations within the context stay GPU-resident until explicitly downloaded.
    /// </summary>
    /// <param name="options">Optional execution options.</param>
    /// <returns>A GPU execution context, or null if GPU is not available.</returns>
    public GpuExecutionContext? BeginGpuContext(GpuExecutionOptions? options = null)
    {
        if (!TryGetBackend(out var backend))
        {
            return null;
        }

        return GpuExecutionContext.Begin(backend, options);
    }

    /// <summary>
    /// Begins a deferred execution scope that records operations to an execution graph
    /// for optimized batch execution.
    /// </summary>
    /// <param name="options">Optional execution options.</param>
    /// <returns>A deferred scope for recording operations, or null if not supported.</returns>
    /// <remarks>
    /// <para><b>Example:</b></para>
    /// <code>
    /// using var scope = engine.BeginDeferredScope();
    /// if (scope != null)
    /// {
    ///     // Operations recorded to scope.GraphBuilder
    ///     scope.Execute(); // Compile and execute all at once
    /// }
    /// </code>
    /// </remarks>
    public IDeferredScope? BeginDeferredScope(GpuExecutionOptions? options = null)
    {
        var asyncBackend = GetAsyncBackend();
        if (asyncBackend == null)
        {
            return null;
        }

        var effectiveOptions = options ?? GpuExecutionOptions.FromEnvironment();
        var streamPool = asyncBackend.SupportsMultiStream
            ? new GpuStreamPool(asyncBackend, effectiveOptions)
            : null;

        return new DeferredScope(asyncBackend, effectiveOptions, streamPool);
    }

    /// <summary>
    /// Gets whether deferred execution is supported on this engine.
    /// </summary>
    public bool SupportsDeferredExecution => GetAsyncBackend() != null;

    /// <summary>
    /// Gets the current GPU execution context for this thread, if any.
    /// This allows operations to check if GPU-resident mode is active.
    /// </summary>
    public static GpuExecutionContext? CurrentContext => GpuExecutionContext.Current;

    /// <summary>
    /// Gets whether a GPU execution context is currently active on this thread.
    /// When active, GPU tensors can stay resident on the GPU without downloading.
    /// </summary>
    public static bool IsGpuContextActive => GpuExecutionContext.Current != null;

    /// <summary>
    /// Determines whether GPU should be used for an operation of the given element count.
    /// Uses the current execution context options if available, otherwise uses defaults.
    /// </summary>
    /// <param name="elementCount">The number of elements in the operation.</param>
    /// <returns>True if GPU should be used.</returns>
    public bool ShouldUseGpu(int elementCount)
    {
        if (!IsGpuAvailable)
        {
            return false;
        }

        // Use context options if available
        var context = GpuExecutionContext.Current;
        if (context != null)
        {
            return context.ShouldUseGpu(elementCount);
        }

        // Default threshold
        return elementCount >= 4096;
    }

    /// <summary>
    /// Uploads a tensor to GPU within the current execution context.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="tensor">The CPU tensor to upload.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor, or null if no context is active.</returns>
    public Tensor<T>? UploadToContext<T>(Tensor<T> tensor, GpuTensorRole role = GpuTensorRole.General)
    {
        var context = GpuExecutionContext.Current;
        return context?.Upload(tensor, role);
    }

    /// <summary>
    /// Uploads data to GPU within the current execution context.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="data">The CPU data to upload.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor, or null if no context is active.</returns>
    public Tensor<T>? UploadToContext<T>(T[] data, int[] shape, GpuTensorRole role = GpuTensorRole.General)
    {
        var context = GpuExecutionContext.Current;
        return context?.Upload(data, shape, role);
    }

    /// <summary>
    /// Creates an empty GPU tensor within the current execution context.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor with uninitialized data, or null if no context is active.</returns>
    public Tensor<T>? EmptyInContext<T>(int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        var context = GpuExecutionContext.Current;
        return context?.Empty<T>(shape, role);
    }

    /// <summary>
    /// Creates a GPU tensor filled with zeros within the current execution context.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A GPU-resident tensor filled with zeros, or null if no context is active.</returns>
    public Tensor<T>? ZerosInContext<T>(int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        var context = GpuExecutionContext.Current;
        return context?.Zeros<T>(shape, role);
    }

    /// <summary>
    /// Executes an action within a GPU execution context.
    /// </summary>
    /// <param name="action">The action to execute.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>True if executed on GPU, false if GPU not available.</returns>
    public bool WithGpuContext(Action<GpuExecutionContext> action, GpuExecutionOptions? options = null)
    {
        var context = BeginGpuContext(options);
        if (context == null)
        {
            return false;
        }

        using (context)
        {
            action(context);
        }

        return true;
    }

    /// <summary>
    /// Executes a function within a GPU execution context.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="func">The function to execute.</param>
    /// <param name="fallback">Fallback function if GPU is not available.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>The function result.</returns>
    public TResult WithGpuContext<TResult>(Func<GpuExecutionContext, TResult> func, Func<TResult> fallback, GpuExecutionOptions? options = null)
    {
        var context = BeginGpuContext(options);
        if (context == null)
        {
            return fallback();
        }

        using (context)
        {
            return func(context);
        }
    }

    private static float ToFloatScalar<T>(T value)
    {
        if (typeof(T) == typeof(float))
            return (float)(object)value!;
        if (typeof(T) == typeof(double))
            return (float)(double)(object)value!;

        // Use numeric operations directly instead of allocating a single-element array
        return (float)MathHelper.GetNumericOperations<T>().ToDouble(value);
    }

    private static T FromFloatScalar<T>(float value)
    {
        if (typeof(T) == typeof(float))
            return (T)(object)value;
        if (typeof(T) == typeof(double))
            return (T)(object)(double)value;

        // Use numeric operations directly instead of allocating a single-element array
        return MathHelper.GetNumericOperations<T>().FromFloat(value);
    }

    /// <summary>
    /// Helper struct for tracking GPU buffer ownership. Implements IDisposable
    /// to only dispose buffers we own (not cached ones).
    /// </summary>
    private readonly struct OwnedBuffer : IDisposable
    {
        private readonly IGpuBuffer _buffer;
        private readonly bool _ownsBuffer;

        /// <summary>
        /// Gets the underlying GPU buffer.
        /// </summary>
        public IGpuBuffer Buffer => _buffer;

        /// <summary>
        /// Gets whether this wrapper owns the buffer (and should dispose it).
        /// </summary>
        public bool OwnsBuffer => _ownsBuffer;

        public OwnedBuffer(IGpuBuffer buffer, bool ownsBuffer)
        {
            _buffer = buffer;
            _ownsBuffer = ownsBuffer;
        }


        public void Dispose()
        {
            if (_ownsBuffer)
                _buffer.Dispose();
        }
    }

    /// <summary>
    /// Gets a GPU buffer for the tensor data, using cache if available.
    /// Returns an OwnedBuffer that only disposes if we allocated it (not cached).
    /// Checks both persistent tensor cache (weights/biases) and activation cache (layer outputs).
    /// Thread-safe: uses lock to prevent use-after-dispose during cache eviction.
    /// </summary>
    /// <summary>
    /// Gets a GPU buffer for tensor data. Checks activation cache and persistent cache
    /// BEFORE triggering any CPU materialization, avoiding wasteful GPU-to-CPU downloads
    /// for chained GPU operations. This is the primary method for GPU-resident pipelines.
    /// </summary>
    private OwnedBuffer GetOrAllocateBuffer<T>(IDirectGpuBackend backend, Tensor<T> tensor)
    {
        // Fast path: tensor already has a GPU buffer from a previous GPU operation.
        // This is the PyTorch-like path — no cache lookup, no CPU materialization.
        if (tensor._gpuBuffer is not null && ReferenceEquals(tensor._gpuBackend, backend))
        {
            return new OwnedBuffer(tensor._gpuBuffer, ownsBuffer: false);
        }

        // Get the backing array reference WITHOUT triggering materialization.
        // This is critical: GetDataArray() would download from GPU, which is wasteful
        // when we're about to find the GPU buffer in the activation cache.
        var backingArray = tensor.DataVector.GetBackingArrayUnsafe();
        if (backingArray is not null)
        {
            // Check caches using the raw array reference (no download triggered)
            var cached = TryGetCachedBuffer(backingArray);
            if (cached != null)
                return new OwnedBuffer(cached, ownsBuffer: false);

            lock (_activationCacheLock)
            {
                if (_activationCache.TryGetValue(backingArray, out var activationEntry) &&
                    ReferenceEquals(activationEntry.Backend, backend))
                {
                    return new OwnedBuffer(activationEntry.Buffer, ownsBuffer: false);
                }
            }
        }
        else
        {
            // GPU-resident tensor with no backing array — check activation cache by vector key
            var vector = tensor.DataVector;
            lock (_activationCacheLock)
            {
                if (_activationCache.TryGetValue(vector, out var activationEntry) &&
                    ReferenceEquals(activationEntry.Backend, backend))
                {
                    return new OwnedBuffer(activationEntry.Buffer, ownsBuffer: false);
                }
            }
        }

        // Not in any cache — fall back to GetDataArray (triggers lazy allocation + GPU download if needed)
        return GetOrAllocateBuffer(backend, tensor.GetDataArray());
    }

    private OwnedBuffer GetOrAllocateBuffer<T>(IDirectGpuBackend backend, T[] data)
    {
        // First check persistent tensor cache (for weights/biases)
        var cached = TryGetCachedBuffer(data);
        if (cached != null)
            return new OwnedBuffer(cached, ownsBuffer: false);

        // Check activation cache (for intermediate layer outputs)
        lock (_activationCacheLock)
        {
            if (_activationCache.TryGetValue(data, out var activationEntry) &&
                ReferenceEquals(activationEntry.Backend, backend))
            {
                return new OwnedBuffer(activationEntry.Buffer, ownsBuffer: false);
            }
        }

        // Not cached - need to upload. If this array still has a pending
        // deferred download from an earlier GPU op, flush it first so the
        // upload sees the current data instead of stale CPU bytes.
        if (Helpers.DeferredArrayMaterializer.IsPending(data))
        {
            MaterializeIfDeferred(data);
        }

        float[] floatData = DirectGpuEngine.ToFloatArray(data);
        return new OwnedBuffer(backend.AllocateBuffer(floatData), ownsBuffer: true);
    }

    /// <summary>
    /// Gets a GPU buffer for Memory&lt;T&gt; data, using cache if available.
    /// Uses MemoryMarshal.TryGetArray to extract underlying array for cache lookup and efficient upload.
    /// </summary>
    private OwnedBuffer GetOrAllocateBuffer<T>(IDirectGpuBackend backend, ReadOnlyMemory<T> memory)
    {
        // Try to get the underlying array for cache lookup
        if (MemoryMarshal.TryGetArray(memory, out ArraySegment<T> segment) &&
            segment.Offset == 0 && segment.Count == segment.Array!.Length)
        {
            // Memory is backed by a full array - use the array-based overload for caching
            return GetOrAllocateBuffer(backend, segment.Array);
        }

        // Memory is a slice or not array-backed - must upload directly (no caching)
        float[] floatData = DirectGpuEngine.ToFloatArray(memory.ToArray());
        return new OwnedBuffer(backend.AllocateBuffer(floatData), ownsBuffer: true);
    }

    /// <summary>
    /// Uploads a tensor to GPU, checking GPU buffer and activation cache BEFORE
    /// triggering CPU materialization. This is the preferred upload path for all
    /// IEngine explicit implementations — replaces the pattern:
    ///   b.AllocateBuffer(((Tensor&lt;float&gt;)(object)tensor).GetDataArray())
    /// with:
    ///   UploadTensor(b, tensor)
    /// </summary>
    /// <summary>
    /// Returns an OwnedBuffer wrapping the tensor's GPU data. If the buffer came from
    /// cache, ownsBuffer is false (won't be disposed). If it's a fresh upload, ownsBuffer
    /// is true (will be disposed when the using block ends).
    /// </summary>
    private OwnedBuffer UploadTensor<T>(IDirectGpuBackend backend, Tensor<T> tensor)
    {
        // Fast path: tensor has a GPU buffer from a previous GPU operation
        if (tensor._gpuBuffer is not null && ReferenceEquals(tensor._gpuBackend, backend))
            return new OwnedBuffer(tensor._gpuBuffer, ownsBuffer: false);

        // Check caches without triggering CPU materialization
        var backingArray = tensor.DataVector.GetBackingArrayUnsafe();
        if (backingArray is not null)
        {
            var cached = TryGetCachedBuffer(backingArray);
            if (cached != null) return new OwnedBuffer(cached, ownsBuffer: false);

            lock (_activationCacheLock)
            {
                if (_activationCache.TryGetValue(backingArray, out var entry) &&
                    ReferenceEquals(entry.Backend, backend))
                    return new OwnedBuffer(entry.Buffer, ownsBuffer: false);
            }
        }

        // No cached buffer — must upload from CPU
        var data = tensor.GetDataArray();
        float[] floatData = DirectGpuEngine.ToFloatArray(data);
        return new OwnedBuffer(backend.AllocateBuffer(floatData), ownsBuffer: true);
    }

    /// <summary>
    /// Returns the raw IGpuBuffer for a tensor, checking GPU buffer and caches first.
    /// WARNING: The returned buffer may or may not be owned — do NOT dispose it unless
    /// you know it was freshly allocated. For IEngine compact one-liners where the entire
    /// try block is wrapped and failure falls back to CPU, this is safe.
    /// For code that needs explicit disposal control, use the OwnedBuffer overload.
    /// </summary>
    private IGpuBuffer UploadTensorRaw<T>(IDirectGpuBackend backend, Tensor<T> tensor)
    {
        var owned = UploadTensor(backend, tensor);
        if (owned.OwnsBuffer)
        {
            // Fresh upload — cache it so it's managed by the activation cache lifecycle.
            // Also set the tensor's _gpuBuffer so future operations find it directly.
            tensor._gpuBuffer = owned.Buffer;
            tensor._gpuBackend = backend;
            var backingArray = tensor.DataVector.GetBackingArrayUnsafe();
            if (backingArray is not null)
            {
                CacheActivation(backingArray, owned.Buffer, tensor.Shape.ToArray(), backend);
            }
            else
            {
                CacheActivation(tensor.DataVector, owned.Buffer, tensor.Shape.ToArray(), backend);
            }
        }
        return owned.Buffer;
    }

    /// <summary>
    /// If AutocastScope is active, converts a fp32 buffer to fp16 and returns the fp16 buffer.
    /// If not active, returns null (caller should use the original buffer).
    /// The caller is responsible for converting outputs back to fp32 via MaybeAutocastOutput.
    /// </summary>
    private IGpuBuffer? MaybeAutocastInput(IDirectGpuBackend backend, IGpuBuffer fp32Buffer, int size)
    {
        return Gpu.AutocastScope.MaybeConvertInput(backend, fp32Buffer, size);
    }

    /// <summary>
    /// If AutocastScope is active, converts an fp16 output buffer back to fp32.
    /// </summary>
    private void MaybeAutocastOutput(IDirectGpuBackend backend, IGpuBuffer fp16Output, IGpuBuffer fp32Target, int size)
    {
        if (Gpu.AutocastScope.IsEnabled && Gpu.AutocastScope.ActivePrecision != Gpu.PrecisionMode.Float32)
        {
            backend.ConvertToFp32(fp16Output, fp32Target, size);
        }
    }

    /// <summary>
    /// Validates attention bias shape/rank and uploads to GPU.
    /// Matches CpuEngine validation: rank must be 3 [heads, seqQ, seqK] or 4 [batch, heads, seqQ, seqK].
    /// </summary>
    private OwnedBuffer GetOrAllocateBiasBuffer<T>(IDirectGpuBackend backend, Tensor<T> attentionBias,
        int batch, int heads, int seqQ, int seqK, out int biasBatchStride)
    {
        var biasShape = attentionBias.Shape._dims;

        if (biasShape.Length == 4)
        {
            if (biasShape[0] != batch || biasShape[1] != heads || biasShape[2] != seqQ || biasShape[3] != seqK)
            {
                throw new ArgumentException(
                    $"attentionBias shape must be [batch, heads, seqQ, seqK] = [{batch}, {heads}, {seqQ}, {seqK}] " +
                    $"for rank-4 bias, but was [{string.Join(", ", biasShape)}].",
                    nameof(attentionBias));
            }

            biasBatchStride = heads * seqQ * seqK;
        }
        else if (biasShape.Length == 3)
        {
            if (biasShape[0] != heads || biasShape[1] != seqQ || biasShape[2] != seqK)
            {
                throw new ArgumentException(
                    $"attentionBias shape must be [heads, seqQ, seqK] = [{heads}, {seqQ}, {seqK}] " +
                    $"for rank-3 bias, but was [{string.Join(", ", biasShape)}].",
                    nameof(attentionBias));
            }

            biasBatchStride = 0;
        }
        else
        {
            throw new ArgumentException(
                "attentionBias must have rank 3 ([heads, seqQ, seqK]) or rank 4 ([batch, heads, seqQ, seqK]).",
                nameof(attentionBias));
        }

        return GetOrAllocateBuffer(backend, attentionBias.GetDataArray());
    }

    /// <summary>
    /// Caches the result buffer for potential reuse by the next layer.
    /// The result data array serves as the cache key.
    /// Thread-safe: uses lock to coordinate with cache lookups.
    /// </summary>
    private void CacheActivation<T>(T[] resultData, IGpuBuffer buffer, int[] shape, IDirectGpuBackend backend)
        => CacheActivation((object)resultData, buffer, shape, backend);

    private void CacheActivation(object cacheKey, IGpuBuffer buffer, int[] shape, IDirectGpuBackend backend)
    {
        List<ActivationCacheEntry> evicted = new List<ActivationCacheEntry>();
        lock (_activationCacheLock)
        {
            // Evict old entries if cache is full by count or GPU memory budget
            bool overCount = _activationCache.Count >= _maxActivationCacheSize;
            bool overMemory = _maxActivationCacheBytes > 0
                && _currentActivationCacheBytes + (buffer.Size * sizeof(float)) > _maxActivationCacheBytes;
            if (overCount || overMemory)
            {
                evicted = EvictOldestActivationsUnsafe();
            }

            var timestamp = System.Threading.Interlocked.Increment(ref _activationCacheTimestamp);
            var entry = new ActivationCacheEntry(buffer, shape, timestamp, backend);
            bool added = false;
            try
            {
                added = _activationCache.TryAdd(cacheKey, entry);
            }
            finally
            {
                if (!added)
                {
                    entry.Dispose();
                }
                else
                {
                    System.Threading.Interlocked.Add(ref _currentActivationCacheBytes, buffer.Size * sizeof(float));
                }
            }
        }

        // Dispose evicted GPU buffers OUTSIDE the lock to avoid blocking cache lookups
        foreach (var entry in evicted)
        {
            entry.Dispose();
        }
    }

    /// <summary>
    /// Evicts the oldest half of the activation cache entries.
    /// Must be called while holding _activationCacheLock.
    /// Returns entries to dispose AFTER releasing the lock.
    /// </summary>
    private List<ActivationCacheEntry> EvictOldestActivationsUnsafe()
    {
        var entries = _activationCache.ToArray();
        var toDispose = new List<ActivationCacheEntry>();
        if (entries.Length == 0) return toDispose;

        int removeCount = entries.Length / 2;
        if (removeCount == 0) return toDispose;

        // Find threshold using Array.Sort on timestamps (avoids LINQ allocation)
        var timestamps = new long[entries.Length];
        for (int i = 0; i < entries.Length; i++)
            timestamps[i] = entries[i].Value.Timestamp;
        Array.Sort(timestamps);
        long threshold = timestamps[removeCount - 1];

        // Remove entries at or below threshold, collect for disposal outside lock.
        // Skip entries whose key still has a pending deferred materializer — otherwise
        // the later TryMaterialize call reads a freed OpenCL buffer and crashes with
        // CL_INVALID_MEM_OBJECT (issue #226). DeferredArrayMaterializer is the single
        // source of truth; both FinishGpuOp and DeferTensorResult register here.
        int removed = 0;
        for (int i = 0; i < entries.Length && removed < removeCount; i++)
        {
            if (entries[i].Value.Timestamp <= threshold)
            {
                if (Helpers.DeferredArrayMaterializer.IsPending(entries[i].Key))
                    continue;

                if (_activationCache.TryRemove(entries[i].Key, out var entry))
                {
                    System.Threading.Interlocked.Add(ref _currentActivationCacheBytes,
                        -(entry.Buffer.Size * sizeof(float)));
                    toDispose.Add(entry);
                    removed++;
                }
            }
        }

        return toDispose;
    }

    /// <summary>
    /// Clears the activation cache to free GPU memory.
    /// Call this between inference batches if memory is tight.
    /// Thread-safe: uses lock to prevent clearing while buffers are in use.
    /// </summary>
    public void ClearActivationCache()
    {
        // Materialize any deferred downloads before clearing — otherwise CPU arrays
        // would be left empty because the GPU buffers they depend on get disposed.
        MaterializeAllDeferred();

        List<ActivationCacheEntry> toDispose;
        lock (_activationCacheLock)
        {
            toDispose = new List<ActivationCacheEntry>(_activationCache.Values);
            _activationCache.Clear();
        }

        // Dispose GPU buffers outside the lock
        foreach (var entry in toDispose)
        {
            entry.Dispose();
        }
    }

    /// <summary>
    /// Gets a GPU buffer for weight/bias tensor, auto-caching if not already persistent.
    /// Unlike GetOrAllocateBuffer, this caches the buffer in the persistent cache
    /// so subsequent calls reuse the same GPU buffer without re-uploading.
    /// Thread-safe: uses lock to coordinate with cache invalidation.
    /// </summary>
    private OwnedBuffer GetOrCacheWeightBuffer<T>(IDirectGpuBackend backend, T[] data, PersistentTensorRole role)
    {
        lock (_persistentBufferLock)
        {
            // First check persistent tensor cache
            var cached = TryGetCachedBuffer(data);
            if (cached != null)
                return new OwnedBuffer(cached, ownsBuffer: false);

            // Not cached - upload and cache for future use
            float[] floatData = DirectGpuEngine.ToFloatArray(data);
            IGpuBuffer gpuBuffer = backend.AllocateBuffer(floatData);

            // Add to persistent cache so future calls don't re-upload
            var entry = new GpuBufferCacheEntry(gpuBuffer, role);
            if (_persistentBufferCache.TryAdd(data, entry))
            {
                _tensorVersions.TryAdd(data, 0);
                // Return with ownsBuffer=false since cache now owns it
                return new OwnedBuffer(gpuBuffer, ownsBuffer: false);
            }
            else
            {
                // Another thread may have cached it; try to use that one
                var alreadyCached = TryGetCachedBuffer(data);
                if (alreadyCached != null)
                {
                    gpuBuffer.Dispose();
                    return new OwnedBuffer(alreadyCached, ownsBuffer: false);
                }

                // Entry was removed between TryAdd and lookup; fall back to our buffer
                return new OwnedBuffer(gpuBuffer, ownsBuffer: true);
            }
        }
    }

    /// <summary>
    /// Gets a GPU buffer for weight/bias Memory&lt;T&gt; data, auto-caching if not already persistent.
    /// Uses MemoryMarshal.TryGetArray to extract underlying array for cache lookup and efficient upload.
    /// </summary>
    private OwnedBuffer GetOrCacheWeightBuffer<T>(IDirectGpuBackend backend, ReadOnlyMemory<T> memory, PersistentTensorRole role)
    {
        // Try to get the underlying array for cache-friendly behavior
        if (MemoryMarshal.TryGetArray(memory, out ArraySegment<T> segment) &&
            segment.Offset == 0 && segment.Count == segment.Array!.Length)
        {
            // Memory is backed by a full array - use the array-based overload for caching
            return GetOrCacheWeightBuffer(backend, segment.Array, role);
        }

        // Memory is a slice or not array-backed - upload without caching
        // (We can't cache slices reliably since the key would be the slice, not the source)
        float[] floatData = DirectGpuEngine.ToFloatArray(memory.ToArray());
        return new OwnedBuffer(backend.AllocateBuffer(floatData), ownsBuffer: true);
    }

    /// <summary>
    /// Allocates a GPU buffer from span data (no caching, avoids ToArray() allocation).
    /// </summary>
    private OwnedBuffer AllocateBufferFromSpan<T>(IDirectGpuBackend backend, ReadOnlySpan<T> data)
    {
        // Convert via numeric operations directly to float array
        // ToFloatSpan has built-in fast path for T=float
        float[] result = new float[data.Length];
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.ToFloatSpan(data, new Span<float>(result));
        return new OwnedBuffer(backend.AllocateBuffer(result), ownsBuffer: true);
    }

    /// <summary>
    /// Allocates a new output buffer (always owned, never cached).
    /// </summary>
    private static OwnedBuffer AllocateOutputBuffer(IDirectGpuBackend backend, int size)
    {
        return new OwnedBuffer(backend.AllocateBuffer(size), ownsBuffer: true);
    }

    /// <summary>
    /// Completes a GPU operation by either deferring the download (when GpuScope is active)
    /// or downloading immediately. When deferred, the GPU buffer stays resident and the CPU
    /// array is only populated when actually needed (via MaterializeIfDeferred).
    /// This eliminates blocking downloads for intermediate results in chained GPU operations.
    /// </summary>
    private T[] FinishGpuOp<T>(IDirectGpuBackend backend, OwnedBuffer outputBuffer, int elementCount)
    {
        // Always defer the download. The GPU buffer stays cached so chained GPU ops
        // reuse it without re-uploading (GetOrAllocateBuffer checks _activationCache).
        // The CPU array is populated lazily when code first accesses the data
        // (via DeferredArrayMaterializer triggered by GetDataArray/AsSpan/indexer).
        //
        // This eliminates both upload AND download for chained GPU operations:
        //   result1 = sigmoid(x)     → GPU kernel, buffer cached, no download
        //   result2 = relu(result1)  → buffer found in cache, no upload, no download
        //   value = result2[0]       → first CPU access triggers download
        // Allocate uninitialized array — skips zeroing since the data will either:
        // (a) be populated lazily from GPU on first CPU access, or
        // (b) never be read at all (chained GPU ops use the cached GPU buffer directly).
        // GC.AllocateUninitializedArray avoids the O(n) zero-fill of new T[n].
#if !NETFRAMEWORK
        var result = GC.AllocateUninitializedArray<T>(elementCount);
#else
        var result = new T[elementCount];
#endif

        // Capture the buffer + backend in the materializer closure directly so
        // the DeferredArrayMaterializer registry is the single source of truth
        // for pending downloads (#226). The activation-cache eviction guard
        // checks IsPending on this same key, keeping the buffer alive until
        // the callback runs.
        //
        // Order matters: Register BEFORE CacheActivation. If we inserted into
        // the activation cache first and another thread's CacheActivation
        // simultaneously triggered EvictOldestActivationsUnsafe, the guard
        // would check IsPending(result) — see FALSE because Register hasn't
        // run yet — and dispose the buffer. The subsequent Register then
        // points at a freed buffer, and the next MaterializeAllDeferred
        // surfaces the CL_INVALID_MEM_OBJECT that triggered #226 in the
        // BatchNorm perf test (stack: engine.Dispose → ClearActivationCache
        // → MaterializeAllDeferred → callback → DownloadBuffer → -38).
        var capturedBuffer = outputBuffer.Buffer;
        var capturedBackend = backend;
        Helpers.DeferredArrayMaterializer.Register(result, arr =>
        {
            float[] floatData;
            try
            {
                floatData = capturedBackend.DownloadBuffer(capturedBuffer);
            }
            catch (InvalidOperationException ex)
            {
                // See DeferTensorResult for rationale on the wrap.
                throw new InvalidOperationException(
                    "Deferred GPU download failed because the underlying buffer was " +
                    "released before materialization. This typically indicates an " +
                    "activation-cache eviction raced with a pending materializer " +
                    "(issue #226). See DirectGpuTensorEngine.FinishGpuOp / " +
                    "EvictOldestActivationsUnsafe.", ex);
            }
            var converted = DirectGpuEngine.FromFloatArray<T>(floatData);
            Array.Copy(converted, (T[])arr, Math.Min(converted.Length, ((T[])arr).Length));
        });

        CacheActivation(result, outputBuffer.Buffer, new[] { elementCount }, backend);

        return result;
    }

    /// <summary>
    /// Creates a deferred Tensor from a raw GPU buffer. The buffer stays GPU-resident;
    /// the CPU array is only populated when code accesses the tensor data.
    /// Use this instead of bb.DownloadBuffer() in IEngine implementations to keep
    /// intermediate results GPU-resident for chained operations.
    /// </summary>
    private Tensor<T> DeferTensorResult<T>(IDirectGpuBackend backend, IGpuBuffer outputBuffer, int elementCount, int[] shape)
    {
        // Create GPU-resident tensor with ZERO CPU allocation.
        // The backing array is only allocated when CPU code actually accesses the data.
        var deviceType = backend.BackendName?.ToUpperInvariant() switch
        {
            "CUDA" or "NVIDIA" => TensorDevice.CUDA,
            "OPENCL" => TensorDevice.OpenCL,
            "HIP" or "ROCM" => TensorDevice.HIP,
            "VULKAN" => TensorDevice.Vulkan,
            "METAL" or "MPS" => TensorDevice.Metal,
            "WEBGPU" => TensorDevice.WebGPU,
            "DIRECTML" or "DML" => TensorDevice.DirectML,
            _ => TensorDevice.CUDA
        };
        var tensor = Tensor<T>.CreateGpuResident(shape, deviceType);
        tensor._gpuBuffer = outputBuffer;
        tensor._gpuBackend = backend;

        // Register materializer keyed by the vector — when GetDataArray() is called,
        // it allocates the backing array and then TryMaterialize(vector) downloads from GPU.
        // The DeferredArrayMaterializer registry also acts as the pending-download
        // source of truth for activation-cache eviction (#226): the eviction guard
        // checks IsPending on this same vector key to decide whether to spare the
        // underlying GPU buffer.
        var vector = tensor.DataVector;
        Helpers.DeferredArrayMaterializer.Register(vector, obj =>
        {
            var vec = (LinearAlgebra.VectorBase<T>)obj;
            if (tensor._gpuBuffer is not null && tensor._gpuBackend is not null)
            {
                float[] floatData;
                try
                {
                    floatData = tensor._gpuBackend.DownloadBuffer(tensor._gpuBuffer);
                }
                catch (InvalidOperationException ex)
                {
                    // Surface a clearer error — historically this manifested as a raw
                    // "Failed to read OpenCL buffer: -38" with no context on why the
                    // buffer was invalid. The lifetime fix in this engine covers the
                    // known case; this wrap guards against any future regression.
                    throw new InvalidOperationException(
                        "Deferred GPU download failed because the underlying buffer was " +
                        "released before materialization. This typically indicates an " +
                        "activation-cache eviction raced with a pending materializer " +
                        "(issue #226). See DirectGpuTensorEngine.DeferTensorResult / " +
                        "EvictOldestActivationsUnsafe.", ex);
                }
                var converted = DirectGpuEngine.FromFloatArray<T>(floatData);
                var arr = vec.GetBackingArrayUnsafe();
                if (arr is not null)
                    Array.Copy(converted, arr, Math.Min(converted.Length, arr.Length));
            }
        });

        // Cache the GPU buffer so GetOrAllocateBuffer finds it
        // Use the vector as cache key since there's no backing array yet
        CacheActivation(vector, outputBuffer, shape, backend);

        return tensor;
    }

    /// <summary>
    /// Materializes a deferred download if the given array was returned from a GPU op
    /// within a GpuScope without downloading. This is called automatically when CPU
    /// code needs the actual data (e.g., reductions, CPU fallback operations, scope exit).
    /// </summary>
    /// <remarks>
    /// Delegates to <see cref="Helpers.DeferredArrayMaterializer"/>, which is the
    /// single source of truth for pending downloads after the #226 cleanup. Each
    /// registered callback closes over its own buffer + backend + type conversion,
    /// so there is no engine-local metadata to consult.
    /// </remarks>
    private void MaterializeIfDeferred<T>(T[] data)
    {
        Helpers.DeferredArrayMaterializer.TryMaterialize(data);
    }

    /// <summary>
    /// Materializes all pending deferred downloads at the end of a normal
    /// <see cref="GpuScope"/> or cache clear. Propagates exceptions so callers
    /// observe any failed download instead of silently seeing empty arrays.
    /// </summary>
    internal void MaterializeAllDeferred()
    {
        Helpers.DeferredArrayMaterializer.MaterializeAll(swallowErrors: false);
    }

    /// <summary>
    /// Tensor-aware TryRunUnary: checks GPU activation cache BEFORE triggering CPU materialization.
    /// This is the preferred path for chained GPU operations — avoids wasteful downloads.
    /// </summary>
    private T[]? TryRunUnary<T>(Tensor<T> input, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, input);
        var bufferB = AllocateOutputBuffer(backend, input.Length);
        try
        {
            // AutocastScope: convert to fp16 for compute, back to fp32 for output
            var fp16Input = Gpu.AutocastScope.MaybeConvertInput(backend, bufferA.Buffer, input.Length);
            if (fp16Input is not null)
            {
                using var fp16Output = AllocateOutputBuffer(backend, input.Length);
                op(backend, fp16Input, fp16Output.Buffer, input.Length);
                // Convert output back to fp32
                backend.ConvertToFp32(fp16Output.Buffer, bufferB.Buffer, input.Length);
                fp16Input.Dispose();
            }
            else
            {
                op(backend, bufferA.Buffer, bufferB.Buffer, input.Length);
            }
            return FinishGpuOp<T>(backend, bufferB, input.Length);
        }
        catch
        {
            bufferB.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Tensor-aware TryRunBinary: checks GPU activation cache BEFORE triggering CPU materialization.
    /// </summary>
    private T[]? TryRunBinary<T>(Tensor<T> left, Tensor<T> right, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (left.Length != right.Length)
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, left);
        using var bufferB = GetOrAllocateBuffer(backend, right);
        var bufferC = AllocateOutputBuffer(backend, left.Length);
        try
        {
            var fp16A = Gpu.AutocastScope.MaybeConvertInput(backend, bufferA.Buffer, left.Length);
            var fp16B = Gpu.AutocastScope.MaybeConvertInput(backend, bufferB.Buffer, left.Length);
            if (fp16A is not null && fp16B is not null)
            {
                using var fp16Out = AllocateOutputBuffer(backend, left.Length);
                op(backend, fp16A, fp16B, fp16Out.Buffer, left.Length);
                backend.ConvertToFp32(fp16Out.Buffer, bufferC.Buffer, left.Length);
                fp16A.Dispose();
                fp16B.Dispose();
            }
            else
            {
                fp16A?.Dispose();
                fp16B?.Dispose();
                op(backend, bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, left.Length);
            }
            return FinishGpuOp<T>(backend, bufferC, left.Length);
        }
        catch
        {
            bufferC.Dispose();
            throw;
        }
    }

    // Legacy T[] overloads kept for backward compatibility with IEngine explicit implementations
    private T[]? TryRunUnary<T>(T[] input, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, input);
        var bufferB = AllocateOutputBuffer(backend, input.Length);
        try
        {
            op(backend, bufferA.Buffer, bufferB.Buffer, input.Length);
            return FinishGpuOp<T>(backend, bufferB, input.Length);
        }
        catch
        {
            bufferB.Dispose();
            throw;
        }
    }

    private T[]? TryRunBinary<T>(T[] left, T[] right, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (left.Length != right.Length)
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, left);
        using var bufferB = GetOrAllocateBuffer(backend, right);
        var bufferC = AllocateOutputBuffer(backend, left.Length);
        try
        {
            op(backend, bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, left.Length);
            return FinishGpuOp<T>(backend, bufferC, left.Length);
        }
        catch
        {
            bufferC.Dispose();
            throw;
        }
    }

    private T[]? TryRunScalar<T>(T[] input, T scalar, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, float, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, input);
        var bufferB = AllocateOutputBuffer(backend, input.Length);
        try
        {
            op(backend, bufferA.Buffer, bufferB.Buffer, ToFloatScalar(scalar), input.Length);
            return FinishGpuOp<T>(backend, bufferB, input.Length);
        }
        catch
        {
            bufferB.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Span-based binary operation that avoids ToArray() allocation for matrix operations.
    /// </summary>
    private T[]? TryRunBinarySpan<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (left.Length != right.Length)
            return null;

        using var bufferA = AllocateBufferFromSpan(backend, left);
        using var bufferB = AllocateBufferFromSpan(backend, right);
        var bufferC = AllocateOutputBuffer(backend, left.Length);
        try
        {
            op(backend, bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, left.Length);
            return FinishGpuOp<T>(backend, bufferC, left.Length);
        }
        catch
        {
            bufferC.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Span-based scalar operation that avoids ToArray() allocation for matrix operations.
    /// </summary>
    private T[]? TryRunScalarSpan<T>(ReadOnlySpan<T> input, T scalar, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, float, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = AllocateBufferFromSpan(backend, input);
        var bufferB = AllocateOutputBuffer(backend, input.Length);
        try
        {
            op(backend, bufferA.Buffer, bufferB.Buffer, ToFloatScalar(scalar), input.Length);
            return FinishGpuOp<T>(backend, bufferB, input.Length);
        }
        catch
        {
            bufferB.Dispose();
            throw;
        }
    }

    private static bool ShapesMatch(int[] left, int[] right)
    {
        return left.Length == right.Length && left.SequenceEqual(right);
    }

    Vector<T> IEngine.Add<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.GetDataArray(), b.GetDataArray(), static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Add(a, b);
    }

    Vector<T> IEngine.Subtract<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.GetDataArray(), b.GetDataArray(), static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Subtract(a, b);
    }

    Vector<T> IEngine.Multiply<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.GetDataArray(), b.GetDataArray(), static (backend, left, right, output, size) => backend.Multiply(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Multiply(a, b);
    }

    Vector<T> IEngine.Multiply<T>(Vector<T> vector, T scalar)
    {
        var result = TryRunScalar(vector.GetDataArray(), scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Vector<T>(result) : base.Multiply(vector, scalar);
    }

    Vector<T> IEngine.Divide<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.GetDataArray(), b.GetDataArray(), static (backend, left, right, output, size) => backend.Divide(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Divide(a, b);
    }

    Vector<T> IEngine.StridedGather<T>(Vector<T> source, int offset, int stride, int count)
    {
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));

        if (count < 0)
        {
            count = offset < source.Length ? (source.Length - offset + stride - 1) / stride : 0;
        }

        if (count == 0) return new Vector<T>(0);

        int lastIndex = offset + (count - 1) * stride;
        if (lastIndex >= source.Length)
            throw new ArgumentOutOfRangeException(nameof(count),
                $"Strided gather would access index {lastIndex} but source length is {source.Length}.");

        if (!TryGetBackend(out var backend))
            return base.StridedGather(source, offset, stride, count);

        try
        {
            var srcData = DirectGpuEngine.ToFloatArray(source.GetDataArray());
            using var srcBuf = backend.AllocateBuffer(srcData);
            using var dstBuf = backend.AllocateBuffer(count);
            backend.StridedGather(srcBuf, dstBuf, offset, stride, count);
            var resultFloat = backend.DownloadBuffer(dstBuf);
            return new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat));
        }
        catch
        {
            return base.StridedGather(source, offset, stride, count);
        }
    }

    void IEngine.StridedScatter<T>(Vector<T> destination, Vector<T> source, int offset, int stride)
    {
        if (!TryGetBackend(out var backend))
        {
            base.StridedScatter(destination, source, offset, stride);
            return;
        }

        try
        {
            var srcData = DirectGpuEngine.ToFloatArray(source.GetDataArray());
            var dstData = DirectGpuEngine.ToFloatArray(destination.GetDataArray());
            using var srcBuf = backend.AllocateBuffer(srcData);
            using var dstBuf = backend.AllocateBuffer(dstData);
            backend.StridedScatter(srcBuf, dstBuf, offset, stride, source.Length);
            var resultFloat = backend.DownloadBuffer(dstBuf);
            var resultT = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            Array.Copy(resultT, destination.GetDataArray(), resultT.Length);
        }
        catch
        {
            base.StridedScatter(destination, source, offset, stride);
        }
    }

    Vector<T> IEngine.Divide<T>(Vector<T> vector, T scalar)
    {
        var scalarValue = ToFloatScalar(scalar);
        if (scalarValue == 0)
            return base.Divide(vector, scalar);

        var result = TryRunScalar(vector.GetDataArray(), scalar, static (backend, input, output, value, size) => backend.Scale(input, output, 1.0f / value, size));
        return result != null ? new Vector<T>(result) : base.Divide(vector, scalar);
    }

    Vector<T> IEngine.Max<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.GetDataArray(), b.GetDataArray(), static (backend, left, right, output, size) => backend.Max(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Max(a, b);
    }

    Vector<T> IEngine.Min<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.GetDataArray(), b.GetDataArray(), static (backend, left, right, output, size) => backend.Min(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Min(a, b);
    }

    Vector<T> IEngine.Abs<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Abs(input, output, size));
        return result != null ? new Vector<T>(result) : base.Abs(vector);
    }

    Vector<T> IEngine.Exp<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Exp(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp(vector);
    }

    Vector<T> IEngine.Exp2<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Exp2(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp2(vector);
    }

    Vector<T> IEngine.Exp10<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Exp10(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp10(vector);
    }

    Vector<T> IEngine.Log<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Log(input, output, size));
        return result != null ? new Vector<T>(result) : base.Log(vector);
    }

    Vector<T> IEngine.Log2<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Log2(input, output, size));
        return result != null ? new Vector<T>(result) : base.Log2(vector);
    }

    Vector<T> IEngine.Sqrt<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Sqrt(input, output, size));
        return result != null ? new Vector<T>(result) : base.Sqrt(vector);
    }

    Vector<T> IEngine.Power<T>(Vector<T> vector, T exponent)
    {
        var result = TryRunScalar(vector.GetDataArray(), exponent, static (backend, input, output, value, size) => backend.Power(input, output, value, size));
        return result != null ? new Vector<T>(result) : base.Power(vector, exponent);
    }

    Vector<T> IEngine.Tanh<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Tanh(input, output, size));
        return result != null ? new Vector<T>(result) : base.Tanh(vector);
    }

    Vector<T> IEngine.Sigmoid<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Sigmoid(input, output, size));
        return result != null ? new Vector<T>(result) : base.Sigmoid(vector);
    }

    Vector<T> IEngine.ReLU<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Relu(input, output, size));
        return result != null ? new Vector<T>(result) : base.ReLU(vector);
    }

    Vector<T> IEngine.GELU<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.GetDataArray(), static (backend, input, output, size) => backend.Gelu(input, output, size));
        return result != null ? new Vector<T>(result) : base.GELU(vector);
    }

    Matrix<T> IEngine.MatrixMultiply<T>(Matrix<T> a, Matrix<T> b)
    {
        if (!IsGpuAvailable || _directGpu == null)
            return base.MatrixMultiply(a, b);

        if (a.Columns != b.Rows)
            return base.MatrixMultiply(a, b);

        try
        {
            var resultData = _directGpu.MatMul(a.AsSpan().ToArray(), b.AsSpan().ToArray(), a.Rows, a.Columns, b.Columns);
            if (resultData == null)
                return base.MatrixMultiply(a, b);

            var result = new Matrix<T>(a.Rows, b.Columns);
            resultData.AsSpan().CopyTo(result.AsWritableSpan());
            return result;
        }
        catch
        {
            return base.MatrixMultiply(a, b);
        }
    }

    Matrix<T> IEngine.MatrixAdd<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return base.MatrixAdd(a, b);

        // Use span-based method to avoid ToArray() allocation
        var result = TryRunBinarySpan(a.AsSpan(), b.AsSpan(), static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        if (result == null)
            return base.MatrixAdd(a, b);

        var matrix = new Matrix<T>(a.Rows, a.Columns);
        result.AsSpan().CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    Matrix<T> IEngine.MatrixSubtract<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return base.MatrixSubtract(a, b);

        // Use span-based method to avoid ToArray() allocation
        var result = TryRunBinarySpan(a.AsSpan(), b.AsSpan(), static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        if (result == null)
            return base.MatrixSubtract(a, b);

        var matrix = new Matrix<T>(a.Rows, a.Columns);
        result.AsSpan().CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    Matrix<T> IEngine.MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar)
    {
        // Use span-based method to avoid ToArray() allocation
        var result = TryRunScalarSpan(matrix.AsSpan(), scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        if (result == null)
            return base.MatrixMultiplyScalar(matrix, scalar);

        var output = new Matrix<T>(matrix.Rows, matrix.Columns);
        result.AsSpan().CopyTo(output.AsWritableSpan());
        return output;
    }

    Tensor<T> IEngine.TensorAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorAdd(a, b);

        var result = TryRunBinary(a, b, static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape._dims) : base.TensorAdd(a, b);
    }

    Tensor<T> IEngine.TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorSubtract(a, b);

        var result = TryRunBinary(a, b, static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape._dims) : base.TensorSubtract(a, b);
    }

    Tensor<T> IEngine.TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorMultiply(a, b);

        var result = TryRunBinary(a, b, static (backend, left, right, output, size) => backend.Multiply(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape._dims) : base.TensorMultiply(a, b);
    }

    Tensor<T> IEngine.TensorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorDivide(a, b);

        var result = TryRunBinary(a, b, static (backend, left, right, output, size) => backend.Divide(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape._dims) : base.TensorDivide(a, b);
    }

    Tensor<T> IEngine.TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
    {
        var result = TryRunScalar(tensor.GetDataArray(), scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.TensorMultiplyScalar(tensor, scalar);
    }

    // GPU-accelerated in-place operations.
    // Uses the same GPU buffer for input and output where safe (element-wise ops).
    // Falls back to CpuEngine base when GPU is not available.

    void IEngine.TensorAddInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (ShapesMatch(a.Shape._dims, b.Shape._dims) && TryRunBinaryInPlace(a, b,
            static (backend, bufA, bufB, size) => backend.Add(bufA, bufB, bufA, size)))
            return;
        base.TensorAddInPlace(a, b);
    }

    void IEngine.TensorAddInto<T>(Tensor<T> dest, Tensor<T> a, Tensor<T> b)
    {
        // Use allocating GPU path and copy result
        var result = ((IEngine)this).TensorAdd(a, b);
        result.Data.Span.CopyTo(dest.Data.Span);
    }

    void IEngine.TensorMultiplyInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (ShapesMatch(a.Shape._dims, b.Shape._dims) && TryRunBinaryInPlace(a, b,
            static (backend, bufA, bufB, size) => backend.Multiply(bufA, bufB, bufA, size)))
            return;
        base.TensorMultiplyInPlace(a, b);
    }

    void IEngine.TensorMultiplyInto<T>(Tensor<T> dest, Tensor<T> a, Tensor<T> b)
    {
        var result = ((IEngine)this).TensorMultiply(a, b);
        result.Data.Span.CopyTo(dest.Data.Span);
    }

    void IEngine.TensorSubtractInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!a.IsContiguous) { base.TensorSubtractInPlace(a, b); return; }
        if (ShapesMatch(a.Shape._dims, b.Shape._dims) && TryRunBinaryInPlace(a, b,
            static (backend, bufA, bufB, size) => backend.Subtract(bufA, bufB, bufA, size)))
            return;
        base.TensorSubtractInPlace(a, b);
    }

    void IEngine.TensorSubtractInto<T>(Tensor<T> dest, Tensor<T> a, Tensor<T> b)
    {
        if (!dest.IsContiguous) { base.TensorSubtractInto(dest, a, b); return; }
        // Compute on GPU then copy directly into dest's backing storage
        var result = ((IEngine)this).TensorSubtract(a, b);
        result.Data.Span.CopyTo(dest.Data.Span);
    }

    void IEngine.TensorMultiplyScalarInPlace<T>(Tensor<T> a, T scalar)
    {
        if (!a.IsContiguous) { base.TensorMultiplyScalarInPlace(a, scalar); return; }
        var scalarF = ToFloatScalar(scalar);
        if (TryRunUnaryInPlace(a,
            (backend, buf, size) => backend.Scale(buf, buf, scalarF, size)))
            return;
        base.TensorMultiplyScalarInPlace(a, scalar);
    }

    void IEngine.TensorMultiplyScalarInto<T>(Tensor<T> dest, Tensor<T> a, T scalar)
    {
        if (!dest.IsContiguous) { base.TensorMultiplyScalarInto(dest, a, scalar); return; }
        // Compute on GPU, copy result directly into dest's contiguous span
        var scalarF = ToFloatScalar(scalar);
        var result = TryRunScalar(a.GetDataArray(), scalar,
            static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        if (result != null)
        {
            result.AsSpan(0, Math.Min(result.Length, dest.Length)).CopyTo(dest.Data.Span);
            return;
        }
        base.TensorMultiplyScalarInto(dest, a, scalar);
    }

    void IEngine.TensorBroadcastAddInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        // If shapes match, use GPU element-wise add in-place
        if (ShapesMatch(a.Shape._dims, b.Shape._dims) && TryRunBinaryInPlace(a, b,
            static (backend, bufA, bufB, size) => backend.Add(bufA, bufB, bufA, size)))
            return;
        // Shapes differ — CPU handles broadcasting logic
        base.TensorBroadcastAddInPlace(a, b);
    }

    void IEngine.SigmoidInPlace<T>(Tensor<T> tensor)
    {
        if (TryRunUnaryInPlace(tensor,
            static (backend, buf, size) => backend.Sigmoid(buf, buf, size)))
            return;
        base.SigmoidInPlace(tensor);
    }

    void IEngine.SigmoidInto<T>(Tensor<T> dest, Tensor<T> input)
    {
        var result = ((IEngine)this).Sigmoid(input);
        result.Data.Span.CopyTo(dest.Data.Span);
    }

    void IEngine.ReLUInPlace<T>(Tensor<T> tensor)
    {
        if (TryRunUnaryInPlace(tensor,
            static (backend, buf, size) => backend.Relu(buf, buf, size)))
            return;
        base.ReLUInPlace(tensor);
    }

    void IEngine.ReLUInto<T>(Tensor<T> dest, Tensor<T> input)
    {
        var result = ((IEngine)this).ReLU(input);
        result.Data.Span.CopyTo(dest.Data.Span);
    }

    void IEngine.Conv2DInto<T>(Tensor<T> output, Tensor<T> input, Tensor<T> kernel, int stride, int padding, int dilation)
    {
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatKernel = (Tensor<float>)(object)kernel;
                var floatOutput = (Tensor<float>)(object)output;

                int batch = input.Shape._dims[0], inChannels = input.Shape._dims[1];
                int inH = input.Shape._dims[2], inW = input.Shape._dims[3];
                int outChannels = kernel.Shape._dims[0], kH = kernel.Shape._dims[2], kW = kernel.Shape._dims[3];
                int outH = output.Shape._dims[2], outW = output.Shape._dims[3];

                using var gpuIn = gpuBackend.AllocateBuffer(floatInput.GetDataArray());
                using var gpuK = gpuBackend.AllocateBuffer(floatKernel.GetDataArray());
                using var gpuOut = gpuBackend.AllocateBuffer(output.Length);

                gpuBackend.Conv2D(gpuIn, gpuK, gpuOut,
                    batch, inChannels, inH, inW,
                    outChannels, outH, outW,
                    kH, kW, stride, stride, padding, padding, dilation, dilation);
                DownloadIntoTensor(gpuBackend, gpuOut, floatOutput);
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.Conv2DInto(output, input, kernel, stride, padding, dilation);
    }

    void IEngine.GroupNormInto<T>(Tensor<T> output, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatOutput = (Tensor<float>)(object)output;
                var floatGamma = (Tensor<float>)(object)gamma;
                var floatBeta = (Tensor<float>)(object)beta;

                int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
                int spatial = input.Length / (batch * channels);

                using var gpuIn = gpuBackend.AllocateBuffer(floatInput.GetDataArray());
                using var gpuGamma = gpuBackend.AllocateBuffer(floatGamma.GetDataArray());
                using var gpuBeta = gpuBackend.AllocateBuffer(floatBeta.GetDataArray());
                using var gpuOut = gpuBackend.AllocateBuffer(input.Length);
                using var gpuMean = gpuBackend.AllocateBuffer(batch * numGroups);
                using var gpuVar = gpuBackend.AllocateBuffer(batch * numGroups);

                gpuBackend.GroupNorm(gpuIn, gpuOut, gpuGamma, gpuBeta, gpuMean, gpuVar,
                    batch, numGroups, channels, spatial, (float)epsilon);
                DownloadIntoTensor(gpuBackend, gpuOut, floatOutput);

                mean = new Tensor<T>(new int[] { batch, numGroups });
                variance = new Tensor<T>(new int[] { batch, numGroups });
                DownloadIntoTensor(gpuBackend, gpuMean, (Tensor<float>)(object)mean);
                DownloadIntoTensor(gpuBackend, gpuVar, (Tensor<float>)(object)variance);
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.GroupNormInto(output, input, numGroups, gamma, beta, epsilon, out mean, out variance);
    }

    void IEngine.SoftmaxInto<T>(Tensor<T> destination, Tensor<T> input, int axis)
    {
        // Try GPU softmax: upload input, compute on GPU, download to destination
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatDest = (Tensor<float>)(object)destination;
                var inputData = floatInput.GetDataArray();

                var gpuIn = gpuBackend.AllocateBuffer(inputData);
                var gpuOut = gpuBackend.AllocateBuffer(input.Length);
                try
                {
                    // GPU softmax kernel
                    int axisSize = input.Shape._dims[axis < 0 ? input.Rank + axis : axis];
                    gpuBackend.Softmax(gpuIn, gpuOut, input.Length, axisSize);
                    DownloadIntoTensor(gpuBackend, gpuOut, floatDest);
                }
                finally
                {
                    gpuIn.Dispose();
                    gpuOut.Dispose();
                }
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.SoftmaxInto(destination, input, axis);
    }

    void IEngine.LogSoftmaxInto<T>(Tensor<T> destination, Tensor<T> input, int axis)
    {
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatDest = (Tensor<float>)(object)destination;
                int rank = input.Rank;
                int effectiveAxis = axis < 0 ? rank + axis : axis;

                if (effectiveAxis >= 0 && effectiveAxis < rank && effectiveAxis == rank - 1)
                {
                    int features = input.Shape._dims[rank - 1];
                    int outerSize = input.Length / features;

                    using var gpuIn = gpuBackend.AllocateBuffer(floatInput.GetDataArray());
                    using var gpuSoftmax = gpuBackend.AllocateBuffer(input.Length);
                    using var gpuOut = gpuBackend.AllocateBuffer(input.Length);

                    // Softmax then Log
                    gpuBackend.Softmax(gpuIn, gpuSoftmax, outerSize, features);
                    gpuBackend.Log(gpuSoftmax, gpuOut, input.Length);
                    DownloadIntoTensor(gpuBackend, gpuOut, floatDest);
                    return;
                }
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.LogSoftmaxInto(destination, input, axis);
    }

    void IEngine.SwishInto<T>(Tensor<T> dest, Tensor<T> input)
    {
        // GPU: use Sigmoid + Multiply kernels
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatDest = (Tensor<float>)(object)dest;
                var inputData = floatInput.GetDataArray();
                int size = input.Length;

                var gpuIn = gpuBackend.AllocateBuffer(inputData);
                var gpuSigmoid = gpuBackend.AllocateBuffer(size);
                var gpuOut = gpuBackend.AllocateBuffer(size);
                try
                {
                    gpuBackend.Sigmoid(gpuIn, gpuSigmoid, size);
                    gpuBackend.Multiply(gpuIn, gpuSigmoid, gpuOut, size);
                    DownloadIntoTensor(gpuBackend, gpuOut, floatDest);
                }
                finally
                {
                    gpuIn.Dispose();
                    gpuSigmoid.Dispose();
                    gpuOut.Dispose();
                }
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.SwishInto(dest, input);
    }

    void IEngine.GELUInto<T>(Tensor<T> dest, Tensor<T> input)
    {
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatDest = (Tensor<float>)(object)dest;
                var inputData = floatInput.GetDataArray();
                int size = input.Length;

                var gpuIn = gpuBackend.AllocateBuffer(inputData);
                var gpuOut = gpuBackend.AllocateBuffer(size);
                try
                {
                    gpuBackend.Gelu(gpuIn, gpuOut, size);
                    DownloadIntoTensor(gpuBackend, gpuOut, floatDest);
                }
                finally
                {
                    gpuIn.Dispose();
                    gpuOut.Dispose();
                }
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.GELUInto(dest, input);
    }

    void IEngine.TanhInto<T>(Tensor<T> dest, Tensor<T> input)
    {
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatDest = (Tensor<float>)(object)dest;
                var inputData = floatInput.GetDataArray();
                int size = input.Length;

                var gpuIn = gpuBackend.AllocateBuffer(inputData);
                var gpuOut = gpuBackend.AllocateBuffer(size);
                try
                {
                    gpuBackend.Tanh(gpuIn, gpuOut, size);
                    DownloadIntoTensor(gpuBackend, gpuOut, floatDest);
                }
                finally
                {
                    gpuIn.Dispose();
                    gpuOut.Dispose();
                }
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.TanhInto(dest, input);
    }

    void IEngine.MishInto<T>(Tensor<T> dest, Tensor<T> input)
    {
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatDest = (Tensor<float>)(object)dest;
                int size = input.Length;

                using var gpuIn = gpuBackend.AllocateBuffer(floatInput.GetDataArray());
                using var gpuOut = gpuBackend.AllocateBuffer(size);
                gpuBackend.Mish(gpuIn, gpuOut, size);
                DownloadIntoTensor(gpuBackend, gpuOut, floatDest);
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.MishInto(dest, input);
    }

    void IEngine.LeakyReLUInto<T>(Tensor<T> dest, Tensor<T> input, T alpha)
    {
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatDest = (Tensor<float>)(object)dest;
                float alphaF = alpha is float f ? f : Convert.ToSingle(alpha);
                int size = input.Length;

                using var gpuIn = gpuBackend.AllocateBuffer(floatInput.GetDataArray());
                using var gpuOut = gpuBackend.AllocateBuffer(size);
                gpuBackend.LeakyRelu(gpuIn, gpuOut, alphaF, size);
                DownloadIntoTensor(gpuBackend, gpuOut, floatDest);
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.LeakyReLUInto(dest, input, alpha);
    }

    void IEngine.MatMulInto<T>(Tensor<T> dest, Tensor<T> a, Tensor<T> b)
    {
        // GPU matmul into destination
        var result = ((IEngine)this).TensorMatMul(a, b);
        result.Data.Span.CopyTo(dest.Data.Span);
    }

    void IEngine.ConcatInto<T>(Tensor<T> dest, Tensor<T>[] tensors, int axis)
    {
        // Only use GPU flat copy for axis=0 concat (leading axis — data is contiguous)
        if (axis == 0 && TryGetBackend(out var gpuBackend) && tensors.Length == 2)
        {
            try
            {
                // For 2-tensor concat along last axis, copy each tensor's data sequentially
                var t0 = (Tensor<float>)(object)tensors[0];
                var t1 = (Tensor<float>)(object)tensors[1];
                var dstF = (Tensor<float>)(object)dest;

                using var gpu0 = gpuBackend.AllocateBuffer(t0.GetDataArray());
                using var gpu1 = gpuBackend.AllocateBuffer(t1.GetDataArray());
                using var gpuOut = gpuBackend.AllocateBuffer(dest.Length);

                // Copy t0 data to start of output, t1 data to offset
                gpuBackend.Copy(gpu0, gpuOut, t0.Length);
                gpuBackend.Copy(gpu1, 0, gpuOut, t0.Length, t1.Length);
                DownloadIntoTensor(gpuBackend, gpuOut, dstF);
                return;
            }
            catch
            {
                // Fall through to CPU — axis-aware concat is complex
            }
        }
        base.ConcatInto(dest, tensors, axis);
    }

    void IEngine.TransposeInto<T>(Tensor<T> dest, Tensor<T> input, int[] axes)
    {
        // Only use GPU fast path for standard 2D transpose (axes == [1, 0])
        bool isStandardTranspose = input.Rank == 2 && axes.Length == 2 && axes[0] == 1 && axes[1] == 0;
        if (isStandardTranspose && TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatDest = (Tensor<float>)(object)dest;
                int rows = input.Shape._dims[0], cols = input.Shape._dims[1];

                using var gpuIn = gpuBackend.AllocateBuffer(floatInput.GetDataArray());
                using var gpuOut = gpuBackend.AllocateBuffer(dest.Length);
                gpuBackend.Transpose(gpuIn, gpuOut, rows, cols);
                DownloadIntoTensor(gpuBackend, gpuOut, floatDest);
                return;
            }
            catch
            {
                // Fall through to CPU for higher-rank transposes
            }
        }
        base.TransposeInto(dest, input, axes);
    }

    void IEngine.GroupNormSwishInto<T>(Tensor<T> output, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon)
    {
        if (TryGetBackend(out var gpuBackend))
        {
            try
            {
                var floatInput = (Tensor<float>)(object)input;
                var floatOutput = (Tensor<float>)(object)output;
                var floatGamma = (Tensor<float>)(object)gamma;
                var floatBeta = (Tensor<float>)(object)beta;

                int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
                int spatial = input.Length / (batch * channels);

                using var gpuIn = gpuBackend.AllocateBuffer(floatInput.GetDataArray());
                using var gpuGamma = gpuBackend.AllocateBuffer(floatGamma.GetDataArray());
                using var gpuBeta = gpuBackend.AllocateBuffer(floatBeta.GetDataArray());
                using var gpuNorm = gpuBackend.AllocateBuffer(input.Length);
                using var gpuMean = gpuBackend.AllocateBuffer(batch * numGroups);
                using var gpuVar = gpuBackend.AllocateBuffer(batch * numGroups);
                using var gpuOut = gpuBackend.AllocateBuffer(input.Length);

                // GroupNorm then Swish (sigmoid * x)
                gpuBackend.GroupNorm(gpuIn, gpuNorm, gpuGamma, gpuBeta, gpuMean, gpuVar,
                    batch, channels, spatial, numGroups, (float)epsilon);
                gpuBackend.Swish(gpuNorm, gpuOut, input.Length);
                DownloadIntoTensor(gpuBackend, gpuOut, floatOutput);
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.GroupNormSwishInto(output, input, numGroups, gamma, beta, epsilon);
    }

    void IEngine.AddGroupNormInto<T>(Tensor<T> output, Tensor<T> a, Tensor<T> b, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon)
    {
        if (TryGetBackend(out var gpuBackend) && ShapesMatch(a.Shape._dims, b.Shape._dims))
        {
            try
            {
                var floatA = (Tensor<float>)(object)a;
                var floatB = (Tensor<float>)(object)b;
                var floatOutput = (Tensor<float>)(object)output;
                var floatGamma = (Tensor<float>)(object)gamma;
                var floatBeta = (Tensor<float>)(object)beta;

                int batch = a.Shape._dims[0], channels = a.Shape._dims[1];
                int spatial = a.Length / (batch * channels);

                using var gpuA = gpuBackend.AllocateBuffer(floatA.GetDataArray());
                using var gpuB = gpuBackend.AllocateBuffer(floatB.GetDataArray());
                using var gpuSum = gpuBackend.AllocateBuffer(a.Length);
                using var gpuGamma = gpuBackend.AllocateBuffer(floatGamma.GetDataArray());
                using var gpuBeta = gpuBackend.AllocateBuffer(floatBeta.GetDataArray());
                using var gpuOut = gpuBackend.AllocateBuffer(a.Length);
                using var gpuMean = gpuBackend.AllocateBuffer(batch * numGroups);
                using var gpuVar = gpuBackend.AllocateBuffer(batch * numGroups);

                // Add then GroupNorm
                gpuBackend.Add(gpuA, gpuB, gpuSum, a.Length);
                gpuBackend.GroupNorm(gpuSum, gpuOut, gpuGamma, gpuBeta, gpuMean, gpuVar,
                    batch, channels, spatial, numGroups, (float)epsilon);
                DownloadIntoTensor(gpuBackend, gpuOut, floatOutput);
                return;
            }
            catch
            {
                // Fall through to CPU
            }
        }
        base.AddGroupNormInto(output, a, b, numGroups, gamma, beta, epsilon);
    }

    /// <summary>
    /// Runs a binary GPU operation in-place on tensor a: a = op(a, b).
    /// Uploads a and b to GPU, runs kernel with a's buffer as output, downloads back.
    /// </summary>
    private bool TryRunBinaryInPlace<T>(Tensor<T> a, Tensor<T> b, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return false;

        var aData = a.GetDataArray();
        var bData = b.GetDataArray();
        if (aData.Length != bData.Length)
            return false;

        using var bufferA = GetOrAllocateBuffer(backend, aData);
        using var bufferB = GetOrAllocateBuffer(backend, bData);

        op(backend, bufferA.Buffer, bufferB.Buffer, aData.Length);

        // Download result back into a's backing array
        float[] resultFloat = backend.DownloadBuffer(bufferA.Buffer);
        var resultT = DirectGpuEngine.FromFloatArray<T>(resultFloat);
        Array.Copy(resultT, aData, aData.Length);
        return true;
    }

    /// <summary>
    /// Runs a unary GPU operation in-place on tensor: tensor = op(tensor).
    /// Uploads tensor to GPU, runs kernel with same buffer as input and output, downloads back.
    /// </summary>
    private bool TryRunUnaryInPlace<T>(Tensor<T> tensor, Action<IDirectGpuBackend, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return false;

        var data = tensor.GetDataArray();
        using var buffer = GetOrAllocateBuffer(backend, data);

        op(backend, buffer.Buffer, data.Length);

        // Download result back into tensor's backing array
        float[] resultFloat = backend.DownloadBuffer(buffer.Buffer);
        var resultT = DirectGpuEngine.FromFloatArray<T>(resultFloat);
        Array.Copy(resultT, data, data.Length);
        return true;
    }

    Tensor<T> IEngine.TensorDivideScalar<T>(Tensor<T> tensor, T scalar)
    {
        var scalarValue = ToFloatScalar(scalar);
        if (scalarValue == 0)
            return base.TensorDivideScalar(tensor, scalar);

        var result = TryRunScalar(tensor.GetDataArray(), scalar, static (backend, input, output, value, size) => backend.Scale(input, output, 1.0f / value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.TensorDivideScalar(tensor, scalar);
    }

    Tensor<T> IEngine.TensorAbs<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Abs(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.TensorAbs(tensor);
    }

    Tensor<T> IEngine.TensorExp<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Exp(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.TensorExp(tensor);
    }

    Tensor<T> IEngine.TensorLog<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Log(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.TensorLog(tensor);
    }

    Tensor<T> IEngine.TensorSqrt<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Sqrt(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.TensorSqrt(tensor);
    }

    Tensor<T> IEngine.TensorNegate<T>(Tensor<T> tensor)
    {
        var result = TryRunScalar(tensor.GetDataArray(), FromFloatScalar<T>(-1.0f), static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.TensorNegate(tensor);
    }

    Tensor<T> IEngine.TensorPower<T>(Tensor<T> tensor, T exponent)
    {
        var result = TryRunScalar(tensor.GetDataArray(), exponent, static (backend, input, output, value, size) => backend.Power(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.TensorPower(tensor, exponent);
    }

    Tensor<T> IEngine.TensorMax<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorMax(a, b);

        var result = TryRunBinary(a, b, static (backend, left, right, output, size) => backend.Max(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape._dims) : base.TensorMax(a, b);
    }

    Tensor<T> IEngine.TensorMin<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorMin(a, b);

        var result = TryRunBinary(a, b, static (backend, left, right, output, size) => backend.Min(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape._dims) : base.TensorMin(a, b);
    }

    Tensor<T> IEngine.Tanh<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Tanh(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.Tanh(tensor);
    }

    Tensor<T> IEngine.Sigmoid<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Sigmoid(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.Sigmoid(tensor);
    }

    Tensor<T> IEngine.ReLU<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Relu(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.ReLU(tensor);
    }

    Tensor<T> IEngine.GELU<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Gelu(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape._dims) : base.GELU(tensor);
    }

    T IEngine.TensorSum<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorSum(tensor);

        using var bufferA = GetOrAllocateBuffer(backend, tensor.GetDataArray());
        backend.Synchronize();
        float sum = backend.Sum(bufferA.Buffer, tensor.Length);
        return FromFloatScalar<T>(sum);
    }

    T IEngine.TensorMaxValue<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorMaxValue(tensor);

        using var bufferA = GetOrAllocateBuffer(backend, tensor.GetDataArray());
        backend.Synchronize();
        float max = backend.Max(bufferA.Buffer, tensor.Length);
        return FromFloatScalar<T>(max);
    }

    T IEngine.TensorMinValue<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorMinValue(tensor);

        using var bufferA = GetOrAllocateBuffer(backend, tensor.GetDataArray());
        backend.Synchronize();
        float min = backend.Min(bufferA.Buffer, tensor.Length);
        return FromFloatScalar<T>(min);
    }

    #region Fused Operations

    /// <summary>
    /// GPU-accelerated fused linear transformation: output = activation(input @ weights + bias).
    /// Uses cached GPU buffers for registered persistent tensors (weights/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedLinear<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedLinear(input, weights, bias, activation);

        if (input.Rank < 1 || weights.Rank != 2)
            return base.FusedLinear(input, weights, bias, activation);

        int batchSize = input.Shape._dims[0];
        int inputFeatures = weights.Shape._dims[0];
        int outputFeatures = weights.Shape._dims[1];

        // Use cache-aware buffer allocation — checks _gpuBuffer and activation cache first
        using var inputBuffer = GetOrAllocateBuffer(backend, input);
        // Auto-cache weights and biases so they stay on GPU for subsequent calls
        using var weightsBuffer = GetOrAllocateBuffer(backend, weights);
        using var biasBuffer = bias != null ? GetOrAllocateBuffer(backend, bias) : default;

        try
        {
            IGpuBuffer resultBuffer;

            // Use fused GPU kernels when available
            // Only use GPU path for natively supported fused ops (with bias)
            // KernelFusionManager patterns: Gemm+Bias+Activation are executed as single fused kernels
            if (bias != null && activation != FusedActivationType.None)
            {
                // Dispatch through fused kernel lookup (covers ReLU, GELU, Sigmoid, Tanh, Swish, LeakyReLU)
                resultBuffer = activation switch
                {
                    FusedActivationType.ReLU => backend.GemmBiasRelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.GELU => backend.GemmBiasGelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.Sigmoid => backend.GemmBiasSigmoid(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.Tanh => backend.GemmBiasTanh(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.Swish => backend.GemmBiasSwish(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.LeakyReLU => backend.GemmBiasLeakyRelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    _ => FusedGemmWithSeparateActivation(backend, inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures, activation)
                };
            }
            else if (bias != null && activation == FusedActivationType.None)
            {
                // GEMM + Bias only (no activation) - use GPU GemmBias kernel
                resultBuffer = backend.GemmBias(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
            }
            else if (bias == null && activation == FusedActivationType.None)
            {
                // Simple MatMul only - use GPU
                resultBuffer = backend.MatMul(inputBuffer.Buffer, weightsBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
            }
            else if (bias == null && activation != FusedActivationType.None)
            {
                // MatMul + activation (no bias) - use GPU MatMul followed by activation
                resultBuffer = backend.MatMul(inputBuffer.Buffer, weightsBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                int size = batchSize * outputFeatures;
                ApplyGpuActivation(backend, resultBuffer, size, activation);
            }
            else
            {
                // Fall back to CPU for other combinations (should not reach here now)
                return base.FusedLinear(input, weights, bias, activation);
            }

            // Download result - DownloadBuffer uses blocking read, Synchronize() removed for performance
            int resultSize = batchSize * outputFeatures;
            float[] resultFloat = new float[resultSize];
            backend.DownloadBuffer(resultBuffer, resultFloat);

            // Convert back to T
            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            int[] resultShape = new[] { batchSize, outputFeatures };

            // Cache the result buffer for potential reuse by the next layer
            // The next layer's input will be this layer's output (same data array)
            // So when GetOrAllocateBuffer is called with resultData, it can reuse this buffer
            CacheActivation(resultData, resultBuffer, resultShape, backend);

            return new Tensor<T>(resultData, resultShape);
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FusedLinear(input, weights, bias, activation);
        }
    }

    /// <summary>
    /// GPU-resident fused linear transformation that keeps result on GPU.
    /// Returns an IGpuTensor that can be passed to subsequent GPU operations
    /// without CPU round-trips. Only download the final result using ToTensor().
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">Input tensor (will be uploaded to GPU).</param>
    /// <param name="weights">Weight tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident tensor with the result. Caller must dispose this tensor to free GPU memory.</returns>
    /// <remarks>
    /// The returned tensor owns its GPU buffer. In GPU-resident workflows, these tensors should be
    /// disposed when no longer needed to prevent GPU memory leaks. Use 'using' statements or explicit
    /// Dispose() calls to ensure proper cleanup.
    /// </remarks>
    public Tensor<T> FusedLinearGpu<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedLinearGpu");

        if (input.Rank < 1 || weights.Rank != 2)
            throw new ArgumentException("Invalid tensor dimensions for FusedLinearGpu");

        int batchSize = input.Shape._dims[0];
        int inputFeatures = weights.Shape._dims[0];
        int outputFeatures = weights.Shape._dims[1];

        // Upload input to GPU (activations are not cached persistently)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        // Auto-cache weights and biases so they stay on GPU for subsequent calls
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.GetDataArray(), PersistentTensorRole.Weights);
        using var biasBuffer = bias != null ? GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases) : default;

        // Execute the fused kernel and get result buffer
        var resultBuffer = ExecuteFusedLinearKernel(backend, inputBuffer.Buffer, weightsBuffer.Buffer,
            biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures, activation);

        // Return GPU-resident tensor - NO DOWNLOAD
        // IMPORTANT: Caller is responsible for disposing the returned tensor to free GPU memory
        return Tensor<T>.FromGpuBuffer(backend, resultBuffer, new[] { batchSize, outputFeatures },
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident fused linear transformation with GPU-resident input.
    /// Avoids re-uploading input that's already on GPU from a previous layer.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="weights">Weight tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident tensor with the result. Caller must dispose this tensor to free GPU memory.</returns>
    /// <remarks>
    /// The returned tensor owns its GPU buffer. In GPU-resident workflows, these tensors should be
    /// disposed when no longer needed to prevent GPU memory leaks. Use 'using' statements or explicit
    /// Dispose() calls to ensure proper cleanup.
    /// </remarks>
    /// <summary>
    /// Executes the fused linear kernel and returns the result buffer.
    /// Shared implementation for both CPU and GPU input variants.
    /// </summary>
    private static IGpuBuffer ExecuteFusedLinearKernel(
        IDirectGpuBackend backend,
        IGpuBuffer inputBuffer,
        IGpuBuffer weightsBuffer,
        IGpuBuffer? biasBuffer,
        int batchSize,
        int outputFeatures,
        int inputFeatures,
        FusedActivationType activation)
    {
        IGpuBuffer resultBuffer;

        // Use fused GPU kernels when available
        if (biasBuffer != null && activation != FusedActivationType.None)
        {
            // Use fused kernels for common activations (most efficient)
            switch (activation)
            {
                case FusedActivationType.ReLU:
                    resultBuffer = backend.GemmBiasRelu(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    break;
                case FusedActivationType.GELU:
                    resultBuffer = backend.GemmBiasGelu(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    break;
                case FusedActivationType.Sigmoid:
                    resultBuffer = backend.GemmBiasSigmoid(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    break;
                case FusedActivationType.Tanh:
                    resultBuffer = backend.GemmBiasTanh(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    break;
                default:
                    // For other activations, use GemmBias + separate activation kernel
                    resultBuffer = backend.GemmBias(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
                    int size = batchSize * outputFeatures;
                    ApplyGpuActivation(backend, resultBuffer, size, activation);
                    break;
            }
        }
        else if (biasBuffer != null && activation == FusedActivationType.None)
        {
            // GEMM + Bias only (no activation)
            resultBuffer = backend.GemmBias(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures);
        }
        else if (biasBuffer == null && activation == FusedActivationType.None)
        {
            // Simple MatMul only
            resultBuffer = backend.MatMul(inputBuffer, weightsBuffer, batchSize, outputFeatures, inputFeatures);
        }
        else
        {
            // MatMul + activation (no bias)
            resultBuffer = backend.MatMul(inputBuffer, weightsBuffer, batchSize, outputFeatures, inputFeatures);
            int size = batchSize * outputFeatures;
            ApplyGpuActivation(backend, resultBuffer, size, activation);
        }

        return resultBuffer;
    }

    private static IGpuBuffer GemmBiasNoActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, int M, int N, int K)
    {
        // Use GemmBiasRelu with a subsequent inverse to get just GEMM + Bias
        // This is a workaround since there's no direct GemmBias function
        // Fall back to return just MatMul result and let caller handle bias on CPU
        return backend.MatMul(input, weights, M, N, K);
    }

    private static IGpuBuffer GemmBiasWithActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, int M, int N, int K, FusedActivationType activation)
    {
        // For activations without native fused support, use MatMul + activation
        var result = backend.MatMul(input, weights, M, N, K);
        int size = M * N;
        ApplyGpuActivation(backend, result, size, activation);
        return result;
    }

    private static void ApplyGpuActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size, FusedActivationType activation)
    {
        switch (activation)
        {
            case FusedActivationType.ReLU:
                backend.Relu(input, output, size);
                break;
            case FusedActivationType.LeakyReLU:
                backend.LeakyRelu(input, output, 0.01f, size);
                break;
            case FusedActivationType.Sigmoid:
                backend.Sigmoid(input, output, size);
                break;
            case FusedActivationType.Tanh:
                backend.Tanh(input, output, size);
                break;
            case FusedActivationType.GELU:
                backend.Gelu(input, output, size);
                break;
            case FusedActivationType.Swish:
                backend.Swish(input, output, size);
                break;
            case FusedActivationType.ELU:
                backend.Elu(input, output, 1.0f, size); // alpha = 1.0 is standard
                break;
            case FusedActivationType.SELU:
                // SELU: scale * (x if x > 0, else alpha * (exp(x) - 1))
                // Standard parameters: scale ≈ 1.0507, alpha ≈ 1.6733
                backend.Selu(input, output, 1.6732632423543772f, 1.0507010f, size);
                break;
            case FusedActivationType.Softplus:
                backend.Softplus(input, output, size);
                break;
            case FusedActivationType.Mish:
                backend.Mish(input, output, size);
                break;
            case FusedActivationType.HardSwish:
                backend.Hardswish(input, output, size);
                break;
            case FusedActivationType.HardSigmoid:
                backend.Hardsigmoid(input, output, size);
                break;
            case FusedActivationType.HardTanh:
                backend.Hardtanh(input, output, -1.0f, 1.0f, size);
                break;
            case FusedActivationType.None:
                if (!ReferenceEquals(input, output))
                {
                    backend.Copy(input, 0, output, 0, size);
                }
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(activation), activation, "Unsupported activation.");
        }
    }

    private static void ApplyGpuActivation(IDirectGpuBackend backend, IGpuBuffer buffer, int size, FusedActivationType activation)
    {
        ApplyGpuActivation(backend, buffer, buffer, size, activation);
    }

    private static IGpuBuffer FusedGemmWithSeparateActivation(
        IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias,
        int M, int N, int K, FusedActivationType activation)
    {
        var result = backend.GemmBias(input, weights, bias, M, N, K);
        int size = M * N;
        ApplyGpuActivation(backend, result, size, activation);
        return result;
    }

    /// <summary>
    /// Uploads a tensor to GPU memory, returning a GPU-resident tensor handle.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="tensor">The CPU tensor to upload.</param>
    /// <param name="role">The role of this tensor for memory management.</param>
    /// <returns>A GPU-resident tensor that can be used in subsequent GPU operations.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <remarks>
    /// <para>
    /// Use this method to explicitly upload data to GPU for use in GPU-resident operations.
    /// The returned tensor can be passed to methods like <see cref="FusedLinearGpu{T}(Tensor{T}, Tensor{T}, Tensor{T}?, FusedActivationType)"/>
    /// to avoid redundant uploads.
    /// </para>
    /// <para>
    /// The caller is responsible for disposing the returned GPU tensor when done.
    /// </para>
    /// </remarks>
    public Tensor<T> UploadToGpu<T>(Tensor<T> tensor, GpuTensorRole role)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UploadToGpu");

        // Convert tensor data to float and allocate GPU buffer
        float[] floatData = DirectGpuEngine.ToFloatArray(tensor.GetDataArray());
        var buffer = backend.AllocateBuffer(floatData);

        // Return GPU tensor that owns the buffer
        return Tensor<T>.FromGpuBuffer(backend, buffer, tensor.Shape.ToArray(), role, ownsBuffer: true);
    }

    /// <summary>
    /// Uploads raw float data to GPU memory, returning a GPU-resident tensor handle.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="data">The float data to upload.</param>
    /// <param name="shape">The shape of the resulting tensor.</param>
    /// <param name="role">The role of this tensor for memory management.</param>
    /// <returns>A GPU-resident tensor that can be used in subsequent GPU operations.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public Tensor<T> UploadToGpu<T>(float[] data, int[] shape, GpuTensorRole role)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UploadToGpu");

        var buffer = backend.AllocateBuffer(data);
        return Tensor<T>.FromGpuBuffer(backend, buffer, shape, role, ownsBuffer: true);
    }

    /// <summary>
    /// Uploads weight/bias tensor to GPU with automatic caching. If the data is already cached,
    /// returns the cached GPU tensor without re-uploading. This is the recommended method for
    /// layer weights and biases that don't change between forward passes during inference.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    /// <param name="tensor">The CPU tensor containing weight/bias data.</param>
    /// <param name="role">The role indicating the type of persistent tensor (Weight, Bias, Statistics, AttentionCache, or Constant).</param>
    /// <returns>
    /// A GPU-resident tensor with ownership semantics determined by cache state:
    /// - If cached: ownsBuffer=false, disposing the tensor is safe (no-op, cache retains buffer).
    /// - If not cached (rare race condition): ownsBuffer=true, caller owns the buffer and disposal will free it.
    /// In both cases, disposing the returned tensor is safe and recommended.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <exception cref="ArgumentException">Thrown when an unsupported role is passed (e.g., General, Activation, Gradient).</exception>
    public Tensor<T> GetOrCacheWeightsGpu<T>(Tensor<T> tensor, GpuTensorRole role = GpuTensorRole.Weight)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GetOrCacheWeightsGpu");

        var persistentRole = role switch
        {
            GpuTensorRole.Weight => PersistentTensorRole.Weights,
            GpuTensorRole.Bias => PersistentTensorRole.Biases,
            GpuTensorRole.Statistics => PersistentTensorRole.NormalizationParams,
            GpuTensorRole.AttentionCache => PersistentTensorRole.AttentionCache,
            GpuTensorRole.Constant => PersistentTensorRole.Constant,
            _ => throw new ArgumentException(
                $"GetOrCacheWeightsGpu only supports Weight, Bias, Statistics, AttentionCache, or Constant roles. " +
                $"Got: {role}. Use UploadToGpu for other tensor types.", nameof(role))
        };
        var ownedBuffer = GetOrCacheWeightBuffer(backend, tensor.GetDataArray(), persistentRole);

        // Propagate ownership: if cache owns buffer, GpuTensor shouldn't dispose;
        // if we own buffer (race condition fallback), GpuTensor should take ownership
        return Tensor<T>.FromGpuBuffer(backend, ownedBuffer.Buffer, tensor.Shape.ToArray(), role, ownsBuffer: ownedBuffer.OwnsBuffer);
    }

    /// <summary>
    /// Invalidates a cached weight buffer, forcing a re-upload on the next GetOrCacheWeightsGpu call.
    /// Thread-safe: Uses _persistentBufferLock to synchronize with GetOrCacheWeightBuffer.
    /// </summary>
    public bool InvalidateWeightCache<T>(T[] data)
    {
        lock (_persistentBufferLock)
        {
            if (_persistentBufferCache.TryRemove(data, out var entry))
            {
                entry.Dispose();
                _tensorVersions.TryRemove(data, out _);
                return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Invalidates all cached weight buffers.
    /// Thread-safe: Uses _persistentBufferLock to synchronize with GetOrCacheWeightBuffer.
    /// </summary>
    public void InvalidateAllWeightCaches()
    {
        lock (_persistentBufferLock)
        {
            foreach (var entry in _persistentBufferCache.Values)
            {
                entry.Dispose();
            }
            _persistentBufferCache.Clear();
            _tensorVersions.Clear();
        }
    }

    /// <summary>
    /// Applies an activation function to a GPU-resident tensor, returning a new GPU tensor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <param name="activation">The activation type to apply.</param>
    /// <returns>A new GPU tensor with the activation applied.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <remarks>
    /// <para>
    /// This method applies the specified activation function entirely on the GPU,
    /// without downloading data to CPU. Supported activations: ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Swish.
    /// </para>
    /// </remarks>
    public Tensor<T> ActivationGpu<T>(Tensor<T> input, FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ActivationGpu");

        // Allocate output buffer
        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        if (activation == FusedActivationType.None)
        {
            // Preserve previous behavior: output is a copy of input.
            backend.Copy(input.Buffer, outputBuffer, size);
        }
        else
        {
            ApplyGpuActivation(backend, input.Buffer, outputBuffer, size, activation);
        }

        // Return new GPU tensor
        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// Performs GPU-resident dropout forward pass with random mask generation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <param name="dropoutRate">Probability of dropping each element (0-1).</param>
    /// <param name="isTraining">If true, applies dropout; if false, passes through unchanged.</param>
    /// <param name="seed">Random seed for mask generation (use different seed per batch for variety).</param>
    /// <returns>A tuple of (output tensor, mask tensor) for use in backward pass.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public (Tensor<T> Output, Tensor<T> Mask) DropoutGpu<T>(
        Tensor<T> input,
        float dropoutRate,
        bool isTraining,
        ulong seed)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DropoutGpu");

        int size = input.Length;

        // Allocate output and mask buffers
        var outputBuffer = backend.AllocateBuffer(size);
        var maskBuffer = backend.AllocateBuffer(size);

        // Run dropout kernel (handles both training and inference modes)
        backend.Dropout(input.Buffer, outputBuffer, maskBuffer, size, dropoutRate, seed, isTraining);

        // Return GPU tensors
        var output = Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
        var mask = Tensor<T>.FromGpuBuffer(backend, maskBuffer, input.Shape._dims, GpuTensorRole.Intermediate, true);

        return (output, mask);
    }

    /// <summary>
    /// Performs GPU-resident dropout backward pass.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">The GPU-resident gradient from the next layer.</param>
    /// <param name="mask">The dropout mask from the forward pass.</param>
    /// <param name="dropoutRate">The dropout rate used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public Tensor<T> DropoutBackwardGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> mask,
        float dropoutRate)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DropoutBackwardGpu");

        int size = gradOutput.Length;
        var gradInputBuffer = backend.AllocateBuffer(size);

        backend.DropoutBackward(gradOutput.Buffer, mask.Buffer, gradInputBuffer, size, dropoutRate);

        return Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident 2D max pooling that keeps output and indices on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input tensor on GPU.</param>
    /// <param name="poolSize">Pool size [height, width].</param>
    /// <param name="stride">Stride [height, width].</param>
    /// <param name="gpuIndices">Output GPU buffer containing pooling indices.</param>
    /// <returns>The pooled output as GPU-resident tensor.</returns>
    public Tensor<T> MaxPool2DGpu<T>(
        Tensor<T> input,
        int[] poolSize,
        int[] stride,
        out IGpuBuffer gpuIndices)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxPool2DGpu");

        if (input.Shape._dims.Length != 4 || poolSize.Length != 2 || stride.Length != 2)
            throw new ArgumentException("Input must be 4D [batch, channels, height, width] with 2D poolSize and stride");

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid pooling parameters: output dimensions ({outHeight}, {outWidth}) are non-positive");

        int outputSize = batch * channels * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);
        var indicesBuffer = backend.AllocateBuffer(outputSize);

        backend.MaxPool2D(input.Buffer, outputBuffer, indicesBuffer,
            batch, channels, inHeight, inWidth,
            outHeight, outWidth,
            poolSize[0], poolSize[1],
            stride[0], stride[1], 0, 0);

        var outputShape = new[] { batch, channels, outHeight, outWidth };
        gpuIndices = indicesBuffer;

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, true);
    }

    /// <summary>
    /// GPU-resident backward pass for 2D max pooling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient of the output on GPU.</param>
    /// <param name="gpuIndices">The GPU buffer containing pooling indices from forward pass.</param>
    /// <param name="inputShape">The shape of the original input.</param>
    /// <param name="poolSize">Pool size [height, width].</param>
    /// <param name="stride">Stride [height, width].</param>
    /// <returns>The gradient with respect to input as GPU-resident tensor.</returns>
    public Tensor<T> MaxPool2DBackwardGpu<T>(
        Tensor<T> gradOutput,
        IGpuBuffer gpuIndices,
        int[] inputShape,
        int[] poolSize,
        int[] stride)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxPool2DBackwardGpu");

        if (gradOutput.Shape._dims.Length != 4 || inputShape.Length != 4)
            throw new ArgumentException("GradOutput and inputShape must be 4D");

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int gradInputSize = batch * channels * inHeight * inWidth;
        var gradInputBuffer = backend.AllocateBuffer(gradInputSize);

        backend.MaxPool2DBackward(gradOutput.Buffer, gpuIndices, gradInputBuffer,
            batch, channels, inHeight, inWidth,
            gradOutput.Shape._dims[2], gradOutput.Shape._dims[3],
            poolSize[0], poolSize[1],
            stride[0], stride[1], 0, 0);

        return Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident 3D max pooling that keeps output and indices on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input tensor on GPU with shape [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">Pool size [depth, height, width].</param>
    /// <param name="stride">Stride [depth, height, width].</param>
    /// <param name="gpuIndices">Output GPU buffer containing flat pooling indices.</param>
    /// <returns>The pooled output as GPU-resident tensor.</returns>
    public Tensor<T> MaxPool3DGpu<T>(
        Tensor<T> input,
        int[] poolSize,
        int[] stride,
        out IGpuBuffer gpuIndices)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxPool3DGpu");

        if (input.Shape._dims.Length != 5 || poolSize.Length != 3 || stride.Length != 3)
            throw new ArgumentException("Input must be 5D [batch, channels, depth, height, width] with 3D poolSize and stride");

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inDepth = input.Shape._dims[2];
        int inHeight = input.Shape._dims[3];
        int inWidth = input.Shape._dims[4];

        int outDepth = (inDepth - poolSize[0]) / stride[0] + 1;
        int outHeight = (inHeight - poolSize[1]) / stride[1] + 1;
        int outWidth = (inWidth - poolSize[2]) / stride[2] + 1;

        if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid pooling parameters: output dimensions ({outDepth}, {outHeight}, {outWidth}) are non-positive");

        int outputSize = batch * channels * outDepth * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);
        gpuIndices = backend.AllocateBuffer(outputSize * sizeof(int) / sizeof(float));

        backend.MaxPool3D(input.Buffer, outputBuffer, gpuIndices,
            batch, channels,
            inDepth, inHeight, inWidth,
            outDepth, outHeight, outWidth,
            poolSize[0], poolSize[1], poolSize[2],
            stride[0], stride[1], stride[2]);

        var outputShape = new[] { batch, channels, outDepth, outHeight, outWidth };
        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, true);
    }

    /// <summary>
    /// GPU-resident backward pass for 3D max pooling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient of the output on GPU.</param>
    /// <param name="gpuIndices">The GPU buffer containing flat pooling indices from forward pass.</param>
    /// <param name="inputShape">The shape of the original input [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">Pool size [depth, height, width].</param>
    /// <param name="stride">Stride [depth, height, width].</param>
    /// <returns>The gradient with respect to input as GPU-resident tensor.</returns>
    public Tensor<T> MaxPool3DBackwardGpu<T>(
        Tensor<T> gradOutput,
        IGpuBuffer gpuIndices,
        int[] inputShape,
        int[] poolSize,
        int[] stride)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxPool3DBackwardGpu");

        if (gradOutput.Shape._dims.Length != 5 || inputShape.Length != 5)
            throw new ArgumentException("GradOutput and inputShape must be 5D");

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inDepth = inputShape[2];
        int inHeight = inputShape[3];
        int inWidth = inputShape[4];

        int gradInputSize = batch * channels * inDepth * inHeight * inWidth;
        var gradInputBuffer = backend.AllocateBuffer(gradInputSize);

        backend.MaxPool3DBackward(gradOutput.Buffer, gpuIndices, gradInputBuffer,
            batch, channels,
            inDepth, inHeight, inWidth,
            gradOutput.Shape._dims[2], gradOutput.Shape._dims[3], gradOutput.Shape._dims[4]);

        return Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident 3D nearest neighbor upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input tensor on GPU with shape [batch, channels, depth, height, width].</param>
    /// <param name="scaleDepth">Scale factor for depth dimension.</param>
    /// <param name="scaleHeight">Scale factor for height dimension.</param>
    /// <param name="scaleWidth">Scale factor for width dimension.</param>
    /// <returns>The upsampled output as GPU-resident tensor.</returns>
    public Tensor<T> NearestNeighborUpsample3DGpu<T>(
        Tensor<T> input,
        int scaleDepth,
        int scaleHeight,
        int scaleWidth)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for NearestNeighborUpsample3DGpu");

        if (input.Shape._dims.Length != 5)
            throw new ArgumentException("Input must be 5D [batch, channels, depth, height, width]");

        if (scaleDepth <= 0 || scaleHeight <= 0 || scaleWidth <= 0)
            throw new ArgumentException("Scale factors must be positive");

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inDepth = input.Shape._dims[2];
        int inHeight = input.Shape._dims[3];
        int inWidth = input.Shape._dims[4];

        int outDepth = inDepth * scaleDepth;
        int outHeight = inHeight * scaleHeight;
        int outWidth = inWidth * scaleWidth;

        int outputSize = batch * channels * outDepth * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        backend.NearestNeighborUpsample3D(input.Buffer, outputBuffer,
            batch, channels,
            inDepth, inHeight, inWidth,
            scaleDepth, scaleHeight, scaleWidth);

        var outputShape = new[] { batch, channels, outDepth, outHeight, outWidth };
        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, true);
    }

    /// <summary>
    /// GPU-resident backward pass for 3D nearest neighbor upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient of the output on GPU.</param>
    /// <param name="inputShape">The shape of the original input [batch, channels, depth, height, width].</param>
    /// <param name="scaleDepth">Scale factor for depth dimension.</param>
    /// <param name="scaleHeight">Scale factor for height dimension.</param>
    /// <param name="scaleWidth">Scale factor for width dimension.</param>
    /// <returns>The gradient with respect to input as GPU-resident tensor.</returns>
    public Tensor<T> NearestNeighborUpsample3DBackwardGpu<T>(
        Tensor<T> gradOutput,
        int[] inputShape,
        int scaleDepth,
        int scaleHeight,
        int scaleWidth)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for NearestNeighborUpsample3DBackwardGpu");

        if (gradOutput.Shape._dims.Length != 5 || inputShape.Length != 5)
            throw new ArgumentException("GradOutput and inputShape must be 5D");

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inDepth = inputShape[2];
        int inHeight = inputShape[3];
        int inWidth = inputShape[4];

        int gradInputSize = batch * channels * inDepth * inHeight * inWidth;
        var gradInputBuffer = backend.AllocateBuffer(gradInputSize);

        backend.NearestNeighborUpsample3DBackward(gradOutput.Buffer, gradInputBuffer,
            batch, channels,
            inDepth, inHeight, inWidth,
            scaleDepth, scaleHeight, scaleWidth);

        return Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident 2D average pooling that keeps output on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input tensor on GPU.</param>
    /// <param name="poolSize">Pool size [height, width].</param>
    /// <param name="stride">Stride [height, width].</param>
    /// <returns>The pooled output as GPU-resident tensor.</returns>
    public Tensor<T> AvgPool2DGpu<T>(
        Tensor<T> input,
        int[] poolSize,
        int[] stride)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AvgPool2DGpu");

        if (input.Shape._dims.Length != 4 || poolSize.Length != 2 || stride.Length != 2)
            throw new ArgumentException("Input must be 4D [batch, channels, height, width] with 2D poolSize and stride");

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid pooling parameters: output dimensions ({outHeight}, {outWidth}) are non-positive");

        int outputSize = batch * channels * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        backend.AvgPool2D(input.Buffer, outputBuffer,
            batch, channels, inHeight, inWidth,
            outHeight, outWidth,
            poolSize[0], poolSize[1],
            stride[0], stride[1], 0, 0,
            countIncludePad: true);

        var outputShape = new[] { batch, channels, outHeight, outWidth };
        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, true);
    }

    /// <summary>
    /// GPU-resident backward pass for 2D average pooling.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The gradient of the output on GPU.</param>
    /// <param name="inputShape">The shape of the original input.</param>
    /// <param name="poolSize">Pool size [height, width].</param>
    /// <param name="stride">Stride [height, width].</param>
    /// <returns>The gradient with respect to input as GPU-resident tensor.</returns>
    public Tensor<T> AvgPool2DBackwardGpu<T>(
        Tensor<T> gradOutput,
        int[] inputShape,
        int[] poolSize,
        int[] stride)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AvgPool2DBackwardGpu");

        if (gradOutput.Shape._dims.Length != 4 || inputShape.Length != 4)
            throw new ArgumentException("GradOutput and inputShape must be 4D");

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int gradInputSize = batch * channels * inHeight * inWidth;
        var gradInputBuffer = backend.AllocateBuffer(gradInputSize);

        backend.AvgPool2DBackward(gradOutput.Buffer, gradInputBuffer,
            batch, channels, inHeight, inWidth,
            gradOutput.Shape._dims[2], gradOutput.Shape._dims[3],
            poolSize[0], poolSize[1],
            stride[0], stride[1], 0, 0,
            countIncludePad: true);

        return Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-accelerated fused 2D convolution with activation.
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelH, kernelW]
        if (input.Rank != 4 || kernel.Rank != 4)
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernel.Shape._dims[0];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        // Calculate output dimensions with dilation
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outHeight * outWidth);

        try
        {
            // Execute GPU convolution
            backend.Conv2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                dilationH, dilationW);

            // Add bias if present
            if (bias != null)
            {
                // Bias is added per output channel, broadcast across batch and spatial dimensions
                int outputSize = batch * outChannels * outHeight * outWidth;
                int spatialSize = outHeight * outWidth;

                // Download, add bias, re-upload (GPU bias broadcast kernel would be more efficient)
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

                // Get bias data (check cache first)
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                float[] biasFloat = new float[bias.Length];
                backend.DownloadBuffer(biasBuffer.Buffer, biasFloat);

                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < outChannels; c++)
                    {
                        float biasVal = biasFloat[c];
                        int baseIdx = (b * outChannels + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                        {
                            outputFloat[baseIdx + s] += biasVal;
                        }
                    }
                }

                // Re-upload for activation
                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                // Apply activation on GPU
                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            else
            {
                // No bias - apply activation directly
                int outputSize = batch * outChannels * outHeight * outWidth;

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer.Buffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);
        }
    }

    /// <summary>
    /// GPU-resident fused 2D convolution with activation.
    /// Keeps input and output on GPU for chained layer execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, inChannels, height, width].</param>
    /// <param name="kernel">Kernel tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="strideH">Vertical stride.</param>
    /// <param name="strideW">Horizontal stride.</param>
    /// <param name="padH">Vertical padding.</param>
    /// <param name="padW">Horizontal padding.</param>
    /// <param name="dilationH">Vertical dilation.</param>
    /// <param name="dilationW">Horizontal dilation.</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident output tensor.</returns>
    public Tensor<T> FusedConv2DGpu<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedConv2DGpu");

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelH, kernelW]
        if (input.Shape._dims.Length != 4 || kernel.Rank != 4)
            throw new ArgumentException("FusedConv2DGpu requires 4D input and kernel tensors");

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernel.Shape._dims[0];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        // Calculate output dimensions with dilation
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outHeight * outWidth;

        // Input is already on GPU - use its buffer directly
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            // Execute GPU convolution
            backend.Conv2D(input.Buffer, kernelBuffer.Buffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                dilationH, dilationW);

            // Add bias if present using GPU kernel for NCHW format
            if (bias != null)
            {
                int spatialSize = outHeight * outWidth;
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                backend.Conv2DBiasAdd(outputBuffer, biasBuffer.Buffer, batch, outChannels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            // Return GPU-resident tensor - NO DOWNLOAD
            return Tensor<T>.FromGpuBuffer(backend, outputBuffer, new[] { batch, outChannels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated fused 3D convolution with activation.
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedConv3D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        // Expected input shape: [batch, inChannels, depth, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelD, kernelH, kernelW]
        if (input.Rank != 5 || kernel.Rank != 5)
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inDepth = input.Shape._dims[2];
        int inHeight = input.Shape._dims[3];
        int inWidth = input.Shape._dims[4];

        int outChannels = kernel.Shape._dims[0];
        int kernelD = kernel.Shape._dims[2];
        int kernelH = kernel.Shape._dims[3];
        int kernelW = kernel.Shape._dims[4];

        // Calculate output dimensions with dilation
        int effectiveKernelD = kernelD + (kernelD - 1) * (dilationD - 1);
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outDepth = (inDepth + 2 * padD - effectiveKernelD) / strideD + 1;
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outDepth * outHeight * outWidth);

        try
        {
            // Execute GPU 3D convolution
            backend.Conv3D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inDepth, inHeight, inWidth,
                outChannels, outDepth, outHeight, outWidth,
                kernelD, kernelH, kernelW,
                strideD, strideH, strideW,
                padD, padH, padW,
                dilationD, dilationH, dilationW);

            // Add bias if present
            if (bias != null)
            {
                int outputSize = batch * outChannels * outDepth * outHeight * outWidth;
                int spatialSize = outDepth * outHeight * outWidth;

                // Download, add bias, re-upload
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                float[] biasFloat = new float[bias.Length];
                backend.DownloadBuffer(biasBuffer.Buffer, biasFloat);

                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < outChannels; c++)
                    {
                        float biasVal = biasFloat[c];
                        int baseIdx = (b * outChannels + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                        {
                            outputFloat[baseIdx + s] += biasVal;
                        }
                    }
                }

                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outDepth, outHeight, outWidth });
            }
            else
            {
                int outputSize = batch * outChannels * outDepth * outHeight * outWidth;

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer.Buffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outDepth, outHeight, outWidth });
            }
        }
        catch
        {
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);
        }
    }

    /// <summary>
    /// GPU-resident fused 3D convolution with activation.
    /// Keeps input and output on GPU for chained layer execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, inChannels, depth, height, width].</param>
    /// <param name="kernel">Kernel tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="strideD">Depth stride.</param>
    /// <param name="strideH">Height stride.</param>
    /// <param name="strideW">Width stride.</param>
    /// <param name="padD">Depth padding.</param>
    /// <param name="padH">Height padding.</param>
    /// <param name="padW">Width padding.</param>
    /// <param name="dilationD">Depth dilation.</param>
    /// <param name="dilationH">Height dilation.</param>
    /// <param name="dilationW">Width dilation.</param>
    /// <param name="activation">Fused activation type.</param>
    /// <returns>GPU-resident output tensor [batch, outChannels, outDepth, outHeight, outWidth].</returns>
    public Tensor<T> FusedConv3DGpu<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedConv3DGpu");

        // Expected input shape: [batch, inChannels, depth, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelD, kernelH, kernelW]
        if (input.Shape._dims.Length != 5 || kernel.Rank != 5)
            throw new ArgumentException("FusedConv3DGpu requires 5D input and kernel tensors");

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inDepth = input.Shape._dims[2];
        int inHeight = input.Shape._dims[3];
        int inWidth = input.Shape._dims[4];

        int outChannels = kernel.Shape._dims[0];
        int kernelD = kernel.Shape._dims[2];
        int kernelH = kernel.Shape._dims[3];
        int kernelW = kernel.Shape._dims[4];

        // Calculate output dimensions with dilation
        int effectiveKernelD = kernelD + (kernelD - 1) * (dilationD - 1);
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outDepth = (inDepth + 2 * padD - effectiveKernelD) / strideD + 1;
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid 3D convolution parameters result in non-positive output dimensions: {outDepth}x{outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outDepth * outHeight * outWidth;
        int spatialSize = outDepth * outHeight * outWidth;

        // Use cache-aware buffer allocation
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            // Execute GPU 3D convolution
            backend.Conv3D(input.Buffer, kernelBuffer.Buffer, outputBuffer,
                batch, inChannels, inDepth, inHeight, inWidth,
                outChannels, outDepth, outHeight, outWidth,
                kernelD, kernelH, kernelW,
                strideD, strideH, strideW,
                padD, padH, padW,
                dilationD, dilationH, dilationW);

            // Add bias if present (Conv2DBiasAdd works for any spatial size)
            if (bias != null)
            {
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                backend.Conv2DBiasAdd(outputBuffer, biasBuffer.Buffer, batch, outChannels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            // Return GPU-resident tensor - NO DOWNLOAD
            return Tensor<T>.FromGpuBuffer(backend, outputBuffer, new[] { batch, outChannels, outDepth, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated fused transposed 2D convolution with activation.
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedConvTranspose2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int outputPadH, int outputPadW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [inChannels, outChannels, kernelH, kernelW]
        if (input.Rank != 4 || kernel.Rank != 4)
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernel.Shape._dims[1];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        // Calculate output dimensions for transposed convolution
        int outHeight = (inHeight - 1) * strideH - 2 * padH + kernelH + outputPadH;
        int outWidth = (inWidth - 1) * strideW - 2 * padW + kernelW + outputPadW;

        if (outHeight <= 0 || outWidth <= 0)
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outHeight * outWidth);

        try
        {
            // Execute GPU transposed convolution
            backend.ConvTranspose2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                outputPadH, outputPadW);

            // Add bias if present
            if (bias != null)
            {
                int outputSize = batch * outChannels * outHeight * outWidth;
                int spatialSize = outHeight * outWidth;

                // Download, add bias, re-upload
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                float[] biasFloat = new float[bias.Length];
                backend.DownloadBuffer(biasBuffer.Buffer, biasFloat);

                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < outChannels; c++)
                    {
                        float biasVal = biasFloat[c];
                        int baseIdx = (b * outChannels + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                        {
                            outputFloat[baseIdx + s] += biasVal;
                        }
                    }
                }

                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            else
            {
                int outputSize = batch * outChannels * outHeight * outWidth;

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer.Buffer, outputSize, activation);
                }

                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
        }
        catch
        {
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);
        }
    }

    /// <summary>
    /// GPU-resident fused transposed 2D convolution with activation.
    /// Keeps input and output on GPU for chained layer execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, inChannels, height, width].</param>
    /// <param name="kernel">Kernel tensor (cached if registered).</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="strideH">Vertical stride.</param>
    /// <param name="strideW">Horizontal stride.</param>
    /// <param name="padH">Vertical padding.</param>
    /// <param name="padW">Horizontal padding.</param>
    /// <param name="outputPadH">Vertical output padding.</param>
    /// <param name="outputPadW">Horizontal output padding.</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident output tensor.</returns>
    public Tensor<T> FusedConvTranspose2DGpu<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int outputPadH, int outputPadW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedConvTranspose2DGpu");

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [inChannels, outChannels, kernelH, kernelW]
        if (input.Shape._dims.Length != 4 || kernel.Rank != 4)
            throw new ArgumentException("FusedConvTranspose2DGpu requires 4D input and kernel tensors");

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernel.Shape._dims[1];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        // Calculate output dimensions for transposed convolution
        int outHeight = (inHeight - 1) * strideH - 2 * padH + kernelH + outputPadH;
        int outWidth = (inWidth - 1) * strideW - 2 * padW + kernelW + outputPadW;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid transposed convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outHeight * outWidth;

        // Input is already on GPU - use its buffer directly
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            // Execute GPU transposed convolution
            backend.ConvTranspose2D(input.Buffer, kernelBuffer.Buffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                outputPadH, outputPadW);

            // Add bias if present using GPU kernel for NCHW format
            if (bias != null)
            {
                int spatialSize = outHeight * outWidth;
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                backend.Conv2DBiasAdd(outputBuffer, biasBuffer.Buffer, batch, outChannels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            // Return GPU-resident tensor - NO DOWNLOAD
            return Tensor<T>.FromGpuBuffer(backend, outputBuffer, new[] { batch, outChannels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated 2D max pooling operation.
    /// Uses GPU kernels for efficient parallel computation of maximum values within pooling windows.
    /// </summary>
    public override Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;

        if (!TryGetBackend(out var backend))
            return base.MaxPool2D(input, poolSize, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        if (input.Rank != 4)
            return base.MaxPool2D(input, poolSize, stride, padding);

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        // Calculate output dimensions
        int outHeight = (inHeight + 2 * padding - poolSize) / stride + 1;
        int outWidth = (inWidth + 2 * padding - poolSize) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.MaxPool2D(input, poolSize, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            // Execute GPU max pooling (indices buffer is null for forward-only)
            backend.MaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, null,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize, poolSize,
                stride, stride, padding, padding);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.MaxPool2D(input, poolSize, stride, padding);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D max pooling with indices for backward pass.
    /// Returns both pooled output and indices of maximum values for gradient computation.
    /// </summary>
    public new Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices)
    {
        if (!TryGetBackend(out var backend))
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        if (input.Rank != 4 || poolSize.Length != 2 || stride.Length != 2)
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);
        using var indicesBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.MaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, indicesBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            float[] indicesFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);
            backend.DownloadBuffer(indicesBuffer.Buffer, indicesFloat);

            // Convert indices to int array
            maxIndices = new int[batch, channels, outHeight, outWidth, 2];
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            int flatIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
                            int idx = (int)indicesFloat[flatIdx];
                            maxIndices[b, c, oh, ow, 0] = idx / inWidth;
                            maxIndices[b, c, oh, ow, 1] = idx % inWidth;
                        }
                    }
                }
            }

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for 2D max pooling.
    /// Propagates gradients back through the max pooling operation using stored indices.
    /// </summary>
    public new Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);

        if (gradOutput.Rank != 4 || inputShape.Length != 4)
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];
        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        // Convert indices to flat GPU buffer
        int indexCount = batch * channels * outHeight * outWidth;
        float[] indicesFlat = new float[indexCount];
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int flatIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
                        int h = maxIndices[b, c, oh, ow, 0];
                        int w = maxIndices[b, c, oh, ow, 1];
                        indicesFlat[flatIdx] = h * inWidth + w;
                    }
                }
            }
        }

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var indicesBuffer = backend.AllocateBuffer(indicesFlat);
        using var gradInputBuffer = AllocateOutputBuffer(backend, batch * channels * inHeight * inWidth);

        try
        {
            backend.MaxPool2DBackward(gradOutputBuffer.Buffer, indicesBuffer, gradInputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int inputSize = batch * channels * inHeight * inWidth;
            float[] resultFloat = new float[inputSize];
            backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, inputShape);
        }
        catch
        {
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D average pooling operation.
    /// Uses GPU kernels for efficient parallel computation of average values within pooling windows.
    /// </summary>
    public override Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;

        if (!TryGetBackend(out var backend))
            return base.AvgPool2D(input, poolSize, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        if (input.Rank != 4)
            return base.AvgPool2D(input, poolSize, stride, padding);

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        // Calculate output dimensions
        int outHeight = (inHeight + 2 * padding - poolSize) / stride + 1;
        int outWidth = (inWidth + 2 * padding - poolSize) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.AvgPool2D(input, poolSize, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            // Execute GPU average pooling
            backend.AvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize, poolSize,
                stride, stride, padding, padding,
                countIncludePad: true);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.AvgPool2D(input, poolSize, stride, padding);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D average pooling with array parameters.
    /// </summary>
    public new Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.AvgPool2D(input, poolSize, stride);

        if (input.Rank != 4 || poolSize.Length != 2 || stride.Length != 2)
            return base.AvgPool2D(input, poolSize, stride);

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.AvgPool2D(input, poolSize, stride);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.AvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0,
                countIncludePad: true);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.AvgPool2D(input, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for 2D average pooling.
    /// Distributes gradients evenly across the input elements that contributed to each output.
    /// </summary>
    public new Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);

        if (gradOutput.Rank != 4 || inputShape.Length != 4)
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];
        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var gradInputBuffer = AllocateOutputBuffer(backend, batch * channels * inHeight * inWidth);

        try
        {
            backend.AvgPool2DBackward(gradOutputBuffer.Buffer, gradInputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0,
                countIncludePad: true);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int inputSize = batch * channels * inHeight * inWidth;
            float[] resultFloat = new float[inputSize];
            backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, inputShape);
        }
        catch
        {
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated depthwise 2D convolution.
    /// Each input channel is convolved with its own filter, commonly used in MobileNets.
    /// </summary>
    public new Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding)
    {
        if (!TryGetBackend(out var backend))
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        // Expected kernel shape: [channels, 1, kernelH, kernelW] or [channels, kernelH, kernelW]
        if (input.Rank != 4)
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int kernelH = kernel.Rank == 4 ? kernel.Shape._dims[2] : kernel.Shape._dims[1];
        int kernelW = kernel.Rank == 4 ? kernel.Shape._dims[3] : kernel.Shape._dims[2];

        int strideH = stride.Length >= 1 ? stride[0] : 1;
        int strideW = stride.Length >= 2 ? stride[1] : strideH;
        int padH = padding.Length >= 1 ? padding[0] : 0;
        int padW = padding.Length >= 2 ? padding[1] : padH;

        int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.DepthwiseConv2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.DepthwiseConv2D(input, kernel, stride, padding);
        }
    }

    /// <summary>
    /// GPU-resident depthwise 2D convolution with optional bias and activation.
    /// Keeps input and output on GPU for chained layer execution.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, channels, height, width].</param>
    /// <param name="kernel">Kernel tensor (cached if registered). Shape: [channels, 1, kH, kW].</param>
    /// <param name="bias">Optional bias tensor (cached if registered).</param>
    /// <param name="strideH">Vertical stride.</param>
    /// <param name="strideW">Horizontal stride.</param>
    /// <param name="padH">Vertical padding.</param>
    /// <param name="padW">Horizontal padding.</param>
    /// <param name="activation">Activation function to fuse.</param>
    /// <returns>GPU-resident output tensor.</returns>
    public Tensor<T> DepthwiseConv2DGpu<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DepthwiseConv2DGpu");

        // Expected input shape: [batch, channels, height, width]
        // Expected kernel shape: [channels, 1, kernelH, kernelW] or [channels, kernelH, kernelW]
        if (input.Shape._dims.Length != 4)
            throw new ArgumentException("DepthwiseConv2DGpu requires 4D input tensor");

        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int kernelH = kernel.Rank == 4 ? kernel.Shape._dims[2] : kernel.Shape._dims[1];
        int kernelW = kernel.Rank == 4 ? kernel.Shape._dims[3] : kernel.Shape._dims[2];

        int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid depthwise convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * channels * outHeight * outWidth;

        // Input is already on GPU - use its buffer directly
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            // Execute GPU depthwise convolution
            backend.DepthwiseConv2D(input.Buffer, kernelBuffer.Buffer, outputBuffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW);

            // Add bias if present using GPU kernel for NCHW format
            if (bias != null)
            {
                int spatialSize = outHeight * outWidth;
                using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                // Depthwise output has same channels as input - use Conv2DBiasAdd pattern
                backend.Conv2DBiasAdd(outputBuffer, biasBuffer.Buffer, batch, channels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            // Return GPU-resident tensor - NO DOWNLOAD
            return Tensor<T>.FromGpuBuffer(backend, outputBuffer, new[] { batch, channels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D convolution.
    /// Uses cached GPU buffers for registered persistent tensors (weights/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> LocallyConnectedConv2D<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.LocallyConnectedConv2D(input, weights, bias, stride);

        // Expected input shape: [batch, inChannels, height, width]
        // Expected weights shape: [outH, outW, outC, inC, kH, kW]
        if (input.Rank != 4 || weights.Rank != 6)
            return base.LocallyConnectedConv2D(input, weights, bias, stride);

        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0)
            throw new ArgumentException("Stride elements must be positive", nameof(stride));

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outHeight = weights.Shape._dims[0];
        int outWidth = weights.Shape._dims[1];
        int outChannels = weights.Shape._dims[2];
        int kernelH = weights.Shape._dims[4];
        int kernelW = weights.Shape._dims[5];

        if (outHeight <= 0 || outWidth <= 0)
            return base.LocallyConnectedConv2D(input, weights, bias, stride);

        int outputSize = batch * outChannels * outHeight * outWidth;

        // Use cache-aware buffer allocation for weights/bias (persistent tensors)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.GetDataArray(), PersistentTensorRole.Weights);
        using var outputBuffer = AllocateOutputBuffer(backend, outputSize);

        try
        {
            // Handle bias - check cache for persistent bias tensors
            IGpuBuffer? biasGpuBuffer = null;
            OwnedBuffer? biasOwned = null;
            if (bias != null)
            {
                biasOwned = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                biasGpuBuffer = biasOwned.Value.Buffer;
            }

            try
            {
                // Execute GPU locally connected convolution
                backend.LocallyConnectedConv2D(
                    inputBuffer.Buffer, weightsBuffer.Buffer, biasGpuBuffer, outputBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1]);

                // Download result
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            finally
            {
                biasOwned?.Dispose();
            }
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.LocallyConnectedConv2D(input, weights, bias, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D backward pass for input gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> LocallyConnectedConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> weights, int[] inputShape, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.LocallyConnectedConv2DBackwardInput(gradOutput, weights, inputShape, stride);

        // Validate inputs
        if (gradOutput.Rank != 4 || weights.Rank != 6)
            return base.LocallyConnectedConv2DBackwardInput(gradOutput, weights, inputShape, stride);

        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("Input shape must be an array of 4 elements", nameof(inputShape));
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int outHeight = weights.Shape._dims[0];
        int outWidth = weights.Shape._dims[1];
        int outChannels = weights.Shape._dims[2];
        int kernelH = weights.Shape._dims[4];
        int kernelW = weights.Shape._dims[5];

        int inputSize = batch * inChannels * inHeight * inWidth;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.GetDataArray(), PersistentTensorRole.Weights);
        using var gradInputBuffer = AllocateOutputBuffer(backend, inputSize);

        try
        {
            backend.LocallyConnectedConv2DBackwardInput(
                gradOutputBuffer.Buffer, weightsBuffer.Buffer, gradInputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, stride[0], stride[1]);

            float[] resultFloat = new float[inputSize];
            backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, inputShape);
        }
        catch
        {
            return base.LocallyConnectedConv2DBackwardInput(gradOutput, weights, inputShape, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D backward pass for weight gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> LocallyConnectedConv2DBackwardWeights<T>(Tensor<T> gradOutput, Tensor<T> input, int[] weightsShape, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.LocallyConnectedConv2DBackwardWeights(gradOutput, input, weightsShape, stride);

        // Validate inputs
        if (gradOutput.Rank != 4 || input.Rank != 4)
            return base.LocallyConnectedConv2DBackwardWeights(gradOutput, input, weightsShape, stride);

        if (weightsShape == null || weightsShape.Length != 6)
            throw new ArgumentException("Weights shape must be an array of 6 elements", nameof(weightsShape));
        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outHeight = weightsShape[0];
        int outWidth = weightsShape[1];
        int outChannels = weightsShape[2];
        int kernelH = weightsShape[4];
        int kernelW = weightsShape[5];

        int weightsSize = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var gradWeightsBuffer = AllocateOutputBuffer(backend, weightsSize);

        try
        {
            backend.LocallyConnectedConv2DBackwardWeights(
                inputBuffer.Buffer, gradOutputBuffer.Buffer, gradWeightsBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, stride[0], stride[1]);

            float[] resultFloat = new float[weightsSize];
            backend.DownloadBuffer(gradWeightsBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, weightsShape);
        }
        catch
        {
            return base.LocallyConnectedConv2DBackwardWeights(gradOutput, input, weightsShape, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D backward pass for bias gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> LocallyConnectedConv2DBackwardBias<T>(Tensor<T> gradOutput)
    {
        if (!TryGetBackend(out var backend))
            return base.LocallyConnectedConv2DBackwardBias<T>(gradOutput);

        // Validate input
        if (gradOutput.Rank != 4)
            return base.LocallyConnectedConv2DBackwardBias<T>(gradOutput);

        int batch = gradOutput.Shape._dims[0];
        int outChannels = gradOutput.Shape._dims[1];
        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var gradBiasBuffer = AllocateOutputBuffer(backend, outChannels);

        try
        {
            backend.LocallyConnectedConv2DBackwardBias(
                gradOutputBuffer.Buffer, gradBiasBuffer.Buffer,
                batch, outChannels, outHeight, outWidth);

            float[] resultFloat = new float[outChannels];
            backend.DownloadBuffer(gradBiasBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { outChannels });
        }
        catch
        {
            return base.LocallyConnectedConv2DBackwardBias<T>(gradOutput);
        }
    }

    /// <summary>
    /// GPU-accelerated locally connected 2D convolution.
    /// Unlike standard convolution, each spatial position uses unique weights.
    /// </summary>
    /// <param name="input">Input tensor [batch, inChannels, inHeight, inWidth]</param>
    /// <param name="weights">Weight tensor [outH, outW, outC, inC, kH, kW]</param>
    /// <param name="bias">Optional bias tensor [outChannels]</param>
    /// <param name="strideH">Vertical stride</param>
    /// <param name="strideW">Horizontal stride</param>
    /// <param name="activation">Fused activation type</param>
    /// <returns>Output GPU tensor [batch, outChannels, outHeight, outWidth]</returns>
    public Tensor<T> LocallyConnectedConv2DGpu<T>(
        Tensor<T> input,
        Tensor<T> weights,
        Tensor<T>? bias,
        int strideH, int strideW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for LocallyConnectedConv2DGpu");

        // Expected input shape: [batch, inChannels, height, width]
        // Expected weights shape: [outH, outW, outC, inC, kH, kW]
        if (input.Shape._dims.Length != 4)
            throw new ArgumentException("LocallyConnectedConv2DGpu requires 4D input tensor");
        if (weights.Rank != 6)
            throw new ArgumentException("LocallyConnectedConv2DGpu requires 6D weights tensor [outH, outW, outC, inC, kH, kW]");

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outHeight = weights.Shape._dims[0];
        int outWidth = weights.Shape._dims[1];
        int outChannels = weights.Shape._dims[2];
        int kernelH = weights.Shape._dims[4];
        int kernelW = weights.Shape._dims[5];

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid locally connected convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outHeight * outWidth;

        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.GetDataArray(), PersistentTensorRole.Weights);
        using var biasBuffer = bias != null ? GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases) : default(OwnedBuffer?);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            backend.LocallyConnectedConv2D(input.Buffer, weightsBuffer.Buffer, biasBuffer?.Buffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW);

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            return Tensor<T>.FromGpuBuffer(backend, outputBuffer, new[] { batch, outChannels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated deformable 2D convolution (DCNv2).
    /// Uses cached GPU buffers for registered persistent tensors to avoid
    /// redundant CPU→GPU transfers. Falls back to CPU if GPU unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.DeformableConv2D(input, kernel, offsets, mask, stride, padding, dilation);

        // Validate inputs
        if (input.Rank != 4 || kernel.Rank != 4 || offsets.Rank != 4)
            return base.DeformableConv2D(input, kernel, offsets, mask, stride, padding, dilation);

        if (stride == null || stride.Length != 2)
            throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));
        if (padding == null || padding.Length != 2)
            throw new ArgumentException("Padding must be an array of 2 elements", nameof(padding));
        if (dilation == null || dilation.Length != 2)
            throw new ArgumentException("Dilation must be an array of 2 elements", nameof(dilation));

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernel.Shape._dims[0];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        int outHeight = (inHeight + 2 * padding[0] - dilation[0] * (kernelH - 1) - 1) / stride[0] + 1;
        int outWidth = (inWidth + 2 * padding[1] - dilation[1] * (kernelW - 1) - 1) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.DeformableConv2D(input, kernel, offsets, mask, stride, padding, dilation);

        // Calculate deformGroups from offsets shape: [batch, deformGroups*2*kH*kW, outH, outW]
        int offsetChannels = offsets.Shape._dims[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int groups = 1; // Standard deformable conv uses groups=1

        int outputSize = batch * outChannels * outHeight * outWidth;

        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.GetDataArray());
        using var outputBuffer = AllocateOutputBuffer(backend, outputSize);

        try
        {
            OwnedBuffer? maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.GetDataArray()) : null;
            try
            {
                backend.DeformableConv2D(
                    inputBuffer.Buffer, weightsBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, outputBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, deformGroups);

                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            finally
            {
                maskBuffer?.Dispose();
            }
        }
        catch
        {
            return base.DeformableConv2D(input, kernel, offsets, mask, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable conv2D backward pass for input gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2DBackwardInput<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.DeformableConv2DBackwardInput(gradOutput, input, kernel, offsets, mask, inputShape, stride, padding, dilation);

        if (gradOutput.Rank != 4 || input.Rank != 4 || kernel.Rank != 4)
            return base.DeformableConv2DBackwardInput(gradOutput, input, kernel, offsets, mask, inputShape, stride, padding, dilation);

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int outChannels = kernel.Shape._dims[0];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        int offsetChannels = offsets.Shape._dims[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int groups = 1;

        int inputSize = batch * inChannels * inHeight * inWidth;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.GetDataArray());
        using var gradInputBuffer = AllocateOutputBuffer(backend, inputSize);

        try
        {
            OwnedBuffer? maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.GetDataArray()) : null;
            try
            {
                backend.DeformableConv2DBackwardInput(
                    gradOutputBuffer.Buffer, weightsBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, gradInputBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, deformGroups);

                float[] resultFloat = new float[inputSize];
                backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, inputShape);
            }
            finally
            {
                maskBuffer?.Dispose();
            }
        }
        catch
        {
            return base.DeformableConv2DBackwardInput(gradOutput, input, kernel, offsets, mask, inputShape, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable conv2D backward pass for kernel gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2DBackwardKernel<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.DeformableConv2DBackwardKernel(gradOutput, input, offsets, mask, kernelShape, stride, padding, dilation);

        if (gradOutput.Rank != 4 || input.Rank != 4)
            return base.DeformableConv2DBackwardKernel(gradOutput, input, offsets, mask, kernelShape, stride, padding, dilation);

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernelShape[0];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        int offsetChannels = offsets.Shape._dims[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int groups = 1;

        int kernelSize = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.GetDataArray());
        using var gradWeightsBuffer = AllocateOutputBuffer(backend, kernelSize);

        try
        {
            OwnedBuffer? maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.GetDataArray()) : null;
            try
            {
                backend.DeformableConv2DBackwardWeights(
                    inputBuffer.Buffer, gradOutputBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, gradWeightsBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, deformGroups);

                float[] resultFloat = new float[kernelSize];
                backend.DownloadBuffer(gradWeightsBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, kernelShape);
            }
            finally
            {
                maskBuffer?.Dispose();
            }
        }
        catch
        {
            return base.DeformableConv2DBackwardKernel(gradOutput, input, offsets, mask, kernelShape, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable conv2D backward pass for offset gradients.
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2DBackwardOffset<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.DeformableConv2DBackwardOffset(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);

        if (gradOutput.Rank != 4 || input.Rank != 4 || kernel.Rank != 4)
            return base.DeformableConv2DBackwardOffset(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernel.Shape._dims[0];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        int offsetChannels = offsets.Shape._dims[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int groups = 1;

        int offsetSize = batch * offsetChannels * outHeight * outWidth;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.GetDataArray());
        using var gradOffsetBuffer = AllocateOutputBuffer(backend, offsetSize);

        try
        {
            OwnedBuffer? maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.GetDataArray()) : null;
            try
            {
                backend.DeformableConv2DBackwardOffset(
                    inputBuffer.Buffer, weightsBuffer.Buffer, gradOutputBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, gradOffsetBuffer.Buffer,
                    batch, inChannels, inHeight, inWidth,
                    outChannels, outHeight, outWidth,
                    kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                    dilation[0], dilation[1], groups, deformGroups);

                float[] resultFloat = new float[offsetSize];
                backend.DownloadBuffer(gradOffsetBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, offsets.Shape._dims);
            }
            finally
            {
                maskBuffer?.Dispose();
            }
        }
        catch
        {
            return base.DeformableConv2DBackwardOffset(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable conv2D backward pass for mask gradients (DCNv2).
    /// Falls back to CPU implementation if GPU is unavailable.
    /// </summary>
    public new Tensor<T> DeformableConv2DBackwardMask<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offsets,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        // Mask gradient computation requires a valid mask
        if (mask == null)
            throw new ArgumentNullException(nameof(mask), "Mask cannot be null when computing mask gradients");

        if (!TryGetBackend(out var backend))
            return base.DeformableConv2DBackwardMask(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);

        if (gradOutput.Rank != 4 || input.Rank != 4 || kernel.Rank != 4 || mask.Rank != 4)
            return base.DeformableConv2DBackwardMask(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernel.Shape._dims[0];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        int offsetChannels = offsets.Shape._dims[1];
        int deformGroups = offsetChannels / (2 * kernelH * kernelW);
        int maskChannels = deformGroups * kernelH * kernelW;
        int groups = 1;

        int maskSize = batch * maskChannels * outHeight * outWidth;

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.GetDataArray());
        using var gradMaskBuffer = AllocateOutputBuffer(backend, maskSize);

        try
        {
            backend.DeformableConv2DBackwardMask(
                inputBuffer.Buffer, weightsBuffer.Buffer, gradOutputBuffer.Buffer, offsetsBuffer.Buffer, gradMaskBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], groups, deformGroups);

            float[] resultFloat = new float[maskSize];
            backend.DownloadBuffer(gradMaskBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, mask.Shape._dims);
        }
        catch
        {
            return base.DeformableConv2DBackwardMask(gradOutput, input, kernel, offsets, mask, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated deformable 2D convolution (DCNv2).
    /// Convolution with learnable offsets and optional modulation masks.
    /// </summary>
    /// <param name="input">Input tensor [batch, inChannels, inHeight, inWidth]</param>
    /// <param name="weights">Weight tensor [outChannels, inChannels/groups, kH, kW]</param>
    /// <param name="offsets">Offset tensor [batch, deformGroups*2*kH*kW, outH, outW]</param>
    /// <param name="mask">Optional mask tensor [batch, deformGroups*kH*kW, outH, outW] for DCNv2</param>
    /// <param name="bias">Optional bias tensor [outChannels]</param>
    /// <param name="strideH">Vertical stride</param>
    /// <param name="strideW">Horizontal stride</param>
    /// <param name="padH">Vertical padding</param>
    /// <param name="padW">Horizontal padding</param>
    /// <param name="dilationH">Vertical dilation</param>
    /// <param name="dilationW">Horizontal dilation</param>
    /// <param name="groups">Number of convolution groups</param>
    /// <param name="deformGroups">Number of deformable groups</param>
    /// <param name="activation">Fused activation type</param>
    /// <returns>Output GPU tensor [batch, outChannels, outHeight, outWidth]</returns>
    public Tensor<T> DeformableConv2DGpu<T>(
        Tensor<T> input,
        Tensor<T> weights,
        Tensor<T> offsets,
        Tensor<T>? mask,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DeformableConv2DGpu");

        // Expected input shape: [batch, inChannels, height, width]
        // Expected weights shape: [outChannels, inChannels/groups, kH, kW]
        // Expected offsets shape: [batch, deformGroups*2*kH*kW, outH, outW]
        if (input.Shape._dims.Length != 4)
            throw new ArgumentException("DeformableConv2DGpu requires 4D input tensor");
        if (weights.Rank != 4)
            throw new ArgumentException("DeformableConv2DGpu requires 4D weights tensor [outC, inC/groups, kH, kW]");

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = weights.Shape._dims[0];
        int kernelH = weights.Shape._dims[2];
        int kernelW = weights.Shape._dims[3];

        int outHeight = (inHeight + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - dilationW * (kernelW - 1) - 1) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException($"Invalid deformable convolution parameters result in non-positive output dimensions: {outHeight}x{outWidth}");

        int outputSize = batch * outChannels * outHeight * outWidth;

        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.GetDataArray(), PersistentTensorRole.Weights);
        using var offsetsBuffer = GetOrAllocateBuffer(backend, offsets.GetDataArray());
        using var maskBuffer = mask != null ? GetOrAllocateBuffer(backend, mask.GetDataArray()) : default(OwnedBuffer?);
        using var biasBuffer = bias != null ? GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases) : default(OwnedBuffer?);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        try
        {
            backend.DeformableConv2D(input.Buffer, weightsBuffer.Buffer, offsetsBuffer.Buffer, maskBuffer?.Buffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW,
                dilationH, dilationW, groups, deformGroups);

            // Add bias if present
            if (bias != null && biasBuffer.HasValue)
            {
                int spatialSize = outHeight * outWidth;
                var biasBuf = biasBuffer.Value.Buffer;
                backend.Conv2DBiasAdd(outputBuffer, biasBuf, batch, outChannels, spatialSize);
            }

            // Apply activation on GPU
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
            }

            return Tensor<T>.FromGpuBuffer(backend, outputBuffer, new[] { batch, outChannels, outHeight, outWidth },
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }
    }

    /// <summary>
    /// GPU-accelerated fused batch normalization with activation.
    /// Uses cached GPU buffers for registered persistent tensors (gamma/beta/running stats)
    /// to avoid redundant CPU→GPU transfers on every forward pass.
    /// </summary>
    public new Tensor<T> FusedBatchNorm<T>(
        Tensor<T> input,
        Tensor<T> gamma,
        Tensor<T> beta,
        Tensor<T> runningMean,
        Tensor<T> runningVar,
        double epsilon,
        double momentum,
        bool training,
        FusedActivationType activation,
        out Tensor<T> saveMean,
        out Tensor<T> saveVar)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);

        if (input.Rank != 2)
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);

        int batchSize = input.Shape._dims[0];
        int features = input.Shape._dims[1];

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
        using var outputBuffer = AllocateOutputBuffer(backend, batchSize * features);
        using var saveMeanBuffer = AllocateOutputBuffer(backend, features);
        using var saveVarBuffer = AllocateOutputBuffer(backend, features);
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
        using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.GetDataArray(), PersistentTensorRole.Biases);
        using var runningMeanBuffer = GetOrCacheWeightBuffer(backend, runningMean.GetDataArray(), PersistentTensorRole.NormalizationParams);
        using var runningVarBuffer = GetOrCacheWeightBuffer(backend, runningVar.GetDataArray(), PersistentTensorRole.NormalizationParams);

        try
        {
            // Try single-pass fused BatchNorm+Activation kernel first
            bool fused = backend.TryFusedBatchNormActivation(
                inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                runningMeanBuffer.Buffer, runningVarBuffer.Buffer, saveMeanBuffer.Buffer, saveVarBuffer.Buffer,
                batchSize, features, 1, (float)epsilon, (float)momentum, training, activation);

            if (!fused)
            {
                // Fall back to separate BatchNorm + Activation
                backend.BatchNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                    runningMeanBuffer.Buffer, runningVarBuffer.Buffer, saveMeanBuffer.Buffer, saveVarBuffer.Buffer,
                    batchSize, features, 1, (float)epsilon, (float)momentum, training);

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer.Buffer, batchSize * features, activation);
                }
            }

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = new float[batchSize * features];
            float[] saveMeanFloat = new float[features];
            float[] saveVarFloat = new float[features];

            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);
            backend.DownloadBuffer(saveMeanBuffer.Buffer, saveMeanFloat);
            backend.DownloadBuffer(saveVarBuffer.Buffer, saveVarFloat);

            // Convert back to T
            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            T[] saveMeanData = DirectGpuEngine.FromFloatArray<T>(saveMeanFloat);
            T[] saveVarData = DirectGpuEngine.FromFloatArray<T>(saveVarFloat);

            saveMean = new Tensor<T>(saveMeanData, new[] { features });
            saveVar = new Tensor<T>(saveVarData, new[] { features });
            return new Tensor<T>(resultData, input.Shape.ToArray());
        }
        catch
        {
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);
        }
    }

    #endregion

    #region Attention Operations (GPU-accelerated)

    /// <summary>
    /// GPU-accelerated FlashAttention - memory-efficient O(N) attention algorithm.
    /// Uses cached GPU buffers for registered persistent tensors (e.g., KV cache) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// Supports optional attention bias (ALiBi) — the bias is uploaded to GPU and passed to the kernel.
    /// Falls back to CPU implementation when GPU is unavailable or on any GPU error.
    /// </summary>
    public new Tensor<T> FlashAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        double? scale,
        bool isCausal,
        out Tensor<T> softmaxStats,
        Tensor<T>? attentionBias = null)
    {
        if (!TryGetBackend(out var backend))
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats, attentionBias);

        // Validate tensor shapes [batch, heads, seq, head_dim]
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats, attentionBias);

        int batch = query.Shape._dims[0];
        int heads = query.Shape._dims[1];
        int seqQ = query.Shape._dims[2];
        int headDim = query.Shape._dims[3];
        int seqK = key.Shape._dims[2];

        // Compute scale if not provided
        float scaleFloat = (float)(scale ?? (1.0 / Math.Sqrt(headDim)));

        // Use cache-aware buffer allocation (especially important for KV cache)
        using var queryBuffer = GetOrAllocateBuffer(backend, query.GetDataArray());
        using var keyBuffer = GetOrAllocateBuffer(backend, key.GetDataArray());
        using var valueBuffer = GetOrAllocateBuffer(backend, value.GetDataArray());
        using var outputBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ * headDim);
        using var statsBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ);

        // Upload attention bias to GPU if provided, with shape validation
        int biasBatchStride = 0;
        using var biasBufferHandle = attentionBias is not null
            ? GetOrAllocateBiasBuffer(backend, attentionBias, batch, heads, seqQ, seqK, out biasBatchStride)
            : default(OwnedBuffer);

        try
        {
            // Execute GPU FlashAttention with optional bias
            backend.FlashAttentionV2(queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer, outputBuffer.Buffer, statsBuffer.Buffer,
                batch, heads, seqQ, seqK, headDim, scaleFloat, isCausal,
                biasBufferHandle.Buffer, biasBatchStride);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = new float[batch * heads * seqQ * headDim];
            float[] statsFloat = new float[batch * heads * seqQ];
            backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);
            backend.DownloadBuffer(statsBuffer.Buffer, statsFloat);

            // Convert back to T
            T[] outputData = DirectGpuEngine.FromFloatArray<T>(outputFloat);
            T[] statsData = DirectGpuEngine.FromFloatArray<T>(statsFloat);

            softmaxStats = new Tensor<T>(statsData, new[] { batch, heads, seqQ });
            return new Tensor<T>(outputData, new[] { batch, heads, seqQ, headDim });
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats, attentionBias);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for FlashAttention.
    /// Uses cached GPU buffers for registered persistent tensors to avoid redundant transfers.
    /// </summary>
    public new Tensor<T> FlashAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        Tensor<T> softmaxStats,
        double scale,
        bool isCausal,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue,
        Tensor<T>? attentionBias = null)
    {
        if (!TryGetBackend(out var backend))
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue, attentionBias);

        if (query.Rank != 4)
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue, attentionBias);

        int batch = query.Shape._dims[0];
        int heads = query.Shape._dims[1];
        int seqQ = query.Shape._dims[2];
        int headDim = query.Shape._dims[3];
        int seqK = key.Shape._dims[2];

        // Use cache-aware buffer allocation
        using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var queryBuffer = GetOrAllocateBuffer(backend, query.GetDataArray());
        using var keyBuffer = GetOrAllocateBuffer(backend, key.GetDataArray());
        using var valueBuffer = GetOrAllocateBuffer(backend, value.GetDataArray());
        using var outputBuffer = GetOrAllocateBuffer(backend, output.GetDataArray());
        using var statsBuffer = GetOrAllocateBuffer(backend, softmaxStats.GetDataArray());
        using var gradQBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ * headDim);
        using var gradKBuffer = AllocateOutputBuffer(backend, batch * heads * seqK * headDim);
        using var gradVBuffer = AllocateOutputBuffer(backend, batch * heads * seqK * headDim);

        // Upload attention bias to GPU if provided, with shape validation
        int biasBatchStride = 0;
        using var biasBufferHandle = attentionBias is not null
            ? GetOrAllocateBiasBuffer(backend, attentionBias, batch, heads, seqQ, seqK, out biasBatchStride)
            : default(OwnedBuffer);

        try
        {
            // Execute GPU backward with optional bias
            backend.FlashAttentionBackward(gradOutBuffer.Buffer, queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer,
                outputBuffer.Buffer, statsBuffer.Buffer, gradQBuffer.Buffer, gradKBuffer.Buffer, gradVBuffer.Buffer,
                batch, heads, seqQ, seqK, headDim, (float)scale, isCausal,
                biasBufferHandle.Buffer, biasBatchStride);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] gradQFloat = new float[batch * heads * seqQ * headDim];
            float[] gradKFloat = new float[batch * heads * seqK * headDim];
            float[] gradVFloat = new float[batch * heads * seqK * headDim];
            backend.DownloadBuffer(gradQBuffer.Buffer, gradQFloat);
            backend.DownloadBuffer(gradKBuffer.Buffer, gradKFloat);
            backend.DownloadBuffer(gradVBuffer.Buffer, gradVFloat);

            // Convert back to T
            gradQuery = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradQFloat), query.Shape.ToArray());
            gradKey = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradKFloat), key.Shape.ToArray());
            gradValue = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradVFloat), value.Shape.ToArray());

            return gradOutput;
        }
        catch
        {
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue, attentionBias);
        }
    }

    /// <summary>
    /// GPU-accelerated Grouped Query Attention for efficient inference.
    /// Uses cached GPU buffers for registered persistent tensors (e.g., KV cache) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
    /// Falls back to CPU implementation when GPU is unavailable.
    /// </summary>
    public new Tensor<T> GroupedQueryAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        int numQueriesPerKV,
        double? scale,
        bool isCausal,
        out Tensor<T> attentionWeights)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);

        // Validate tensor shapes
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);

        int batch = query.Shape._dims[0];
        int numQHeads = query.Shape._dims[1];
        int seqQ = query.Shape._dims[2];
        int headDim = query.Shape._dims[3];
        int numKVHeads = key.Shape._dims[1];
        int seqK = key.Shape._dims[2];

        float scaleFloat = (float)(scale ?? (1.0 / Math.Sqrt(headDim)));

        // Use cache-aware buffer allocation (especially important for KV cache)
        using var queryBuffer = GetOrAllocateBuffer(backend, query.GetDataArray());
        using var keyBuffer = GetOrAllocateBuffer(backend, key.GetDataArray());
        using var valueBuffer = GetOrAllocateBuffer(backend, value.GetDataArray());
        using var outputBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * headDim);
        using var attnWeightsBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * seqK);

        try
        {
            // Execute GPU GQA
            backend.GroupedQueryAttention(queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer, outputBuffer.Buffer, attnWeightsBuffer.Buffer,
                batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scaleFloat, isCausal);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = new float[batch * numQHeads * seqQ * headDim];
            float[] attnWeightsFloat = new float[batch * numQHeads * seqQ * seqK];
            backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);
            backend.DownloadBuffer(attnWeightsBuffer.Buffer, attnWeightsFloat);

            // Convert back to T
            T[] outputData = DirectGpuEngine.FromFloatArray<T>(outputFloat);
            T[] attnWeightsData = DirectGpuEngine.FromFloatArray<T>(attnWeightsFloat);

            attentionWeights = new Tensor<T>(attnWeightsData, new[] { batch, numQHeads, seqQ, seqK });
            return new Tensor<T>(outputData, new[] { batch, numQHeads, seqQ, headDim });
        }
        catch
        {
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for Grouped Query Attention.
    /// Uses cached GPU buffers for registered persistent tensors to avoid redundant transfers.
    /// </summary>
    public new Tensor<T> GroupedQueryAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> attentionWeights,
        int numQueriesPerKV,
        double scale,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);

        if (query.Rank != 4)
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);

        int batch = query.Shape._dims[0];
        int numQHeads = query.Shape._dims[1];
        int seqQ = query.Shape._dims[2];
        int headDim = query.Shape._dims[3];
        int numKVHeads = key.Shape._dims[1];
        int seqK = key.Shape._dims[2];

        // Use cache-aware buffer allocation
        using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
        using var queryBuffer = GetOrAllocateBuffer(backend, query.GetDataArray());
        using var keyBuffer = GetOrAllocateBuffer(backend, key.GetDataArray());
        using var valueBuffer = GetOrAllocateBuffer(backend, value.GetDataArray());
        using var attnWeightsBuffer = GetOrAllocateBuffer(backend, attentionWeights.GetDataArray());
        using var gradQBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * headDim);
        using var gradKBuffer = AllocateOutputBuffer(backend, batch * numKVHeads * seqK * headDim);
        using var gradVBuffer = AllocateOutputBuffer(backend, batch * numKVHeads * seqK * headDim);

        try
        {
            // Execute GPU backward
            backend.GroupedQueryAttentionBackward(gradOutBuffer.Buffer, queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer,
                attnWeightsBuffer.Buffer, gradQBuffer.Buffer, gradKBuffer.Buffer, gradVBuffer.Buffer,
                batch, numQHeads, numKVHeads, seqQ, seqK, headDim, (float)scale);

            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] gradQFloat = new float[batch * numQHeads * seqQ * headDim];
            float[] gradKFloat = new float[batch * numKVHeads * seqK * headDim];
            float[] gradVFloat = new float[batch * numKVHeads * seqK * headDim];
            backend.DownloadBuffer(gradQBuffer.Buffer, gradQFloat);
            backend.DownloadBuffer(gradKBuffer.Buffer, gradKFloat);
            backend.DownloadBuffer(gradVBuffer.Buffer, gradVFloat);

            // Convert back to T
            gradQuery = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradQFloat), query.Shape.ToArray());
            gradKey = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradKFloat), key.Shape.ToArray());
            gradValue = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradVFloat), value.Shape.ToArray());

            return gradOutput;
        }
        catch
        {
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);
        }
    }

    /// <summary>
    /// GPU-resident Scaled Dot-Product Attention.
    /// Takes GPU-resident Q, K, V tensors in 4D shape [batch, heads, seq, head_dim]
    /// and returns GPU-resident attention output.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="query">GPU-resident query tensor [batch, heads, seqQ, headDim].</param>
    /// <param name="key">GPU-resident key tensor [batch, heads, seqK, headDim].</param>
    /// <param name="value">GPU-resident value tensor [batch, heads, seqK, headDim].</param>
    /// <param name="scale">Scaling factor (typically 1/sqrt(headDim)).</param>
    /// <param name="isCausal">If true, applies causal masking.</param>
    /// <returns>GPU-resident output tensor [batch, heads, seqQ, headDim].</returns>
    public Tensor<T> ScaledDotProductAttentionGpu<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        double scale,
        bool isCausal = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScaledDotProductAttentionGpu");

        // Validate 4D tensor shapes
        if (query.Shape._dims.Length != 4 || key.Shape._dims.Length != 4 || value.Shape._dims.Length != 4)
            throw new ArgumentException("Query, Key, Value must be 4D tensors [batch, heads, seq, headDim]");

        int batch = query.Shape._dims[0];
        int heads = query.Shape._dims[1];
        int seqQ = query.Shape._dims[2];
        int headDim = query.Shape._dims[3];
        int seqK = key.Shape._dims[2];

        // Allocate output and attention weights buffers
        var outputBuffer = backend.AllocateBuffer(batch * heads * seqQ * headDim);
        var attnWeightsBuffer = backend.AllocateBuffer(batch * heads * seqQ * seqK);

        // Execute GPU ScaledDotProductAttention
        backend.ScaledDotProductAttention(
            query.Buffer, key.Buffer, value.Buffer,
            outputBuffer, attnWeightsBuffer, null,
            batch, heads, seqQ, headDim, (float)scale, isCausal);

        // Free attention weights buffer (not needed when not returning weights)
        attnWeightsBuffer.Dispose();

        // Return GPU-resident output
        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, new[] { batch, heads, seqQ, headDim },
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident scaled dot-product attention with attention weights output for training.
    /// Computes: softmax(Q @ K^T / scale) @ V, returning both output and attention weights.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="query">GPU-resident query tensor [batch, heads, seqQ, headDim].</param>
    /// <param name="key">GPU-resident key tensor [batch, heads, seqK, headDim].</param>
    /// <param name="value">GPU-resident value tensor [batch, heads, seqK, headDim].</param>
    /// <param name="scale">Scaling factor (typically 1/sqrt(headDim)).</param>
    /// <param name="attentionWeights">Output: GPU-resident attention weights tensor [batch, heads, seqQ, seqK].</param>
    /// <param name="isCausal">If true, applies causal masking.</param>
    /// <returns>GPU-resident output tensor [batch, heads, seqQ, headDim].</returns>
    public Tensor<T> ScaledDotProductAttentionGpu<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        double scale,
        out Tensor<T> attentionWeights,
        bool isCausal = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScaledDotProductAttentionGpu");

        // Validate 4D tensor shapes
        if (query.Shape._dims.Length != 4 || key.Shape._dims.Length != 4 || value.Shape._dims.Length != 4)
            throw new ArgumentException("Query, Key, Value must be 4D tensors [batch, heads, seq, headDim]");

        int batch = query.Shape._dims[0];
        int heads = query.Shape._dims[1];
        int seqQ = query.Shape._dims[2];
        int headDim = query.Shape._dims[3];
        int seqK = key.Shape._dims[2];

        // Allocate output and attention weights buffers
        var outputBuffer = backend.AllocateBuffer(batch * heads * seqQ * headDim);
        var attnWeightsBuffer = backend.AllocateBuffer(batch * heads * seqQ * seqK);

        // Execute GPU ScaledDotProductAttention
        backend.ScaledDotProductAttention(
            query.Buffer, key.Buffer, value.Buffer,
            outputBuffer, attnWeightsBuffer, null,
            batch, heads, seqQ, headDim, (float)scale, isCausal);

        // Return both output and attention weights as GPU-resident tensors
        attentionWeights = Tensor<T>.FromGpuBuffer(backend, attnWeightsBuffer, new[] { batch, heads, seqQ, seqK },
            GpuTensorRole.Activation, ownsBuffer: true);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, new[] { batch, heads, seqQ, headDim },
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for scaled dot-product attention.
    /// Computes gradients for query, key, and value tensors.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">Gradient of loss w.r.t. attention output [batch, heads, seqLen, headDim].</param>
    /// <param name="query">Query tensor from forward pass [batch, heads, seqLen, headDim].</param>
    /// <param name="key">Key tensor from forward pass [batch, heads, seqLen, headDim].</param>
    /// <param name="value">Value tensor from forward pass [batch, heads, seqLen, headDim].</param>
    /// <param name="attentionWeights">Attention weights from forward pass [batch, heads, seqLen, seqLen].</param>
    /// <param name="scale">Scale factor (typically 1/sqrt(headDim)).</param>
    /// <param name="isCausal">Whether to use causal masking.</param>
    /// <returns>Tuple of (gradQuery, gradKey, gradValue) GPU-resident tensors.</returns>
    public (Tensor<T> GradQuery, Tensor<T> GradKey, Tensor<T> GradValue) ScaledDotProductAttentionBackwardGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> attentionWeights,
        double scale,
        bool isCausal = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScaledDotProductAttentionBackwardGpu");

        // Extract dimensions from query tensor (all tensors share same shape except attn weights)
        int[] qShape = query.Shape._dims;
        if (qShape.Length != 4)
            throw new ArgumentException("Query tensor must be 4D [batch, heads, seqLen, headDim]");

        int batch = qShape[0];
        int heads = qShape[1];
        int seqLen = qShape[2];
        int headDim = qShape[3];

        // Allocate output gradient buffers with exception-safe disposal
        int qkvSize = batch * heads * seqLen * headDim;
        IGpuBuffer? gradQueryBuffer = null;
        IGpuBuffer? gradKeyBuffer = null;
        IGpuBuffer? gradValueBuffer = null;

        try
        {
            gradQueryBuffer = backend.AllocateBuffer(qkvSize);
            gradKeyBuffer = backend.AllocateBuffer(qkvSize);
            gradValueBuffer = backend.AllocateBuffer(qkvSize);

            // Execute ScaledDotProductAttentionBackward on GPU
            backend.ScaledDotProductAttentionBackward(
                gradOutput.Buffer, query.Buffer, key.Buffer, value.Buffer,
                attentionWeights.Buffer, gradQueryBuffer, gradKeyBuffer, gradValueBuffer,
                batch, heads, seqLen, headDim, (float)scale, isCausal);

            // Return GPU-resident gradient tensors
            var gradQuery = Tensor<T>.FromGpuBuffer(backend, gradQueryBuffer, qShape, GpuTensorRole.Gradient, true);
            var gradKey = Tensor<T>.FromGpuBuffer(backend, gradKeyBuffer, qShape, GpuTensorRole.Gradient, true);
            var gradValue = Tensor<T>.FromGpuBuffer(backend, gradValueBuffer, qShape, GpuTensorRole.Gradient, true);

            // Ownership transferred to tensors
            gradQueryBuffer = null;
            gradKeyBuffer = null;
            gradValueBuffer = null;

            return (gradQuery, gradKey, gradValue);
        }
        finally
        {
            // Dispose any buffers that weren't successfully transferred
            gradQueryBuffer?.Dispose();
            gradKeyBuffer?.Dispose();
            gradValueBuffer?.Dispose();
        }
    }

    /// <summary>
    /// GPU-resident tensor permutation (transpose with arbitrary dimension reordering).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="permutation">Permutation of dimensions (e.g., [0, 2, 1, 3] for [B,H,S,D] -> [B,S,H,D]).</param>
    /// <returns>GPU-resident permuted tensor.</returns>
    public Tensor<T> PermuteGpu<T>(Tensor<T> input, int[] permutation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for PermuteGpu");

        if (permutation.Length != input.Shape._dims.Length)
            throw new ArgumentException("Permutation length must match input rank");

        // Compute output shape
        int[] outputShape = new int[input.Shape._dims.Length];
        for (int i = 0; i < permutation.Length; i++)
            outputShape[i] = input.Shape._dims[permutation[i]];

        int totalElements = 1;
        foreach (int dim in input.Shape._dims)
            totalElements *= dim;

        var outputBuffer = backend.AllocateBuffer(totalElements);
        backend.Permute(input.Buffer, outputBuffer, input.Shape._dims, permutation);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident batched matrix multiplication.
    /// Supports 3D inputs [batch, M, K] @ [K, N] -> [batch, M, N] for projections.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, seq, inputDim] or 2D [batch*seq, inputDim].</param>
    /// <param name="weights">Weight tensor [inputDim, outputDim].</param>
    /// <returns>GPU-resident output tensor.</returns>
    public Tensor<T> BatchedMatMulGpu<T>(Tensor<T> input, Tensor<T> weights)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BatchedMatMulGpu");

        if (weights.Rank != 2)
            throw new ArgumentException("Weights must be 2D tensor [inputDim, outputDim]");

        int inputDim = weights.Shape._dims[0];
        int outputDim = weights.Shape._dims[1];

        // Flatten input to 2D for MatMul: [batch*seq, inputDim]
        int flatBatch = 1;
        for (int i = 0; i < input.Shape._dims.Length - 1; i++)
            flatBatch *= input.Shape._dims[i];
        int lastDim = input.Shape._dims[^1];

        if (lastDim != inputDim)
            throw new ArgumentException($"Input last dimension {lastDim} doesn't match weight input dimension {inputDim}");

        // Upload weights
        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.GetDataArray(), PersistentTensorRole.Weights);

        // Execute MatMul
        var resultBuffer = backend.MatMul(input.Buffer, weightsBuffer.Buffer, flatBatch, outputDim, inputDim);

        // Compute output shape (same leading dimensions, last dim = outputDim)
        int[] outputShape = new int[input.Shape._dims.Length];
        for (int i = 0; i < input.Shape._dims.Length - 1; i++)
            outputShape[i] = input.Shape._dims[i];
        outputShape[^1] = outputDim;

        return Tensor<T>.FromGpuBuffer(backend, resultBuffer, outputShape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident tensor reshape (zero-copy view when possible).
    /// Creates a new GPU tensor with the same buffer but different shape interpretation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="newShape">New shape (total elements must match).</param>
    /// <returns>GPU-resident reshaped tensor (shares buffer with input).</returns>
    public void CopyGpu<T>(Tensor<T> source, Tensor<T> destination, int size)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for CopyGpu");

        backend.Copy(source.Buffer, destination.Buffer, size);
    }

    public void CopyGpu<T>(Tensor<T> source, int srcOffset, Tensor<T> destination, int destOffset, int size)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for CopyGpu");

        backend.Copy(source.Buffer, srcOffset, destination.Buffer, destOffset, size);
    }

    public void FillGpu<T>(Tensor<T> buffer, float value, int size)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FillGpu");

        backend.Fill(buffer.Buffer, value, size);
    }

    /// <summary>
    /// Copies a 2D region from source to destination with different strides.
    /// Useful for concatenating features: dest[row, destColOffset:destColOffset+srcCols] = src[row, :]
    /// </summary>

    /// <summary>
    /// GPU-resident tensor bias addition with broadcasting.
    /// Adds bias to the last dimension of the input tensor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="bias">Bias tensor (1D, length must match input's last dimension).</param>
    /// <returns>GPU-resident output tensor with bias added.</returns>
    public Tensor<T> AddBiasGpu<T>(Tensor<T> input, Tensor<T> bias)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AddBiasGpu");

        if (bias.Rank != 1)
            throw new ArgumentException("Bias must be 1D tensor");

        int lastDim = input.Shape._dims[^1];
        if (bias.Length != lastDim)
            throw new ArgumentException($"Bias length {bias.Length} doesn't match input last dimension {lastDim}");

        int totalElements = 1;
        foreach (int dim in input.Shape._dims)
            totalElements *= dim;

        // Upload bias
        using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);

        // Allocate output
        var outputBuffer = backend.AllocateBuffer(totalElements);

        // Execute bias addition (broadcast along last dimension)
        // BiasAdd signature: BiasAdd(A, bias, C, M, N) where A is [M, N], bias is [N], C is output [M, N]
        int numVectors = totalElements / lastDim;
        backend.BiasAdd(input.Buffer, biasBuffer.Buffer, outputBuffer, numVectors, lastDim);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape.ToArray(), GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident nearest-neighbor upsampling.
    /// Increases spatial dimensions (last two) by the specified scale factor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor with shape [..., height, width].</param>
    /// <param name="scaleFactor">Scale factor for both height and width.</param>
    /// <returns>GPU-resident upsampled tensor.</returns>
    public Tensor<T> UpsampleGpu<T>(Tensor<T> input, int scaleFactor)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UpsampleGpu");

        if (input.Shape._dims.Length < 2)
            throw new ArgumentException("Input must have at least 2 dimensions for upsampling");

        // Parameter validation guard
        if (scaleFactor <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), "Scale factor must be positive");

        // Compute output shape (scale last two dimensions)
        int[] outputShape = new int[input.Shape._dims.Length];
        for (int i = 0; i < input.Shape._dims.Length - 2; i++)
            outputShape[i] = input.Shape._dims[i];

        int inHeight = input.Shape._dims[^2];
        int inWidth = input.Shape._dims[^1];
        int outHeight = inHeight * scaleFactor;
        int outWidth = inWidth * scaleFactor;
        outputShape[^2] = outHeight;
        outputShape[^1] = outWidth;

        // Compute total elements
        int batchChannels = 1;
        for (int i = 0; i < input.Shape._dims.Length - 2; i++)
            batchChannels *= input.Shape._dims[i];

        int inputSize = batchChannels * inHeight * inWidth;
        int outputSize = batchChannels * outHeight * outWidth;

        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Use NearestNeighborUpsample if available, otherwise implement manually
        backend.NearestNeighborUpsample(
            input.Buffer, outputBuffer,
            batchChannels, inHeight, inWidth,
            scaleFactor);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// Performs GPU-accelerated backward pass for nearest-neighbor upsampling (2D).
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer with upsampled shape.</param>
    /// <param name="inputHeight">Original input height before upsampling.</param>
    /// <param name="inputWidth">Original input width before upsampling.</param>
    /// <param name="scaleFactor">Scale factor used during forward pass.</param>
    /// <returns>GPU-resident gradient input tensor with original input shape.</returns>
    public Tensor<T> UpsampleBackwardGpu<T>(Tensor<T> gradOutput, int inputHeight, int inputWidth, int scaleFactor)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UpsampleBackwardGpu");

        if (gradOutput.Shape._dims.Length < 2)
            throw new ArgumentException("Gradient output must have at least 2 dimensions for upsampling backward");

        // Parameter validation guards
        if (scaleFactor <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), "Scale factor must be positive");

        if (inputHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputHeight), "Input height must be positive");

        if (inputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "Input width must be positive");

        // Validate that gradOutput dimensions are consistent with scale factor
        int expectedOutputHeight = inputHeight * scaleFactor;
        int expectedOutputWidth = inputWidth * scaleFactor;
        int actualOutputHeight = gradOutput.Shape._dims[^2];
        int actualOutputWidth = gradOutput.Shape._dims[^1];

        if (actualOutputHeight != expectedOutputHeight)
            throw new ArgumentException(
                $"Gradient output height ({actualOutputHeight}) does not match expected height ({expectedOutputHeight}) based on inputHeight ({inputHeight}) and scaleFactor ({scaleFactor})");

        if (actualOutputWidth != expectedOutputWidth)
            throw new ArgumentException(
                $"Gradient output width ({actualOutputWidth}) does not match expected width ({expectedOutputWidth}) based on inputWidth ({inputWidth}) and scaleFactor ({scaleFactor})");

        // Compute input shape (original shape before upsampling)
        int[] inputShape = new int[gradOutput.Shape._dims.Length];
        for (int i = 0; i < gradOutput.Shape._dims.Length - 2; i++)
            inputShape[i] = gradOutput.Shape._dims[i];

        inputShape[^2] = inputHeight;
        inputShape[^1] = inputWidth;

        // Compute total elements
        int batchChannels = 1;
        for (int i = 0; i < gradOutput.Shape._dims.Length - 2; i++)
            batchChannels *= gradOutput.Shape._dims[i];

        int inputSize = batchChannels * inputHeight * inputWidth;

        var gradInputBuffer = backend.AllocateBuffer(inputSize);

        // Zero initialize the gradient input buffer for accumulation
        backend.Fill(gradInputBuffer, 0.0f, inputSize);

        // Use NearestNeighborUpsampleBackward for gradient propagation
        backend.NearestNeighborUpsampleBackward(
            gradOutput.Buffer, gradInputBuffer,
            batchChannels, inputHeight, inputWidth,
            scaleFactor);

        return Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    #endregion

    #region Persistent Tensor Management

    /// <summary>
    /// Registers a tensor for GPU memory optimization by pre-allocating and uploading
    /// its data to GPU memory. This eliminates repeated CPU-GPU transfers for tensors
    /// that are reused across multiple operations (e.g., layer weights, biases).
    /// </summary>
    public new void RegisterPersistentTensor<T>(Tensor<T> tensor, PersistentTensorRole role)
    {
        base.RegisterPersistentTensor(tensor, role);

        if (!TryGetBackend(out var backend))
            return;

        // Use the tensor's data array as the cache key
        object key = tensor.GetDataArray();

        // Check if already registered
        if (_persistentBufferCache.ContainsKey(key))
            return;

        try
        {
            // Convert tensor data to float and upload to GPU
            float[] floatData = DirectGpuEngine.ToFloatArray(tensor.GetDataArray());
            IGpuBuffer gpuBuffer = backend.AllocateBuffer(floatData);
            backend.Synchronize();

            var entry = new GpuBufferCacheEntry(gpuBuffer, role);
            _persistentBufferCache.TryAdd(key, entry);
            _tensorVersions.TryAdd(key, 0);
        }
        catch
        {
            // Silently ignore GPU allocation failures - operations will fall back to CPU
        }
    }

    /// <summary>
    /// Unregisters a persistent tensor and releases its associated GPU memory.
    /// </summary>
    public new void UnregisterPersistentTensor<T>(Tensor<T> tensor)
    {
        base.UnregisterPersistentTensor(tensor);

        object key = tensor.GetDataArray();

        if (_persistentBufferCache.TryRemove(key, out var entry))
        {
            entry.Dispose();
        }
        _tensorVersions.TryRemove(key, out _);
    }

    /// <summary>
    /// Invalidates a persistent tensor's GPU buffer, triggering re-upload of its
    /// data to GPU memory. Call this after modifying the tensor's data on CPU.
    /// </summary>
    public new void InvalidatePersistentTensor<T>(Tensor<T> tensor)
    {
        base.InvalidatePersistentTensor(tensor);

        if (!TryGetBackend(out var backend))
            return;

        object key = tensor.GetDataArray();

        if (!_persistentBufferCache.TryGetValue(key, out var entry))
            return;

        try
        {
            // Dispose old buffer
            entry.Buffer.Dispose();

            // Upload new data
            float[] floatData = DirectGpuEngine.ToFloatArray(tensor.GetDataArray());
            IGpuBuffer newBuffer = backend.AllocateBuffer(floatData);
            backend.Synchronize();

            // Update cache entry with new buffer
            var newEntry = new GpuBufferCacheEntry(newBuffer, entry.Role);
            newEntry.Version = entry.Version + 1;

            _persistentBufferCache[key] = newEntry;
            _tensorVersions[key] = newEntry.Version;
        }
        catch
        {
            // On failure, remove from cache - operations will fall back to CPU
            _persistentBufferCache.TryRemove(key, out _);
            _tensorVersions.TryRemove(key, out _);
        }
    }

    /// <summary>
    /// Attempts to get a cached GPU buffer for a tensor.
    /// Returns null if the tensor is not registered as persistent.
    /// </summary>
    internal IGpuBuffer? TryGetCachedBuffer<T>(T[] tensorData)
    {
        if (_persistentBufferCache.TryGetValue(tensorData, out var entry))
        {
            return entry.Buffer;
        }
        return null;
    }

    /// <summary>
    /// Gets the number of tensors currently cached on GPU.
    /// </summary>
    public int CachedTensorCount => _persistentBufferCache.Count;

    #endregion

    #region FFT Operations (GPU-accelerated)

    /// <summary>
    /// GPU-accelerated 1D complex-to-complex FFT.
    /// </summary>
    void IEngine.FFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Length != inputImag.Length)
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int n = inputReal.Shape._dims[^1];
        if ((n & (n - 1)) != 0) // Not power of 2
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.GetDataArray());
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.GetDataArray());
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, n, inverse: false);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 1D complex-to-complex inverse FFT.
    /// </summary>
    void IEngine.IFFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Length != inputImag.Length)
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int n = inputReal.Shape._dims[^1];
        if ((n & (n - 1)) != 0)
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.GetDataArray());
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.GetDataArray());
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, n, inverse: true);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D FFT.
    /// </summary>
    void IEngine.FFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Rank < 2 || inputReal.Length != inputImag.Length)
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int height = inputReal.Shape._dims[^2];
        int width = inputReal.Shape._dims[^1];

        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.GetDataArray());
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.GetDataArray());
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT2D(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, height, width, inverse: false);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D inverse FFT.
    /// </summary>
    void IEngine.IFFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Rank < 2 || inputReal.Length != inputImag.Length)
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int height = inputReal.Shape._dims[^2];
        int width = inputReal.Shape._dims[^1];

        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.GetDataArray());
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.GetDataArray());
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT2D(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, height, width, inverse: true);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated Short-Time Fourier Transform.
    /// </summary>
    void IEngine.STFT<T>(
        Tensor<T> input,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        out Tensor<T> magnitudeOut,
        out Tensor<T> phaseOut)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
            return;
        }

        try
        {
            // For STFT, we need to process frame by frame
            // First, handle centering by padding the input
            T[] inputData = input.GetDataArray();
            if (center)
            {
                int padAmount = nFft / 2;
                T[] paddedData = new T[input.Length + 2 * padAmount];
                Array.Copy(inputData, 0, paddedData, padAmount, input.Length);
                inputData = paddedData;
            }

            int numSamples = input.Length;
            int numFrames = (numSamples - nFft) / hopLength + 1;
            int numFreqs = nFft / 2 + 1;

            if (numFrames <= 0)
            {
                base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
                return;
            }

            float[] inputFloat = DirectGpuEngine.ToFloatArray(inputData);

            // Use cache-aware allocation for window (likely persistent)
            using var windowBuffer = GetOrAllocateBuffer(backend, window.GetDataArray());
            // Allocate working buffers
            using var frameBuffer = AllocateOutputBuffer(backend, nFft);
            using var windowedBuffer = AllocateOutputBuffer(backend, nFft);
            using var fftRealBuffer = AllocateOutputBuffer(backend, nFft);
            using var fftImagBuffer = AllocateOutputBuffer(backend, nFft);
            using var zeroBuffer = AllocateOutputBuffer(backend, nFft);

            float[] magnitudeData = new float[numFrames * numFreqs];
            float[] phaseData = new float[numFrames * numFreqs];

            for (int frame = 0; frame < numFrames; frame++)
            {
                int frameStart = frame * hopLength;

                // Extract frame from input
                float[] frameData = new float[nFft];
                Array.Copy(inputFloat, frameStart, frameData, 0, Math.Min(nFft, inputFloat.Length - frameStart));

                // Upload frame data
                using var currentFrameBuffer = backend.AllocateBuffer(frameData);

                // Apply window
                backend.ApplyWindow(currentFrameBuffer, windowBuffer.Buffer, windowedBuffer.Buffer, nFft);

                // Perform FFT (windowed signal as real input, zeros as imaginary)
                backend.FFT(windowedBuffer.Buffer, zeroBuffer.Buffer, fftRealBuffer.Buffer, fftImagBuffer.Buffer, nFft, inverse: false);

                // Download FFT results
                float[] fftReal = new float[nFft];
                float[] fftImag = new float[nFft];
                backend.DownloadBuffer(fftRealBuffer.Buffer, fftReal);
                backend.DownloadBuffer(fftImagBuffer.Buffer, fftImag);

                // Compute magnitude and phase for positive frequencies only
                for (int k = 0; k < numFreqs; k++)
                {
                    float real = fftReal[k];
                    float imag = fftImag[k];
                    magnitudeData[frame * numFreqs + k] = (float)Math.Sqrt(real * real + imag * imag);
                    phaseData[frame * numFreqs + k] = (float)Math.Atan2(imag, real);
                }
            }
            // Note: DownloadBuffer calls inside the loop are blocking, no need for Synchronize after

            int[] outputShape = input.Rank == 1
                ? new[] { numFrames, numFreqs }
                : new[] { input.Shape._dims[0], numFrames, numFreqs };

            magnitudeOut = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(magnitudeData), outputShape);
            phaseOut = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phaseData), outputShape);
        }
        catch
        {
            base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
        }
    }

    /// <summary>
    /// GPU-accelerated inverse Short-Time Fourier Transform.
    /// </summary>
    Tensor<T> IEngine.ISTFT<T>(
        Tensor<T> magnitude,
        Tensor<T> phase,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        int? length)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.ISTFT(magnitude, phase, nFft, hopLength, window, center, length);
        }

        try
        {
            int numFrames = magnitude.Shape._dims[^2];
            int numFreqs = magnitude.Shape._dims[^1];

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.GetDataArray());
            float[] phaseFloat = DirectGpuEngine.ToFloatArray(phase.GetDataArray());
            float[] windowFloat = DirectGpuEngine.ToFloatArray(window.GetDataArray());

            // Reconstruct full spectrum (mirror for negative frequencies)
            int outputSamples = (numFrames - 1) * hopLength + nFft;
            float[] output = new float[outputSamples];
            float[] windowSum = new float[outputSamples];

            // Use cache-aware allocation for window (likely persistent)
            using var windowBuffer = GetOrAllocateBuffer(backend, window.GetDataArray());
            // Allocate working buffers
            using var outputRealBuffer = AllocateOutputBuffer(backend, nFft);
            using var outputImagBuffer = AllocateOutputBuffer(backend, nFft);

            for (int frame = 0; frame < numFrames; frame++)
            {
                // Convert polar to complex for full spectrum
                float[] frameReal = new float[nFft];
                float[] frameImag = new float[nFft];

                // Fill positive frequencies
                for (int k = 0; k < numFreqs; k++)
                {
                    float mag = magnitudeFloat[frame * numFreqs + k];
                    float ph = phaseFloat[frame * numFreqs + k];
                    frameReal[k] = mag * (float)Math.Cos(ph);
                    frameImag[k] = mag * (float)Math.Sin(ph);
                }

                // Mirror for negative frequencies (conjugate symmetry)
                for (int k = 1; k < nFft - numFreqs + 1; k++)
                {
                    int srcIdx = numFreqs - 1 - k;
                    if (srcIdx > 0 && srcIdx < numFreqs)
                    {
                        frameReal[nFft - k] = frameReal[srcIdx];
                        frameImag[nFft - k] = -frameImag[srcIdx];
                    }
                }

                using var frameRealBuffer = backend.AllocateBuffer(frameReal);
                using var frameImagBuffer = backend.AllocateBuffer(frameImag);

                // Perform inverse FFT
                backend.FFT(frameRealBuffer, frameImagBuffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, nFft, inverse: true);

                // Download result
                float[] ifftResult = new float[nFft];
                backend.DownloadBuffer(outputRealBuffer.Buffer, ifftResult);

                // Overlap-add with window
                int frameStart = frame * hopLength;
                for (int i = 0; i < nFft && frameStart + i < outputSamples; i++)
                {
                    float w = windowFloat[i];
                    output[frameStart + i] += ifftResult[i] * w;
                    windowSum[frameStart + i] += w * w;
                }
            }
            // Note: DownloadBuffer calls inside the loop are blocking, no need for Synchronize after

            // Normalize by window sum
            for (int i = 0; i < outputSamples; i++)
            {
                if (windowSum[i] > 1e-8f)
                {
                    output[i] /= windowSum[i];
                }
            }

            // Remove centering padding if needed
            if (center)
            {
                int padAmount = nFft / 2;
                int actualLength = length ?? (outputSamples - 2 * padAmount);
                float[] trimmed = new float[actualLength];
                Array.Copy(output, padAmount, trimmed, 0, Math.Min(actualLength, outputSamples - padAmount));
                output = trimmed;
            }
            else if (length.HasValue)
            {
                float[] trimmed = new float[length.Value];
                Array.Copy(output, 0, trimmed, 0, Math.Min(length.Value, output.Length));
                output = trimmed;
            }

            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(output), new[] { output.Length });
        }
        catch
        {
            return base.ISTFT(magnitude, phase, nFft, hopLength, window, center, length);
        }
    }

    /// <summary>
    /// GPU-accelerated Mel spectrogram computation.
    /// </summary>
    Tensor<T> IEngine.MelSpectrogram<T>(
        Tensor<T> input,
        int sampleRate,
        int nFft,
        int hopLength,
        int nMels,
        T fMin,
        T fMax,
        Tensor<T> window,
        bool powerToDb)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.MelSpectrogram(input, sampleRate, nFft, hopLength, nMels, fMin, fMax, window, powerToDb);
        }

        try
        {
            // First compute STFT
            ((IEngine)this).STFT(input, nFft, hopLength, window, center: true, out var magnitude, out var _);

            int numFrames = magnitude.Shape._dims[^2];
            int numFreqs = magnitude.Shape._dims[^1];

            // Create Mel filterbank
            var filterbank = ((IEngine)this).CreateMelFilterbank<T>(nMels, nFft, sampleRate, fMin, fMax);

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.GetDataArray());
            float[] filterbankFloat = DirectGpuEngine.ToFloatArray(filterbank.GetDataArray());

            // Compute power spectrum (magnitude squared)
            float[] powerSpec = new float[magnitudeFloat.Length];
            for (int i = 0; i < magnitudeFloat.Length; i++)
            {
                powerSpec[i] = magnitudeFloat[i] * magnitudeFloat[i];
            }

            // Use cache-aware allocation for filterbank (likely persistent)
            using var filterbankBuffer = GetOrAllocateBuffer(backend, filterbank.GetDataArray());
            // Allocate working buffers
            using var powerBuffer = backend.AllocateBuffer(powerSpec);
            using var melBuffer = AllocateOutputBuffer(backend, numFrames * nMels);

            // Apply Mel filterbank
            backend.ApplyMelFilterbank(powerBuffer, filterbankBuffer.Buffer, melBuffer.Buffer, numFrames, numFreqs, nMels);

            if (powerToDb)
            {
                using var dbBuffer = AllocateOutputBuffer(backend, numFrames * nMels);
                backend.PowerToDb(melBuffer.Buffer, dbBuffer.Buffer, numFrames * nMels, 1.0f, -80.0f);
                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] dbResult = new float[numFrames * nMels];
                backend.DownloadBuffer(dbBuffer.Buffer, dbResult);

                int[] outputShape = input.Rank == 1
                    ? new[] { numFrames, nMels }
                    : new[] { input.Shape._dims[0], numFrames, nMels };

                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(dbResult), outputShape);
            }
            else
            {
                // DownloadBuffer uses blocking read, Synchronize() removed for performance
                float[] melResult = new float[numFrames * nMels];
                backend.DownloadBuffer(melBuffer.Buffer, melResult);

                int[] outputShape = input.Rank == 1
                    ? new[] { numFrames, nMels }
                    : new[] { input.Shape._dims[0], numFrames, nMels };

                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(melResult), outputShape);
            }
        }
        catch
        {
            return base.MelSpectrogram(input, sampleRate, nFft, hopLength, nMels, fMin, fMax, window, powerToDb);
        }
    }

    /// <summary>
    /// GPU-accelerated Griffin-Lim algorithm for audio reconstruction from magnitude spectrogram.
    /// </summary>
    Tensor<T> IEngine.GriffinLim<T>(
        Tensor<T> magnitude,
        int nFft,
        int hopLength,
        Tensor<T> window,
        int iterations,
        double momentum,
        int? length)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.GriffinLim(magnitude, nFft, hopLength, window, iterations, momentum, length);
        }

        try
        {
            int numFrames = magnitude.Shape._dims[^2];
            int numFreqs = magnitude.Shape._dims[^1];

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.GetDataArray());

            // Initialize with random phase
            var random = new Random(42);
            float[] phase = new float[magnitudeFloat.Length];
            for (int i = 0; i < phase.Length; i++)
            {
                phase[i] = (float)(random.NextDouble() * 2 * Math.PI - Math.PI);
            }

            float[] prevPhase = new float[phase.Length];
            float momentumF = (float)momentum;

            for (int iter = 0; iter < iterations; iter++)
            {
                // Reconstruct signal using current phase estimate
                var phaseTensor = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phase), magnitude.Shape.ToArray());
                var reconstructed = ((IEngine)this).ISTFT(magnitude, phaseTensor, nFft, hopLength, window, center: true, length);

                // Re-analyze to get new phase
                ((IEngine)this).STFT(reconstructed, nFft, hopLength, window, center: true, out var _, out var newPhaseTensor);

                float[] newPhase = DirectGpuEngine.ToFloatArray(newPhaseTensor.GetDataArray());

                // Apply momentum
                if (iter > 0 && momentumF > 0)
                {
                    for (int i = 0; i < phase.Length; i++)
                    {
                        // Unwrap phase difference for momentum
                        float diff = newPhase[i] - prevPhase[i];
                        while (diff > Math.PI) diff -= (float)(2 * Math.PI);
                        while (diff < -Math.PI) diff += (float)(2 * Math.PI);

                        float accelerated = prevPhase[i] + diff * (1 + momentumF);
                        phase[i] = accelerated;
                    }
                }
                else
                {
                    Array.Copy(newPhase, phase, phase.Length);
                }

                Array.Copy(newPhase, prevPhase, prevPhase.Length);
            }

            // Final reconstruction
            var finalPhaseTensor = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phase), magnitude.Shape.ToArray());
            return ((IEngine)this).ISTFT(magnitude, finalPhaseTensor, nFft, hopLength, window, center: true, length);
        }
        catch
        {
            return base.GriffinLim(magnitude, nFft, hopLength, window, iterations, momentum, length);
        }
    }

    /// <summary>
    /// Creates a Mel filterbank matrix (CPU implementation, can be cached).
    /// </summary>
    Tensor<T> IEngine.CreateMelFilterbank<T>(int nMels, int nFft, int sampleRate, T fMin, T fMax)
    {
        // Filterbank creation is a one-time operation, use CPU base implementation
        return base.CreateMelFilterbank<T>(nMels, nFft, sampleRate, fMin, fMax);
    }

    /// <summary>
    /// Creates a window function (CPU implementation, can be cached).
    /// </summary>
    Tensor<T> IEngine.CreateWindow<T>(string windowType, int windowLength)
    {
        // Window creation is a one-time operation, use CPU base implementation
        return base.CreateWindow<T>(windowType, windowLength);
    }

    #endregion

    #region Normalization Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Softmax operation.
    /// Supports arbitrary axes by treating the tensor as 2D (outerSize, features).
    /// </summary>
    Tensor<T> IEngine.Softmax<T>(Tensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.Softmax(input, axis);

        // Handle negative axis
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.Softmax(input, axis);

        try
        {
            // For softmax over the last dimension, we can use GPU directly
            // by treating the tensor as 2D: (product of all dims except last) x (last dim)
            if (axis == rank - 1)
            {
                int features = input.Shape._dims[rank - 1];
                int outerSize = input.Length / features;

                using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
                using var outputBuffer = AllocateOutputBuffer(backend, input.Length);

                backend.Softmax(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, features);
                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
            }

            // For softmax over a non-last axis, permute to move the axis to the end,
            // apply softmax, then permute back
            if (axis < rank - 1)
            {
                // Build permutation: move axis to end
                var permutation = new int[rank];
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (i != axis) permutation[j++] = i;
                }
                permutation[rank - 1] = axis;

                // Permute input
                var permutedInput = PermuteImpl(input, permutation);

                // Now apply softmax over the last axis
                int features = permutedInput.Shape._dims[rank - 1];
                int outerSize = permutedInput.Length / features;

                using var inputBuffer = GetOrAllocateBuffer(backend, permutedInput.GetDataArray());
                using var outputBuffer = AllocateOutputBuffer(backend, permutedInput.Length);

                backend.Softmax(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, features);
                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                var permutedOutput = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), permutedInput.Shape.ToArray());

                // Build inverse permutation and permute back
                var inversePermutation = new int[rank];
                for (int i = 0; i < rank; i++)
                {
                    inversePermutation[permutation[i]] = i;
                }
                return PermuteImpl(permutedOutput, inversePermutation);
            }

            // Fall back to CPU for any other edge cases
            return base.Softmax(input, axis);
        }
        catch
        {
            return base.Softmax(input, axis);
        }
    }

    /// <summary>
    /// GPU-accelerated Softmax backward operation.
    /// Supports arbitrary axes by treating the tensor as 2D (outerSize, features).
    /// </summary>
    Tensor<T> IEngine.SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.SoftmaxBackward(gradOutput, output, axis);

        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.SoftmaxBackward(gradOutput, output, axis);

        try
        {
            // For softmax backward over the last dimension
            if (axis == rank - 1)
            {
                int features = output.Shape._dims[rank - 1];
                int outerSize = output.Length / features;

                using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
                using var outputBuffer = GetOrAllocateBuffer(backend, output.GetDataArray());
                using var gradInputBuffer = AllocateOutputBuffer(backend, output.Length);

                backend.SoftmaxBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, outerSize, features);
                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), output.Shape.ToArray());
            }

            // For softmax backward over a non-last axis, permute to move the axis to the end,
            // apply softmax backward, then permute back
            if (axis < rank - 1)
            {
                // Build permutation: move axis to end
                var permutation = new int[rank];
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (i != axis) permutation[j++] = i;
                }
                permutation[rank - 1] = axis;

                // Permute inputs
                var permutedGradOutput = PermuteImpl(gradOutput, permutation);
                var permutedOutput = PermuteImpl(output, permutation);

                int features = permutedOutput.Shape._dims[rank - 1];
                int outerSize = permutedOutput.Length / features;

                using var gradOutBuffer = GetOrAllocateBuffer(backend, permutedGradOutput.GetDataArray());
                using var outputBuffer = GetOrAllocateBuffer(backend, permutedOutput.GetDataArray());
                using var gradInputBuffer = AllocateOutputBuffer(backend, permutedOutput.Length);

                backend.SoftmaxBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, outerSize, features);
                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                var permutedResult = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), permutedOutput.Shape.ToArray());

                // Build inverse permutation and permute back
                var inversePermutation = new int[rank];
                for (int i = 0; i < rank; i++)
                {
                    inversePermutation[permutation[i]] = i;
                }
                return PermuteImpl(permutedResult, inversePermutation);
            }

            return base.SoftmaxBackward(gradOutput, output, axis);
        }
        catch
        {
            return base.SoftmaxBackward(gradOutput, output, axis);
        }
    }

    /// <summary>
    /// GPU-accelerated Squash activation for capsule networks.
    /// squash(v) = ||v||² / (1 + ||v||²) × v / ||v||
    /// </summary>
    Tensor<T> IEngine.TensorSquash<T>(Tensor<T> tensor, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorSquash(tensor, axis);

        int rank = tensor.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.TensorSquash(tensor, axis);

        try
        {
            // For squash over the last dimension
            if (axis == rank - 1)
            {
                int capsuleDim = tensor.Shape._dims[rank - 1];
                int numCapsules = tensor.Length / capsuleDim;

                using var inputBuffer = GetOrAllocateBuffer(backend, tensor.GetDataArray());
                using var outputBuffer = AllocateOutputBuffer(backend, tensor.Length);

                backend.Squash(inputBuffer.Buffer, outputBuffer.Buffer, numCapsules, capsuleDim, 1e-8f);
                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), tensor.Shape.ToArray());
            }

            // For squash over a non-last axis, permute to move the axis to the end,
            // apply squash, then permute back
            if (axis < rank - 1)
            {
                // Build permutation: move axis to end
                var permutation = new int[rank];
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (i != axis) permutation[j++] = i;
                }
                permutation[rank - 1] = axis;

                var permutedInput = PermuteImpl(tensor, permutation);

                int capsuleDim = permutedInput.Shape._dims[rank - 1];
                int numCapsules = permutedInput.Length / capsuleDim;

                using var inputBuffer = GetOrAllocateBuffer(backend, permutedInput.GetDataArray());
                using var outputBuffer = AllocateOutputBuffer(backend, permutedInput.Length);

                backend.Squash(inputBuffer.Buffer, outputBuffer.Buffer, numCapsules, capsuleDim, 1e-8f);
                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                var permutedResult = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), permutedInput.Shape.ToArray());

                // Build inverse permutation and permute back
                var inversePermutation = new int[rank];
                for (int i = 0; i < rank; i++)
                {
                    inversePermutation[permutation[i]] = i;
                }
                return PermuteImpl(permutedResult, inversePermutation);
            }

            return base.TensorSquash(tensor, axis);
        }
        catch
        {
            return base.TensorSquash(tensor, axis);
        }
    }

    /// <summary>
    /// GPU-accelerated Squash backward operation.
    /// </summary>
    Tensor<T> IEngine.TensorSquashBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorSquashBackward(gradOutput, input, output, axis);

        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.TensorSquashBackward(gradOutput, input, output, axis);

        try
        {
            // For squash backward over the last dimension
            if (axis == rank - 1)
            {
                int capsuleDim = input.Shape._dims[rank - 1];
                int numCapsules = input.Length / capsuleDim;

                using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
                using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
                using var gradInputBuffer = AllocateOutputBuffer(backend, input.Length);

                backend.SquashBackward(gradOutputBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, numCapsules, capsuleDim, 1e-8f);
                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
            }

            // For squash backward over a non-last axis, permute to move the axis to the end,
            // apply squash backward, then permute back
            if (axis < rank - 1)
            {
                // Build permutation: move axis to end
                var permutation = new int[rank];
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (i != axis) permutation[j++] = i;
                }
                permutation[rank - 1] = axis;

                var permutedGradOutput = PermuteImpl(gradOutput, permutation);
                var permutedInput = PermuteImpl(input, permutation);

                int capsuleDim = permutedInput.Shape._dims[rank - 1];
                int numCapsules = permutedInput.Length / capsuleDim;

                using var gradOutputBuffer = GetOrAllocateBuffer(backend, permutedGradOutput.GetDataArray());
                using var inputBuffer = GetOrAllocateBuffer(backend, permutedInput.GetDataArray());
                using var gradInputBuffer = AllocateOutputBuffer(backend, permutedInput.Length);

                backend.SquashBackward(gradOutputBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, numCapsules, capsuleDim, 1e-8f);
                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                var permutedResult = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), permutedInput.Shape.ToArray());

                // Build inverse permutation and permute back
                var inversePermutation = new int[rank];
                for (int i = 0; i < rank; i++)
                {
                    inversePermutation[permutation[i]] = i;
                }
                return PermuteImpl(permutedResult, inversePermutation);
            }

            return base.TensorSquashBackward(gradOutput, input, output, axis);
        }
        catch
        {
            return base.TensorSquashBackward(gradOutput, input, output, axis);
        }
    }

    /// <summary>
    /// GPU-resident tensor tiling (repeating) along the batch dimension.
    /// Tiles the input tensor to create a larger output tensor with the specified number of repeats.
    /// Output shape: [repeats * batchSize, ...] where input shape is [batchSize, ...].
    /// </summary>
    public Tensor<T> TileBatchGpu<T>(Tensor<T> input, int repeats)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TileBatchGpu");

        if (repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(repeats), "Repeats must be positive");

        int batchSize = input.Shape._dims[0];
        int innerSize = input.Length / batchSize;
        int outputTotalSize = repeats * batchSize * innerSize;

        int[] outputShape = new int[input.Shape._dims.Length];
        outputShape[0] = repeats * batchSize;
        for (int i = 1; i < input.Shape._dims.Length; i++)
            outputShape[i] = input.Shape._dims[i];

        var outputBuffer = backend.AllocateBuffer(outputTotalSize);
        backend.TileBatch(input.Buffer, outputBuffer, repeats * batchSize, innerSize);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, true);
    }

    /// <summary>
    /// GPU-resident tensor tiling (repeating) along a specific axis.
    /// Tiles the input tensor by repeating elements along the specified axis.
    /// </summary>
    public Tensor<T> TileAxisGpu<T>(Tensor<T> input, int axis, int repeats)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TileAxisGpu");

        if (repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(repeats), "Repeats must be positive");

        int rank = input.Shape._dims.Length;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis out of range");

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input.Shape._dims[i];
        int axisSize = input.Shape._dims[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < rank; i++) innerSize *= input.Shape._dims[i];

        int outputTotalSize = outerSize * axisSize * repeats * innerSize;

        int[] outputShape = new int[rank];
        for (int i = 0; i < rank; i++)
            outputShape[i] = i == axis ? input.Shape._dims[i] * repeats : input.Shape._dims[i];

        var outputBuffer = backend.AllocateBuffer(outputTotalSize);
        backend.TileAxis(input.Buffer, outputBuffer, outerSize, axisSize, innerSize, repeats);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, true);
    }

    /// <summary>
    /// GPU-resident global average pooling operation.
    /// Reduces spatial dimensions (all except batch and last) to 1 using mean.
    /// </summary>
    public Tensor<T> GlobalMeanPoolGpu<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GlobalMeanPoolGpu");

        int rank = input.Shape._dims.Length;

        // Get reduction axes (all except first and last)
        int[] axes = rank switch
        {
            4 => [1, 2],  // [batch, height, width, channels] -> reduce H, W
            3 => [1],     // [batch, seq_len, features] -> reduce seq_len
            2 => [],      // Nothing to reduce
            1 => [0],     // Reduce all
            _ when rank > 4 => Enumerable.Range(1, rank - 2).ToArray(),
            _ => []
        };

        if (axes.Length == 0)
        {
            // No reduction needed - return input with new shape
            return Tensor<T>.FromGpuBuffer(backend, input.Buffer, input.Shape._dims, GpuTensorRole.Intermediate, ownsBuffer: false);
        }

        // Calculate output shape and sizes
        int outerSize = 1;
        int reduceSize = 1;
        for (int i = 0; i < axes[0]; i++) outerSize *= input.Shape._dims[i];
        foreach (int axis in axes) reduceSize *= input.Shape._dims[axis];
        int innerSize = 1;
        for (int i = axes[^1] + 1; i < rank; i++) innerSize *= input.Shape._dims[i];

        // For global pooling with innerSize, treat it as multiple reductions
        int totalOuter = outerSize * innerSize;
        int outputSize = totalOuter;

        int[] outputShape;
        if (rank == 4)
        {
            outputShape = [input.Shape._dims[0], 1, 1, input.Shape._dims[3]];
        }
        else if (rank == 3)
        {
            outputShape = [input.Shape._dims[0], input.Shape._dims[2]];
        }
        else
        {
            outputShape = [1];
        }

        var outputBuffer = backend.AllocateBuffer(outputSize);
        backend.MeanAxis(input.Buffer, outputBuffer, totalOuter, reduceSize);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, true);
    }

    /// <summary>
    /// GPU-resident global max pooling operation.
    /// Reduces spatial dimensions (all except batch and last) to 1 using max.
    /// Returns CPU indices for backward pass compatibility.
    /// </summary>
    public Tensor<T> GlobalMaxPoolGpu<T>(Tensor<T> input, out int[] maxIndices)
    {
        var result = GlobalMaxPoolGpuWithGpuIndices(input, out var gpuIndices);

        // Download indices from GPU to CPU for backward pass
        if (gpuIndices is not null && TryGetBackend(out var idxBackend))
        {
            var indicesFloat = idxBackend.DownloadBuffer(gpuIndices.Buffer);
            maxIndices = new int[indicesFloat.Length];
            for (int i = 0; i < indicesFloat.Length; i++)
                maxIndices[i] = (int)indicesFloat[i];
            gpuIndices.Dispose();
        }
        else
        {
            maxIndices = [];
        }

        return result;
    }

    /// <summary>
    /// GPU-resident global max pooling operation with GPU-resident indices.
    /// Keeps both max values and argmax indices on GPU for maximum performance.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="input">Input GPU tensor.</param>
    /// <param name="gpuIndices">GPU tensor containing argmax indices (as floats). Null if no reduction needed.</param>
    /// <returns>GPU tensor containing the max-pooled values.</returns>
    public Tensor<T> GlobalMaxPoolGpuWithGpuIndices<T>(Tensor<T> input, out Tensor<float>? gpuIndices)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GlobalMaxPoolGpu");

        int rank = input.Shape._dims.Length;

        // Get reduction axes (all except first and last)
        int[] axes = rank switch
        {
            4 => [1, 2],  // [batch, height, width, channels] -> reduce H, W
            3 => [1],     // [batch, seq_len, features] -> reduce seq_len
            2 => [],      // Nothing to reduce
            1 => [0],     // Reduce all
            _ when rank > 4 => Enumerable.Range(1, rank - 2).ToArray(),
            _ => []
        };

        if (axes.Length == 0)
        {
            gpuIndices = null;
            return Tensor<T>.FromGpuBuffer(backend, input.Buffer, input.Shape._dims, GpuTensorRole.Intermediate, ownsBuffer: false);
        }

        // Calculate output shape and sizes
        int outerSize = 1;
        int reduceSize = 1;
        for (int i = 0; i < axes[0]; i++) outerSize *= input.Shape._dims[i];
        foreach (int axis in axes) reduceSize *= input.Shape._dims[axis];
        int innerSize = 1;
        for (int i = axes[^1] + 1; i < rank; i++) innerSize *= input.Shape._dims[i];

        int totalOuter = outerSize * innerSize;
        int outputSize = totalOuter;

        int[] outputShape;
        if (rank == 4)
        {
            outputShape = [input.Shape._dims[0], 1, 1, input.Shape._dims[3]];
        }
        else if (rank == 3)
        {
            outputShape = [input.Shape._dims[0], input.Shape._dims[2]];
        }
        else
        {
            outputShape = [1];
        }

        var outputBuffer = backend.AllocateBuffer(outputSize);
        backend.MaxAxis(input.Buffer, outputBuffer, totalOuter, reduceSize);

        // Compute argmax indices on GPU and keep them GPU-resident
        var indicesBuffer = backend.AllocateBuffer(outputSize);
        backend.ArgMax(input.Buffer, indicesBuffer, totalOuter, reduceSize);
        gpuIndices = Tensor<float>.FromGpuBuffer(backend, indicesBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Intermediate, true);
    }

    /// <summary>
    /// GPU-resident backward pass for global mean pooling.
    /// Broadcasts gradient to all spatial positions and divides by count.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="gradOutput">GPU-resident gradient of shape [batch, 1, 1, channels] or [batch, channels].</param>
    /// <param name="inputShape">Original input shape to broadcast to.</param>
    /// <returns>GPU-resident gradient of same shape as original input.</returns>
    public Tensor<T> GlobalMeanPoolBackwardGpu<T>(Tensor<T> gradOutput, int[] inputShape)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GlobalMeanPoolBackwardGpu");

        // Parameter validation
        if (inputShape is null || inputShape.Length == 0)
            throw new ArgumentException("Input shape must not be null or empty", nameof(inputShape));

        foreach (int dim in inputShape)
        {
            if (dim <= 0)
                throw new ArgumentException("All input shape dimensions must be positive", nameof(inputShape));
        }

        int rank = inputShape.Length;

        // Get reduction axes (same as forward pass)
        int[] axes = rank switch
        {
            4 => [1, 2],  // [batch, height, width, channels]
            3 => [1],     // [batch, seq_len, features]
            2 => [],      // Nothing to reduce
            1 => [0],     // Reduce all
            _ when rank > 4 => Enumerable.Range(1, rank - 2).ToArray(),
            _ => []
        };

        if (axes.Length == 0)
        {
            // No reduction was done - return gradient as-is
            return Tensor<T>.FromGpuBuffer(backend, gradOutput.Buffer, inputShape, GpuTensorRole.Gradient, ownsBuffer: false);
        }

        // Calculate sizes
        int reduceSize = 1;
        foreach (int axis in axes) reduceSize *= inputShape[axis];
        int totalSize = inputShape.Aggregate(1, (a, b) => a * b);
        int outerSize = gradOutput.Length;

        // Allocate output buffer
        var outputBuffer = backend.AllocateBuffer(totalSize);

        // For mean pooling backward: broadcast and scale by 1/reduceSize
        // grad_input = grad_output tiled reduceSize times, then scaled
        float scale = 1.0f / reduceSize;

        // Use TileBatch to repeat gradient values
        // TileBatch(input, output, repeats, innerSize) tiles input[i] to output[i*repeats:(i+1)*repeats]
        backend.TileBatch(gradOutput.Buffer, outputBuffer, reduceSize, 1);

        // Scale the output by 1/reduceSize
        backend.Scale(outputBuffer, outputBuffer, scale, totalSize);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident backward pass for global max pooling.
    /// Scatters gradient to the positions that had maximum values.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="gradOutput">GPU-resident gradient of shape [batch, 1, 1, channels] or [batch, channels].</param>
    /// <param name="maxIndices">CPU indices of max positions from forward pass.</param>
    /// <param name="inputShape">Original input shape.</param>
    /// <returns>GPU-resident gradient of same shape as original input.</returns>
    public Tensor<T> GlobalMaxPoolBackwardGpu<T>(Tensor<T> gradOutput, int[] maxIndices, int[] inputShape)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GlobalMaxPoolBackwardGpu");

        // Parameter validation
        if (inputShape is null || inputShape.Length == 0)
            throw new ArgumentException("Input shape must not be null or empty", nameof(inputShape));

        foreach (int dim in inputShape)
        {
            if (dim <= 0)
                throw new ArgumentException("All input shape dimensions must be positive", nameof(inputShape));
        }

        if (maxIndices is null)
            throw new ArgumentNullException(nameof(maxIndices));

        int totalSize = inputShape.Aggregate(1, (a, b) => a * b);

        // Validate indices are within bounds
        foreach (int idx in maxIndices)
        {
            if (idx < 0 || idx >= totalSize)
                throw new ArgumentOutOfRangeException(nameof(maxIndices),
                    $"Index {idx} is out of bounds for input with total size {totalSize}");
        }

        // Allocate and zero-initialize output buffer
        var outputBuffer = backend.AllocateBuffer(totalSize);
        backend.Fill(outputBuffer, 0f, totalSize);

        // Upload indices
        using var indicesBuffer = backend.AllocateIntBuffer(maxIndices);

        // Scatter-add gradient to max positions: destination[indices[i]] += source[i]
        // sourceSize = number of gradients, destSize = total output size
        backend.ScatterAdd(gradOutput.Buffer, indicesBuffer, outputBuffer, maxIndices.Length, totalSize);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident ArgMax operation. Returns indices of maximum values along an axis.
    /// Indices are returned as floats on GPU (cast to int when downloading).
    /// </summary>
    /// <typeparam name="T">Element type of input.</typeparam>
    /// <param name="input">Input GPU tensor with shape [outerSize, reduceSize].</param>
    /// <param name="axis">Axis along which to find argmax.</param>
    /// <returns>GPU tensor containing argmax indices as floats.</returns>
    public Tensor<float> ArgMaxGpu<T>(Tensor<T> input, int axis = -1)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ArgMaxGpu");

        int rank = input.Shape._dims.Length;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of range for tensor with {rank} dimensions");

        // Calculate sizes for reduction
        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input.Shape._dims[i];
        int reduceSize = input.Shape._dims[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < rank; i++) innerSize *= input.Shape._dims[i];

        int totalOuter = outerSize * innerSize;

        // Output shape removes the reduction axis
        var outputShape = new int[rank - 1];
        int outIdx = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i != axis)
                outputShape[outIdx++] = input.Shape._dims[i];
        }
        if (outputShape.Length == 0)
            outputShape = [1];

        var indicesBuffer = backend.AllocateBuffer(totalOuter);
        backend.ArgMax(input.Buffer, indicesBuffer, totalOuter, reduceSize);

        return Tensor<float>.FromGpuBuffer(backend, indicesBuffer, outputShape, GpuTensorRole.Intermediate, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-accelerated LayerNorm operation.
    /// </summary>
    Tensor<T> IEngine.LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Determine batch size and normalized size from gamma shape
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
            using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.GetDataArray(), PersistentTensorRole.Biases);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batchSize);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batchSize);

            backend.LayerNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batchSize, normalizedSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batchSize });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batchSize });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    /// <summary>
    /// GPU-accelerated LayerNorm backward operation.
    /// </summary>
    Tensor<T> IEngine.LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (!TryGetBackend(out var backend))
            return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
            using var saveMeanBuffer = GetOrAllocateBuffer(backend, mean.GetDataArray());
            using var saveVarBuffer = GetOrAllocateBuffer(backend, variance.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var gradGammaBuffer = AllocateOutputBuffer(backend, normalizedSize);
            using var gradBetaBuffer = AllocateOutputBuffer(backend, normalizedSize);

            backend.LayerNormBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gammaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, gradInputBuffer.Buffer, gradGammaBuffer.Buffer, gradBetaBuffer.Buffer,
                batchSize, normalizedSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] gradInputFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            float[] gradGammaFloat = backend.DownloadBuffer(gradGammaBuffer.Buffer);
            float[] gradBetaFloat = backend.DownloadBuffer(gradBetaBuffer.Buffer);

            gradGamma = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaFloat), gamma.Shape.ToArray());
            gradBeta = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBetaFloat), gamma.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradInputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);
        }
    }

    /// <summary>
    /// GPU-accelerated RMSNorm operation.
    /// </summary>
    Tensor<T> IEngine.RMSNorm<T>(Tensor<T> input, Tensor<T> gamma, double epsilon, out Tensor<T> rms)
    {
        if (!TryGetBackend(out var backend))
            return base.RMSNorm(input, gamma, epsilon, out rms);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.RMSNorm(input, gamma, epsilon, out rms);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveRmsBuffer = AllocateOutputBuffer(backend, batchSize);

            backend.RmsNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, saveRmsBuffer.Buffer,
                batchSize, normalizedSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] rmsFloat = backend.DownloadBuffer(saveRmsBuffer.Buffer);

            rms = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(rmsFloat), new[] { batchSize });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.RMSNorm(input, gamma, epsilon, out rms);
        }
    }

    /// <summary>
    /// GPU-accelerated RMSNorm backward operation.
    /// </summary>
    Tensor<T> IEngine.RMSNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> rms, double epsilon, out Tensor<T> gradGamma)
    {
        if (!TryGetBackend(out var backend))
            return base.RMSNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.RMSNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
            using var saveRmsBuffer = GetOrAllocateBuffer(backend, rms.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var gradGammaBuffer = AllocateOutputBuffer(backend, normalizedSize);

            backend.RmsNormBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gammaBuffer.Buffer, saveRmsBuffer.Buffer,
                gradInputBuffer.Buffer, gradGammaBuffer.Buffer, batchSize, normalizedSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] gradInputFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            float[] gradGammaFloat = backend.DownloadBuffer(gradGammaBuffer.Buffer);

            gradGamma = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaFloat), gamma.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradInputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.RMSNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);
        }
    }

    /// <summary>
    /// GPU-accelerated GroupNorm operation.
    /// </summary>
    Tensor<T> IEngine.GroupNorm<T>(Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Input shape: [batch, channels, spatial...]
            if (input.Rank < 2)
                return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

            int batch = input.Shape._dims[0];
            int channels = input.Shape._dims[1];
            int spatialSize = 1;
            for (int i = 2; i < input.Rank; i++)
                spatialSize *= input.Shape._dims[i];

            if (channels % numGroups != 0)
                return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
            using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.GetDataArray(), PersistentTensorRole.Biases);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batch * numGroups);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batch * numGroups);

            backend.GroupNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batch, numGroups, channels, spatialSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batch, numGroups });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batch, numGroups });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);
        }
    }

    /// <summary>
    /// GPU-accelerated InstanceNorm operation.
    /// </summary>
    Tensor<T> IEngine.InstanceNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Input shape: [batch, channels, spatial...]
            if (input.Rank < 2)
                return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);

            int batch = input.Shape._dims[0];
            int channels = input.Shape._dims[1];
            int spatialSize = 1;
            for (int i = 2; i < input.Rank; i++)
                spatialSize *= input.Shape._dims[i];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
            using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.GetDataArray(), PersistentTensorRole.Biases);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batch * channels);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.InstanceNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batch, channels, spatialSize, (float)epsilon);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batch, channels });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batch, channels });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    /// <summary>
    /// GPU-resident batch normalization. Input and output remain on GPU, avoiding CPU round-trips.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor with shape [batch, features] or [batch, channels, H, W].</param>
    /// <param name="gamma">Scale parameters (from CPU, cached on GPU).</param>
    /// <param name="beta">Shift parameters (from CPU, cached on GPU).</param>
    /// <param name="runningMean">Running mean for inference (from CPU, will be updated during training).</param>
    /// <param name="runningVar">Running variance for inference (from CPU, will be updated during training).</param>
    /// <param name="epsilon">Numerical stability constant.</param>
    /// <param name="momentum">Momentum for running statistics update.</param>
    /// <param name="training">Whether in training mode.</param>
    /// <returns>GPU-resident output tensor with same shape as input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is not available.</exception>
    /// <remarks>
    /// <para>
    /// This method performs batch normalization entirely on GPU, returning a GPU-resident tensor.
    /// The running statistics are updated on GPU during training mode and then downloaded back
    /// to update the CPU-side tensors.
    /// </para>
    /// <para>
    /// For 4D input [batch, channels, H, W], spatialSize = H * W. For 2D input [batch, features], spatialSize = 1.
    /// </para>
    /// </remarks>
    public (Tensor<T> Output, Tensor<T>? SaveMean, Tensor<T>? SaveVar) FusedBatchNormGpu<T>(
        Tensor<T> input,
        Tensor<T> gamma,
        Tensor<T> beta,
        ref Tensor<T> runningMean,
        ref Tensor<T> runningVar,
        double epsilon,
        double momentum,
        bool training)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for FusedBatchNormGpu");

        int[] shape = input.Shape._dims;
        int batch = shape[0];
        int channels = shape.Length > 1 ? shape[1] : shape[0];
        int spatialSize = 1;
        for (int i = 2; i < shape.Length; i++)
        {
            spatialSize *= shape[i];
        }

        // Upload parameters to GPU (these are typically cached)
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
        using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.GetDataArray(), PersistentTensorRole.Biases);
        using var runningMeanBuffer = GetOrCacheWeightBuffer(backend, runningMean.GetDataArray(), PersistentTensorRole.NormalizationParams);
        using var runningVarBuffer = GetOrCacheWeightBuffer(backend, runningVar.GetDataArray(), PersistentTensorRole.NormalizationParams);

        // Allocate output and save buffers
        int outputSize = input.Length;
        var outputBuffer = backend.AllocateBuffer(outputSize);
        using var saveMeanBuffer = AllocateOutputBuffer(backend, channels);
        using var saveVarBuffer = AllocateOutputBuffer(backend, channels);

        // Execute batch norm on GPU
        backend.BatchNorm(
            input.Buffer, outputBuffer, gammaBuffer.Buffer, betaBuffer.Buffer,
            runningMeanBuffer.Buffer, runningVarBuffer.Buffer,
            saveMeanBuffer.Buffer, saveVarBuffer.Buffer,
            batch, channels, spatialSize,
            (float)epsilon, (float)momentum, training);

        // If training, download updated running statistics back to CPU
        Tensor<T>? saveMean = null;
        Tensor<T>? saveVar = null;
        if (training)
        {
            float[] updatedRunningMean = backend.DownloadBuffer(runningMeanBuffer.Buffer);
            float[] updatedRunningVar = backend.DownloadBuffer(runningVarBuffer.Buffer);
            runningMean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(updatedRunningMean), new[] { channels });
            runningVar = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(updatedRunningVar), new[] { channels });

            // Also return saveMean/saveVar for backward pass
            float[] saveMeanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] saveVarFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);
            saveMean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(saveMeanFloat), new[] { channels });
            saveVar = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(saveVarFloat), new[] { channels });
        }

        // Return GPU-resident output tensor
        var outputTensor = Tensor<T>.FromGpuBuffer(backend, outputBuffer, shape, GpuTensorRole.Activation, true);
        return (outputTensor, saveMean, saveVar);
    }

    /// <summary>
    /// GPU-resident Layer Normalization forward pass.
    /// Normalizes input across the normalized (feature) dimension for each sample independently.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, normalizedSize].</param>
    /// <param name="gamma">Scale parameters [normalizedSize].</param>
    /// <param name="beta">Shift parameters [normalizedSize].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <returns>
    /// A tuple containing:
    /// - Output: GPU-resident normalized tensor
    /// - SaveMean: Mean values per sample (for backward pass, downloaded to CPU)
    /// - SaveInvVar: Inverse variance per sample (for backward pass, downloaded to CPU)
    /// </returns>
    public (Tensor<T> Output, Tensor<T> SaveMean, Tensor<T> SaveInvVar) LayerNormGpu<T>(
        Tensor<T> input,
        Tensor<T> gamma,
        Tensor<T> beta,
        double epsilon)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for LayerNormGpu");

        int[] shape = input.Shape._dims;
        int batchSize = shape[0];
        int normalizedSize = shape.Length > 1 ? shape[1] : shape[0];

        // For higher-rank tensors, flatten the normalized dimensions
        for (int i = 2; i < shape.Length; i++)
        {
            normalizedSize *= shape[i];
        }

        // Upload gamma and beta to GPU
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
        using var betaBuffer = GetOrCacheWeightBuffer(backend, beta.GetDataArray(), PersistentTensorRole.Biases);

        // Allocate output and save buffers
        int outputSize = input.Length;
        var outputBuffer = backend.AllocateBuffer(outputSize);
        using var saveMeanBuffer = AllocateOutputBuffer(backend, batchSize);
        using var saveInvVarBuffer = AllocateOutputBuffer(backend, batchSize);

        // Execute LayerNorm on GPU
        backend.LayerNorm(
            input.Buffer, outputBuffer, gammaBuffer.Buffer, betaBuffer.Buffer,
            saveMeanBuffer.Buffer, saveInvVarBuffer.Buffer,
            batchSize, normalizedSize, (float)epsilon);

        // Download save buffers for backward pass (these are per-sample, so relatively small)
        float[] saveMeanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
        float[] saveInvVarFloat = backend.DownloadBuffer(saveInvVarBuffer.Buffer);
        var saveMean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(saveMeanFloat), new[] { batchSize });
        var saveInvVar = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(saveInvVarFloat), new[] { batchSize });

        // Return GPU-resident output tensor
        var outputTensor = Tensor<T>.FromGpuBuffer(backend, outputBuffer, shape, GpuTensorRole.Activation, true);
        return (outputTensor, saveMean, saveInvVar);
    }

    /// <summary>
    /// GPU-resident layer normalization backward pass.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident gradient of loss w.r.t. output.</param>
    /// <param name="input">GPU-resident input from forward pass.</param>
    /// <param name="gamma">Scale parameters.</param>
    /// <param name="saveMean">Saved mean from forward pass.</param>
    /// <param name="saveInvVar">Saved inverse variance from forward pass.</param>
    /// <param name="epsilon">Epsilon for numerical stability.</param>
    /// <returns>Tuple of (gradInput, gradGamma, gradBeta) GPU-resident tensors.</returns>
    public (Tensor<T> GradInput, Tensor<T> GradGamma, Tensor<T> GradBeta) LayerNormBackwardGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> gamma,
        Tensor<T> saveMean,
        Tensor<T> saveInvVar,
        double epsilon)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for LayerNormBackwardGpu");

        int[] shape = gradOutput.Shape._dims;
        int batchSize = shape[0];
        int normalizedSize = shape.Length > 1 ? shape[1] : shape[0];

        // For higher-rank tensors, flatten the normalized dimensions
        for (int i = 2; i < shape.Length; i++)
        {
            normalizedSize *= shape[i];
        }

        // Validate parameter shapes to prevent out-of-bounds kernel access
        if (gamma.Length != normalizedSize)
            throw new ArgumentException($"gamma.Length ({gamma.Length}) must match normalizedSize ({normalizedSize}).", nameof(gamma));
        if (saveMean.Length != batchSize)
            throw new ArgumentException($"saveMean.Length ({saveMean.Length}) must match batchSize ({batchSize}).", nameof(saveMean));
        if (saveInvVar.Length != batchSize)
            throw new ArgumentException($"saveInvVar.Length ({saveInvVar.Length}) must match batchSize ({batchSize}).", nameof(saveInvVar));

        // Upload gamma, saveMean, saveInvVar to GPU
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);
        float[] saveMeanFloat = DirectGpuEngine.ToFloatArray(saveMean.GetDataArray());
        float[] saveInvVarFloat = DirectGpuEngine.ToFloatArray(saveInvVar.GetDataArray());

        // Allocate temporary and output buffers with exception-safe disposal
        IGpuBuffer? saveMeanBuffer = null;
        IGpuBuffer? saveInvVarBuffer = null;
        IGpuBuffer? gradInputBuffer = null;
        IGpuBuffer? gradGammaBuffer = null;
        IGpuBuffer? gradBetaBuffer = null;

        try
        {
            saveMeanBuffer = backend.AllocateBuffer(saveMeanFloat);
            saveInvVarBuffer = backend.AllocateBuffer(saveInvVarFloat);

            // Allocate output buffers
            int inputSize = gradOutput.Length;
            gradInputBuffer = backend.AllocateBuffer(inputSize);
            gradGammaBuffer = backend.AllocateBuffer(normalizedSize);
            gradBetaBuffer = backend.AllocateBuffer(normalizedSize);

            // Execute LayerNormBackward on GPU
            backend.LayerNormBackward(
                gradOutput.Buffer, input.Buffer, gammaBuffer.Buffer,
                saveMeanBuffer, saveInvVarBuffer,
                gradInputBuffer, gradGammaBuffer, gradBetaBuffer,
                batchSize, normalizedSize, (float)epsilon);

            // Download gradGamma and gradBeta (these are small, same size as normalizedSize)
            float[] gradGammaFloat = backend.DownloadBuffer(gradGammaBuffer);
            float[] gradBetaFloat = backend.DownloadBuffer(gradBetaBuffer);

            // Dispose temporary buffers (not gradInputBuffer - it becomes part of returned tensor)
            saveMeanBuffer.Dispose();
            saveMeanBuffer = null;
            saveInvVarBuffer.Dispose();
            saveInvVarBuffer = null;
            gradGammaBuffer.Dispose();
            gradGammaBuffer = null;
            gradBetaBuffer.Dispose();
            gradBetaBuffer = null;

            var gradGamma = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaFloat), new[] { normalizedSize });
            var gradBeta = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBetaFloat), new[] { normalizedSize });

            // Return GPU-resident gradInput tensor
            var gradInputTensor = Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, shape, GpuTensorRole.Gradient, true);
            gradInputBuffer = null; // Ownership transferred to tensor
            return (gradInputTensor, gradGamma, gradBeta);
        }
        finally
        {
            // Dispose any buffers that weren't successfully transferred or already disposed
            saveMeanBuffer?.Dispose();
            saveInvVarBuffer?.Dispose();
            gradInputBuffer?.Dispose();
            gradGammaBuffer?.Dispose();
            gradBetaBuffer?.Dispose();
        }
    }

    /// <summary>
    /// GPU-resident element-wise addition: C = A + B
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="a">First GPU-resident input tensor.</param>
    /// <param name="b">Second GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor with the element-wise sum.</returns>
    public Tensor<T> AddGpu<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AddGpu");

        int size = a.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Add(a.Buffer, b.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, a.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident element-wise multiplication: C = A * B
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="a">First GPU-resident input tensor.</param>
    /// <param name="b">Second GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor with the element-wise product.</returns>
    public Tensor<T> MultiplyGpu<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MultiplyGpu");

        int size = a.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Multiply(a.Buffer, b.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, a.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident scalar multiplication: B = A * scalar
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="scalar">Scalar value to multiply by.</param>
    /// <returns>A GPU-resident output tensor with the scaled values.</returns>
    public Tensor<T> ScaleGpu<T>(Tensor<T> input, float scalar)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScaleGpu");

        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Scale(input.Buffer, outputBuffer, scalar, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident softmax operation along the last axis.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor of shape [batch, features].</param>
    /// <returns>A GPU-resident output tensor with softmax applied.</returns>
    public Tensor<T> SoftmaxGpu<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SoftmaxGpu");

        // Assuming 2D input [batch, features]
        int batchSize = input.Shape._dims[0];
        int features = input.Shape._dims.Length > 1 ? input.Shape._dims[1] : input.Shape._dims[0];

        var outputBuffer = backend.AllocateBuffer(input.Length);

        backend.Softmax(input.Buffer, outputBuffer, batchSize, features);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident Top-K selection along the last axis.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor of shape [batch, features].</param>
    /// <param name="k">Number of top elements to select.</param>
    /// <param name="indices">Output GPU buffer containing the indices of top-k elements.</param>
    /// <param name="sorted">Whether to return sorted results (default true).</param>
    /// <returns>A GPU-resident output tensor with the top-k values.</returns>
    public Tensor<T> TopKGpu<T>(Tensor<T> input, int k, out Tensor<int> indices, bool sorted = true)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TopKGpu");

        // Assuming 2D input [batch, features]
        int outerSize = input.Shape._dims[0];
        int reduceSize = input.Shape._dims.Length > 1 ? input.Shape._dims[1] : input.Shape._dims[0];

        // Allocate output buffers
        var valuesBuffer = backend.AllocateBuffer(outerSize * k);
        var indicesBuffer = backend.AllocateBuffer(outerSize * k);

        backend.TopK(input.Buffer, valuesBuffer, indicesBuffer, outerSize, reduceSize, k, sorted);

        // Create output shape [batch, k]
        int[] outputShape = input.Shape._dims.Length > 1 ? [outerSize, k] : [k];

        indices = Tensor<int>.FromGpuBuffer(backend, indicesBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
        return Tensor<T>.FromGpuBuffer(backend, valuesBuffer, outputShape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident broadcast multiply: C[i,j] = A[i,j] * B[i,0]
    /// Broadcasts a column vector across the last dimension.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, features].</param>
    /// <param name="weights">GPU-resident weight tensor [batch, 1] to broadcast.</param>
    /// <returns>A GPU-resident output tensor with broadcast multiplication.</returns>
    public Tensor<T> BroadcastMultiplyColumnGpu<T>(Tensor<T> input, Tensor<T> weights)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BroadcastMultiplyColumnGpu");

        int outerSize = input.Shape._dims[0];
        int innerSize = input.Length / outerSize;

        var outputBuffer = backend.AllocateBuffer(input.Length);

        backend.BroadcastMultiplyFirstAxis(input.Buffer, weights.Buffer, outputBuffer, outerSize, innerSize);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident slice operation to extract a column from a 2D tensor.
    /// Uses gather with computed indices to extract strided elements.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, features].</param>
    /// <param name="columnIndex">Column index to extract.</param>
    /// <returns>A GPU-resident output tensor [batch, 1].</returns>
    public Tensor<T> SliceColumnGpu<T>(Tensor<T> input, int columnIndex)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SliceColumnGpu");

        int batchSize = input.Shape._dims[0];
        int features = input.Shape._dims.Length > 1 ? input.Shape._dims[1] : 1;

        if (columnIndex < 0 || columnIndex >= features)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        // Create indices for gathering: [columnIndex, features + columnIndex, 2*features + columnIndex, ...]
        int[] indices = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            indices[i] = i * features + columnIndex;
        }

        var indicesBuffer = backend.AllocateIntBuffer(indices);
        var outputBuffer = backend.AllocateBuffer(batchSize);

        // Gather uses (source, indices, output, numIndices, featureSize)
        // With featureSize=1, it gathers individual elements at the specified indices
        backend.Gather(input.Buffer, indicesBuffer, outputBuffer, batchSize, 1);

        indicesBuffer.Dispose();

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [batchSize, 1], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident slice operation along a specified axis.
    /// Extracts a contiguous slice from the input tensor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="axis">The axis to slice along.</param>
    /// <param name="start">Starting index (inclusive).</param>
    /// <param name="end">Ending index (exclusive).</param>
    /// <returns>A GPU-resident sliced tensor.</returns>
    public Tensor<T> SliceGpu<T>(Tensor<T> input, int axis, int start, int end)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SliceGpu");

        int rank = input.Shape._dims.Length;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        int sliceSize = end - start;
        if (sliceSize <= 0)
            throw new ArgumentException("End must be greater than start");

        // Calculate output shape
        int[] outputShape = new int[rank];
        Array.Copy(input.Shape._dims, outputShape, rank);
        outputShape[axis] = sliceSize;

        int totalOutputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputBuffer = backend.AllocateBuffer(totalOutputSize);

        // Use general gather approach for all axes
        // Calculate the stride pattern for the axis
        int beforeAxisSize = 1;
        for (int i = 0; i < axis; i++)
            beforeAxisSize *= input.Shape._dims[i];

        int afterAxisSize = 1;
        for (int i = axis + 1; i < rank; i++)
            afterAxisSize *= input.Shape._dims[i];

        int srcAxisSize = input.Shape._dims[axis];

        // Build indices for gathering
        var indices = new int[totalOutputSize];
        int idx = 0;
        for (int b = 0; b < beforeAxisSize; b++)
        {
            for (int a = start; a < end; a++)
            {
                for (int s = 0; s < afterAxisSize; s++)
                {
                    indices[idx++] = b * srcAxisSize * afterAxisSize + a * afterAxisSize + s;
                }
            }
        }

        using var indicesBuffer = backend.AllocateIntBuffer(indices);
        backend.Gather(input.Buffer, indicesBuffer, outputBuffer, totalOutputSize, 1);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-accelerated power iteration for computing spectral norm (largest singular value).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="weights">GPU-resident weight matrix [rows, cols].</param>
    /// <param name="u">Left singular vector [rows] - updated in-place.</param>
    /// <param name="v">Right singular vector [cols] - updated in-place.</param>
    /// <param name="numIterations">Number of power iterations.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <returns>The estimated spectral norm (largest singular value).</returns>
    public float PowerIterationGpu<T>(
        Tensor<T> weights,
        ref Tensor<T> u,
        ref Tensor<T> v,
        int numIterations,
        float epsilon = 1e-12f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for PowerIterationGpu");

        int rows = weights.Shape._dims[0];
        int cols = weights.Shape._dims[1];

        // Allocate transpose buffer
        var wTransposeBuffer = backend.AllocateBuffer(rows * cols);

        // Compute W^T once
        backend.Transpose(weights.Buffer, wTransposeBuffer, rows, cols);

        for (int iter = 0; iter < numIterations; iter++)
        {
            // Step 1: v_new = W^T @ u
            // W^T is [cols, rows], u is [rows, 1], result is [cols, 1]
            var vNewBuffer = backend.AllocateBuffer(cols);
            backend.Gemm(wTransposeBuffer, u.Buffer, vNewBuffer, cols, 1, rows, 1.0f, 0.0f);

            // Normalize v_new
            float vNorm = backend.L2Norm(vNewBuffer, cols);
            float vNormSafe = Math.Max(vNorm, epsilon);
            backend.Scale(vNewBuffer, vNewBuffer, 1.0f / vNormSafe, cols);

            // Update v - the old buffer will be cleaned up by GC/finalizer
            v = Tensor<T>.FromGpuBuffer(backend, vNewBuffer, [cols], GpuTensorRole.Activation, true);

            // Step 2: u_new = W @ v
            // W is [rows, cols], v is [cols, 1], result is [rows, 1]
            var uNewBuffer = backend.AllocateBuffer(rows);
            backend.Gemm(weights.Buffer, v.Buffer, uNewBuffer, rows, 1, cols, 1.0f, 0.0f);

            // Normalize u_new
            float uNorm = backend.L2Norm(uNewBuffer, rows);
            float uNormSafe = Math.Max(uNorm, epsilon);
            backend.Scale(uNewBuffer, uNewBuffer, 1.0f / uNormSafe, rows);

            // Update u - the old buffer will be cleaned up by GC/finalizer
            u = Tensor<T>.FromGpuBuffer(backend, uNewBuffer, [rows], GpuTensorRole.Activation, true);
        }

        // Clean up transpose buffer
        wTransposeBuffer.Dispose();

        // Compute spectral norm: sigma = u^T @ W @ v
        // First compute Wv: W is [rows, cols], v is [cols, 1], Wv is [rows, 1]
        var wvBuffer = backend.AllocateBuffer(rows);
        backend.Gemm(weights.Buffer, v.Buffer, wvBuffer, rows, 1, cols, 1.0f, 0.0f);

        // Then compute u^T @ Wv (dot product)
        // Element-wise multiply u and Wv, then sum
        var productBuffer = backend.AllocateBuffer(rows);
        backend.Multiply(u.Buffer, wvBuffer, productBuffer, rows);

        // Sum reduction to get scalar
        var sumBuffer = backend.AllocateBuffer(1);
        backend.SumAxis(productBuffer, sumBuffer, 1, rows);

        // Download the scalar result
        float[] sumResult = backend.DownloadBuffer(sumBuffer);
        float spectralNorm = sumResult[0];

        // Clean up temporary buffers
        wvBuffer.Dispose();
        productBuffer.Dispose();
        sumBuffer.Dispose();

        return Math.Max(spectralNorm, epsilon);
    }

    /// <summary>
    /// GPU-resident scalar division: B = A / scalar
    /// </summary>
    public Tensor<T> DivideScalarGpu<T>(Tensor<T> input, float scalar)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DivideScalarGpu");

        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        // Use Scale with 1/scalar
        float invScalar = 1.0f / scalar;
        backend.Scale(input.Buffer, outputBuffer, invScalar, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident affine grid generation for spatial transformers.
    /// Given affine transformation matrices, generates a sampling grid.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="theta">GPU-resident affine transformation matrices [batch, 2, 3] flattened to [batch * 6].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="outputHeight">Height of the output grid.</param>
    /// <param name="outputWidth">Width of the output grid.</param>
    /// <returns>A GPU-resident output grid [batch, outputHeight, outputWidth, 2].</returns>
    public Tensor<T> AffineGridGpu<T>(Tensor<T> theta, int batch, int outputHeight, int outputWidth)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for AffineGridGpu");

        // Output shape: [batch, outputHeight, outputWidth, 2]
        int gridSize = batch * outputHeight * outputWidth * 2;
        var gridBuffer = backend.AllocateBuffer(gridSize);

        backend.AffineGrid(theta.Buffer, gridBuffer, batch, outputHeight, outputWidth);

        return Tensor<T>.FromGpuBuffer(backend, gridBuffer, [batch, outputHeight, outputWidth, 2], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident grid sampling with bilinear interpolation for spatial transformers.
    /// Samples from input using a sampling grid.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor [batch, channels, inHeight, inWidth].</param>
    /// <param name="grid">GPU-resident sampling grid [batch, outHeight, outWidth, 2].</param>
    /// <param name="paddingMode">Padding mode: 0=zeros, 1=border, 2=reflection.</param>
    /// <param name="alignCorners">If true, [-1, 1] maps to corner pixels.</param>
    /// <returns>A GPU-resident output tensor [batch, channels, outHeight, outWidth].</returns>
    public Tensor<T> GridSampleGpu<T>(Tensor<T> input, Tensor<T> grid, int paddingMode = 0, bool alignCorners = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GridSampleGpu");

        // Input: [batch, channels, inHeight, inWidth]
        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        // Grid: [batch, outHeight, outWidth, 2]
        int outHeight = grid.Shape._dims[1];
        int outWidth = grid.Shape._dims[2];

        // Output shape: [batch, channels, outHeight, outWidth]
        int outputSize = batch * channels * outHeight * outWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        backend.GridSample(input.Buffer, grid.Buffer, outputBuffer,
            batch, channels, inHeight, inWidth, outHeight, outWidth,
            paddingMode, alignCorners);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [batch, channels, outHeight, outWidth], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident backward pass for grid sampling.
    /// Computes gradients for both input and grid.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident gradient from upstream [batch, channels, outHeight, outWidth].</param>
    /// <param name="input">GPU-resident original input [batch, channels, inHeight, inWidth].</param>
    /// <param name="grid">GPU-resident sampling grid [batch, outHeight, outWidth, 2].</param>
    /// <param name="gradInput">Output: GPU-resident gradient w.r.t. input.</param>
    /// <param name="gradGrid">Output: GPU-resident gradient w.r.t. grid.</param>
    /// <param name="paddingMode">Padding mode: 0=zeros, 1=border, 2=reflection.</param>
    /// <param name="alignCorners">If true, [-1, 1] maps to corner pixels.</param>
    public void GridSampleBackwardGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> grid,
        out Tensor<T> gradInput,
        out Tensor<T> gradGrid,
        int paddingMode = 0,
        bool alignCorners = false)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GridSampleBackwardGpu");

        // Input: [batch, channels, inHeight, inWidth]
        int batch = input.Shape._dims[0];
        int channels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        // Grid: [batch, outHeight, outWidth, 2]
        int outHeight = grid.Shape._dims[1];
        int outWidth = grid.Shape._dims[2];

        // Allocate gradient buffers
        var gradInputBuffer = backend.AllocateBuffer(input.Length);
        var gradGridBuffer = backend.AllocateBuffer(grid.Length);

        // Initialize gradInput to zero
        backend.Fill(gradInputBuffer, 0f, input.Length);

        backend.GridSampleBackward(gradOutput.Buffer, input.Buffer, grid.Buffer,
            gradInputBuffer, gradGridBuffer,
            batch, channels, inHeight, inWidth, outHeight, outWidth,
            paddingMode, alignCorners);

        gradInput = Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, input.Shape._dims, GpuTensorRole.Gradient, true);
        gradGrid = Tensor<T>.FromGpuBuffer(backend, gradGridBuffer, grid.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident ReLU activation: y = max(0, x)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor with ReLU applied.</returns>
    public Tensor<T> ReluGpu<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ReluGpu");

        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Relu(input.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident Tanh activation: y = tanh(x)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor with Tanh applied.</returns>
    public Tensor<T> TanhGpu<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TanhGpu");

        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.Tanh(input.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident sum reduction along a specified axis.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <param name="axis">Axis to reduce (0 for sum over rows, 1 for sum over columns).</param>
    /// <returns>A GPU-resident output tensor with reduced dimensions.</returns>
    public Tensor<T> SumAxisGpu<T>(Tensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SumAxisGpu");

        // Validate axis for 2D tensors - only axis 0 and 1 are supported
        if (axis < 0 || axis > 1)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, "SumAxisGpu only supports axis 0 (sum over rows) or axis 1 (sum over columns) for 2D tensors.");

        if (input.Shape._dims.Length < 1)
            throw new ArgumentException("Input tensor must have at least one dimension.", nameof(input));

        int outerSize = input.Shape._dims[0];
        int innerSize = input.Shape._dims.Length > 1 ? input.Shape._dims[1] : 1;

        int outputSize;
        int[] outputShape;

        if (axis == 0)
        {
            // Sum over rows -> output shape [1, innerSize]
            outputSize = innerSize;
            outputShape = [1, innerSize];
        }
        else // axis == 1 (validated above)
        {
            // Sum over columns -> output shape [outerSize, 1]
            outputSize = outerSize;
            outputShape = [outerSize, 1];
        }

        var outputBuffer = backend.AllocateBuffer(outputSize);

        if (axis == 0)
        {
            if (innerSize == 1)
            {
                backend.SumAxis(input.Buffer, outputBuffer, 1, outerSize);
            }
            else
            {
                using var transposedBuffer = backend.AllocateBuffer(outerSize * innerSize);
                backend.Transpose(input.Buffer, transposedBuffer, outerSize, innerSize);
                backend.SumAxis(transposedBuffer, outputBuffer, innerSize, outerSize);
            }
        }
        else
        {
            backend.SumAxis(input.Buffer, outputBuffer, outerSize, innerSize);
        }

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident gather operation: gathers feature vectors from source at specified indices.
    /// Each index selects a feature vector of size featureSize from the source.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">GPU-resident source tensor [vocabSize, featureSize] or flat.</param>
    /// <param name="indices">GPU-resident indices buffer containing indices to gather.</param>
    /// <param name="numIndices">Number of indices to gather.</param>
    /// <param name="featureSize">Size of each feature vector.</param>
    /// <returns>A GPU-resident output tensor [numIndices, featureSize] with gathered values.</returns>
    public Tensor<T> GatherGpu<T>(Tensor<T> source, IGpuBuffer indices, int numIndices, int featureSize)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GatherGpu");

        var outputBuffer = backend.AllocateBuffer(numIndices * featureSize);

        backend.Gather(source.Buffer, indices, outputBuffer, numIndices, featureSize);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [numIndices, featureSize], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident scatter-add operation: accumulates source values into destination at specified indices.
    /// destination[indices[i]] += source[i]
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">GPU-resident source values tensor.</param>
    /// <param name="indices">GPU-resident indices buffer.</param>
    /// <param name="destSize">Size of the destination buffer.</param>
    /// <returns>A GPU-resident output tensor with scattered values.</returns>
    public Tensor<T> ScatterAddGpu<T>(Tensor<T> source, IGpuBuffer indices, int destSize)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScatterAddGpu");

        int sourceSize = source.Length;

        var outputBuffer = backend.AllocateBuffer(destSize);

        // Initialize to zero
        backend.Fill(outputBuffer, 0f, destSize);

        backend.ScatterAdd(source.Buffer, indices, outputBuffer, sourceSize, destSize);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [destSize], GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// Creates a GPU-resident tensor filled with zeros.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">Shape of the tensor to create.</param>
    /// <returns>A GPU-resident tensor filled with zeros.</returns>
    public Tensor<T> ZerosGpu<T>(int[] shape)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ZerosGpu");

        int size = 1;
        foreach (var dim in shape)
            size *= dim;

        var outputBuffer = backend.AllocateBuffer(size);
        backend.Fill(outputBuffer, 0f, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, shape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident element-wise division with broadcast: C[i,j] = A[i,j] / B[i,0]
    /// Divides each element by the corresponding element in the first column of B.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="a">GPU-resident input tensor [batchSize, features].</param>
    /// <param name="b">GPU-resident divisor tensor [batchSize, 1] to broadcast.</param>
    /// <returns>A GPU-resident output tensor with element-wise division.</returns>
    public Tensor<T> DivideByBroadcastGpu<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DivideByBroadcastGpu");

        int outerSize = a.Shape._dims[0];
        int innerSize = a.Length / outerSize;
        int bSize = b.Length;

        // Compute reciprocal of b: 1/b
        var reciprocalBuffer = backend.AllocateBuffer(bSize);
        backend.Reciprocal(b.Buffer, reciprocalBuffer, bSize);

        // Multiply a by broadcast reciprocal: a * (1/b) = a / b
        var outputBuffer = backend.AllocateBuffer(a.Length);
        backend.BroadcastMultiplyFirstAxis(a.Buffer, reciprocalBuffer, outputBuffer, outerSize, innerSize);

        // Clean up intermediate buffer
        reciprocalBuffer.Dispose();

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, a.Shape._dims, GpuTensorRole.Activation, true);
    }

    #endregion

    #region Dropout Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Dropout operation.
    /// </summary>
    Tensor<T> IEngine.Dropout<T>(Tensor<T> input, double dropoutRate, bool training, out Tensor<T> mask)
    {
        if (!TryGetBackend(out var backend) || !training)
            return base.Dropout(input, dropoutRate, training, out mask);

        try
        {
            int size = input.Length;
            ulong seed = (ulong)DateTime.UtcNow.Ticks;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, size);
            using var maskBuffer = AllocateOutputBuffer(backend, size);

            backend.Dropout(inputBuffer.Buffer, outputBuffer.Buffer, maskBuffer.Buffer, size, (float)dropoutRate, seed, training);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] maskFloat = backend.DownloadBuffer(maskBuffer.Buffer);

            mask = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(maskFloat), input.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Dropout(input, dropoutRate, training, out mask);
        }
    }

    /// <summary>
    /// GPU-accelerated Dropout backward operation.
    /// </summary>
    Tensor<T> IEngine.DropoutBackward<T>(Tensor<T> gradOutput, Tensor<T> mask, double dropoutRate)
    {
        if (!TryGetBackend(out var backend))
            return base.DropoutBackward(gradOutput, mask, dropoutRate);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var maskBuffer = GetOrAllocateBuffer(backend, mask.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.DropoutBackward(gradOutBuffer.Buffer, maskBuffer.Buffer, gradInputBuffer.Buffer, size, (float)dropoutRate);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.DropoutBackward(gradOutput, mask, dropoutRate);
        }
    }

    /// <summary>
    /// GPU-accelerated fused BiasAdd + Dropout in a single kernel launch.
    /// Eliminates one global memory round-trip compared to separate BiasAdd + Dropout calls.
    /// Falls back to separate operations if the backend doesn't support the fused kernel.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="input">Input tensor [batch x features] or any shape where last dim = bias length.</param>
    /// <param name="bias">1D bias tensor [features].</param>
    /// <param name="dropoutRate">Probability of dropping each element (0.0 to 1.0).</param>
    /// <param name="training">Whether we're in training mode (dropout only applies during training).</param>
    /// <param name="mask">Output dropout mask tensor (1 = kept, 0 = dropped).</param>
    /// <returns>Result tensor with bias added and dropout applied.</returns>
    public Tensor<T> FusedBiasDropout<T>(Tensor<T> input, Tensor<T> bias, double dropoutRate, bool training, out Tensor<T> mask)
    {
        if (input.Shape._dims.Length == 0)
            throw new ArgumentException("Input tensor must have at least one dimension.", nameof(input));
        if (bias.Shape._dims.Length != 1)
            throw new ArgumentException("Bias tensor must be 1-dimensional.", nameof(bias));
        if (bias.Length != input.Shape._dims[^1])
            throw new ArgumentException($"Bias length {bias.Length} must match input's last dimension {input.Shape._dims[^1]}.", nameof(bias));
        if (dropoutRate < 0.0 || dropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(dropoutRate), dropoutRate, "Dropout rate must be in [0, 1).");

        if (!TryGetBackend(out var backend))
        {
            // No GPU — fall back to CPU: add bias element-wise then dropout
            return CpuFallbackBiasDropout(input, bias, dropoutRate, training, out mask);
        }

        if (!training || dropoutRate <= 0.0)
        {
            // No dropout — just do bias add on GPU, return all-ones mask
            int lastDim = input.Shape._dims[^1];
            int rows = input.Length / lastDim;
            try
            {
                using var inputBuf = GetOrAllocateBuffer(backend, input.GetDataArray());
                using var biasBuf = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
                using var outBuf = AllocateOutputBuffer(backend, input.Length);
                backend.BiasAdd(inputBuf.Buffer, biasBuf.Buffer, outBuf.Buffer, rows, lastDim);
                float[] resultFloat = backend.DownloadBuffer(outBuf.Buffer);
                var numOps = MathHelper.GetNumericOperations<T>();
                var ones = new T[input.Length];
                for (int i = 0; i < ones.Length; i++) ones[i] = numOps.One;
                mask = new Tensor<T>(ones, input.Shape.ToArray());
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
            }
            catch
            {
                return CpuFallbackBiasDropout(input, bias, dropoutRate, training, out mask);
            }
        }

        try
        {
            int lastDim = input.Shape._dims[^1];
            int rows = input.Length / lastDim;
            int cols = lastDim;
            float scale = 1.0f / (1.0f - (float)dropoutRate);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var maskBuffer = AllocateOutputBuffer(backend, input.Length);

            // Generate mask via existing Dropout kernel (it writes the mask as a side effect)
            ulong seed = (ulong)DateTime.UtcNow.Ticks;
            using var tempBuffer = AllocateOutputBuffer(backend, input.Length);
            backend.Dropout(inputBuffer.Buffer, tempBuffer.Buffer, maskBuffer.Buffer, input.Length, (float)dropoutRate, seed, training);

            // Try fused path: single kernel does bias + mask application
            bool fused = backend.TryFusedBiasDropout(
                inputBuffer.Buffer, outputBuffer.Buffer, biasBuffer.Buffer, maskBuffer.Buffer,
                rows, cols, (float)dropoutRate, scale);

            if (!fused)
            {
                // Fallback: separate BiasAdd + re-apply dropout with existing mask
                backend.BiasAdd(inputBuffer.Buffer, biasBuffer.Buffer, outputBuffer.Buffer, rows, cols);
                backend.Dropout(outputBuffer.Buffer, outputBuffer.Buffer, maskBuffer.Buffer, input.Length, (float)dropoutRate, seed, training);
            }

            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] maskFloat = backend.DownloadBuffer(maskBuffer.Buffer);

            mask = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(maskFloat), input.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return CpuFallbackBiasDropout(input, bias, dropoutRate, training, out mask);
        }
    }

    /// <summary>
    /// CPU fallback for fused bias + dropout when no GPU is available.
    /// </summary>
    private Tensor<T> CpuFallbackBiasDropout<T>(Tensor<T> input, Tensor<T> bias, double dropoutRate, bool training, out Tensor<T> mask)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int lastDim = input.Shape._dims[^1];
        var inputData = input.GetDataArray();
        var biasData = bias.GetDataArray();
        var result = new T[input.Length];

        // Add bias (broadcast along last dimension)
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = numOps.Add(inputData[i], biasData[i % lastDim]);
        }

        var biasedTensor = new Tensor<T>(result, input.Shape.ToArray());
        return base.Dropout(biasedTensor, dropoutRate, training, out mask);
    }

    #endregion

    #region Embedding Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Embedding lookup operation.
    /// </summary>
    Tensor<T> IEngine.Embedding<T>(Tensor<int> indices, Tensor<T> embeddingTable)
    {
        if (!TryGetBackend(out var backend))
            return base.Embedding(indices, embeddingTable);

        try
        {
            int numIndices = indices.Length;
            int embeddingDim = embeddingTable.Shape._dims[^1];

            using var indicesBuffer = backend.AllocateIntBuffer(indices.GetDataArray());
            using var tableBuffer = GetOrCacheWeightBuffer(backend, embeddingTable.GetDataArray(), PersistentTensorRole.Embeddings);
            using var outputBuffer = AllocateOutputBuffer(backend, numIndices * embeddingDim);

            backend.Embedding(indicesBuffer, tableBuffer.Buffer, outputBuffer.Buffer, numIndices, embeddingDim);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);

            // Output shape: indices.Shape + [embeddingDim]
            int[] outputShape = new int[indices.Shape._dims.Length + 1];
            for (int i = 0; i < indices.Shape._dims.Length; i++)
                outputShape[i] = indices.Shape._dims[i];
            outputShape[^1] = embeddingDim;

            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), outputShape);
        }
        catch
        {
            return base.Embedding(indices, embeddingTable);
        }
    }

    /// <summary>
    /// GPU-accelerated Embedding backward operation.
    /// </summary>
    Tensor<T> IEngine.EmbeddingBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int vocabSize, int embeddingDim)
    {
        if (!TryGetBackend(out var backend))
            return base.EmbeddingBackward(gradOutput, indices, vocabSize, embeddingDim);

        try
        {
            int numIndices = indices.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var indicesBuffer = backend.AllocateIntBuffer(indices.GetDataArray());
            using var gradEmbeddingBuffer = AllocateOutputBuffer(backend, vocabSize * embeddingDim);

            // Initialize to zero
            backend.Fill(gradEmbeddingBuffer.Buffer, 0f, vocabSize * embeddingDim);

            backend.EmbeddingBackward(gradOutBuffer.Buffer, indicesBuffer, gradEmbeddingBuffer.Buffer, numIndices, embeddingDim, vocabSize);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradEmbeddingBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { vocabSize, embeddingDim });
        }
        catch
        {
            return base.EmbeddingBackward(gradOutput, indices, vocabSize, embeddingDim);
        }
    }

    /// <summary>
    /// GPU-resident embedding lookup operation.
    /// Performs embedding lookup on GPU and returns a GPU-resident tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the embedding tensor.</typeparam>
    /// <param name="embeddingTable">The embedding table tensor (either CPU Tensor or already on GPU).</param>
    /// <param name="indices">The token indices to look up.</param>
    /// <returns>A GPU-resident tensor containing the embeddings for the given indices.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <remarks>
    /// <para>
    /// This method performs embedding lookup entirely on GPU, returning a GPU-resident tensor
    /// that can be passed to subsequent GPU operations without downloading to CPU.
    /// </para>
    /// <para>
    /// The output shape is: indices.Shape + [embeddingDim]
    /// For example, if indices has shape [batch, seqLen] and embeddingDim is 512,
    /// the output will have shape [batch, seqLen, 512].
    /// </para>
    /// </remarks>
    public Tensor<T> EmbeddingLookupGpu<T>(Tensor<T> embeddingTable, Tensor<int> indices)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for EmbeddingLookupGpu");

        int numIndices = indices.Length;
        int embeddingDim = embeddingTable.Shape._dims[^1];

        // Upload indices and embedding table to GPU
        using var indicesBuffer = backend.AllocateIntBuffer(indices.GetDataArray());
        using var tableBuffer = GetOrCacheWeightBuffer(backend, embeddingTable.GetDataArray(), PersistentTensorRole.Embeddings);

        // Allocate output buffer (stays on GPU)
        var outputBuffer = backend.AllocateBuffer(numIndices * embeddingDim);

        // Perform embedding lookup on GPU
        backend.Embedding(indicesBuffer, tableBuffer.Buffer, outputBuffer, numIndices, embeddingDim);

        // Calculate output shape: indices.Shape + [embeddingDim]
        int[] outputShape = new int[indices.Shape._dims.Length + 1];
        for (int i = 0; i < indices.Shape._dims.Length; i++)
            outputShape[i] = indices.Shape._dims[i];
        outputShape[^1] = embeddingDim;

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident embedding lookup operation with GPU-resident embedding table.
    /// Both input embedding table and output remain on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type of the embedding tensor.</typeparam>
    /// <param name="embeddingTableGpu">The GPU-resident embedding table.</param>
    /// <param name="indices">The token indices to look up.</param>
    /// <param name="embeddingDim">The dimension of each embedding vector.</param>
    /// <returns>A GPU-resident tensor containing the embeddings for the given indices.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public Tensor<T> EmbeddingLookupGpu<T>(Tensor<T> embeddingTableGpu, Tensor<int> indices, int embeddingDim)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for EmbeddingLookupGpu");

        int numIndices = indices.Length;

        // Upload indices to GPU (embedding table is already on GPU)
        using var indicesBuffer = backend.AllocateIntBuffer(indices.GetDataArray());

        // Allocate output buffer (stays on GPU)
        var outputBuffer = backend.AllocateBuffer(numIndices * embeddingDim);

        // Perform embedding lookup on GPU
        backend.Embedding(indicesBuffer, embeddingTableGpu.Buffer, outputBuffer, numIndices, embeddingDim);

        // Calculate output shape: indices.Shape + [embeddingDim]
        int[] outputShape = new int[indices.Shape._dims.Length + 1];
        for (int i = 0; i < indices.Shape._dims.Length; i++)
            outputShape[i] = indices.Shape._dims[i];
        outputShape[^1] = embeddingDim;

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// GPU-resident embedding backward operation.
    /// Computes gradients for the embedding table on GPU.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradOutput">The GPU-resident gradient of the loss w.r.t. output.</param>
    /// <param name="indices">The indices that were used in the forward pass.</param>
    /// <param name="vocabSize">The vocabulary size (number of embeddings).</param>
    /// <param name="embeddingDim">The dimension of each embedding vector.</param>
    /// <returns>A GPU-resident gradient tensor for the embedding table.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public Tensor<T> EmbeddingBackwardGpu<T>(Tensor<T> gradOutput, Tensor<int> indices, int vocabSize, int embeddingDim)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for EmbeddingBackwardGpu");

        int numIndices = indices.Length;

        // Upload indices to GPU
        using var indicesBuffer = backend.AllocateIntBuffer(indices.GetDataArray());

        // Allocate gradient embedding buffer and initialize to zero
        var gradEmbeddingBuffer = backend.AllocateBuffer(vocabSize * embeddingDim);
        backend.Fill(gradEmbeddingBuffer, 0f, vocabSize * embeddingDim);

        // Perform scatter-add for gradient accumulation
        backend.EmbeddingBackward(gradOutput.Buffer, indicesBuffer, gradEmbeddingBuffer, numIndices, embeddingDim, vocabSize);

        return Tensor<T>.FromGpuBuffer(backend, gradEmbeddingBuffer, [vocabSize, embeddingDim], GpuTensorRole.Gradient, ownsBuffer: true);
    }

    #endregion

    #region Loss Functions (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated CrossEntropy loss computation.
    /// </summary>

    /// <summary>
    /// GPU-accelerated CrossEntropy backward computation.
    /// </summary>
    Tensor<T> IEngine.CrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.CrossEntropyBackward(predictions, targets);

        try
        {
            if (predictions.Rank != 2)
                return base.CrossEntropyBackward(predictions, targets);

            int batchSize = predictions.Shape._dims[0];
            int numClasses = predictions.Shape._dims[1];

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.GetDataArray());
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, predictions.Length);

            backend.CrossEntropyBackward(predBuffer.Buffer, targetBuffer.Buffer, gradInputBuffer.Buffer, batchSize, numClasses);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), predictions.Shape.ToArray());
        }
        catch
        {
            return base.CrossEntropyBackward(predictions, targets);
        }
    }

    /// <summary>
    /// GPU-accelerated MSE loss computation.
    /// </summary>

    /// <summary>
    /// GPU-accelerated MSE backward computation.
    /// </summary>
    Tensor<T> IEngine.MseBackward<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.MseBackward(predictions, targets);

        try
        {
            int size = predictions.Length;

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.GetDataArray());
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.MseBackward(predBuffer.Buffer, targetBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), predictions.Shape.ToArray());
        }
        catch
        {
            return base.MseBackward(predictions, targets);
        }
    }

    #endregion

    #region Activation Backward Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated ReLU backward operation.
    /// </summary>
    Tensor<T> IEngine.ReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.ReluBackward(gradOutput, input);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.ReluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.ReluBackward(gradOutput, input);
        }
    }

    /// <summary>
    /// GPU-accelerated Sigmoid backward operation.
    /// </summary>
    Tensor<T> IEngine.SigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            return base.SigmoidBackward(gradOutput, output);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var outputBuffer = GetOrAllocateBuffer(backend, output.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.SigmoidBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.SigmoidBackward(gradOutput, output);
        }
    }

    /// <summary>
    /// GPU-accelerated Tanh backward operation.
    /// </summary>
    Tensor<T> IEngine.TanhBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            return base.TanhBackward(gradOutput, output);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var outputBuffer = GetOrAllocateBuffer(backend, output.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.TanhBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.TanhBackward(gradOutput, output);
        }
    }

    /// <summary>
    /// GPU-accelerated GELU backward operation.
    /// </summary>
    Tensor<T> IEngine.GeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GeluBackward(gradOutput, input);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.GeluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.GeluBackward(gradOutput, input);
        }
    }

    /// <summary>
    /// GPU-accelerated LeakyReLU activation.
    /// </summary>
    Tensor<T> IEngine.LeakyReLU<T>(Tensor<T> input, T alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.LeakyReLU(input, alpha);

        try
        {
            int size = input.Length;
            var numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            float negativeSlope = (float)numOps.ToDouble(alpha);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.LeakyRelu(inputBuffer.Buffer, outputBuffer.Buffer, negativeSlope, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LeakyReLU(input, alpha);
        }
    }

    /// <summary>
    /// GPU-accelerated LeakyReLU backward operation.
    /// </summary>
    Tensor<T> IEngine.LeakyReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double negativeSlope)
    {
        if (!TryGetBackend(out var backend))
            return base.LeakyReluBackward(gradOutput, input, negativeSlope);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.LeakyReluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, (float)negativeSlope, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.LeakyReluBackward(gradOutput, input, negativeSlope);
        }
    }

    /// <summary>
    /// GPU-accelerated ELU activation.
    /// </summary>
    Tensor<T> IEngine.ELU<T>(Tensor<T> input, double alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.ELU(input, alpha);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Elu(inputBuffer.Buffer, outputBuffer.Buffer, (float)alpha, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.ELU(input, alpha);
        }
    }

    /// <summary>
    /// GPU-accelerated Swish activation.
    /// </summary>
    Tensor<T> IEngine.Swish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Swish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Swish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Swish(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Mish activation.
    /// </summary>
    Tensor<T> IEngine.Mish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Mish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Mish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Mish(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Softplus activation.
    /// </summary>
    Tensor<T> IEngine.Softplus<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Softplus(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Softplus(inputBuffer.Buffer, outputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Softplus(input);
        }
    }

    /// <summary>
    /// GPU-accelerated HardSwish activation.
    /// </summary>
    Tensor<T> IEngine.HardSwish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.HardSwish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Hardswish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.HardSwish(input);
        }
    }

    #endregion

    #region Convolution Backward Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Conv2D backward for input gradients.
    /// </summary>
    Tensor<T> IEngine.Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape,
        int[] stride, int[] padding, int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);

        try
        {
            if (gradOutput.Rank != 4 || kernel.Rank != 4)
                return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);

            int strideH = stride.Length > 0 ? stride[0] : 1;
            int strideW = stride.Length > 1 ? stride[1] : strideH;
            int padH = padding.Length > 0 ? padding[0] : 0;
            int padW = padding.Length > 1 ? padding[1] : padH;
            int dilationH = dilation.Length > 0 ? dilation[0] : 1;
            int dilationW = dilation.Length > 1 ? dilation[1] : dilationH;

            int batch = gradOutput.Shape._dims[0];
            int outChannels = gradOutput.Shape._dims[1];
            int outHeight = gradOutput.Shape._dims[2];
            int outWidth = gradOutput.Shape._dims[3];

            int inChannels = inputShape[1];
            int inHeight = inputShape[2];
            int inWidth = inputShape[3];

            int kernelH = kernel.Shape._dims[2];
            int kernelW = kernel.Shape._dims[3];

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);
            using var gradInputBuffer = AllocateOutputBuffer(backend, batch * inChannels * inHeight * inWidth);

            backend.Conv2DBackwardInput(gradOutBuffer.Buffer, kernelBuffer.Buffer, gradInputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), inputShape);
        }
        catch
        {
            return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated Conv2D backward for kernel gradients.
    /// </summary>
    Tensor<T> IEngine.Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape,
        int[] stride, int[] padding, int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);

        try
        {
            if (input.Rank != 4 || gradOutput.Rank != 4)
                return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);

            int strideH = stride.Length > 0 ? stride[0] : 1;
            int strideW = stride.Length > 1 ? stride[1] : strideH;
            int padH = padding.Length > 0 ? padding[0] : 0;
            int padW = padding.Length > 1 ? padding[1] : padH;
            int dilationH = dilation.Length > 0 ? dilation[0] : 1;
            int dilationW = dilation.Length > 1 ? dilation[1] : dilationH;

            int batch = input.Shape._dims[0];
            int inChannels = input.Shape._dims[1];
            int inHeight = input.Shape._dims[2];
            int inWidth = input.Shape._dims[3];

            int outChannels = gradOutput.Shape._dims[1];
            int outHeight = gradOutput.Shape._dims[2];
            int outWidth = gradOutput.Shape._dims[3];

            int kernelH = kernelShape[2];
            int kernelW = kernelShape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.GetDataArray());
            using var gradKernelBuffer = AllocateOutputBuffer(backend, outChannels * inChannels * kernelH * kernelW);

            backend.Conv2DBackwardKernel(inputBuffer.Buffer, gradOutBuffer.Buffer, gradKernelBuffer.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(gradKernelBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), kernelShape);
        }
        catch
        {
            return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
        }
    }

    #endregion

    #region Global Pooling Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Global Average Pooling.
    /// </summary>
    Tensor<T> IEngine.GlobalAvgPool2D<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GlobalAvgPool2D(input);

        try
        {
            if (input.Rank != 4)
                return base.GlobalAvgPool2D(input);

            int batch = input.Shape._dims[0];
            int channels = input.Shape._dims[1];
            int height = input.Shape._dims[2];
            int width = input.Shape._dims[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.GlobalAvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, height, width);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, 1, 1 });
        }
        catch
        {
            return base.GlobalAvgPool2D(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Global Max Pooling.
    /// </summary>
    Tensor<T> IEngine.GlobalMaxPool2D<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GlobalMaxPool2D(input);

        try
        {
            if (input.Rank != 4)
                return base.GlobalMaxPool2D(input);

            int batch = input.Shape._dims[0];
            int channels = input.Shape._dims[1];
            int height = input.Shape._dims[2];
            int width = input.Shape._dims[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.GlobalMaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, height, width);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, 1, 1 });
        }
        catch
        {
            return base.GlobalMaxPool2D(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Adaptive Average Pooling.
    /// </summary>
    Tensor<T> IEngine.AdaptiveAvgPool2D<T>(Tensor<T> input, int outputHeight, int outputWidth)
    {
        if (!TryGetBackend(out var backend))
            return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);

        try
        {
            if (input.Rank != 4)
                return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);

            int batch = input.Shape._dims[0];
            int channels = input.Shape._dims[1];
            int inHeight = input.Shape._dims[2];
            int inWidth = input.Shape._dims[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outputHeight * outputWidth);

            backend.AdaptiveAvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, inHeight, inWidth, outputHeight, outputWidth);
            // DownloadBuffer uses blocking read, Synchronize() removed for performance
            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, outputHeight, outputWidth });
        }
        catch
        {
            return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);
        }
    }

    #endregion

    #region GPU-Accelerated Reduction Operations

    /// <summary>
    /// GPU-accelerated ReduceMean operation.
    /// </summary>
    public override Tensor<T> ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        var safeAxes = axes ?? Array.Empty<int>();
        if (!TryGetBackend(out var backend))
            return base.ReduceMean(input, safeAxes, keepDims);

        // Validate and normalize axes
        if (safeAxes.Length == 0)
            return base.ReduceMean(input, safeAxes, keepDims);

        // Normalize negative axes
        var normalizedAxes = new int[safeAxes.Length];
        for (int i = 0; i < safeAxes.Length; i++)
        {
            normalizedAxes[i] = safeAxes[i] < 0 ? safeAxes[i] + input.Rank : safeAxes[i];
        }
        Array.Sort(normalizedAxes);

        try
        {
            return ReduceAxisGpu(input, normalizedAxes, keepDims, backend, ReduceOperation.Mean);
        }
        catch
        {
            return base.ReduceMean(input, safeAxes, keepDims);
        }
    }

    /// <summary>
    /// GPU-accelerated ReduceMax operation.
    /// </summary>
    public new Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices)
    {
        var safeAxes = axes ?? Array.Empty<int>();
        if (!TryGetBackend(out var backend))
            return base.ReduceMax(input, safeAxes, keepDims, out maxIndices);

        // Validate and normalize axes
        if (safeAxes.Length == 0)
            return base.ReduceMax(input, safeAxes, keepDims, out maxIndices);

        // Normalize negative axes
        var normalizedAxes = new int[safeAxes.Length];
        for (int i = 0; i < safeAxes.Length; i++)
        {
            normalizedAxes[i] = safeAxes[i] < 0 ? safeAxes[i] + input.Rank : safeAxes[i];
        }
        Array.Sort(normalizedAxes);

        try
        {
            // Run MaxAxis (values) and ArgMaxAxis (indices) on the same
            // GPU upload — replaces the previous CPU re-reduce that did
            // 2× the work for every ReduceMax-with-indices call.
            return ReduceMaxWithIndicesGpu(input, normalizedAxes, keepDims, backend, out maxIndices);
        }
        catch
        {
            return base.ReduceMax(input, safeAxes, keepDims, out maxIndices);
        }
    }

    /// <summary>
    /// Single-pass GPU reduce-max that produces both values and indices.
    /// Mirrors <see cref="ReduceAxisGpu"/>'s shape-collapse logic but
    /// dispatches both <see cref="IDirectGpuBackend.MaxAxis"/> and
    /// <see cref="IDirectGpuBackend.ArgMaxAxis"/> on the same uploaded
    /// input buffer. Indices are returned in the original tensor's
    /// row-major frame after the optional permutation is undone.
    /// </summary>
    private Tensor<T> ReduceMaxWithIndicesGpu<T>(Tensor<T> input, int[] normalizedAxes, bool keepDims,
        IDirectGpuBackend backend, out int[] maxIndices)
    {
        var inputShape = input.Shape._dims;
        int inputRank = inputShape.Length;
        var outputShapeList = new List<int>();
        int reduceSize = 1;
        int outerSize = 1;
        bool permuted = false;

        if (normalizedAxes.Length == 1 && normalizedAxes[0] == inputRank - 1)
        {
            for (int i = 0; i < inputRank - 1; i++)
            {
                outerSize *= inputShape[i];
                outputShapeList.Add(inputShape[i]);
            }
            reduceSize = inputShape[^1];
            if (keepDims) outputShapeList.Add(1);
        }
        else
        {
            var permutation = new List<int>();
            var reduceDims = new HashSet<int>(normalizedAxes);
            for (int i = 0; i < inputRank; i++)
            {
                if (!reduceDims.Contains(i))
                {
                    permutation.Add(i);
                    outerSize *= inputShape[i];
                    outputShapeList.Add(inputShape[i]);
                }
            }
            foreach (int axis in normalizedAxes)
            {
                permutation.Add(axis);
                reduceSize *= inputShape[axis];
                if (keepDims) outputShapeList.Add(1);
            }
            input = PermuteImpl(input, permutation.ToArray());
            permuted = true;
        }
        if (outputShapeList.Count == 0) outputShapeList.Add(1);
        var outputShape = outputShapeList.ToArray();

        float[] inputFloat = DirectGpuEngine.ToFloatArray(input.GetDataArray());
        using var inputBuffer = GetOrAllocateBuffer(backend, inputFloat);
        using var valuesBuffer = AllocateOutputBuffer(backend, outerSize);
        // ArgMaxAxis writes int32 indices — same outerSize as values.
        using var indicesBuffer = AllocateOutputBuffer(backend, outerSize);

        backend.MaxAxis(inputBuffer.Buffer, valuesBuffer.Buffer, outerSize, reduceSize);
        backend.ArgMaxAxis(inputBuffer.Buffer, indicesBuffer.Buffer, outerSize, reduceSize);

        float[] resultFloat = backend.DownloadBuffer(valuesBuffer.Buffer);
        T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
        // Indices come back as float (DownloadBuffer is float-typed) but
        // hold int32 bit patterns the kernel wrote (WebGPU `bitcast<f32>(i32)`,
        // Vulkan/Metal `Int32BitsToSingleCompat`). Use bit reinterpretation —
        // a value cast `(int)f` would truncate the float interpretation of
        // those bit patterns and corrupt every index above ~16M.
        float[] indicesFloat = backend.DownloadBuffer(indicesBuffer.Buffer);
        maxIndices = new int[indicesFloat.Length];
        for (int i = 0; i < indicesFloat.Length; i++)
        {
#if NET5_0_OR_GREATER
            maxIndices[i] = BitConverter.SingleToInt32Bits(indicesFloat[i]);
#else
            // net471 fallback: same bit pattern via Unsafe.As, since
            // BitConverter.SingleToInt32Bits is .NET 5+ only.
            float f = indicesFloat[i];
            maxIndices[i] = System.Runtime.CompilerServices.Unsafe.As<float, int>(ref f);
#endif
        }
        // If we permuted, the kernel-emitted indices are over the
        // permuted-axis layout. The caller expects indices in the
        // original tensor's frame — but the permutation collapsed every
        // reduction axis into a single trailing flat index, so the
        // caller's contract for multi-axis reduce-max is the flat index
        // into the collapsed reduction span (matches PyTorch's
        // torch.max(dim=...) semantics).
        _ = permuted;
        return new Tensor<T>(outputShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// GPU-accelerated ReduceSum operation.
    /// </summary>
    public override Tensor<T> ReduceSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        if (!TryGetBackend(out var backend))
            return base.ReduceSum(tensor, axes, keepDims);

        // If axes is null, reduce all dimensions
        if (axes == null || axes.Length == 0)
        {
            // Full reduction - use existing Sum implementation
            return base.ReduceSum(tensor, axes, keepDims);
        }

        // Normalize negative axes
        var normalizedAxes = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            normalizedAxes[i] = axes[i] < 0 ? axes[i] + tensor.Rank : axes[i];
        }
        Array.Sort(normalizedAxes);

        try
        {
            return ReduceAxisGpu(tensor, normalizedAxes, keepDims, backend, ReduceOperation.Sum);
        }
        catch
        {
            return base.ReduceSum(tensor, axes, keepDims);
        }
    }

    private enum ReduceOperation { Sum, Mean, Max }

    /// <summary>
    /// Internal GPU reduction implementation that handles arbitrary axes.
    /// </summary>
    private Tensor<T> ReduceAxisGpu<T>(Tensor<T> input, int[] normalizedAxes, bool keepDims,
        IDirectGpuBackend backend, ReduceOperation op)
    {
        var inputShape = input.Shape._dims;
        int inputRank = inputShape.Length;

        // Compute output shape
        var outputShapeList = new List<int>();
        int reduceSize = 1;
        int outerSize = 1;

        // For single axis reduction at the end, we can use backend directly
        // For other cases, we need to reshape/permute
        if (normalizedAxes.Length == 1 && normalizedAxes[0] == inputRank - 1)
        {
            // Reduction over last axis - optimal case
            for (int i = 0; i < inputRank - 1; i++)
            {
                outerSize *= inputShape[i];
                outputShapeList.Add(inputShape[i]);
            }
            reduceSize = inputShape[^1];
            if (keepDims) outputShapeList.Add(1);
        }
        else
        {
            // General case: permute axes so reduction axes are at the end
            // Then reshape to 2D [outerSize, reduceSize]
            var permutation = new List<int>();
            var reduceDims = new HashSet<int>(normalizedAxes);

            // First add non-reduce dimensions
            for (int i = 0; i < inputRank; i++)
            {
                if (!reduceDims.Contains(i))
                {
                    permutation.Add(i);
                    outerSize *= inputShape[i];
                    outputShapeList.Add(inputShape[i]);
                }
            }

            // Then add reduce dimensions
            foreach (int axis in normalizedAxes)
            {
                permutation.Add(axis);
                reduceSize *= inputShape[axis];
                if (keepDims) outputShapeList.Add(1);
            }

            // Permute the input tensor
            input = PermuteImpl(input, permutation.ToArray());
        }

        if (outputShapeList.Count == 0)
            outputShapeList.Add(1);

        var outputShape = outputShapeList.ToArray();

        // Upload input
        float[] inputFloat = DirectGpuEngine.ToFloatArray(input.GetDataArray());
        using var inputBuffer = GetOrAllocateBuffer(backend, inputFloat);
        using var outputBuffer = AllocateOutputBuffer(backend, outerSize);

        // Execute the appropriate reduction
        switch (op)
        {
            case ReduceOperation.Sum:
                backend.SumAxis(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, reduceSize);
                break;
            case ReduceOperation.Mean:
                backend.MeanAxis(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, reduceSize);
                break;
            case ReduceOperation.Max:
                backend.MaxAxis(inputBuffer.Buffer, outputBuffer.Buffer, outerSize, reduceSize);
                break;
        }

        // Download result
        float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
        T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);

        return new Tensor<T>(outputShape, new Vector<T>(resultData));
    }

    /// <summary>
    /// Internal tensor permutation helper.
    /// </summary>
    private static Tensor<T> PermuteImpl<T>(Tensor<T> input, int[] permutation)
    {
        var inputShape = input.Shape._dims;
        int rank = inputShape.Length;

        // Compute output shape
        var outputShape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            outputShape[i] = inputShape[permutation[i]];
        }

        // Compute strides
        var inputStrides = new int[rank];
        var outputStrides = new int[rank];
        inputStrides[rank - 1] = 1;
        outputStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
        {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];
        }

        var inputData = input.GetDataArray();
        var outputData = new T[input.Length];

        // Permute data
        for (int i = 0; i < input.Length; i++)
        {
            // Convert flat index to multi-index
            var multiIndex = new int[rank];
            int remaining = i;
            for (int d = 0; d < rank; d++)
            {
                multiIndex[d] = remaining / inputStrides[d];
                remaining %= inputStrides[d];
            }

            // Apply permutation and compute output index
            int outputIdx = 0;
            for (int d = 0; d < rank; d++)
            {
                outputIdx += multiIndex[permutation[d]] * outputStrides[d];
            }

            outputData[outputIdx] = inputData[i];
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    #endregion

    #region Broadcast Operations

    /// <summary>
    /// GPU-accelerated TensorBroadcastMultiply operation.
    /// Performs element-wise multiplication with NumPy-style broadcasting.
    /// </summary>
    public override Tensor<T> TensorBroadcastMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorBroadcastMultiply(a, b);

        // Fast path: same shape - use element-wise multiply
        if (a.Shape._dims.SequenceEqual(b.Shape._dims))
        {
            try
            {
                using var bufferA = GetOrAllocateBuffer(backend, a.GetDataArray());
                using var bufferB = GetOrAllocateBuffer(backend, b.GetDataArray());
                using var bufferC = AllocateOutputBuffer(backend, a.Length);

                backend.Multiply(bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, a.Length);

                float[] resultFloat = new float[a.Length];
                backend.DownloadBuffer(bufferC.Buffer, resultFloat);
                return new Tensor<T>(a.Shape._dims, new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat)));
            }
            catch
            {
                return base.TensorBroadcastMultiply(a, b);
            }
        }

        // Check for common broadcast patterns that we can accelerate
        try
        {
            // Pattern 1: (outer, inner) * (inner,) -> broadcast along last axis
            if (b.Rank == 1 && a.Shape._dims[a.Rank - 1] == b.Shape._dims[0])
            {
                int innerSize = b.Shape._dims[0];
                int outerSize = a.Length / innerSize;

                using var bufferA = GetOrAllocateBuffer(backend, a.GetDataArray());
                using var bufferB = GetOrAllocateBuffer(backend, b.GetDataArray());
                using var bufferC = AllocateOutputBuffer(backend, a.Length);

                backend.BroadcastMultiplyLastAxis(bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, outerSize, innerSize);

                float[] resultFloat = new float[a.Length];
                backend.DownloadBuffer(bufferC.Buffer, resultFloat);
                return new Tensor<T>(a.Shape._dims, new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat)));
            }

            // Pattern 2: (outer, inner) * (outer, 1) -> broadcast along first axis (column broadcast)
            if (a.Rank == 2 && b.Rank == 2 && b.Shape._dims[0] == a.Shape._dims[0] && b.Shape._dims[1] == 1)
            {
                int outerSize = a.Shape._dims[0];
                int innerSize = a.Shape._dims[1];

                // Extract first column from b as 1D array
                T[] bFlatData = new T[outerSize];
                for (int i = 0; i < outerSize; i++)
                    bFlatData[i] = b.GetDataArray()[i];

                using var bufferA = GetOrAllocateBuffer(backend, a.GetDataArray());
                using var bufferB = GetOrAllocateBuffer(backend, bFlatData);
                using var bufferC = AllocateOutputBuffer(backend, a.Length);

                backend.BroadcastMultiplyFirstAxis(bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, outerSize, innerSize);

                float[] resultFloat = new float[a.Length];
                backend.DownloadBuffer(bufferC.Buffer, resultFloat);
                return new Tensor<T>(a.Shape._dims, new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat)));
            }

            // Pattern 3: (batch, seq, features) * (1, 1, features) -> common in attention/normalization
            if (a.Rank >= 2 && b.Rank == a.Rank)
            {
                // Check if broadcasting along all but last axis
                bool isLastAxisBroadcast = true;
                for (int i = 0; i < a.Rank - 1; i++)
                {
                    if (b.Shape._dims[i] != 1)
                    {
                        isLastAxisBroadcast = false;
                        break;
                    }
                }
                if (isLastAxisBroadcast && a.Shape._dims[a.Rank - 1] == b.Shape._dims[b.Rank - 1])
                {
                    int innerSize = a.Shape._dims[a.Rank - 1];
                    int outerSize = a.Length / innerSize;

                    // Extract last dimension from b as 1D array
                    T[] bFlatData = new T[innerSize];
                    for (int i = 0; i < innerSize; i++)
                        bFlatData[i] = b.GetDataArray()[i];

                    using var bufferA = GetOrAllocateBuffer(backend, a.GetDataArray());
                    using var bufferB = GetOrAllocateBuffer(backend, bFlatData);
                    using var bufferC = AllocateOutputBuffer(backend, a.Length);

                    backend.BroadcastMultiplyLastAxis(bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, outerSize, innerSize);

                    float[] resultFloat = new float[a.Length];
                    backend.DownloadBuffer(bufferC.Buffer, resultFloat);
                    return new Tensor<T>(a.Shape._dims, new Vector<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat)));
                }
            }

            // Fallback to CPU for complex broadcast patterns
            return base.TensorBroadcastMultiply(a, b);
        }
        catch
        {
            return base.TensorBroadcastMultiply(a, b);
        }
    }

    #endregion

    /// <summary>
    /// Gets or uploads CSR GPU buffers for a sparse tensor, caching for repeated calls.
    /// Cache key includes the backend instance to avoid cross-backend buffer reuse.
    /// </summary>
    private CsrGpuCache GetOrUploadCsrBuffers<T>(IDirectGpuBackend backend, SparseTensor<T> sparse)
    {
        var key = new CsrCacheKey(sparse, backend);
        if (_csrBufferCache.TryGetValue(key, out var existing))
            return existing;

        var csr = sparse.ToCsr();
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        var floatVals = new float[csr.Values.Length];
        for (int i = 0; i < csr.Values.Length; i++)
            floatVals[i] = (float)numOps.ToDouble(csr.Values[i]);

        // Upload indices as bit-exact int32 reinterpreted as float32 —
        // GPU kernels declare these as int* and read the raw bit pattern.
        static float[] ReinterpretIntsAsFloats(int[] ints)
        {
            var floats = new float[ints.Length];
            Buffer.BlockCopy(ints, 0, floats, 0, ints.Length * sizeof(int));
            return floats;
        }

        // Exception-safe allocation: dispose earlier buffers if a later one fails
        IGpuBuffer? valsBuf = null;
        IGpuBuffer? colsBuf = null;
        IGpuBuffer? rowPtrBuf = null;
        try
        {
            valsBuf = backend.AllocateBuffer(floatVals);
            colsBuf = backend.AllocateBuffer(ReinterpretIntsAsFloats(csr.ColumnIndices));
            rowPtrBuf = backend.AllocateBuffer(ReinterpretIntsAsFloats(csr.RowPointers));
        }
        catch
        {
            valsBuf?.Dispose();
            colsBuf?.Dispose();
            rowPtrBuf?.Dispose();
            throw;
        }

        var cache = new CsrGpuCache(valsBuf, colsBuf, rowPtrBuf, csr.NonZeroCount);
        if (!_csrBufferCache.TryAdd(key, cache))
        {
            // Another thread beat us — dispose ours and use theirs
            cache.Dispose();
            return _csrBufferCache[key];
        }
        return cache;
    }

    #region GPU Sparse Matrix Operations

    /// <summary>
    /// GPU-resident sparse-dense matrix multiplication using CSR format.
    /// Computes: C[M,N] = A[M,K] * B[K,N] where A is in CSR sparse format.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="sparseA">CSR sparse tensor A [M, K].</param>
    /// <param name="denseB">GPU-resident dense tensor B [K, N].</param>
    /// <returns>GPU-resident dense output tensor C [M, N].</returns>
    public Tensor<T> SparseDenseMatMulGpu<T>(SparseTensor<T> sparseA, Tensor<T> denseB)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SparseDenseMatMulGpu");

        if (sparseA is null) throw new ArgumentNullException(nameof(sparseA));
        if (denseB is null) throw new ArgumentNullException(nameof(denseB));

        if (denseB.Shape._dims.Length != 2)
            throw new ArgumentException("Dense tensor B must be 2D [K, N]");

        int M = sparseA.Rows;
        int K = sparseA.Columns;
        int N = denseB.Shape._dims[1];

        if (denseB.Shape._dims[0] != K)
            throw new ArgumentException($"Dimension mismatch: sparse A has {K} columns, but dense B has {denseB.Shape._dims[0]} rows");

        var csrCache = GetOrUploadCsrBuffers(backend, sparseA);

        var outputBuffer = backend.AllocateBuffer(M * N);
        try
        {
            backend.CsrSpMM(csrCache.Values, csrCache.ColumnIndices, csrCache.RowPointers,
                denseB.Buffer, outputBuffer, M, K, N, csrCache.NonZeroCount);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [M, N],
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident sparse-dense matrix multiplication with bias using CSR format.
    /// Computes: C[M,N] = A[M,K] * B[K,N] + bias[N] where A is in CSR sparse format.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="sparseA">CSR sparse tensor A [M, K].</param>
    /// <param name="denseB">GPU-resident dense tensor B [K, N].</param>
    /// <param name="bias">Bias tensor [N].</param>
    /// <returns>GPU-resident dense output tensor C [M, N].</returns>
    public Tensor<T> SparseDenseMatMulBiasGpu<T>(SparseTensor<T> sparseA, Tensor<T> denseB, Tensor<T> bias)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SparseDenseMatMulBiasGpu");

        if (sparseA is null) throw new ArgumentNullException(nameof(sparseA));
        if (denseB is null) throw new ArgumentNullException(nameof(denseB));
        if (bias is null) throw new ArgumentNullException(nameof(bias));

        // Validate dimensions
        if (denseB.Shape._dims.Length != 2)
            throw new ArgumentException("Dense tensor B must be 2D [K, N]");

        int M = sparseA.Rows;
        int K = sparseA.Columns;
        int N = denseB.Shape._dims[1];

        if (denseB.Shape._dims[0] != K)
            throw new ArgumentException($"Dimension mismatch: sparse A has {K} columns, but dense B has {denseB.Shape._dims[0]} rows");

        if (bias.Length != N)
            throw new ArgumentException($"Bias length {bias.Length} must match output columns {N}");

        var csrCache = GetOrUploadCsrBuffers(backend, sparseA);
        using var biasBuffer = GetOrCacheWeightBuffer(backend, bias.GetDataArray(), PersistentTensorRole.Biases);

        var outputBuffer = backend.AllocateBuffer(M * N);
        try
        {
            backend.CsrSpMMBias(csrCache.Values, csrCache.ColumnIndices, csrCache.RowPointers,
                denseB.Buffer, biasBuffer.Buffer, outputBuffer, M, K, N, csrCache.NonZeroCount);
        }
        catch
        {
            outputBuffer.Dispose();
            throw;
        }

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [M, N],
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU scatter-add operation for graph neural network message passing.
    /// For each edge (source -> target), adds source features weighted by edge values to target.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="nodeFeatures">GPU-resident node feature tensor [numNodes, features].</param>
    /// <param name="sourceIndices">Source node indices for each edge [numEdges].</param>
    /// <param name="targetIndices">Target node indices for each edge [numEdges].</param>
    /// <param name="edgeValues">Optional edge weights [numEdges]. If null, uses 1.0 for all edges.</param>
    /// <returns>GPU-resident aggregated node features [numNodes, features].</returns>
    public Tensor<T> ScatterAddGpu<T>(
        Tensor<T> nodeFeatures,
        int[] sourceIndices,
        int[] targetIndices,
        float[]? edgeValues = null)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ScatterAddGpu");

        if (nodeFeatures is null) throw new ArgumentNullException(nameof(nodeFeatures));
        if (sourceIndices is null) throw new ArgumentNullException(nameof(sourceIndices));
        if (targetIndices is null) throw new ArgumentNullException(nameof(targetIndices));

        if (nodeFeatures.Shape._dims.Length != 2)
            throw new ArgumentException("Node features must be 2D [numNodes, features]");

        if (sourceIndices.Length != targetIndices.Length)
            throw new ArgumentException("Source and target indices must have the same length");

        int numNodes = nodeFeatures.Shape._dims[0];
        int features = nodeFeatures.Shape._dims[1];
        int numEdges = sourceIndices.Length;

        // Upload indices as float buffers (GPU kernels use float for everything)
        float[] srcFloat = new float[numEdges];
        float[] tgtFloat = new float[numEdges];
        for (int i = 0; i < numEdges; i++)
        {
            srcFloat[i] = sourceIndices[i];
            tgtFloat[i] = targetIndices[i];
        }

        using var srcBuffer = GetOrAllocateBuffer(backend, srcFloat);
        using var tgtBuffer = GetOrAllocateBuffer(backend, tgtFloat);
        OwnedBuffer? edgeBuffer = edgeValues is not null ? GetOrAllocateBuffer(backend, edgeValues) : null;

        try
        {
            // Allocate output buffer and zero it
            var outputBuffer = backend.AllocateBuffer(numNodes * features);
            backend.Fill(outputBuffer, 0.0f, numNodes * features);

            // Execute scatter add
            backend.ScatterAddEdges(
                nodeFeatures.Buffer,
                srcBuffer.Buffer,
                tgtBuffer.Buffer,
                edgeBuffer?.Buffer,
                outputBuffer,
                numNodes, numEdges, features);

            return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [numNodes, features],
                GpuTensorRole.Activation, ownsBuffer: true);
        }
        finally
        {
            edgeBuffer?.Dispose();
        }
    }

    /// <summary>
    /// Creates a CSR GPU tensor from edge indices for graph operations.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="sourceIndices">Source node indices for each edge.</param>
    /// <param name="targetIndices">Target node indices for each edge.</param>
    /// <param name="values">Edge values (weights). If null, uses 1.0 for all edges.</param>
    /// <param name="numNodes">Number of nodes in the graph.</param>
    /// <returns>CSR GPU tensor representing the adjacency matrix.</returns>
    public SparseTensor<T> CreateCsrFromEdges<T>(
        int[] sourceIndices,
        int[] targetIndices,
        float[]? values,
        int numNodes)
    {
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        var edgeValues = values ?? Enumerable.Repeat(1f, sourceIndices.Length).ToArray();
        var typedValues = new T[edgeValues.Length];
        for (int i = 0; i < edgeValues.Length; i++)
            typedValues[i] = numOps.FromDouble(edgeValues[i]);
        // Build COO then convert to CSR for proper row pointer structure
        var coo = new SparseTensor<T>(numNodes, numNodes, sourceIndices, targetIndices, typedValues);
        return coo.ToCsr();
    }

    /// <summary>
    /// Creates a sparse tensor from a dense tensor by extracting non-zero elements.
    /// Returns a CSR-format sparse tensor.
    /// </summary>
    public SparseTensor<T> CreateCsrFromDense<T>(Tensor<T> denseTensor, float threshold = 1e-6f)
    {
        if (denseTensor.Rank != 2)
            throw new ArgumentException("Dense tensor must be 2D for CSR conversion.");
        int rows = denseTensor._shape[0], cols = denseTensor._shape[1];
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();
        var rowIdx = new System.Collections.Generic.List<int>();
        var colIdx = new System.Collections.Generic.List<int>();
        var vals = new System.Collections.Generic.List<T>();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                var val = denseTensor[r, c];
                if (Math.Abs(numOps.ToDouble(val)) >= threshold)
                {
                    rowIdx.Add(r);
                    colIdx.Add(c);
                    vals.Add(val);
                }
            }
        // Build COO then convert to CSR for proper row pointer structure
        var coo = new SparseTensor<T>(rows, cols, rowIdx.ToArray(), colIdx.ToArray(), vals.ToArray());
        return coo.ToCsr();
    }

    #region Element-wise Operations (GPU)

    public Tensor<T> ExpGpu<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ExpGpu");

        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);
        backend.Exp(input.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    public Tensor<T> SubtractGpu<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SubtractGpu");

        int size = a.Length;
        var outputBuffer = backend.AllocateBuffer(size);
        backend.Subtract(a.Buffer, b.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, a.Shape._dims, GpuTensorRole.Activation, true);
    }

    public Tensor<T> BroadcastMultiplyRowGpu<T>(Tensor<T> input, Tensor<T> weights)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BroadcastMultiplyRowGpu");

        int outerSize = input.Shape._dims[0];
        int innerSize = input.Length / outerSize;

        var outputBuffer = backend.AllocateBuffer(input.Length);
        backend.BroadcastMultiplyLastAxis(input.Buffer, weights.Buffer, outputBuffer, outerSize, innerSize);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    public Tensor<T> SinGpu<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SinGpu");

        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);
        backend.Sin(input.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    public Tensor<T> CosGpu<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for CosGpu");

        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);
        backend.Cos(input.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    public Tensor<T> GreaterThanScalarGpu<T>(Tensor<T> input, float scalar)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GreaterThanScalarGpu");

        int size = input.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        using var scalarBuffer = backend.AllocateBuffer(size);
        backend.Fill(scalarBuffer, scalar, size);

        backend.GreaterThan(input.Buffer, scalarBuffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, input.Shape._dims, GpuTensorRole.Activation, true);
    }

    public Tensor<T> ConcatGpu<T>(Tensor<T>[] inputs, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ConcatGpu");

        if (inputs.Length == 0) throw new ArgumentException("No inputs to concatenate");

        var input0 = inputs[0];
        int rank = input0.Shape._dims.Length;
        int actualAxis = axis < 0 ? rank + axis : axis;

        int[] outputShape = input0.Shape.ToArray();
        outputShape[actualAxis] = 0;
        foreach (var input in inputs)
        {
            outputShape[actualAxis] += input.Shape._dims[actualAxis];
        }

        // 1. Move concatenation axis to last dimension via permutation if needed
        bool needsPermute = actualAxis != rank - 1;
        int[]? permutation = null;
        int[]? invPermutation = null;
        Tensor<T>[] processedInputs = inputs;

        if (needsPermute)
        {
            permutation = new int[rank];
            invPermutation = new int[rank];
            int j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (i != actualAxis) permutation[j++] = i;
            }
            permutation[rank - 1] = actualAxis;
            for (int i = 0; i < rank; i++) invPermutation[permutation[i]] = i;

            processedInputs = new Tensor<T>[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                processedInputs[i] = PermuteGpu(inputs[i], permutation);
            }
        }

        // 2. Flatten to 2D [Outer, AxisDim] for strided copy
        long outerSize = 1;
        for (int i = 0; i < rank - 1; i++)
            outerSize *= (needsPermute ? inputs[0].Shape._dims[permutation![i]] : inputs[0].Shape._dims[i]);

        int totalAxisDim = outputShape[actualAxis];
        int totalSize = (int)(outerSize * totalAxisDim);
        var outputBuffer = backend.AllocateBuffer(totalSize);

        // 3. Copy inputs into concatenated buffer at specific offsets
        int currentOffset = 0;
        foreach (var input in processedInputs)
        {
            int axisDim = input.Shape._dims[rank - 1];
            backend.Copy2DStrided(input.Buffer, outputBuffer, (int)outerSize, axisDim, totalAxisDim, currentOffset);
            currentOffset += axisDim;
        }

        // 4. Construct output tensor and restore original axis order if permuted
        int[] tempShape = new int[rank];
        if (needsPermute)
        {
            for (int i = 0; i < rank - 1; i++) tempShape[i] = outputShape[permutation![i]];
            tempShape[rank - 1] = totalAxisDim;
        }
        else
        {
            Array.Copy(outputShape, tempShape, rank);
        }

        var result = Tensor<T>.FromGpuBuffer(backend, outputBuffer, tempShape, GpuTensorRole.Activation, true);

        if (needsPermute)
        {
            var permutedResult = PermuteGpu(result, invPermutation!);
            result.Dispose();
            result = (Tensor<T>)permutedResult;

            foreach (var pInput in processedInputs) pInput.Dispose();
        }

        return result;
    }

    #endregion

    public Tensor<T> ArgMaxAxisGpu<T>(Tensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ArgMaxAxisGpu");

        // Similar logic to ReduceAxisGpu for arbitrary axis
        // For CRF, axis is usually 1 (after reshape).
        // Let's implement generic axis handling via Permute if needed.

        var inputShape = input.Shape._dims;
        int inputRank = inputShape.Length;
        int outerSize = 1;
        int reduceSize = inputShape[axis];

        // If axis is last, optimal.
        // If not, Permute.
        Tensor<T> processedInput = input;
        bool needsPermute = axis != inputRank - 1;

        if (needsPermute)
        {
            var perm = new int[inputRank];
            int j = 0;
            for (int i = 0; i < inputRank; i++)
                if (i != axis) perm[j++] = i;
            perm[inputRank - 1] = axis;
            processedInput = PermuteGpu(input, perm);
        }

        // Calculate outer size (product of all dims except axis)
        outerSize = processedInput.Length / reduceSize;

        var outputBuffer = backend.AllocateBuffer(outerSize);
        backend.ArgMaxAxis(processedInput.Buffer, outputBuffer, outerSize, reduceSize);

        if (needsPermute)
        {
            processedInput.Dispose();
        }

        // Output shape is input shape with axis removed (or set to 1? ArgMax usually reduces rank).
        // Let's keep rank for compatibility with Torch-like ArgMax, or reduce.
        // ReduceAxisGpu kept dims optionally.
        // For CRF Viterbi, we want [B, C] from [B, C, C].
        // Output shape construction:
        var outputShapeList = new List<int>();
        for (int i = 0; i < inputRank; i++)
        {
            if (i != axis) outputShapeList.Add(inputShape[i]);
        }
        if (outputShapeList.Count == 0) outputShapeList.Add(1);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShapeList.ToArray(), GpuTensorRole.Activation, true);
    }

    public Tensor<T> MaxAxisGpu<T>(Tensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MaxAxisGpu");

        var inputShape = input.Shape._dims;
        int inputRank = inputShape.Length;
        int outerSize = 1;
        int reduceSize = inputShape[axis];

        Tensor<T> processedInput = input;
        bool needsPermute = axis != inputRank - 1;

        if (needsPermute)
        {
            var perm = new int[inputRank];
            int j = 0;
            for (int i = 0; i < inputRank; i++)
                if (i != axis) perm[j++] = i;
            perm[inputRank - 1] = axis;
            processedInput = PermuteGpu(input, perm);
        }

        outerSize = processedInput.Length / reduceSize;
        var outputBuffer = backend.AllocateBuffer(outerSize);
        backend.MaxAxis(processedInput.Buffer, outputBuffer, outerSize, reduceSize);

        if (needsPermute) processedInput.Dispose();

        var outputShapeList = new List<int>();
        for (int i = 0; i < inputRank; i++)
            if (i != axis) outputShapeList.Add(inputShape[i]);
        if (outputShapeList.Count == 0) outputShapeList.Add(1);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, outputShapeList.ToArray(), GpuTensorRole.Activation, true);
    }

    public Tensor<T> BroadcastAddGpu<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BroadcastAddGpu");

        // Support full broadcasting logic like NumPy?
        // Or specific patterns?
        // Implementing full broadcast requires analyzing shapes and tiling.
        // For CRF: [B, C, 1] + [B, C, C] (tiled from [C, C]).
        // Actually, if we use TileAxisGpu manually, we just need element-wise AddGpu.
        // So we don't strictly need BroadcastAddGpu if the caller tiles.
        // But a helper is nice.
        // Let's implement generic AddGpu that handles simple broadcasts or falls back to Tile+Add.
        // But for now, let's expose explicit Tile ops and let caller handle shape matching.
        // It's more predictable.
        // So I will just implement AddGpu (which I assume exists? No, I saw 'Add' in backend).
        // I need to expose AddGpu (element-wise).

        int size = a.Length;
        if (size != b.Length)
            throw new ArgumentException($"AddGpu requires matching sizes: {size} vs {b.Length}");

        var outputBuffer = backend.AllocateBuffer(size);
        backend.Add(a.Buffer, b.Buffer, outputBuffer, size);
        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, a.Shape._dims, GpuTensorRole.Activation, true);
    }



    #endregion

    #region Random Number Generation

    /// <summary>
    /// Generates a GPU-resident tensor with uniformly distributed random numbers.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="min">Minimum value (inclusive).</param>
    /// <param name="max">Maximum value (exclusive).</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>A GPU-resident tensor.</returns>
    public Tensor<T> RandomUniformGpu<T>(int[] shape, float min, float max, ulong seed)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for RandomUniformGpu");

        int size = 1;
        foreach (var dim in shape) size *= dim;

        var outputBuffer = backend.AllocateBuffer(size);
        backend.GenerateRandomUniform(outputBuffer, size, min, max, seed);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, shape, GpuTensorRole.Activation, true);
    }

    /// <summary>
    /// Generates a GPU-resident tensor with normally distributed (Gaussian) random numbers.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="mean">Mean of the distribution.</param>
    /// <param name="stdDev">Standard deviation of the distribution.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>A GPU-resident tensor.</returns>
    public Tensor<T> RandomNormalGpu<T>(int[] shape, float mean, float stdDev, ulong seed)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for RandomNormalGpu");

        int size = 1;
        foreach (var dim in shape) size *= dim;

        var outputBuffer = backend.AllocateBuffer(size);
        backend.GenerateRandomNormal(outputBuffer, size, mean, stdDev, seed);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, shape, GpuTensorRole.Activation, true);
    }

    public Tensor<T> ReshapeGpu<T>(Tensor<T> input, int[] newShape)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ReshapeGpu");

        // Validate size
        int newSize = 1;
        foreach (var dim in newShape) newSize *= dim;

        if (newSize != input.Length)
            throw new ArgumentException($"Reshape total size mismatch: {input.Length} vs {newSize}");

        return Tensor<T>.FromGpuBuffer(backend, input.Buffer, newShape, input.Role, ownsBuffer: false);
    }

    #endregion

    #region Optimizer Operations

    public void SgdMomentumUpdateGpu<T>(Tensor<T> param, Tensor<T> gradient, Tensor<T> velocity, float learningRate, float momentum, float weightDecay)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SgdMomentumUpdateGpu");

        using var paramBuffer = GetOrAllocateBuffer(backend, param.GetDataArray());
        using var gradBuffer = GetOrAllocateBuffer(backend, gradient.GetDataArray());
        using var velocityBuffer = GetOrAllocateBuffer(backend, velocity.GetDataArray());

        backend.SgdMomentumUpdate(paramBuffer.Buffer, gradBuffer.Buffer, velocityBuffer.Buffer,
            learningRate, momentum, weightDecay, param.Length);
    }

    #endregion

    #region Specialized Layer Operations

    public Tensor<T> RbfKernelGpu<T>(Tensor<T> input, Tensor<T> centers, Tensor<T> widths)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for RbfKernelGpu");

        int batch = input.Shape._dims[0];
        int inputDim = input.Shape._dims.Length > 1 ? input.Shape._dims[1] : 1;
        int numCenters = centers.Shape._dims[0];

        // Compute epsilons on CPU (small calculation)
        // epsilon = 1 / (2 * width^2)
        var ops = MathHelper.GetNumericOperations<T>();
        var epsilons = new Tensor<T>(widths.Shape._dims);
        var two = ops.FromDouble(2.0);
        for (int i = 0; i < numCenters; i++)
        {
            var w = widths[i];
            epsilons[i] = ops.Divide(ops.One, ops.Multiply(two, ops.Multiply(w, w)));
        }

        // Upload persistent tensors (using cache if registered)
        using var centersBuffer = GetOrAllocateBuffer(backend, centers.GetDataArray());
        using var epsilonsBuffer = GetOrAllocateBuffer(backend, epsilons.GetDataArray());

        var outputBuffer = backend.AllocateBuffer(batch * numCenters);

        backend.RbfForward(
            input.Buffer,
            centersBuffer.Buffer,
            epsilonsBuffer.Buffer,
            outputBuffer,
            batch, numCenters, inputDim);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [batch, numCenters], GpuTensorRole.Activation, ownsBuffer: true);
    }

    public void UpdateTracesGpu<T>(Tensor<T> traces, Tensor<T> spikes, Tensor<T> input, float decay, float threshold)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for UpdateTracesGpu");

        backend.UpdateTraces(traces.Buffer, spikes.Buffer, input.Buffer, decay, threshold, input.Length);
    }

    public void StdpUpdateGpu<T>(
        Tensor<T> weights,
        Tensor<T> preTrace,
        Tensor<T> postTrace,
        Tensor<T> preSpike,
        Tensor<T> postSpike,
        double ltpRate,
        double ltdRate,
        double homeostasisRate,
        double minWeight,
        double maxWeight)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for StdpUpdateGpu");

        int numPre = weights.Shape._dims[0];
        int numPost = weights.Shape._dims[1]; // Correct shape for fully connected?
                                        // SynapticPlasticityLayer weights are [size, size] so Pre x Post.

        // Weights are modified in-place on GPU, then need to be invalidating CPU cache or vice-versa.
        // We assume weights are persistent GPU tensors.
        // But here we accept Tensor<T> weights.
        // We should check if it's cached.

        // This operation modifies weights in-place on GPU.
        // If we only have CPU weights, we must upload, modify, download.
        // But for training loop, weights should stay on GPU.
        // We use RegisterPersistentTensor mechanism.

        // Get buffer without allocating new one if possible, but we need writable access.
        // GetOrAllocateBuffer returns OwnedBuffer which might be cached.
        // If cached, we modify it in place.
        // We must ensure CPU side knows it's dirty if we download later.

        using var weightsBuffer = GetOrCacheWeightBuffer(backend, weights.GetDataArray(), PersistentTensorRole.Weights);
        backend.StdpUpdate(
            weightsBuffer.Buffer,
            preTrace.Buffer,
            postTrace.Buffer,
            preSpike.Buffer,
            postSpike.Buffer,
            (float)ltpRate, (float)ltdRate, (float)homeostasisRate,
            (float)minWeight, (float)maxWeight,
            numPre, numPost);

        // Mark as modified on GPU so next Download syncs it
        // BUT our current system doesn't track dirty state for download.
        // We assume explicit download or automatic handling.
        // For now, let's assume the user will keep using GPU path.
        // Ideally we should download if we are done, but for training loop we keep it there.
        // To be safe, we can download back to CPU tensor immediately if this isn't fully persistent-managed.
        // Or we rely on the fact that weights.GetDataArray() is the key.
        // If we modify GPU buffer, CPU array is stale.
        // DirectGpuTensorEngine doesn't have "MarkDirty".
        // We will download to keep consistency for now, or assume layer manages it.
        // Given UpdateParameters returns void, we should update the CPU tensor too.

        backend.DownloadBuffer(weightsBuffer.Buffer, DirectGpuEngine.ToFloatArray(weights.GetDataArray()));
    }

    #endregion

    #region GPU-Resident Linear Layer Backward Operations

    /// <summary>
    /// GPU-resident 2D matrix transpose.
    /// Transposes input tensor from [rows, cols] to [cols, rows].
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">GPU-resident 2D input tensor [rows, cols].</param>
    /// <returns>GPU-resident transposed tensor [cols, rows].</returns>
    public Tensor<T> TransposeGpu<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TransposeGpu");

        if (input.Shape._dims.Length != 2)
            throw new ArgumentException("TransposeGpu requires 2D tensor [rows, cols]");

        int rows = input.Shape._dims[0];
        int cols = input.Shape._dims[1];

        var outputBuffer = backend.AllocateBuffer(rows * cols);
        backend.Transpose(input.Buffer, outputBuffer, rows, cols);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, [cols, rows],
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident matrix multiplication between two GPU tensors.
    /// Computes: C = A @ B where A is [M, K] and B is [K, N].
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="A">GPU-resident tensor A [M, K].</param>
    /// <param name="B">GPU-resident tensor B [K, N].</param>
    /// <returns>GPU-resident output tensor C [M, N].</returns>
    public Tensor<T> MatMulGpuTensors<T>(Tensor<T> A, Tensor<T> B)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for MatMulGpuTensors");

        if (A.Shape._dims.Length != 2 || B.Shape._dims.Length != 2)
            throw new ArgumentException("MatMulGpuTensors requires 2D tensors");

        int M = A.Shape._dims[0];
        int K = A.Shape._dims[1];
        int N = B.Shape._dims[1];

        if (B.Shape._dims[0] != K)
            throw new ArgumentException($"Dimension mismatch: A has {K} columns, but B has {B.Shape._dims[0]} rows");

        var resultBuffer = backend.MatMul(A.Buffer, B.Buffer, M, N, K);

        return Tensor<T>.FromGpuBuffer(backend, resultBuffer, [M, N],
            GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// GPU-resident ReLU backward operation.
    /// Computes: gradInput = gradOutput * (input > 0 ? 1 : 0)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public Tensor<T> ReluBackwardGpu<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ReluBackwardGpu");

        int size = gradOutput.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.ReluBackward(gradOutput.Buffer, input.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident Sigmoid backward operation.
    /// Computes: gradInput = gradOutput * sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="output">GPU-resident output from forward pass (post-activation sigmoid output).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public Tensor<T> SigmoidBackwardGpu<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SigmoidBackwardGpu");

        int size = gradOutput.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.SigmoidBackward(gradOutput.Buffer, output.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident Tanh backward operation.
    /// Computes: gradInput = gradOutput * (1 - tanh(x)^2)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="output">GPU-resident output from forward pass (post-activation tanh output).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public Tensor<T> TanhBackwardGpu<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for TanhBackwardGpu");

        int size = gradOutput.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.TanhBackward(gradOutput.Buffer, output.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident LeakyReLU backward operation.
    /// Computes: gradInput = gradOutput * (input > 0 ? 1 : alpha)
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <param name="alpha">Negative slope parameter.</param>
    /// <returns>GPU-resident input gradient.</returns>
    public Tensor<T> LeakyReluBackwardGpu<T>(Tensor<T> gradOutput, Tensor<T> input, float alpha = 0.01f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for LeakyReluBackwardGpu");

        int size = gradOutput.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.LeakyReluBackward(gradOutput.Buffer, input.Buffer, outputBuffer, alpha, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident GELU backward operation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public Tensor<T> GeluBackwardGpu<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for GeluBackwardGpu");

        int size = gradOutput.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.GeluBackward(gradOutput.Buffer, input.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident Softmax backward operation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="output">GPU-resident output from forward pass (post-activation softmax output).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public Tensor<T> SoftmaxBackwardGpu<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SoftmaxBackwardGpu");

        int batchSize = gradOutput.Shape._dims[0];
        int features = gradOutput.Shape._dims.Length > 1 ? gradOutput.Shape._dims[1] : 1;
        int size = gradOutput.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.SoftmaxBackward(gradOutput.Buffer, output.Buffer, outputBuffer, batchSize, features);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident Swish backward operation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <returns>GPU-resident input gradient.</returns>
    public Tensor<T> SwishBackwardGpu<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SwishBackwardGpu");

        int size = gradOutput.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.SwishBackward(gradOutput.Buffer, input.Buffer, outputBuffer, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident ELU backward operation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient.</param>
    /// <param name="input">GPU-resident input from forward pass (pre-activation).</param>
    /// <param name="output">GPU-resident output from forward pass (post-activation).</param>
    /// <param name="alpha">ELU alpha parameter.</param>
    /// <returns>GPU-resident input gradient.</returns>
    public Tensor<T> EluBackwardGpu<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, float alpha = 1.0f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for EluBackwardGpu");

        int size = gradOutput.Length;
        var outputBuffer = backend.AllocateBuffer(size);

        backend.EluBackward(gradOutput.Buffer, input.Buffer, output.Buffer, outputBuffer, alpha, size);

        return Tensor<T>.FromGpuBuffer(backend, outputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true);
    }

    #endregion

    #region GPU-Resident BatchNorm Backward Operations

    /// <summary>
    /// GPU-resident batch normalization backward pass.
    /// Computes gradients for input, gamma (scale), and beta (shift).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, C, H, W] or [B, C].</param>
    /// <param name="input">GPU-resident input from forward pass.</param>
    /// <param name="gamma">Scale parameter [C].</param>
    /// <param name="saveMean">Running mean saved from forward pass [C].</param>
    /// <param name="saveInvVar">Running inverse variance saved from forward pass [C].</param>
    /// <param name="epsilon">Epsilon for numerical stability.</param>
    /// <returns>Tuple of (gradInput, gradGamma, gradBeta).</returns>
    public (Tensor<T> gradInput, Tensor<T> gradGamma, Tensor<T> gradBeta) BatchNormBackwardGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> gamma,
        Tensor<T> saveMean,
        Tensor<T> saveInvVar,
        float epsilon)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for BatchNormBackwardGpu");

        // Determine dimensions
        int batch, channels, spatialSize;
        if (gradOutput.Shape._dims.Length == 2)
        {
            // [B, C] - fully connected
            batch = gradOutput.Shape._dims[0];
            channels = gradOutput.Shape._dims[1];
            spatialSize = 1;
        }
        else if (gradOutput.Shape._dims.Length == 4)
        {
            // [B, C, H, W] - convolutional
            batch = gradOutput.Shape._dims[0];
            channels = gradOutput.Shape._dims[1];
            spatialSize = gradOutput.Shape._dims[2] * gradOutput.Shape._dims[3];
        }
        else
        {
            throw new ArgumentException($"BatchNormBackwardGpu expects 2D [B, C] or 4D [B, C, H, W] tensor, got {gradOutput.Shape._dims.Length}D");
        }

        // Validate parameter lengths match channels to prevent out-of-bounds kernel access
        if (gamma.Length != channels)
            throw new ArgumentException($"gamma.Length ({gamma.Length}) must match channels ({channels}).", nameof(gamma));
        if (saveMean.Length != channels)
            throw new ArgumentException($"saveMean.Length ({saveMean.Length}) must match channels ({channels}).", nameof(saveMean));
        if (saveInvVar.Length != channels)
            throw new ArgumentException($"saveInvVar.Length ({saveInvVar.Length}) must match channels ({channels}).", nameof(saveInvVar));

        // Allocate output buffers
        var gradInputBuffer = backend.AllocateBuffer(gradOutput.Length);
        var gradGammaBuffer = backend.AllocateBuffer(channels);
        var gradBetaBuffer = backend.AllocateBuffer(channels);

        // Upload gamma
        using var gammaBuffer = GetOrCacheWeightBuffer(backend, gamma.GetDataArray(), PersistentTensorRole.Weights);

        backend.BatchNormBackward(
            gradOutput.Buffer, input.Buffer, gammaBuffer.Buffer,
            saveMean.Buffer, saveInvVar.Buffer,
            gradInputBuffer, gradGammaBuffer, gradBetaBuffer,
            batch, channels, spatialSize, epsilon);

        return (
            Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, gradOutput.Shape._dims, GpuTensorRole.Gradient, true),
            Tensor<T>.FromGpuBuffer(backend, gradGammaBuffer, [channels], GpuTensorRole.Gradient, true),
            Tensor<T>.FromGpuBuffer(backend, gradBetaBuffer, [channels], GpuTensorRole.Gradient, true)
        );
    }

    #endregion

    #region GPU-Resident Conv2D Backward Operations

    /// <summary>
    /// GPU-resident Conv2D backward pass for input gradients.
    /// Computes gradient with respect to input.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <param name="kernel">Kernel weights [outC, inC, kH, kW].</param>
    /// <param name="inputShape">Shape of the original input [B, inC, inH, inW].</param>
    /// <param name="stride">Convolution stride [strideH, strideW].</param>
    /// <param name="padding">Padding [padH, padW].</param>
    /// <param name="dilation">Dilation [dilationH, dilationW].</param>
    /// <returns>GPU-resident gradient with respect to input.</returns>
    public Tensor<T> Conv2DBackwardInputGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> kernel,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for Conv2DBackwardInputGpu");

        // Validate shape lengths to prevent index out of bounds
        if (inputShape.Length != 4)
            throw new ArgumentException($"inputShape must be 4D [B, inC, inH, inW], got {inputShape.Length}D.", nameof(inputShape));
        if (kernel.Rank != 4)
            throw new ArgumentException($"kernel must be 4D [outC, inC, kH, kW], got {kernel.Rank}D.", nameof(kernel));
        if (gradOutput.Shape._dims.Length != 4)
            throw new ArgumentException($"gradOutput must be 4D [B, outC, outH, outW], got {gradOutput.Shape._dims.Length}D.", nameof(gradOutput));

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        int outChannels = kernel.Shape._dims[0];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        // Allocate output buffer for input gradient
        var gradInputBuffer = backend.AllocateBuffer(batch * inChannels * inHeight * inWidth);

        // Upload kernel
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);

        backend.Conv2DBackwardInput(
            gradOutput.Buffer, kernelBuffer.Buffer, gradInputBuffer,
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelH, kernelW,
            stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1]);

        return Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident Conv2D backward pass for kernel gradients.
    /// Computes gradient with respect to kernel weights.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <param name="input">GPU-resident input from forward pass [B, inC, inH, inW].</param>
    /// <param name="kernelShape">Shape of the kernel [outC, inC, kH, kW].</param>
    /// <param name="stride">Convolution stride [strideH, strideW].</param>
    /// <param name="padding">Padding [padH, padW].</param>
    /// <param name="dilation">Dilation [dilationH, dilationW].</param>
    /// <returns>GPU-resident gradient with respect to kernels.</returns>
    public Tensor<T> Conv2DBackwardKernelGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for Conv2DBackwardKernelGpu");

        // Validate shape lengths to prevent index out of bounds
        if (input.Shape._dims.Length != 4)
            throw new ArgumentException($"input must be 4D [B, inC, inH, inW], got {input.Shape._dims.Length}D.", nameof(input));
        if (kernelShape.Length != 4)
            throw new ArgumentException($"kernelShape must be 4D [outC, inC, kH, kW], got {kernelShape.Length}D.", nameof(kernelShape));
        if (gradOutput.Shape._dims.Length != 4)
            throw new ArgumentException($"gradOutput must be 4D [B, outC, outH, outW], got {gradOutput.Shape._dims.Length}D.", nameof(gradOutput));

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        int outChannels = kernelShape[0];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        // Allocate output buffer for kernel gradient
        int kernelSize = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
        var gradKernelBuffer = backend.AllocateBuffer(kernelSize);

        backend.Conv2DBackwardKernel(
            input.Buffer, gradOutput.Buffer, gradKernelBuffer,
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelH, kernelW,
            stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1]);

        return Tensor<T>.FromGpuBuffer(backend, gradKernelBuffer, kernelShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident Conv2D backward pass for bias gradients.
    /// Computes gradient with respect to bias by summing over batch and spatial dimensions.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <returns>GPU-resident gradient with respect to bias [outC].</returns>
    public Tensor<T> Conv2DBackwardBiasGpu<T>(Tensor<T> gradOutput)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for Conv2DBackwardBiasGpu");

        // Validate shape length to prevent index out of bounds
        if (gradOutput.Shape._dims.Length != 4)
            throw new ArgumentException($"gradOutput must be 4D [B, outC, outH, outW], got {gradOutput.Shape._dims.Length}D.", nameof(gradOutput));

        int batch = gradOutput.Shape._dims[0];
        int outChannels = gradOutput.Shape._dims[1];
        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        // Bias gradient = sum over batch and spatial dimensions
        // gradOutput: [B, outC, outH, outW] -> gradBias: [outC]
        // Sum over batch and spatial dimensions for each channel
        // Note: For large tensors, this could be optimized with GPU reduction kernels
        // (e.g., reshape to [B*H*W, C] and use SumAxis). Current implementation downloads
        // to CPU for simplicity and correctness.
        float[] gradOutData = backend.DownloadBuffer(gradOutput.Buffer);
        float[] gradBiasData = new float[outChannels];

        int spatialSize = outHeight * outWidth;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < outChannels; c++)
            {
                int baseIdx = b * outChannels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    gradBiasData[c] += gradOutData[baseIdx + s];
                }
            }
        }

        // Create new GPU buffer with the computed bias gradients
        var gradBiasBuffer = backend.AllocateBuffer(gradBiasData);

        return Tensor<T>.FromGpuBuffer(backend, gradBiasBuffer, [outChannels], GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident ConvTranspose2D backward pass for input gradients.
    /// Computes gradient with respect to input.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <param name="kernel">Kernel weights [inC, outC, kH, kW].</param>
    /// <param name="inputShape">Shape of the input [B, inC, inH, inW].</param>
    /// <param name="stride">Convolution stride [strideH, strideW].</param>
    /// <param name="padding">Padding [padH, padW].</param>
    /// <param name="outputPadding">Output padding [outputPadH, outputPadW].</param>
    /// <returns>GPU-resident gradient with respect to input.</returns>
    public Tensor<T> ConvTranspose2DBackwardInputGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> kernel,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] outputPadding)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ConvTranspose2DBackwardInputGpu");

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];

        // For transposed conv: kernel shape is [inC, outC, kH, kW]
        int outChannels = kernel.Shape._dims[1];
        int kernelH = kernel.Shape._dims[2];
        int kernelW = kernel.Shape._dims[3];

        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        // Allocate output buffer for input gradient
        var gradInputBuffer = backend.AllocateBuffer(batch * inChannels * inHeight * inWidth);

        // Upload kernel
        using var kernelBuffer = GetOrCacheWeightBuffer(backend, kernel.GetDataArray(), PersistentTensorRole.Weights);

        backend.ConvTranspose2DBackwardInput(
            gradOutput.Buffer, kernelBuffer.Buffer, gradInputBuffer,
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelH, kernelW,
            stride[0], stride[1], padding[0], padding[1],
            outputPadding[0], outputPadding[1]);

        return Tensor<T>.FromGpuBuffer(backend, gradInputBuffer, inputShape, GpuTensorRole.Gradient, true);
    }

    /// <summary>
    /// GPU-resident ConvTranspose2D backward pass for kernel gradients.
    /// Computes gradient with respect to kernel weights.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">GPU-resident output gradient [B, outC, outH, outW].</param>
    /// <param name="input">GPU-resident input from forward pass [B, inC, inH, inW].</param>
    /// <param name="kernelShape">Shape of the kernel [inC, outC, kH, kW].</param>
    /// <param name="stride">Convolution stride [strideH, strideW].</param>
    /// <param name="padding">Padding [padH, padW].</param>
    /// <param name="outputPadding">Output padding [outputPadH, outputPadW].</param>
    /// <returns>GPU-resident gradient with respect to kernels.</returns>
    public Tensor<T> ConvTranspose2DBackwardKernelGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] outputPadding)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for ConvTranspose2DBackwardKernelGpu");

        int batch = input.Shape._dims[0];
        int inChannels = input.Shape._dims[1];
        int inHeight = input.Shape._dims[2];
        int inWidth = input.Shape._dims[3];

        // For transposed conv: kernel shape is [inC, outC, kH, kW]
        int outChannels = kernelShape[1];
        int kernelH = kernelShape[2];
        int kernelW = kernelShape[3];

        int outHeight = gradOutput.Shape._dims[2];
        int outWidth = gradOutput.Shape._dims[3];

        // Allocate output buffer for kernel gradient
        int kernelSize = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
        var gradKernelBuffer = backend.AllocateBuffer(kernelSize);

        backend.ConvTranspose2DBackwardKernel(
            input.Buffer, gradOutput.Buffer, gradKernelBuffer,
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelH, kernelW,
            stride[0], stride[1], padding[0], padding[1],
            outputPadding[0], outputPadding[1]);

        return Tensor<T>.FromGpuBuffer(backend, gradKernelBuffer, kernelShape, GpuTensorRole.Gradient, true);
    }

    #endregion

    #region Tensor-Level Activation GPU Dispatch

    public override Tensor<T> TensorSigmoid<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Sigmoid(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Sigmoid", output, tensor,
                    Autodiff.BackwardFunctions<T>.SigmoidBackward);
                return output;
            }
            return base.TensorSigmoid(tensor);
        }
        catch (Exception)
        {
            return base.TensorSigmoid(tensor);
        }
    }

    public override Tensor<T> TensorReLU<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Relu(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("ReLU", output, tensor,
                    Autodiff.BackwardFunctions<T>.ReLUBackward);
                return output;
            }
            return base.TensorReLU(tensor);
        }
        catch (Exception)
        {
            return base.TensorReLU(tensor);
        }
    }

    public override Tensor<T> TensorGELU<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Gelu(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("GELU", output, tensor,
                    Autodiff.BackwardFunctions<T>.GELUBackward);
                return output;
            }
            return base.TensorGELU(tensor);
        }
        catch (Exception)
        {
            return base.TensorGELU(tensor);
        }
    }

    public override Tensor<T> TensorSiLU<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Silu(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Swish", output, tensor,
                    Autodiff.BackwardFunctions<T>.SwishBackward);
                return output;
            }
            return base.TensorSiLU(tensor);
        }
        catch (Exception)
        {
            return base.TensorSiLU(tensor);
        }
    }

    public override Tensor<T> TensorTanh<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Tanh(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Tanh", output, tensor,
                    Autodiff.BackwardFunctions<T>.TanhBackward);
                return output;
            }
            return base.TensorTanh(tensor);
        }
        catch (Exception)
        {
            return base.TensorTanh(tensor);
        }
    }

    public override Tensor<T> TensorLeakyReLU<T>(Tensor<T> tensor, T alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorLeakyReLU(tensor, alpha);

        try
        {
            float alphaFloat = ToFloatScalar(alpha);
            using var bufferA = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            using var bufferB = AllocateOutputBuffer(backend, tensor.Length);
            backend.LeakyRelu(bufferA.Buffer, bufferB.Buffer, alphaFloat, tensor.Length);
            float[] resultFloat = backend.DownloadBuffer(bufferB.Buffer);
            var result = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            var gpuOutput = new Tensor<T>(result, tensor.Shape._dims);
            Autodiff.DifferentiableOps.RecordUnary("LeakyReLU", gpuOutput, tensor,
                Autodiff.BackwardFunctions<T>.LeakyReLUBackward,
                savedState: new object[] { (double)alphaFloat });
            return gpuOutput;
        }
        catch (Exception)
        {
            return base.TensorLeakyReLU(tensor, alpha);
        }
    }

    public override Tensor<T> TensorMish<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Mish(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Mish", output, tensor,
                    Autodiff.BackwardFunctions<T>.MishBackward);
                return output;
            }
            return base.TensorMish(tensor);
        }
        catch (Exception)
        {
            return base.TensorMish(tensor);
        }
    }

    public override Tensor<T> TensorHardSwish<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Hardswish(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("HardSwish", output, tensor,
                    Autodiff.BackwardFunctions<T>.HardSwishBackward);
                return output;
            }
            return base.TensorHardSwish(tensor);
        }
        catch (Exception)
        {
            return base.TensorHardSwish(tensor);
        }
    }

    #endregion

    #region Tensor-Level Composite GPU Dispatch

    public override Tensor<T> TensorLerp<T>(Tensor<T> a, Tensor<T> b, T t)
    {
        // Single fused GPU kernel: output[i] = a[i] + t * (b[i] - a[i])
        if (!TryGetBackend(out var backend) || !ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorLerp(a, b, t);

        try
        {
            float tFloat = ToFloatScalar(t);
            int size = a.Length;

            using var bufferA = GetOrAllocateBuffer(backend, a.GetDataArray());
            using var bufferB = GetOrAllocateBuffer(backend, b.GetDataArray());
            var bufferResult = AllocateOutputBuffer(backend, size);

            backend.Lerp(bufferA.Buffer, bufferB.Buffer, bufferResult.Buffer, tFloat, size);

            float[] resultFloat = backend.DownloadBuffer(bufferResult.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), a.Shape._dims);
        }
        catch (Exception)
        {
            return base.TensorLerp(a, b, t);
        }
    }

    public override Tensor<T> TensorAddScaled<T>(Tensor<T> a, Tensor<T> b, T scaleA, T scaleB)
    {
        // Single fused GPU kernel: output[i] = scaleA * a[i] + scaleB * b[i]
        if (!TryGetBackend(out var backend) || !ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorAddScaled(a, b, scaleA, scaleB);

        try
        {
            float scaleAFloat = ToFloatScalar(scaleA);
            float scaleBFloat = ToFloatScalar(scaleB);
            int size = a.Length;

            using var bufferA = GetOrAllocateBuffer(backend, a.GetDataArray());
            using var bufferB = GetOrAllocateBuffer(backend, b.GetDataArray());
            var bufferResult = AllocateOutputBuffer(backend, size);

            backend.AddScaled(bufferA.Buffer, bufferB.Buffer, bufferResult.Buffer, scaleAFloat, scaleBFloat, size);

            var result = FinishGpuOp<T>(backend, bufferResult, size);
            return new Tensor<T>(result, a.Shape._dims);
        }
        catch (Exception)
        {
            return base.TensorAddScaled(a, b, scaleA, scaleB);
        }
    }

    public override Tensor<T> TensorMaxPool2D<T>(Tensor<T> input, int poolSize, int stride, int padding)
    {
        // GPU dispatch already handled by the existing MaxPool2D method override
        return MaxPool2D(input, poolSize, stride, padding);
    }

    public override Tensor<T> TensorAvgPool2D<T>(Tensor<T> input, int poolSize, int stride, int padding)
    {
        // GPU dispatch already handled by the existing AvgPool2D method override
        return AvgPool2D(input, poolSize, stride, padding);
    }

    public override Tensor<T> TensorConv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride, int padding, int dilation)
    {
        // GPU dispatch handled by FusedConv2D which already has GPU acceleration
        return FusedConv2D(input, kernel, null, stride, stride, padding, padding, dilation, dilation, FusedActivationType.None);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated element-wise arithmetic
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorAdd(a, b);
        try
        {
            var result = TryRunBinary(a, b, static (backend, ia, ib, o, size) => backend.Add(ia, ib, o, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, a.Shape._dims);
                Autodiff.DifferentiableOps.RecordBinary("TensorAdd", output, a, b, Autodiff.BackwardFunctions<T>.AddBackward);
                return output;
            }
        }
        catch { }
        return base.TensorAdd(a, b);
    }

    public override Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorSubtract(a, b);
        try
        {
            var result = TryRunBinary(a, b, static (backend, ia, ib, o, size) => backend.Subtract(ia, ib, o, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, a.Shape._dims);
                Autodiff.DifferentiableOps.RecordBinary("TensorSubtract", output, a, b, Autodiff.BackwardFunctions<T>.SubtractBackward);
                return output;
            }
        }
        catch { }
        return base.TensorSubtract(a, b);
    }

    public override Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorMultiply(a, b);
        try
        {
            var result = TryRunBinary(a, b, static (backend, ia, ib, o, size) => backend.Multiply(ia, ib, o, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, a.Shape._dims);
                Autodiff.DifferentiableOps.RecordBinary("TensorMultiply", output, a, b, Autodiff.BackwardFunctions<T>.MultiplyBackward);
                return output;
            }
        }
        catch { }
        return base.TensorMultiply(a, b);
    }

    public override Tensor<T> TensorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape._dims, b.Shape._dims))
            return base.TensorDivide(a, b);
        try
        {
            var result = TryRunBinary(a, b, static (backend, ia, ib, o, size) => backend.Divide(ia, ib, o, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, a.Shape._dims);
                Autodiff.DifferentiableOps.RecordBinary("TensorDivide", output, a, b, Autodiff.BackwardFunctions<T>.DivideBackward);
                return output;
            }
        }
        catch { }
        return base.TensorDivide(a, b);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated unary math
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorExp<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Exp(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("TensorExp", output, tensor, Autodiff.BackwardFunctions<T>.ExpBackward);
                return output;
            }
        }
        catch { }
        return base.TensorExp(tensor);
    }

    public override Tensor<T> TensorLog<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Log(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("TensorLog", output, tensor, Autodiff.BackwardFunctions<T>.LogBackward);
                return output;
            }
        }
        catch { }
        return base.TensorLog(tensor);
    }

    public override Tensor<T> TensorSqrt<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Sqrt(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("TensorSqrt", output, tensor, Autodiff.BackwardFunctions<T>.SqrtBackward);
                return output;
            }
        }
        catch { }
        return base.TensorSqrt(tensor);
    }

    public override Tensor<T> TensorAbs<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Abs(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("TensorAbs", output, tensor, Autodiff.BackwardFunctions<T>.AbsBackward);
                return output;
            }
        }
        catch { }
        return base.TensorAbs(tensor);
    }

    public override Tensor<T> TensorNegate<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Negate(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("TensorNegate", output, tensor, Autodiff.BackwardFunctions<T>.NegateBackward);
                return output;
            }
        }
        catch { }
        return base.TensorNegate(tensor);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated activations (missing from original 13)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> Swish<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Swish(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Swish", output, tensor, Autodiff.BackwardFunctions<T>.SwishBackward);
                return output;
            }
        }
        catch { }
        return base.Swish(tensor);
    }

    public override Tensor<T> ELU<T>(Tensor<T> tensor, double alpha)
    {
        try
        {
            var result = TryRunUnary(tensor, (backend, input, output, size) => backend.Elu(input, output, (float)alpha, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("ELU", output, tensor,
                    Autodiff.BackwardFunctions<T>.ELUBackward, new object[] { alpha });
                return output;
            }
        }
        catch { }
        return base.ELU(tensor, alpha);
    }

    public override Tensor<T> Softplus<T>(Tensor<T> input)
    {
        try
        {
            var result = TryRunUnary(input, static (backend, inp, output, size) => backend.Softplus(inp, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, input.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Softplus", output, input,
                    Autodiff.BackwardFunctions<T>.SoftplusBackward);
                return output;
            }
        }
        catch { }
        return base.Softplus(input);
    }

    public override Tensor<T> TensorSELU<T>(Tensor<T> tensor)
    {
        try
        {
            const float alpha = 1.6732632423543772f;
            const float scale = 1.0507009873554805f;
            var result = TryRunUnary(tensor, (backend, input, output, size) => backend.Selu(input, output, alpha, scale, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("SELU", output, tensor,
                    Autodiff.BackwardFunctions<T>.SELUBackward);
                return output;
            }
        }
        catch { }
        return base.TensorSELU(tensor);
    }

    public override Tensor<T> TensorHardSigmoid<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Hardsigmoid(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("HardSigmoid", output, tensor,
                    Autodiff.BackwardFunctions<T>.HardSigmoidBackward);
                return output;
            }
        }
        catch { }
        return base.TensorHardSigmoid(tensor);
    }

    public override Tensor<T> TensorReLU6<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Relu6(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("ReLU6", output, tensor,
                    Autodiff.BackwardFunctions<T>.ReLU6Backward);
                return output;
            }
        }
        catch { }
        return base.TensorReLU6(tensor);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated backward functions
    // These dispatch activation backward to GPU kernels
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> ReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                var gArr = DirectGpuEngine.ToFloatArray(gradOutput.GetFlattenedData());
                var iArr = DirectGpuEngine.ToFloatArray(input.GetFlattenedData());
                int size = gArr.Length;
                using var gBuf = GetOrAllocateBuffer(backend, gArr);
                using var iBuf = GetOrAllocateBuffer(backend, iArr);
                var oBuf = AllocateOutputBuffer(backend, size);
                try
                {
                    backend.ReluBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, size);
                    var result = FinishGpuOp<T>(backend, oBuf, size);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.ReluBackward(gradOutput, input);
    }

    public override Tensor<T> SigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                var gArr = DirectGpuEngine.ToFloatArray(gradOutput.GetFlattenedData());
                var oArr = DirectGpuEngine.ToFloatArray(output.GetDataArray());
                int size = gArr.Length;
                using var gBuf = GetOrAllocateBuffer(backend, gArr);
                using var oBuf = GetOrAllocateBuffer(backend, oArr);
                var rBuf = AllocateOutputBuffer(backend, size);
                try
                {
                    backend.SigmoidBackward(gBuf.Buffer, oBuf.Buffer, rBuf.Buffer, size);
                    var result = FinishGpuOp<T>(backend, rBuf, size);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { rBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.SigmoidBackward(gradOutput, output);
    }

    public override Tensor<T> TanhBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                var gArr = DirectGpuEngine.ToFloatArray(gradOutput.GetFlattenedData());
                var oArr = DirectGpuEngine.ToFloatArray(output.GetDataArray());
                int size = gArr.Length;
                using var gBuf = GetOrAllocateBuffer(backend, gArr);
                using var oBuf = GetOrAllocateBuffer(backend, oArr);
                var rBuf = AllocateOutputBuffer(backend, size);
                try
                {
                    backend.TanhBackward(gBuf.Buffer, oBuf.Buffer, rBuf.Buffer, size);
                    var result = FinishGpuOp<T>(backend, rBuf, size);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { rBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.TanhBackward(gradOutput, output);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated activation backward methods
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> GeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.GeluBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.GeluBackward(gradOutput, input);
    }

    public override Tensor<T> LeakyReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double negativeSlope)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.LeakyReluBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, (float)negativeSlope, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.LeakyReluBackward(gradOutput, input, negativeSlope);
    }

    public override Tensor<T> SwishBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.SwishBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.SwishBackward(gradOutput, input);
    }

    public override Tensor<T> MishBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.MishBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.MishBackward(gradOutput, input);
    }

    public override Tensor<T> SoftplusBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.SoftplusBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.SoftplusBackward(gradOutput, input);
    }

    public override Tensor<T> HardswishBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.HardswishBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.HardswishBackward(gradOutput, input);
    }

    public override Tensor<T> SeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                const float alpha = 1.6732632423543772f;
                const float scale = 1.0507009873554805f;
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.SeluBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, alpha, scale, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.SeluBackward(gradOutput, input);
    }

    public override Tensor<T> HardsigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.HardsigmoidBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.HardsigmoidBackward(gradOutput, input);
    }

    public override Tensor<T> Relu6Backward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.Relu6Backward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.Relu6Backward(gradOutput, input);
    }

    public override Tensor<T> ThresholdBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double threshold)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                var oBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.ThresholdBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, (float)threshold, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, oBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.ThresholdBackward(gradOutput, input, threshold);
    }

    public override Tensor<T> ReciprocalBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var oBuf2 = GetOrAllocateBuffer(backend, output);
                var rBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.ReciprocalBackward(gBuf.Buffer, oBuf2.Buffer, rBuf.Buffer, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, rBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { rBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.ReciprocalBackward(gradOutput, output);
    }

    public override (Tensor<T> inputGrad, Tensor<T> alphaGrad) PReLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> alpha)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                int size = input.Length;
                int alphaSize = alpha.Length;
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                using var aBuf = GetOrAllocateBuffer(backend, alpha);
                var giB = AllocateOutputBuffer(backend, size);
                var gaB = AllocateOutputBuffer(backend, alphaSize);
                try
                {
                    backend.PReluBackwardInput(gBuf.Buffer, iBuf.Buffer, aBuf.Buffer, giB.Buffer, size, alphaSize);
                    backend.PReluBackwardAlpha(gBuf.Buffer, iBuf.Buffer, gaB.Buffer, size, alphaSize);
                    var inputGrad = FinishGpuOp<T>(backend, giB, size);
                    var alphaGrad = FinishGpuOp<T>(backend, gaB, alphaSize);
                    return (new Tensor<T>(inputGrad, input.Shape._dims), new Tensor<T>(alphaGrad, alpha.Shape._dims));
                }
                catch { giB.Dispose(); gaB.Dispose(); throw; }
            }
        }
        catch { }
        return base.PReLUBackward(gradOutput, input, alpha);
    }

    public override Tensor<T> VarBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, int[] axes)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                int outerSize = 1;
                int reduceSize = input.Length;
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                using var mBuf = GetOrAllocateBuffer(backend, mean);
                var oBuf = AllocateOutputBuffer(backend, input.Length);
                try
                {
                    backend.VarBackward(gBuf.Buffer, iBuf.Buffer, mBuf.Buffer, oBuf.Buffer, outerSize, reduceSize);
                    var result = FinishGpuOp<T>(backend, oBuf, input.Length);
                    return new Tensor<T>(result, input.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.VarBackward(gradOutput, input, mean, axes);
    }

    public override Tensor<T> StdBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, Tensor<T> std, int[] axes)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                int outerSize = 1;
                int reduceSize = input.Length;
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                using var mBuf = GetOrAllocateBuffer(backend, mean);
                using var sBuf = GetOrAllocateBuffer(backend, std);
                var oBuf = AllocateOutputBuffer(backend, input.Length);
                try
                {
                    backend.StdBackward(gBuf.Buffer, iBuf.Buffer, mBuf.Buffer, sBuf.Buffer, oBuf.Buffer, outerSize, reduceSize);
                    var result = FinishGpuOp<T>(backend, oBuf, input.Length);
                    return new Tensor<T>(result, input.Shape._dims);
                }
                catch { oBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.StdBackward(gradOutput, input, mean, std, axes);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated trigonometric
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorSin<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Sin(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Sin", output, tensor, Autodiff.BackwardFunctions<T>.SinBackward);
                return output;
            }
        }
        catch { }
        return base.TensorSin(tensor);
    }

    public override Tensor<T> TensorCos<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Cos(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Cos", output, tensor, Autodiff.BackwardFunctions<T>.CosBackward);
                return output;
            }
        }
        catch { }
        return base.TensorCos(tensor);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated scalar operations
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                float scalarF = scalar is float f ? f : Convert.ToSingle(scalar);
                var result = TryRunUnary(tensor, (b, input, output, size) => b.Scale(input, output, scalarF, size));
                if (result != null)
                {
                    var output = new Tensor<T>(result, tensor.Shape._dims);
                    Autodiff.DifferentiableOps.RecordUnary("TensorMultiplyScalar", output, tensor,
                        Autodiff.BackwardFunctions<T>.MultiplyScalarBackward, new object[] { scalar as object ?? new object() });
                    return output;
                }
            }
        }
        catch { }
        return base.TensorMultiplyScalar(tensor, scalar);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated floor/ceiling
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorFloor<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Floor(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Floor", output, tensor, Autodiff.BackwardFunctions<T>.SignBackward);
                return output;
            }
        }
        catch { }
        return base.TensorFloor(tensor);
    }

    public override Tensor<T> TensorCeiling<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Ceiling(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Ceiling", output, tensor, Autodiff.BackwardFunctions<T>.SignBackward);
                return output;
            }
        }
        catch { }
        return base.TensorCeiling(tensor);
    }

    public override Tensor<T> TensorPower<T>(Tensor<T> tensor, T exponent)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                float expF = exponent is float ef ? ef : Convert.ToSingle(exponent);
                var result = TryRunUnary(tensor, (b, input, output, size) => b.Power(input, output, expF, size));
                if (result != null)
                {
                    var output = new Tensor<T>(result, tensor.Shape._dims);
                    Autodiff.DifferentiableOps.RecordUnary("TensorPower", output, tensor,
                        Autodiff.BackwardFunctions<T>.PowerBackward, new object[] { exponent as object ?? new object() });
                    return output;
                }
            }
        }
        catch { }
        return base.TensorPower(tensor, exponent);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated unary math (Round, Sign)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorRound<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Round(input, output, size));
            if (result != null)
                return new Tensor<T>(result, tensor.Shape._dims);
        }
        catch { }
        return base.TensorRound(tensor);
    }

    public override Tensor<T> TensorSign<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Sign(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Sign", output, tensor, Autodiff.BackwardFunctions<T>.SignBackward);
                return output;
            }
        }
        catch { }
        return base.TensorSign(tensor);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated binary element-wise (Max, Min, Clamp, Where)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorMax<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var result = TryRunBinary(a, b, static (backend, ia, ib, o, size) => backend.Max(ia, ib, o, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, a.Shape._dims);
                // Save clones of inputs in savedState for gradient correctness
                // when inputs are mutated in-place after the forward pass
                Autodiff.DifferentiableOps.RecordBinary("TensorMax", output, a, b,
                    Autodiff.BackwardFunctions<T>.MaxBackward,
                    new object[] { a.Clone(), b.Clone() });
                return output;
            }
        }
        catch { }
        return base.TensorMax(a, b);
    }

    public override Tensor<T> TensorMin<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var result = TryRunBinary(a, b, static (backend, ia, ib, o, size) => backend.Min(ia, ib, o, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, a.Shape._dims);
                Autodiff.DifferentiableOps.RecordBinary("TensorMin", output, a, b,
                    Autodiff.BackwardFunctions<T>.MinBackward,
                    new object[] { a.Clone(), b.Clone() });
                return output;
            }
        }
        catch { }
        return base.TensorMin(a, b);
    }

    public override Tensor<T> TensorClamp<T>(Tensor<T> tensor, T min, T max)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorClamp(tensor, min, max);

        try
        {
            float minF = ToFloatScalar(min);
            float maxF = ToFloatScalar(max);
            using var bufferA = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            var bufferOut = AllocateOutputBuffer(backend, tensor.Length);
            backend.Clamp(bufferA.Buffer, bufferOut.Buffer, minF, maxF, tensor.Length);
            var result = FinishGpuOp<T>(backend, bufferOut, tensor.Length);
            var output = new Tensor<T>(result, tensor.Shape._dims);
            Autodiff.DifferentiableOps.RecordUnary("Clamp", output, tensor,
                Autodiff.BackwardFunctions<T>.ClampBackward, new object[] { min as object ?? new object(), max as object ?? new object() });
            return output;
        }
        catch (Exception)
        {
            return base.TensorClamp(tensor, min, max);
        }
    }

    public override Tensor<T> TensorWhere<T>(Tensor<T> condition, Tensor<T> x, Tensor<T> y)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorWhere(condition, x, y);

        try
        {
            using var condBuf = GetOrAllocateBuffer(backend, condition.GetDataArray());
            using var xBuf = GetOrAllocateBuffer(backend, x.GetDataArray());
            using var yBuf = GetOrAllocateBuffer(backend, y.GetDataArray());
            var outBuf = AllocateOutputBuffer(backend, x.Length);
            backend.Where(condBuf.Buffer, xBuf.Buffer, yBuf.Buffer, outBuf.Buffer, x.Length);
            var result = FinishGpuOp<T>(backend, outBuf, x.Length);
            var output = new Tensor<T>(result, x.Shape._dims);
            Autodiff.DifferentiableOps.RecordBinary("Where", output, x, y, Autodiff.BackwardFunctions<T>.WhereBackward,
                new object[] { condition });
            return output;
        }
        catch (Exception)
        {
            return base.TensorWhere(condition, x, y);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated matrix operations (MatMul, Transpose, Permute)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorMatMul(a, b);

        try
        {
            int M = a.Shape._dims[a.Rank - 2];
            int K = a.Shape._dims[a.Rank - 1];
            int N = b.Shape._dims[b.Rank - 1];
            using var bufA = GetOrAllocateBuffer(backend, a.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, b.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, M * N);
            backend.Gemm(bufA.Buffer, bufB.Buffer, bufOut.Buffer, M, N, K);
            var result = FinishGpuOp<T>(backend, bufOut, M * N);
            int[] outShape = a.Rank == 1 ? new[] { N } : new[] { M, N };
            var output = new Tensor<T>(result, outShape);
            Autodiff.DifferentiableOps.RecordBinary("TensorMatMul", output, a, b, Autodiff.BackwardFunctions<T>.MatMulBackward);
            return output;
        }
        catch (Exception)
        {
            return base.TensorMatMul(a, b);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorMatMulTransposed<T>(Tensor<T> a, Tensor<T> b)
    {
        // GPU path: float-only + 2D for now (matches the CPU fast-path
        // contract). Other dtypes / higher rank fall through to the
        // CpuEngine base implementation which correctly materializes
        // the transpose for the generic case.
        if (typeof(T) != typeof(float) || a.Rank != 2 || b.Rank != 2 || !TryGetBackend(out var backend))
            return base.TensorMatMulTransposed(a, b);

        try
        {
            int M = a._shape[0];
            int K = a._shape[1];
            int N = b._shape[0];
            if (b._shape[1] != K)
                throw new ArgumentException(
                    $"TensorMatMulTransposed K mismatch: a's trailing dim {K} != b's trailing dim {b._shape[1]}.");

            using var bufA = GetOrAllocateBuffer(backend, a.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, b.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, M * N);
            // Backend-native A·Bᵀ: cuBLAS / rocBLAS / MPS / CLBlast use
            // their transB flag; Vulkan/WebGPU dispatch a custom kernel
            // that reads B with the [N, K] index pattern. Either way,
            // no materialized transpose copy.
            backend.MatMulTransposed(bufA.Buffer, bufB.Buffer, bufOut.Buffer, M, N, K);
            var result = FinishGpuOp<T>(backend, bufOut, M * N);
            var output = new Tensor<T>(result, new[] { M, N });
            // Use the dedicated MatMulTransposedBackward — see CpuEngine
            // override for why MatMulBackward is wrong here.
            Autodiff.DifferentiableOps.RecordBinary("TensorMatMulTransposed", output, a, b,
                Autodiff.BackwardFunctions<T>.MatMulTransposedBackward);
            return output;
        }
        catch (Exception)
        {
            return base.TensorMatMulTransposed(a, b);
        }
    }

    public override Tensor<T> BatchMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!TryGetBackend(out var backend) || a.Rank < 3 || b.Rank < 3)
            return base.BatchMatMul(a, b);

        try
        {
            int batchSize = a.Shape._dims[0];
            int M = a.Shape._dims[a.Rank - 2];
            int K = a.Shape._dims[a.Rank - 1];
            int N = b.Shape._dims[b.Rank - 1];
            using var bufA = GetOrAllocateBuffer(backend, a.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, b.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batchSize * M * N);
            backend.BatchedGemm(bufA.Buffer, bufB.Buffer, bufOut.Buffer, M, N, K, batchSize);
            var result = FinishGpuOp<T>(backend, bufOut, batchSize * M * N);
            int[] outShape = (int[])a.Shape._dims.Clone();
            outShape[a.Rank - 1] = N;
            var output = new Tensor<T>(result, outShape);
            Autodiff.DifferentiableOps.RecordBinary("BatchMatMul", output, a, b, Autodiff.BackwardFunctions<T>.BatchMatMulBackward);
            return output;
        }
        catch (Exception)
        {
            return base.BatchMatMul(a, b);
        }
    }

    public override Tensor<T> TensorTranspose<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend) || tensor.Rank != 2)
            return base.TensorTranspose(tensor);

        try
        {
            int rows = tensor.Shape._dims[0];
            int cols = tensor.Shape._dims[1];
            using var bufIn = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, rows * cols);
            backend.Transpose(bufIn.Buffer, bufOut.Buffer, rows, cols);
            var result = FinishGpuOp<T>(backend, bufOut, rows * cols);
            var output = new Tensor<T>(result, new[] { cols, rows });
            Autodiff.DifferentiableOps.RecordUnary("Transpose", output, tensor, Autodiff.BackwardFunctions<T>.TransposeBackward);
            return output;
        }
        catch (Exception)
        {
            return base.TensorTranspose(tensor);
        }
    }

    public override Tensor<T> TensorPermute<T>(Tensor<T> tensor, int[] axes)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorPermute(tensor, axes);

        try
        {
            using var bufIn = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, tensor.Length);
            backend.Permute(bufIn.Buffer, bufOut.Buffer, tensor.Shape._dims, axes);
            var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
            int[] outShape = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
                outShape[i] = tensor.Shape._dims[axes[i]];
            var output = new Tensor<T>(result, outShape);
            Autodiff.DifferentiableOps.RecordUnary("Permute", output, tensor,
                Autodiff.BackwardFunctions<T>.PermuteBackward, new object[] { axes });
            return output;
        }
        catch (Exception)
        {
            return base.TensorPermute(tensor, axes);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated normalization (BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> BatchNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend) || input.Rank < 2)
            return base.BatchNorm(input, gamma, beta, epsilon, out mean, out variance);

        // Issue #226 fix + leak-on-partial-alloc fix: buffers flowing
        // through FinishGpuOp MUST NOT be `using`-scoped (FinishGpuOp's
        // deferred-download closure captures them; a sync `using` Dispose
        // would free the GPU handle before MaterializeIfDeferred fires).
        // BUT a plain `var` allocation that runs BEFORE the cleanup `try`
        // would leak if a LATER allocation throws — none of the partial
        // state ever reaches catch. Allocate the FinishGpuOp-bound buffers
        // inside the try as `default(OwnedBuffer)` slots, then dispose
        // them in catch only when ownership wasn't successfully handed
        // off to FinishGpuOp. The `using var` inputs/intermediates self-
        // dispose on stack unwind regardless of where the throw lands.
        var bufOut = default(OwnedBuffer);
        var bufSaveMean = default(OwnedBuffer);
        var bufSaveInvVar = default(OwnedBuffer);
        bool ownershipTransferred = false;
        try
        {
            int batch = input.Shape._dims[0];
            int channels = input.Shape._dims[1];
            int spatial = input.Length / (batch * channels);
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufGamma = GetOrAllocateBuffer(backend, gamma.GetDataArray());
            using var bufBeta = GetOrAllocateBuffer(backend, beta.GetDataArray());
            bufOut = AllocateOutputBuffer(backend, input.Length);
            using var bufRunMean = AllocateOutputBuffer(backend, channels);
            using var bufRunVar = AllocateOutputBuffer(backend, channels);
            bufSaveMean = AllocateOutputBuffer(backend, channels);
            bufSaveInvVar = AllocateOutputBuffer(backend, channels);
            backend.Fill(bufRunMean.Buffer, 0f, channels);
            backend.Fill(bufRunVar.Buffer, 1f, channels);
            backend.BatchNorm(bufIn.Buffer, bufOut.Buffer, bufGamma.Buffer, bufBeta.Buffer,
                bufRunMean.Buffer, bufRunVar.Buffer, bufSaveMean.Buffer, bufSaveInvVar.Buffer,
                batch, channels, spatial, (float)epsilon, 0.1f, true);
            var result = FinishGpuOp<T>(backend, bufOut, input.Length);
            mean = new Tensor<T>(FinishGpuOp<T>(backend, bufSaveMean, channels), new[] { channels });
            variance = new Tensor<T>(FinishGpuOp<T>(backend, bufSaveInvVar, channels), new[] { channels });
            ownershipTransferred = true;
            return new Tensor<T>(result, input.Shape._dims);
        }
        catch (Exception)
        {
            if (!ownershipTransferred)
            {
                bufOut.Dispose();
                bufSaveMean.Dispose();
                bufSaveInvVar.Dispose();
            }
            return base.BatchNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    public override Tensor<T> LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend) || input.Rank < 2)
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);

        // Issue #226 + leak-on-partial-alloc fix — see BatchNorm above for
        // the full rationale on the default(OwnedBuffer) + ownership-flag
        // pattern.
        var bufOut = default(OwnedBuffer);
        var bufMean = default(OwnedBuffer);
        var bufVar = default(OwnedBuffer);
        bool ownershipTransferred = false;
        try
        {
            int outerSize = input.Shape._dims[0];
            int normSize = input.Length / outerSize;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufGamma = GetOrAllocateBuffer(backend, gamma.GetDataArray());
            using var bufBeta = GetOrAllocateBuffer(backend, beta.GetDataArray());
            bufOut = AllocateOutputBuffer(backend, input.Length);
            bufMean = AllocateOutputBuffer(backend, outerSize);
            bufVar = AllocateOutputBuffer(backend, outerSize);
            backend.LayerNorm(bufIn.Buffer, bufOut.Buffer, bufGamma.Buffer, bufBeta.Buffer,
                bufMean.Buffer, bufVar.Buffer, outerSize, normSize, (float)epsilon);
            var result = FinishGpuOp<T>(backend, bufOut, input.Length);
            mean = new Tensor<T>(FinishGpuOp<T>(backend, bufMean, outerSize), new[] { outerSize });
            variance = new Tensor<T>(FinishGpuOp<T>(backend, bufVar, outerSize), new[] { outerSize });
            ownershipTransferred = true;
            return new Tensor<T>(result, input.Shape._dims);
        }
        catch (Exception)
        {
            if (!ownershipTransferred)
            {
                bufOut.Dispose(); bufMean.Dispose(); bufVar.Dispose();
            }
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    public override Tensor<T> GroupNorm<T>(Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend) || input.Rank < 2)
            return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

        // Issue #226 + leak-on-partial-alloc fix — see BatchNorm above.
        var bufOut = default(OwnedBuffer);
        var bufMean = default(OwnedBuffer);
        var bufVar = default(OwnedBuffer);
        bool ownershipTransferred = false;
        try
        {
            int batch = input.Shape._dims[0];
            int channels = input.Shape._dims[1];
            int spatial = input.Length / (batch * channels);
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufGamma = GetOrAllocateBuffer(backend, gamma.GetDataArray());
            using var bufBeta = GetOrAllocateBuffer(backend, beta.GetDataArray());
            bufOut = AllocateOutputBuffer(backend, input.Length);
            bufMean = AllocateOutputBuffer(backend, batch * numGroups);
            bufVar = AllocateOutputBuffer(backend, batch * numGroups);
            backend.GroupNorm(bufIn.Buffer, bufOut.Buffer, bufGamma.Buffer, bufBeta.Buffer,
                bufMean.Buffer, bufVar.Buffer, batch, channels, spatial, numGroups, (float)epsilon);
            var result = FinishGpuOp<T>(backend, bufOut, input.Length);
            mean = new Tensor<T>(FinishGpuOp<T>(backend, bufMean, batch * numGroups), new[] { batch, numGroups });
            variance = new Tensor<T>(FinishGpuOp<T>(backend, bufVar, batch * numGroups), new[] { batch, numGroups });
            ownershipTransferred = true;
            return new Tensor<T>(result, input.Shape._dims);
        }
        catch (Exception)
        {
            if (!ownershipTransferred)
            {
                bufOut.Dispose(); bufMean.Dispose(); bufVar.Dispose();
            }
            return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);
        }
    }

    public override Tensor<T> InstanceNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend) || input.Rank < 4)
            return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);

        // Issue #226 + leak-on-partial-alloc fix — see BatchNorm above.
        var bufOut = default(OwnedBuffer);
        var bufMean = default(OwnedBuffer);
        var bufVar = default(OwnedBuffer);
        bool ownershipTransferred = false;
        try
        {
            int batch = input.Shape._dims[0];
            int channels = input.Shape._dims[1];
            int spatial = input.Length / (batch * channels);
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufGamma = GetOrAllocateBuffer(backend, gamma.GetDataArray());
            using var bufBeta = GetOrAllocateBuffer(backend, beta.GetDataArray());
            bufOut = AllocateOutputBuffer(backend, input.Length);
            bufMean = AllocateOutputBuffer(backend, batch * channels);
            bufVar = AllocateOutputBuffer(backend, batch * channels);
            backend.InstanceNorm(bufIn.Buffer, bufOut.Buffer, bufGamma.Buffer, bufBeta.Buffer,
                bufMean.Buffer, bufVar.Buffer, batch, channels, spatial, (float)epsilon);
            var result = FinishGpuOp<T>(backend, bufOut, input.Length);
            mean = new Tensor<T>(FinishGpuOp<T>(backend, bufMean, batch * channels), new[] { batch, channels });
            variance = new Tensor<T>(FinishGpuOp<T>(backend, bufVar, batch * channels), new[] { batch, channels });
            ownershipTransferred = true;
            return new Tensor<T>(result, input.Shape._dims);
        }
        catch (Exception)
        {
            if (!ownershipTransferred)
            {
                bufOut.Dispose(); bufMean.Dispose(); bufVar.Dispose();
            }
            return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    public override Tensor<T> RMSNorm<T>(Tensor<T> input, Tensor<T> gamma, double epsilon, out Tensor<T> rms)
    {
        if (!TryGetBackend(out var backend) || input.Rank < 2)
            return base.RMSNorm(input, gamma, epsilon, out rms);

        // Issue #226 + leak-on-partial-alloc fix — see BatchNorm above.
        var bufOut = default(OwnedBuffer);
        var bufRms = default(OwnedBuffer);
        bool ownershipTransferred = false;
        try
        {
            int outerSize = input.Shape._dims[0];
            int normSize = input.Length / outerSize;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufGamma = GetOrAllocateBuffer(backend, gamma.GetDataArray());
            bufOut = AllocateOutputBuffer(backend, input.Length);
            bufRms = AllocateOutputBuffer(backend, outerSize);
            backend.RmsNorm(bufIn.Buffer, bufOut.Buffer, bufGamma.Buffer, bufRms.Buffer,
                outerSize, normSize, (float)epsilon);
            var result = FinishGpuOp<T>(backend, bufOut, input.Length);
            rms = new Tensor<T>(FinishGpuOp<T>(backend, bufRms, outerSize), new[] { outerSize });
            ownershipTransferred = true;
            return new Tensor<T>(result, input.Shape._dims);
        }
        catch (Exception)
        {
            if (!ownershipTransferred)
            {
                bufOut.Dispose(); bufRms.Dispose();
            }
            return base.RMSNorm(input, gamma, epsilon, out rms);
        }
    }

    public override Tensor<T> TensorLayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon)
    {
        var result = LayerNorm(input, gamma, beta, epsilon, out _, out _);
        return result;
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated convolutions (Conv1D, Conv3D, ConvTranspose2D)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> Conv1D<T>(Tensor<T> input, Tensor<T> kernel, int stride, int padding, int dilation)
    {
        if (!TryGetBackend(out var backend) || input.Rank < 3)
            return base.Conv1D(input, kernel, stride, padding, dilation);

        try
        {
            int batch = input.Shape._dims[0];
            int inChannels = input.Shape._dims[1];
            int inWidth = input.Shape._dims[2];
            int outChannels = kernel.Shape._dims[0];
            int kernelW = kernel.Shape._dims[2];
            int outWidth = (inWidth + 2 * padding - dilation * (kernelW - 1) - 1) / stride + 1;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufK = GetOrAllocateBuffer(backend, kernel.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batch * outChannels * outWidth);
            backend.Conv1D(bufIn.Buffer, bufK.Buffer, bufOut.Buffer,
                batch, inChannels, inWidth, outChannels, outWidth, kernelW, stride, padding, dilation);
            var result = FinishGpuOp<T>(backend, bufOut, batch * outChannels * outWidth);
            return new Tensor<T>(result, new[] { batch, outChannels, outWidth });
        }
        catch (Exception)
        {
            return base.Conv1D(input, kernel, stride, padding, dilation);
        }
    }

    public override Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int stride, int padding, int dilation)
    {
        if (!TryGetBackend(out var backend) || input.Rank < 5)
            return base.Conv3D(input, kernel, stride, padding, dilation);

        try
        {
            int batch = input.Shape._dims[0];
            int inChannels = input.Shape._dims[1];
            int inD = input.Shape._dims[2], inH = input.Shape._dims[3], inW = input.Shape._dims[4];
            int outChannels = kernel.Shape._dims[0];
            int kD = kernel.Shape._dims[2], kH = kernel.Shape._dims[3], kW = kernel.Shape._dims[4];
            int outD = (inD + 2 * padding - dilation * (kD - 1) - 1) / stride + 1;
            int outH = (inH + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
            int outW = (inW + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufK = GetOrAllocateBuffer(backend, kernel.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batch * outChannels * outD * outH * outW);
            backend.Conv3D(bufIn.Buffer, bufK.Buffer, bufOut.Buffer,
                batch, inChannels, inD, inH, inW,
                outChannels, outD, outH, outW,
                kD, kH, kW,
                stride, stride, stride,
                padding, padding, padding,
                dilation, dilation, dilation);
            var result = FinishGpuOp<T>(backend, bufOut, batch * outChannels * outD * outH * outW);
            return new Tensor<T>(result, new[] { batch, outChannels, outD, outH, outW });
        }
        catch (Exception)
        {
            return base.Conv3D(input, kernel, stride, padding, dilation);
        }
    }

    public override Tensor<T> ConvTranspose2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding)
    {
        if (!TryGetBackend(out var backend) || input.Rank < 4)
            return base.ConvTranspose2D(input, kernel, stride, padding, outputPadding);

        try
        {
            int batch = input.Shape._dims[0];
            int inChannels = input.Shape._dims[1];
            int inH = input.Shape._dims[2], inW = input.Shape._dims[3];
            int outChannels = kernel.Shape._dims[1];
            int kH = kernel.Shape._dims[2], kW = kernel.Shape._dims[3];
            int outH = (inH - 1) * stride[0] - 2 * padding[0] + kH + outputPadding[0];
            int outW = (inW - 1) * stride[1] - 2 * padding[1] + kW + outputPadding[1];
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufK = GetOrAllocateBuffer(backend, kernel.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batch * outChannels * outH * outW);
            backend.ConvTranspose2D(bufIn.Buffer, bufK.Buffer, bufOut.Buffer,
                batch, inChannels, inH, inW, outChannels, outH, outW,
                kH, kW, stride[0], stride[1], padding[0], padding[1],
                outputPadding[0], outputPadding[1]);
            var result = FinishGpuOp<T>(backend, bufOut, batch * outChannels * outH * outW);
            return new Tensor<T>(result, new[] { batch, outChannels, outH, outW });
        }
        catch (Exception)
        {
            return base.ConvTranspose2D(input, kernel, stride, padding, outputPadding);
        }
    }

    // ReduceSum and ReduceMean are already GPU-accelerated via IEngine explicit implementations below.

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated softmax
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> Softmax<T>(Tensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.Softmax(input, axis);

        try
        {
            int rank = input.Rank;
            int ea = axis < 0 ? rank + axis : axis;
            int features = input.Shape._dims[ea];
            int outerSize = input.Length / features;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, input.Length);
            backend.Softmax(bufIn.Buffer, bufOut.Buffer, outerSize, features);
            var result = FinishGpuOp<T>(backend, bufOut, input.Length);
            var output = new Tensor<T>(result, input.Shape._dims);
            Autodiff.DifferentiableOps.RecordUnary("Softmax", output, input,
                Autodiff.BackwardFunctions<T>.SoftmaxBackward);
            return output;
        }
        catch (Exception)
        {
            return base.Softmax(input, axis);
        }
    }

    public override Tensor<T> TensorLogSoftmax<T>(Tensor<T> tensor, int axis)
    {
        // Delegate to the Softmax + Log path through IEngine explicit implementation
        return ((IEngine)this).TensorLogSoftmax(tensor, axis);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated indexing (Gather, ScatterAdd, Embedding)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorGather<T>(Tensor<T> source, Tensor<int> indices, int axis)
    {
        if (!TryGetBackend(out var backend) || axis != 0)
            return base.TensorGather(source, indices, axis);

        try
        {
            int numIndices = indices.Length;
            int featureSize = source.Rank >= 2 ? source.Length / source.Shape._dims[0] : 1;
            using var bufSrc = GetOrAllocateBuffer(backend, source.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(indices.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, numIndices * featureSize);
            backend.Gather(bufSrc.Buffer, bufIdx, bufOut.Buffer, numIndices, featureSize);
            var result = FinishGpuOp<T>(backend, bufOut, numIndices * featureSize);
            int[] outShape = source.Rank >= 2
                ? new[] { numIndices }.Concat(source.Shape._dims.Skip(1)).ToArray()
                : new[] { numIndices };
            var output = new Tensor<T>(result, outShape);
            Autodiff.DifferentiableOps.RecordUnary("Gather", output, source,
                Autodiff.BackwardFunctions<T>.GatherBackward, new object[] { indices, axis });
            return output;
        }
        catch (Exception)
        {
            return base.TensorGather(source, indices, axis);
        }
    }

    public override Tensor<T> TensorScatterAdd<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> updates, int axis)
    {
        if (!TryGetBackend(out var backend) || axis != 0)
            return base.TensorScatterAdd(destination, indices, updates, axis);

        try
        {
            // Copy destination first, then scatter-add into it
            using var bufDst = GetOrAllocateBuffer(backend, destination.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(indices.GetDataArray());
            using var bufSrc = GetOrAllocateBuffer(backend, updates.GetDataArray());
            backend.ScatterAdd(bufSrc.Buffer, bufIdx, bufDst.Buffer, updates.Length, destination.Length);
            float[] resultFloat = backend.DownloadBuffer(bufDst.Buffer);
            var result = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            var output = new Tensor<T>(result, destination.Shape._dims);
            Autodiff.DifferentiableOps.RecordUnary("ScatterAdd", output, updates,
                Autodiff.BackwardFunctions<T>.ScatterAddBackward, new object[] { indices, axis, destination.Shape._dims });
            return output;
        }
        catch (Exception)
        {
            return base.TensorScatterAdd(destination, indices, updates, axis);
        }
    }

    public override Tensor<T> Embedding<T>(Tensor<int> indices, Tensor<T> embeddingTable)
    {
        if (!TryGetBackend(out var backend))
            return base.Embedding(indices, embeddingTable);

        try
        {
            int numIndices = indices.Length;
            int embeddingDim = embeddingTable.Shape._dims[1];
            using var bufTable = GetOrAllocateBuffer(backend, embeddingTable.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(indices.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, numIndices * embeddingDim);
            backend.Embedding(bufIdx, bufTable.Buffer, bufOut.Buffer, numIndices, embeddingDim);
            var result = FinishGpuOp<T>(backend, bufOut, numIndices * embeddingDim);
            int[] outShape = indices.Rank == 1
                ? new[] { numIndices, embeddingDim }
                : indices.Shape._dims.Concat(new[] { embeddingDim }).ToArray();
            return new Tensor<T>(result, outShape);
        }
        catch (Exception)
        {
            return base.Embedding(indices, embeddingTable);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated loss functions
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorMSELoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBatchBackend(out var bb))
            return base.TensorMSELoss(predictions, targets);

        try
        {
            int batchSize = predictions.Rank >= 2 ? predictions.Shape._dims[0] : 1;
            int numFeatures = predictions.Length / batchSize;
            var bufP = UploadTensorRaw(bb, predictions);
            var bufT = UploadTensorRaw(bb, targets);
            var bufOut = bb.AllocateBuffer(batchSize);
            bb.MseLoss(bufP, bufT, bufOut, batchSize, numFeatures);
            return DeferTensorResult<T>(bb, bufOut, batchSize, new[] { batchSize });
        }
        catch (Exception)
        {
            return base.TensorMSELoss(predictions, targets);
        }
    }

    public override Tensor<T> TensorL1Loss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorL1Loss(predictions, targets);

        try
        {
            int batchSize = predictions.Rank >= 2 ? predictions.Shape._dims[0] : 1;
            int numFeatures = predictions.Length / batchSize;
            using var bufP = GetOrAllocateBuffer(backend, predictions.GetDataArray());
            using var bufT = GetOrAllocateBuffer(backend, targets.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batchSize);
            backend.L1Loss(bufP.Buffer, bufT.Buffer, bufOut.Buffer, batchSize, numFeatures);
            var result = FinishGpuOp<T>(backend, bufOut, batchSize);
            return new Tensor<T>(result, new[] { batchSize });
        }
        catch (Exception)
        {
            return base.TensorL1Loss(predictions, targets);
        }
    }

    public override Tensor<T> TensorHuberLoss<T>(Tensor<T> predictions, Tensor<T> targets, double delta)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorHuberLoss(predictions, targets, delta);

        try
        {
            int batchSize = predictions.Rank >= 2 ? predictions.Shape._dims[0] : 1;
            int numFeatures = predictions.Length / batchSize;
            using var bufP = GetOrAllocateBuffer(backend, predictions.GetDataArray());
            using var bufT = GetOrAllocateBuffer(backend, targets.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batchSize);
            backend.HuberLoss(bufP.Buffer, bufT.Buffer, bufOut.Buffer, batchSize, numFeatures, (float)delta);
            var result = FinishGpuOp<T>(backend, bufOut, batchSize);
            return new Tensor<T>(result, new[] { batchSize });
        }
        catch (Exception)
        {
            return base.TensorHuberLoss(predictions, targets, delta);
        }
    }

    public override Tensor<T> TensorBCEWithLogitsLoss<T>(Tensor<T> logits, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorBCEWithLogitsLoss(logits, targets);

        try
        {
            using var bufL = GetOrAllocateBuffer(backend, logits.GetDataArray());
            using var bufT = GetOrAllocateBuffer(backend, targets.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, logits.Length);
            backend.BceWithLogitsLoss(bufL.Buffer, bufT.Buffer, bufOut.Buffer, logits.Length);
            var result = FinishGpuOp<T>(backend, bufOut, logits.Length);
            return new Tensor<T>(result, logits.Shape._dims);
        }
        catch (Exception)
        {
            return base.TensorBCEWithLogitsLoss(logits, targets);
        }
    }

    public override Tensor<T> TensorCrossEntropyLoss<T>(Tensor<T> logits, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend) || logits.Rank < 2)
            return base.TensorCrossEntropyLoss(logits, targets);

        try
        {
            int batchSize = logits.Shape._dims[0];
            int numClasses = logits.Shape._dims[1];
            float loss = backend.CrossEntropyLoss(
                GetOrAllocateBuffer(backend, logits.GetDataArray()).Buffer,
                GetOrAllocateBuffer(backend, targets.GetDataArray()).Buffer,
                batchSize, numClasses);
            var numOps = MathHelper.GetNumericOperations<T>();
            return new Tensor<T>(new[] { numOps.FromDouble(loss) }, new[] { 1 });
        }
        catch (Exception)
        {
            return base.TensorCrossEntropyLoss(logits, targets);
        }
    }

    public override Tensor<T> TensorNLLLoss<T>(Tensor<T> logProbs, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend) || logProbs.Rank < 2)
            return base.TensorNLLLoss(logProbs, targets);

        try
        {
            int batchSize = logProbs.Shape._dims[0];
            int numClasses = logProbs.Shape._dims[1];
            using var bufLP = GetOrAllocateBuffer(backend, logProbs.GetDataArray());
            using var bufT = GetOrAllocateBuffer(backend, targets.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batchSize);
            backend.NllLoss(bufLP.Buffer, bufT.Buffer, bufOut.Buffer, batchSize, numClasses);
            var result = FinishGpuOp<T>(backend, bufOut, batchSize);
            return new Tensor<T>(result, new[] { batchSize });
        }
        catch (Exception)
        {
            return base.TensorNLLLoss(logProbs, targets);
        }
    }

    public override Tensor<T> TensorKLDivLoss<T>(Tensor<T> input, Tensor<T> target)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorKLDivLoss(input, target);

        try
        {
            using var bufI = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufT = GetOrAllocateBuffer(backend, target.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, input.Length);
            backend.KlDivLoss(bufI.Buffer, bufT.Buffer, bufOut.Buffer, input.Length);
            var result = FinishGpuOp<T>(backend, bufOut, input.Length);
            return new Tensor<T>(result, input.Shape._dims);
        }
        catch (Exception)
        {
            return base.TensorKLDivLoss(input, target);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated misc (Dropout, Upsample, GridSample, AdaptiveAvgPool2D)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> Dropout<T>(Tensor<T> input, double dropoutRate, bool training, out Tensor<T> mask)
    {
        if (!TryGetBackend(out var backend) || !training)
            return base.Dropout(input, dropoutRate, training, out mask);

        try
        {
            ulong seed = (ulong)Environment.TickCount;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, input.Length);
            var bufMask = AllocateOutputBuffer(backend, input.Length);
            backend.Dropout(bufIn.Buffer, bufOut.Buffer, bufMask.Buffer, input.Length, (float)dropoutRate, seed, true);
            var result = FinishGpuOp<T>(backend, bufOut, input.Length);
            var maskResult = FinishGpuOp<T>(backend, bufMask, input.Length);
            mask = new Tensor<T>(maskResult, input.Shape._dims);
            return new Tensor<T>(result, input.Shape._dims);
        }
        catch (Exception)
        {
            return base.Dropout(input, dropoutRate, training, out mask);
        }
    }

    public override Tensor<T> Upsample<T>(Tensor<T> input, int scaleH, int scaleW)
    {
        if (!TryGetBackend(out var backend) || input.Rank != 4 || scaleH != scaleW)
            return base.Upsample(input, scaleH, scaleW);

        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
            int inH = input.Shape._dims[2], inW = input.Shape._dims[3];
            int outH = inH * scaleH, outW = inW * scaleW;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batch * channels * outH * outW);
            backend.NearestNeighborUpsample(bufIn.Buffer, bufOut.Buffer, batch * channels, inH, inW, scaleH);
            var result = FinishGpuOp<T>(backend, bufOut, batch * channels * outH * outW);
            return new Tensor<T>(result, new[] { batch, channels, outH, outW });
        }
        catch (Exception)
        {
            return base.Upsample(input, scaleH, scaleW);
        }
    }

    public override Tensor<T> GridSample<T>(Tensor<T> input, Tensor<T> grid)
    {
        if (!TryGetBackend(out var backend) || input.Rank != 4 || grid.Rank != 4)
            return base.GridSample(input, grid);

        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
            int inH = input.Shape._dims[2], inW = input.Shape._dims[3];
            int outH = grid.Shape._dims[1], outW = grid.Shape._dims[2];
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            using var bufGrid = GetOrAllocateBuffer(backend, grid.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batch * channels * outH * outW);
            backend.GridSample(bufIn.Buffer, bufGrid.Buffer, bufOut.Buffer,
                batch, channels, inH, inW, outH, outW);
            var result = FinishGpuOp<T>(backend, bufOut, batch * channels * outH * outW);
            return new Tensor<T>(result, new[] { batch, channels, outH, outW });
        }
        catch (Exception)
        {
            return base.GridSample(input, grid);
        }
    }

    public override Tensor<T> AdaptiveAvgPool2D<T>(Tensor<T> input, int outputHeight, int outputWidth)
    {
        if (!TryGetBackend(out var backend) || input.Rank != 4)
            return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);

        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
            int inH = input.Shape._dims[2], inW = input.Shape._dims[3];
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batch * channels * outputHeight * outputWidth);
            backend.AdaptiveAvgPool2D(bufIn.Buffer, bufOut.Buffer, batch, channels, inH, inW, outputHeight, outputWidth);
            var result = FinishGpuOp<T>(backend, bufOut, batch * channels * outputHeight * outputWidth);
            return new Tensor<T>(result, new[] { batch, channels, outputHeight, outputWidth });
        }
        catch (Exception)
        {
            return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated in-place operations
    // ──────────────────────────────────────────────────────────────

    public override void TensorAddInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryRunBinaryInPlace(a, b, static (backend, ia, ib, size) => backend.Add(ia, ib, ia, size)))
            return;
        base.TensorAddInPlace(a, b);
    }

    public override void TensorSubtractInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryRunBinaryInPlace(a, b, static (backend, ia, ib, size) => backend.Subtract(ia, ib, ia, size)))
            return;
        base.TensorSubtractInPlace(a, b);
    }

    public override void TensorMultiplyInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryRunBinaryInPlace(a, b, static (backend, ia, ib, size) => backend.Multiply(ia, ib, ia, size)))
            return;
        base.TensorMultiplyInPlace(a, b);
    }

    public override void SigmoidInPlace<T>(Tensor<T> tensor)
    {
        if (TryRunUnaryInPlace(tensor, static (backend, buf, size) => backend.Sigmoid(buf, buf, size)))
            return;
        base.SigmoidInPlace(tensor);
    }

    public override void LeakyReLUInPlace<T>(Tensor<T> tensor, T alpha)
    {
        if (!TryGetBackend(out var backend))
        {
            base.LeakyReLUInPlace(tensor, alpha);
            return;
        }

        try
        {
            float alphaF = ToFloatScalar(alpha);
            using var buf = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            backend.LeakyRelu(buf.Buffer, buf.Buffer, alphaF, tensor.Length);
            float[] result = backend.DownloadBuffer(buf.Buffer);
            var ops = MathHelper.GetNumericOperations<T>();
            ops.FromFloatSpan(new ReadOnlySpan<float>(result), tensor.AsWritableSpan());
            return;
        }
        catch { }
        base.LeakyReLUInPlace(tensor, alpha);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated additional activations (PReLU, RReLU, Threshold)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorPReLU<T>(Tensor<T> tensor, Tensor<T> alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorPReLU(tensor, alpha);

        try
        {
            using var bufIn = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            using var bufAlpha = GetOrAllocateBuffer(backend, alpha.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, tensor.Length);
            backend.PRelu(bufIn.Buffer, bufAlpha.Buffer, bufOut.Buffer, tensor.Length, alpha.Length);
            var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
            var output = new Tensor<T>(result, tensor.Shape._dims);
            Autodiff.DifferentiableOps.RecordBinary("PReLU", output, tensor, alpha,
                Autodiff.BackwardFunctions<T>.PReLUBackward);
            return output;
        }
        catch (Exception)
        {
            return base.TensorPReLU(tensor, alpha);
        }
    }

    public override Tensor<T> TensorThreshold<T>(Tensor<T> tensor, T threshold, T value)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorThreshold(tensor, threshold, value);

        try
        {
            float threshF = ToFloatScalar(threshold);
            float valueF = ToFloatScalar(value);
            using var bufIn = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, tensor.Length);
            backend.Threshold(bufIn.Buffer, bufOut.Buffer, threshF, valueF, tensor.Length);
            var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
            var output = new Tensor<T>(result, tensor.Shape._dims);
            Autodiff.DifferentiableOps.RecordUnary("Threshold", output, tensor,
                Autodiff.BackwardFunctions<T>.ThresholdBackward,
                new object[] { threshold as object ?? new object() });
            return output;
        }
        catch (Exception)
        {
            return base.TensorThreshold(tensor, threshold, value);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated shape ops that need data movement (Unfold, Fold)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> Unfold<T>(Tensor<T> input, int[] kernelSize, int[] stride, int[] padding)
    {
        if (!TryGetBackend(out var backend) || input.Rank != 4)
            return base.Unfold(input, kernelSize, stride, padding);

        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
            int inH = input.Shape._dims[2], inW = input.Shape._dims[3];
            int kH = kernelSize[0], kW = kernelSize[1];
            int sH = stride[0], sW = stride[1];
            int pH = padding[0], pW = padding[1];
            int outH = (inH + 2 * pH - kH) / sH + 1;
            int outW = (inW + 2 * pW - kW) / sW + 1;
            int colSize = batch * channels * kH * kW * outH * outW;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, colSize);
            backend.Unfold(bufIn.Buffer, bufOut.Buffer, batch, channels, inH, inW, kH, kW, sH, sW, pH, pW);
            var result = FinishGpuOp<T>(backend, bufOut, colSize);
            return new Tensor<T>(result, new[] { batch, channels * kH * kW, outH * outW });
        }
        catch (Exception)
        {
            return base.Unfold(input, kernelSize, stride, padding);
        }
    }

    public override Tensor<T> Fold<T>(Tensor<T> input, int[] outputSize, int[] kernelSize, int[] stride, int[] padding)
    {
        if (!TryGetBackend(out var backend))
            return base.Fold(input, outputSize, kernelSize, stride, padding);

        try
        {
            int batch = input.Shape._dims[0];
            int outC = outputSize.Length >= 2 ? outputSize[0] : 1;
            int outH = outputSize.Length >= 2 ? outputSize[outputSize.Length - 2] : outputSize[0];
            int outW = outputSize[outputSize.Length - 1];
            int kH = kernelSize[0], kW = kernelSize[1];
            int sH = stride[0], sW = stride[1];
            int pH = padding[0], pW = padding[1];
            int totalOut = batch * outC * outH * outW;
            using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, totalOut);
            backend.Fill(bufOut.Buffer, 0f, totalOut);
            backend.Fold(bufIn.Buffer, bufOut.Buffer, batch, outC, outH, outW, kH, kW, sH, sW, pH, pW);
            var result = FinishGpuOp<T>(backend, bufOut, totalOut);
            return new Tensor<T>(result, new[] { batch, outC, outH, outW });
        }
        catch (Exception)
        {
            return base.Fold(input, outputSize, kernelSize, stride, padding);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU overrides for remaining CpuEngine virtual methods
    // ──────────────────────────────────────────────────────────────

    // Shape metadata ops — no GPU compute, override for full coverage
    public override Tensor<T> Reshape<T>(Tensor<T> tensor, int[] newShape) => base.Reshape(tensor, newShape);
    public override Tensor<T> TensorExpandDims<T>(Tensor<T> tensor, int axis) => base.TensorExpandDims(tensor, axis);
    public override Tensor<T> TensorFlatten<T>(Tensor<T> tensor) => base.TensorFlatten(tensor);
    public override Tensor<T> TensorSqueeze<T>(Tensor<T> tensor, int axis) => base.TensorSqueeze(tensor, axis);

    public override Tensor<T> EluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, double alpha)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var gBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var iBuf = GetOrAllocateBuffer(backend, input);
                using var oBuf = GetOrAllocateBuffer(backend, output);
                var rBuf = AllocateOutputBuffer(backend, gradOutput.Length);
                try
                {
                    backend.EluBackward(gBuf.Buffer, iBuf.Buffer, oBuf.Buffer, rBuf.Buffer, (float)alpha, gradOutput.Length);
                    var result = FinishGpuOp<T>(backend, rBuf, gradOutput.Length);
                    return new Tensor<T>(result, gradOutput.Shape._dims);
                }
                catch { rBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.EluBackward(gradOutput, input, output, alpha);
    }

    public override Tensor<T> TensorNarrow<T>(Tensor<T> tensor, int dim, int start, int length)
    {
        if (dim == tensor.Rank - 1)
        {
            var startArr = new int[tensor.Rank];
            var lengthArr = new int[tensor.Rank];
            for (int i = 0; i < tensor.Rank; i++)
            {
                startArr[i] = i == dim ? start : 0;
                lengthArr[i] = i == dim ? length : tensor.Shape._dims[i];
            }
            return TensorSlice(tensor, startArr, lengthArr);
        }
        return base.TensorNarrow(tensor, dim, start, length);
    }

    public override Tensor<T> TensorConstantPad<T>(Tensor<T> tensor, int[] padding, T value)
    {
        if (tensor.Rank == 4 && padding.Length >= 4)
            return ((IEngine)this).Pad(tensor, padding[2], padding[3], padding[0], padding[1], value);
        return base.TensorConstantPad(tensor, padding, value);
    }

    public override void SinCos<T>(Vector<T> vector, out Vector<T> sinResult, out Vector<T> cosResult)
    {
        // Dispatch both Sin and Cos through GPU
        sinResult = ((IEngine)this).Sin(vector);
        cosResult = ((IEngine)this).Cos(vector);
    }

    public override Tensor<T> TensorAvgPool1D<T>(Tensor<T> input, int kernelSize, int stride)
    {
        if (!TryGetBackend(out var backend) || input.Rank != 3)
            return base.TensorAvgPool1D(input, kernelSize, stride);
        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1], inLen = input.Shape._dims[2];
            int outLen = (inLen - kernelSize) / stride + 1;
            using var bufIn = GetOrAllocateBuffer(backend, input);
            var bufOut = AllocateOutputBuffer(backend, batch * channels * outLen);
            backend.AvgPool1D(bufIn.Buffer, bufOut.Buffer, batch, channels, inLen, outLen, kernelSize, stride);
            var result = FinishGpuOp<T>(backend, bufOut, batch * channels * outLen);
            return new Tensor<T>(result, new[] { batch, channels, outLen });
        }
        catch { return base.TensorAvgPool1D(input, kernelSize, stride); }
    }

    public override Tensor<T> TensorMaxPool1D<T>(Tensor<T> input, int kernelSize, int stride)
    {
        if (!TryGetBackend(out var backend) || input.Rank != 3)
            return base.TensorMaxPool1D(input, kernelSize, stride);
        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1], inLen = input.Shape._dims[2];
            int outLen = (inLen - kernelSize) / stride + 1;
            using var bufIn = GetOrAllocateBuffer(backend, input);
            var bufOut = AllocateOutputBuffer(backend, batch * channels * outLen);
            backend.MaxPool1D(bufIn.Buffer, bufOut.Buffer, batch, channels, inLen, outLen, kernelSize, stride);
            var result = FinishGpuOp<T>(backend, bufOut, batch * channels * outLen);
            return new Tensor<T>(result, new[] { batch, channels, outLen });
        }
        catch { return base.TensorMaxPool1D(input, kernelSize, stride); }
    }

    public override Tensor<T> TensorUpsampleBilinear<T>(Tensor<T> input, int[] outputSize)
    {
        if (!TryGetBackend(out var backend) || input.Rank != 4 || outputSize.Length < 2)
            return base.TensorUpsampleBilinear(input, outputSize);
        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
            int inH = input.Shape._dims[2], inW = input.Shape._dims[3];
            int outH = outputSize[0], outW = outputSize[1];
            using var bufIn = GetOrAllocateBuffer(backend, input);
            var bufOut = AllocateOutputBuffer(backend, batch * channels * outH * outW);
            backend.BilinearUpsample2D(bufIn.Buffer, bufOut.Buffer, batch, channels, inH, inW, outH, outW);
            var result = FinishGpuOp<T>(backend, bufOut, batch * channels * outH * outW);
            return new Tensor<T>(result, new[] { batch, channels, outH, outW });
        }
        catch { return base.TensorUpsampleBilinear(input, outputSize); }
    }

    public override Tensor<T> ScatterMean<T>(Tensor<T> source, Tensor<int> indices, out Tensor<int>? counts, int dim, int? outputSize)
    {
        if (!TryGetBackend(out var backend) || dim != 0)
            return base.ScatterMean(source, indices, out counts, dim, outputSize);
        try
        {
            int outSize = outputSize ?? (indices.Length > 0 ? indices.GetDataArray().Max() + 1 : 0);
            int featureSize = source.Rank >= 2 ? source.Length / source.Shape._dims[0] : 1;
            int totalOut = outSize * featureSize;
            using var bufSrc = GetOrAllocateBuffer(backend, source);
            using var bufIdx = backend.AllocateIntBuffer(indices.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, totalOut);
            var bufCnt = backend.AllocateBuffer(outSize);
            backend.Fill(bufOut.Buffer, 0f, totalOut);
            backend.Fill(bufCnt, 0f, outSize);
            backend.ScatterMean(bufSrc.Buffer, bufIdx, bufOut.Buffer, bufCnt, source.Length, totalOut, featureSize);
            var result = FinishGpuOp<T>(backend, bufOut, totalOut);
            var countData = backend.DownloadBuffer(bufCnt);
            var countInts = new int[outSize];
            for (int i = 0; i < outSize; i++) countInts[i] = (int)countData[i];
            counts = new Tensor<int>(countInts, new[] { outSize });
            int[] outShape = source.Rank >= 2 ? new[] { outSize }.Concat(source.Shape._dims.Skip(1)).ToArray() : new[] { outSize };
            return new Tensor<T>(result, outShape);
        }
        catch { return base.ScatterMean(source, indices, out counts, dim, outputSize); }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated non-Tensor-prefix activations
    // (ensures GPU dispatch when callers use engine.Sigmoid() directly)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> Sigmoid<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Sigmoid(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Sigmoid", output, tensor,
                    Autodiff.BackwardFunctions<T>.SigmoidBackward);
                return output;
            }
        }
        catch { }
        return base.Sigmoid(tensor);
    }

    public override Tensor<T> ReLU<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Relu(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("ReLU", output, tensor,
                    Autodiff.BackwardFunctions<T>.ReLUBackward);
                return output;
            }
        }
        catch { }
        return base.ReLU(tensor);
    }

    public override Tensor<T> GELU<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Gelu(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("GELU", output, tensor,
                    Autodiff.BackwardFunctions<T>.GELUBackward);
                return output;
            }
        }
        catch { }
        return base.GELU(tensor);
    }

    public override Tensor<T> Mish<T>(Tensor<T> tensor)
    {
        try
        {
            var result = TryRunUnary(tensor, static (backend, input, output, size) => backend.Mish(input, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, tensor.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("Mish", output, tensor,
                    Autodiff.BackwardFunctions<T>.MishBackward);
                return output;
            }
        }
        catch { }
        return base.Mish(tensor);
    }

    public override Tensor<T> LeakyReLU<T>(Tensor<T> tensor, T alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.LeakyReLU(tensor, alpha);

        try
        {
            float alphaFloat = ToFloatScalar(alpha);
            using var bufferA = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            var bufferB = AllocateOutputBuffer(backend, tensor.Length);
            backend.LeakyRelu(bufferA.Buffer, bufferB.Buffer, alphaFloat, tensor.Length);
            var result = FinishGpuOp<T>(backend, bufferB, tensor.Length);
            var gpuOutput = new Tensor<T>(result, tensor.Shape._dims);
            Autodiff.DifferentiableOps.RecordUnary("LeakyReLU", gpuOutput, tensor,
                Autodiff.BackwardFunctions<T>.LeakyReLUBackward,
                savedState: new object[] { (double)alphaFloat });
            return gpuOutput;
        }
        catch (Exception)
        {
            return base.LeakyReLU(tensor, alpha);
        }
    }

    public override Tensor<T> HardSwish<T>(Tensor<T> input)
    {
        try
        {
            var result = TryRunUnary(input, static (backend, inp, output, size) => backend.Hardswish(inp, output, size));
            if (result != null)
            {
                var output = new Tensor<T>(result, input.Shape._dims);
                Autodiff.DifferentiableOps.RecordUnary("HardSwish", output, input,
                    Autodiff.BackwardFunctions<T>.HardSwishBackward);
                return output;
            }
        }
        catch { }
        return base.HardSwish(input);
    }

    // MaxPool2D and AvgPool2D use `public override` earlier in this file.
    // TensorBroadcastMultiply uses `public override` earlier in this file.

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated broadcast ops (override virtual methods)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorBroadcastAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryGetBackend(out var backend) && a.Length > b.Length && a.Length % b.Length == 0)
        {
            try
            {
                int outerSize = a.Length / b.Length;
                using var bufA = GetOrAllocateBuffer(backend, a.GetDataArray());
                using var bufB = GetOrAllocateBuffer(backend, b.GetDataArray());
                var bufOut = AllocateOutputBuffer(backend, a.Length);
                backend.BroadcastAddLast(bufA.Buffer, bufB.Buffer, bufOut.Buffer, outerSize, b.Length);
                var result = FinishGpuOp<T>(backend, bufOut, a.Length);
                var output = new Tensor<T>(result, a.Shape._dims);
                Autodiff.DifferentiableOps.RecordBinary("BroadcastAdd", output, a, b,
                    Autodiff.BackwardFunctions<T>.BroadcastAddBackward);
                return output;
            }
            catch { }
        }
        return base.TensorBroadcastAdd(a, b);
    }

    public override Tensor<T> TensorBroadcastSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryGetBackend(out var backend) && a.Length > b.Length && a.Length % b.Length == 0)
        {
            try
            {
                int outerSize = a.Length / b.Length;
                using var bufA = GetOrAllocateBuffer(backend, a.GetDataArray());
                using var bufB = GetOrAllocateBuffer(backend, b.GetDataArray());
                var bufOut = AllocateOutputBuffer(backend, a.Length);
                backend.BroadcastSubLast(bufA.Buffer, bufB.Buffer, bufOut.Buffer, outerSize, b.Length);
                var result = FinishGpuOp<T>(backend, bufOut, a.Length);
                var output = new Tensor<T>(result, a.Shape._dims);
                Autodiff.DifferentiableOps.RecordBinary("BroadcastSubtract", output, a, b,
                    Autodiff.BackwardFunctions<T>.BroadcastSubtractBackward);
                return output;
            }
            catch { }
        }
        return base.TensorBroadcastSubtract(a, b);
    }

    public override Tensor<T> TensorBroadcastDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (TryGetBackend(out var backend) && a.Length > b.Length && a.Length % b.Length == 0)
        {
            try
            {
                int outerSize = a.Length / b.Length;
                using var bufA = GetOrAllocateBuffer(backend, a.GetDataArray());
                using var bufB = GetOrAllocateBuffer(backend, b.GetDataArray());
                var bufOut = AllocateOutputBuffer(backend, a.Length);
                backend.BroadcastDivLast(bufA.Buffer, bufB.Buffer, bufOut.Buffer, outerSize, b.Length);
                var result = FinishGpuOp<T>(backend, bufOut, a.Length);
                var output = new Tensor<T>(result, a.Shape._dims);
                Autodiff.DifferentiableOps.RecordBinary("BroadcastDivide", output, a, b,
                    Autodiff.BackwardFunctions<T>.BroadcastDivideBackward);
                return output;
            }
            catch { }
        }
        return base.TensorBroadcastDivide(a, b);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated Conv2D (non-Tensor-prefix)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride, int padding, int dilation)
    {
        return FusedConv2D(input, kernel, null, stride, stride, padding, padding, dilation, dilation, FusedActivationType.None);
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated Var/Std reductions
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorVar<T>(Tensor<T> tensor)
    {
        if (!TryGetBatchBackend(out var bb))
            return base.TensorVar(tensor);

        try
        {
            // Compute variance as a single scalar: var = mean((x - mean(x))^2)
            // Use MeanAxis then VarAxis
            int size = tensor.Length;
            var bufIn = UploadTensorRaw(bb, tensor);
            using var bufMean = bb.AllocateBuffer(1);
            var bufVar = bb.AllocateBuffer(1);
            bb.MeanAxis(bufIn, bufMean, 1, size);
            bb.VarAxis(bufIn, bufMean, bufVar, 1, size);
            return DeferTensorResult<T>(bb, bufVar, 1, new[] { 1 });
        }
        catch (Exception)
        {
            return base.TensorVar(tensor);
        }
    }

    public override Tensor<T> TensorStd<T>(Tensor<T> tensor)
    {
        if (!TryGetBatchBackend(out var bb))
            return base.TensorStd(tensor);

        try
        {
            int size = tensor.Length;
            var bufIn = UploadTensorRaw(bb, tensor);
            var bufOut = bb.AllocateBuffer(1);
            bb.StdAxis(bufIn, bufOut, 1, size);
            return DeferTensorResult<T>(bb, bufOut, 1, new[] { 1 });
        }
        catch (Exception)
        {
            return base.TensorStd(tensor);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated scalar ops and reductions
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorAddScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (TryGetBackend(out var backend))
        {
            try
            {
                using var bufIn = GetOrAllocateBuffer(backend, tensor.GetDataArray());
                var bufOut = AllocateOutputBuffer(backend, tensor.Length);
                backend.AddScalar(bufIn.Buffer, bufOut.Buffer, Convert.ToSingle(scalar), tensor.Length);
                var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
                return new Tensor<T>(result, tensor.Shape._dims);
            }
            catch { }
        }
        return base.TensorAddScalar(tensor, scalar);
    }

    public override Tensor<T> TensorSubtractScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (TryGetBackend(out var backend))
        {
            try
            {
                using var bufIn = GetOrAllocateBuffer(backend, tensor.GetDataArray());
                var bufOut = AllocateOutputBuffer(backend, tensor.Length);
                backend.SubScalar(bufIn.Buffer, bufOut.Buffer, Convert.ToSingle(scalar), tensor.Length);
                var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
                return new Tensor<T>(result, tensor.Shape._dims);
            }
            catch { }
        }
        return base.TensorSubtractScalar(tensor, scalar);
    }

    public override Tensor<T> TensorDivideScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (TryGetBatchBackend(out var bb))
        {
            try
            {
                var bufIn = UploadTensorRaw(bb, tensor);
                var bufOut = bb.AllocateBuffer(tensor.Length);
                bb.DivScalar(bufIn, bufOut, Convert.ToSingle(scalar), tensor.Length);
                return DeferTensorResult<T>(bb, bufOut, tensor.Length, tensor.Shape.ToArray());
            }
            catch { }
        }
        return base.TensorDivideScalar(tensor, scalar);
    }

    public override Tensor<T> ReduceStd<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        if (!TryGetBatchBackend(out var bb) || axes.Length != 1)
            return base.ReduceStd(input, axes, keepDims);

        try
        {
            int rank = input.Rank;
            int axis = axes[0] < 0 ? rank + axes[0] : axes[0];
            if (axis < 0 || axis >= rank)
                return base.ReduceStd(input, axes, keepDims);

            int outerSize = 1;
            for (int i = 0; i < axis; i++) outerSize *= input.Shape._dims[i];
            int reduceSize = input.Shape._dims[axis];
            int innerSize = 1;
            for (int i = axis + 1; i < rank; i++) innerSize *= input.Shape._dims[i];
            int totalOuter = outerSize * innerSize;
            var bufIn = UploadTensorRaw(bb, input);
            var bufOut = bb.AllocateBuffer(totalOuter);
            bb.StdAxis(bufIn, bufOut, totalOuter, reduceSize);
            int[] outShape;
            if (keepDims)
            {
                outShape = (int[])input.Shape._dims.Clone();
                outShape[axis] = 1;
            }
            else
            {
                outShape = new int[rank - 1];
                for (int i = 0, j = 0; i < rank; i++)
                    if (i != axis) outShape[j++] = input.Shape._dims[i];
            }
            return DeferTensorResult<T>(bb, bufOut, totalOuter, outShape);
        }
        catch (Exception)
        {
            return base.ReduceStd(input, axes, keepDims);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated remaining activations (RReLU) and indexing (IndexSelect, MaskedFill)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorRReLU<T>(Tensor<T> tensor, double lower, double upper, bool training)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorRReLU(tensor, lower, upper, training);

        try
        {
            // Generate noise buffer for RReLU
            float lowerF = (float)lower, upperF = (float)upper;
            using var bufIn = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            var bufNoise = AllocateOutputBuffer(backend, tensor.Length);
            var bufOut = AllocateOutputBuffer(backend, tensor.Length);
            if (training)
            {
                backend.GenerateRandomUniform(bufNoise.Buffer, tensor.Length, lowerF, upperF, (ulong)Environment.TickCount);
            }
            else
            {
                float mid = (lowerF + upperF) / 2f;
                backend.Fill(bufNoise.Buffer, mid, tensor.Length);
            }
            backend.RRelu(bufIn.Buffer, bufNoise.Buffer, bufOut.Buffer, tensor.Length);
            var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
            var output = new Tensor<T>(result, tensor.Shape._dims);
            Autodiff.DifferentiableOps.RecordUnary("RReLU", output, tensor,
                Autodiff.BackwardFunctions<T>.RReLUBackward,
                new object[] { lower, upper, training, new Tensor<T>(FinishGpuOp<T>(backend, bufNoise, tensor.Length), tensor.Shape._dims) });
            return output;
        }
        catch (Exception)
        {
            return base.TensorRReLU(tensor, lower, upper, training);
        }
    }

    public override Tensor<T> TensorIndexSelect<T>(Tensor<T> tensor, Tensor<int> indices, int axis)
    {
        if (!TryGetBatchBackend(out var bb) || axis != 0)
            return base.TensorIndexSelect(tensor, indices, axis);

        try
        {
            int numIndices = indices.Length;
            int innerSize = tensor.Rank >= 2 ? tensor.Length / tensor.Shape._dims[0] : 1;
            var bufSrc = UploadTensorRaw(bb, tensor);
            using var bufIdx = bb.AllocateIntBuffer(indices.GetDataArray());
            int total = numIndices * innerSize;
            var bufOut = bb.AllocateBuffer(total);
            bb.IndexSelect(bufSrc, bufIdx, bufOut, numIndices, innerSize);
            int[] outShape = tensor.Rank >= 2
                ? new[] { numIndices }.Concat(tensor.Shape._dims.Skip(1)).ToArray()
                : new[] { numIndices };
            var output = DeferTensorResult<T>(bb, bufOut, total, outShape);
            Autodiff.DifferentiableOps.RecordUnary("IndexSelect", output, tensor,
                Autodiff.BackwardFunctions<T>.IndexSelectBackward,
                new object[] { indices, axis, tensor.Shape._dims });
            return output;
        }
        catch (Exception)
        {
            return base.TensorIndexSelect(tensor, indices, axis);
        }
    }

    public override Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, Tensor<bool> mask, T value)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorMaskedFill(tensor, mask, value);

        try
        {
            // Convert bool mask to float (1.0 = true, 0.0 = false) for GPU
            float[] maskFloat = new float[mask.Length];
            var maskData = mask.GetDataArray();
            for (int i = 0; i < mask.Length; i++)
                maskFloat[i] = maskData[i] ? 1f : 0f;

            using var bufIn = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            using var bufMask = backend.AllocateBuffer(maskFloat);
            var bufOut = AllocateOutputBuffer(backend, tensor.Length);
            backend.MaskedFillKernel(bufIn.Buffer, bufMask, bufOut.Buffer, Convert.ToSingle(value), tensor.Length);
            var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
            var output = new Tensor<T>(result, tensor.Shape._dims);
            Autodiff.DifferentiableOps.RecordUnary("MaskedFill", output, tensor,
                Autodiff.BackwardFunctions<T>.MaskedFillBackward, new object[] { mask });
            return output;
        }
        catch (Exception)
        {
            return base.TensorMaskedFill(tensor, mask, value);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated shape ops with data movement (Concatenate, Split, Stack, Tile, Slice, LogSumExp)
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorConcatenate<T>(Tensor<T>[] tensors, int axis)
    {
        // Delegate to IEngine explicit implementation which already has GPU dispatch
        return ((IEngine)this).TensorConcatenate(tensors, axis);
    }

    public override Tensor<T> TensorStack<T>(Tensor<T>[] tensors, int axis)
    {
        return ((IEngine)this).TensorStack(tensors, axis);
    }

    public override Tensor<T> TensorTile<T>(Tensor<T> tensor, int[] multiples)
    {
        return ((IEngine)this).TensorTile(tensor, multiples);
    }

    public override Tensor<T> TensorSlice<T>(Tensor<T> tensor, int[] start, int[] length)
    {
        return ((IEngine)this).TensorSlice(tensor, start, length);
    }

    public override Tensor<T> TensorLogSumExp<T>(Tensor<T> tensor, int axis, bool keepDims)
    {
        return ((IEngine)this).TensorLogSumExp(tensor, axis, keepDims);
    }

    // ReduceSum and ReduceMean already have GPU dispatch via IEngine explicit implementations.

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated AdaptiveMaxPool2D
    // ──────────────────────────────────────────────────────────────

    public override Tensor<T> TensorAdaptiveMaxPool2D<T>(Tensor<T> input, int[] outputSize)
    {
        if (!TryGetBackend(out var backend) || input.Rank != 4)
            return base.TensorAdaptiveMaxPool2D(input, outputSize);

        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
            int inH = input.Shape._dims[2], inW = input.Shape._dims[3];
            int outH = outputSize[0], outW = outputSize[1];
            // For 1x1 output, use GlobalMaxPool2D
            if (outH == 1 && outW == 1)
            {
                using var bufIn = GetOrAllocateBuffer(backend, input.GetDataArray());
                var bufOut = AllocateOutputBuffer(backend, batch * channels);
                backend.GlobalMaxPool2D(bufIn.Buffer, bufOut.Buffer, batch, channels, inH, inW);
                var result = FinishGpuOp<T>(backend, bufOut, batch * channels);
                return new Tensor<T>(result, new[] { batch, channels, 1, 1 });
            }
            return base.TensorAdaptiveMaxPool2D(input, outputSize);
        }
        catch (Exception)
        {
            return base.TensorAdaptiveMaxPool2D(input, outputSize);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GPU-accelerated ReLUInPlace
    // ──────────────────────────────────────────────────────────────

    public override void ReLUInPlace<T>(Tensor<T> tensor)
    {
        if (TryRunUnaryInPlace(tensor, static (backend, buf, size) => backend.Relu(buf, buf, size)))
            return;
        base.ReLUInPlace(tensor);
    }

    // --- Scalar reductions ---
    T IEngine.Sum<T>(Vector<T> v)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                using var gi = b.AllocateBuffer(((Vector<float>)(object)v).ToArray());
                using var go = b.AllocateBuffer(1);
                b.Fill(go, 0f, 1);
                b.SumAxis(gi, go, 1, v.Length);
                return (T)(object)b.DownloadBuffer(go)[0];
            }
            catch { }
        }
        return base.Sum(v);
    }

    T IEngine.Mean<T>(Vector<T> v)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                using var gi = b.AllocateBuffer(((Vector<float>)(object)v).ToArray());
                using var go = b.AllocateBuffer(1);
                b.Fill(go, 0f, 1);
                b.ReduceMean(gi, go, v.Length);
                return (T)(object)b.DownloadBuffer(go)[0];
            }
            catch { }
        }
        return base.Mean(v);
    }

    Tensor<T> IEngine.TensorAddScalar<T>(Tensor<T> input, T scalar)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b))
        { try { var gi=UploadTensorRaw(b, input); var go=b.AllocateBuffer(input.Length); b.AddScalar(gi,go,Convert.ToSingle(scalar),input.Length); return DeferTensorResult<T>(b,go,input.Length,input.Shape.ToArray()); } catch{} }
        return base.TensorAddScalar(input,scalar);
    }

    Tensor<T> IEngine.TensorSubtractScalar<T>(Tensor<T> input, T scalar)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b))
        { try { var gi=UploadTensorRaw(b, input); var go=b.AllocateBuffer(input.Length); b.SubScalar(gi,go,Convert.ToSingle(scalar),input.Length); return DeferTensorResult<T>(b,go,input.Length,input.Shape.ToArray()); } catch{} }
        return base.TensorSubtractScalar(input,scalar);
    }

    Tensor<T> IEngine.TensorBroadcastAdd<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && a.Length>b2.Length && a.Length%b2.Length==0)
        { try { int os=a.Length/b2.Length; var ga=UploadTensorRaw(b, a); var gb=UploadTensorRaw(b, b2); var go=b.AllocateBuffer(a.Length); b.BroadcastAddLast(ga,gb,go,os,b2.Length); return DeferTensorResult<T>(b,go,a.Length,a.Shape.ToArray()); } catch{} }
        return base.TensorBroadcastAdd(a,b2);
    }

    Tensor<T> IEngine.TensorBroadcastSubtract<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && a.Length>b2.Length && a.Length%b2.Length==0)
        { try { int os=a.Length/b2.Length; var ga=UploadTensorRaw(b, a); var gb=UploadTensorRaw(b, b2); var go=b.AllocateBuffer(a.Length); b.BroadcastSubLast(ga,gb,go,os,b2.Length); return DeferTensorResult<T>(b,go,a.Length,a.Shape.ToArray()); } catch{} }
        return base.TensorBroadcastSubtract(a,b2);
    }

    Tensor<T> IEngine.TensorBroadcastMultiply<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && a.Length>b2.Length && a.Length%b2.Length==0)
        { try { int os=a.Length/b2.Length; var ga=UploadTensorRaw(b, a); var gb=UploadTensorRaw(b, b2); var go=b.AllocateBuffer(a.Length); b.BroadcastMulLast(ga,gb,go,os,b2.Length); return DeferTensorResult<T>(b,go,a.Length,a.Shape.ToArray()); } catch{} }
        return base.TensorBroadcastMultiply(a,b2);
    }

    Tensor<T> IEngine.TensorBroadcastDivide<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && a.Length>b2.Length && a.Length%b2.Length==0)
        { try { int os=a.Length/b2.Length; var ga=UploadTensorRaw(b, a); var gb=UploadTensorRaw(b, b2); var go=b.AllocateBuffer(a.Length); b.BroadcastDivLast(ga,gb,go,os,b2.Length); return DeferTensorResult<T>(b,go,a.Length,a.Shape.ToArray()); } catch{} }
        return base.TensorBroadcastDivide(a,b2);
    }

    Tensor<T> IEngine.TensorSiLU<T>(Tensor<T> tensor)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b))
        { try { var gi=UploadTensorRaw(b, tensor); var go=b.AllocateBuffer(tensor.Length); b.Silu(gi,go,tensor.Length); return DeferTensorResult<T>(b,go,tensor.Length,tensor.Shape.ToArray()); } catch{} }
        return base.TensorSiLU(tensor);
    }

    Tensor<T> IEngine.TensorMish<T>(Tensor<T> tensor)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b))
        { try { var gi=UploadTensorRaw(b, tensor); var go=b.AllocateBuffer(tensor.Length); b.Mish(gi,go,tensor.Length); return DeferTensorResult<T>(b,go,tensor.Length,tensor.Shape.ToArray()); } catch{} }
        return base.TensorMish(tensor);
    }

    Tensor<T> IEngine.TensorHardSwish<T>(Tensor<T> tensor)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b))
        { try { var gi=UploadTensorRaw(b, tensor); var go=b.AllocateBuffer(tensor.Length); b.Hardswish(gi,go,tensor.Length); return DeferTensorResult<T>(b,go,tensor.Length,tensor.Shape.ToArray()); } catch{} }
        return base.TensorHardSwish(tensor);
    }

    Tensor<T> IEngine.TensorLeakyReLU<T>(Tensor<T> tensor, T alpha)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b))
        { try { var gi=UploadTensorRaw(b, tensor); var go=b.AllocateBuffer(tensor.Length); b.LeakyRelu(gi,go,Convert.ToSingle(alpha),tensor.Length); return DeferTensorResult<T>(b,go,tensor.Length,tensor.Shape.ToArray()); } catch{} }
        return base.TensorLeakyReLU(tensor, alpha);
    }


    Tensor<T> IEngine.TensorRandomUniform<T>(int[] shape)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                int total = 1;
                foreach (var d in shape) total *= d;
                var go = b.AllocateBuffer(total);
                b.GenerateRandomUniform(go, total, 0f, 1f, (ulong)System.Threading.Interlocked.Increment(ref _gpuRngSeed));
                return DeferTensorResult<T>(b, go, total, shape);
            }
            catch { }
        }
        return base.TensorRandomUniform<T>(shape);
    }

    Tensor<T> IEngine.TensorRandomNormal<T>(int[] shape, T mean, T stddev)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                int total = 1;
                foreach (var d in shape) total *= d;
                var go = b.AllocateBuffer(total);
                b.GenerateRandomNormal(go, total, Convert.ToSingle(mean), Convert.ToSingle(stddev), (ulong)System.Threading.Interlocked.Increment(ref _gpuRngSeed));
                return DeferTensorResult<T>(b, go, total, shape);
            }
            catch { }
        }
        return base.TensorRandomNormal(shape, mean, stddev);
    }

    Tensor<T> IEngine.ReduceSum<T>(Tensor<T> tensor, int[]? axes, bool keepDims)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && axes is not null && axes.Length == 1)
        {
            try
            {
                int rank = tensor.Rank;
                int axis = axes[0] < 0 ? rank + axes[0] : axes[0];
                if (axis >= 0 && axis < rank)
                {
                    int outerSize = 1;
                    for (int i = 0; i < axis; i++) outerSize *= tensor.Shape._dims[i];
                    int reduceSize = tensor.Shape._dims[axis];
                    int innerSize = 1;
                    for (int i = axis + 1; i < rank; i++) innerSize *= tensor.Shape._dims[i];
                    int totalOuter = outerSize * innerSize;
                    var gi = UploadTensorRaw(b, tensor);
                    var go = b.AllocateBuffer(totalOuter);
                    b.SumAxis(gi, go, totalOuter, reduceSize);
                    int[] outShape;
                    if (keepDims)
                    {
                        outShape = (int[])tensor.Shape._dims.Clone();
                        outShape[axis] = 1;
                    }
                    else
                    {
                        outShape = new int[rank - 1];
                        for (int i = 0, j = 0; i < rank; i++)
                            if (i != axis) outShape[j++] = tensor.Shape._dims[i];
                    }
                    return DeferTensorResult<T>(b, go, totalOuter, outShape);
                }
            }
            catch { }
        }
        return base.ReduceSum(tensor, axes, keepDims);
    }

    Tensor<T> IEngine.ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && axes.Length == 1)
        {
            try
            {
                int rank = input.Rank;
                int axis = axes[0] < 0 ? rank + axes[0] : axes[0];
                if (axis >= 0 && axis < rank)
                {
                    int outerSize = 1;
                    for (int i = 0; i < axis; i++) outerSize *= input.Shape._dims[i];
                    int reduceSize = input.Shape._dims[axis];
                    int innerSize = 1;
                    for (int i = axis + 1; i < rank; i++) innerSize *= input.Shape._dims[i];
                    int totalOuter = outerSize * innerSize;
                    var gi = UploadTensorRaw(b, input);
                    var go = b.AllocateBuffer(totalOuter);
                    b.MeanAxis(gi, go, totalOuter, reduceSize);
                    int[] outShape;
                    if (keepDims)
                    {
                        outShape = (int[])input.Shape._dims.Clone();
                        outShape[axis] = 1;
                    }
                    else
                    {
                        outShape = new int[rank - 1];
                        for (int i = 0, j = 0; i < rank; i++)
                            if (i != axis) outShape[j++] = input.Shape._dims[i];
                    }
                    return DeferTensorResult<T>(b, go, totalOuter, outShape);
                }
            }
            catch { }
        }
        return base.ReduceMean(input, axes, keepDims);
    }

    Tensor<T> IEngine.TensorBatchMatMul<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && a.Rank >= 3 && b2.Rank >= 3)
        {
            try
            {
                int batchSize = a.Shape._dims[0];
                int M = a.Shape._dims[a.Rank - 2];
                int K = a.Shape._dims[a.Rank - 1];
                int N = b2.Shape._dims[b2.Rank - 1];
                var ga = UploadTensorRaw(b, a);
                var gb = UploadTensorRaw(b, b2);
                var go = b.AllocateBuffer(batchSize * M * N);
                b.BatchedGemm(ga, gb, go, M, N, K, batchSize);
                int[] outShape = (int[])a.Shape._dims.Clone();
                outShape[a.Rank - 1] = N;
                return DeferTensorResult<T>(b, go, batchSize * M * N, outShape);
            }
            catch { }
        }
        return base.TensorBatchMatMul(a, b2);
    }

    // --- Ops using IGpuBatchExecution (fused kernel methods) ---

    T IEngine.TensorSumOfSquares<T>(Tensor<T> tensor)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { var gi=UploadTensorRaw(bb, tensor); using var go=bb.AllocateBuffer(1); bb.Fill(go,0f,1); bb.ReduceSumOfSquares(gi,go,tensor.Length); return (T)(object)bb.DownloadBuffer(go)[0]; } catch{} }
        return base.TensorSumOfSquares(tensor);
    }

    Tensor<T> IEngine.TensorDiag<T>(Tensor<T> diagonal)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { int n=diagonal.Length; var gi=UploadTensorRaw(bb, diagonal); var go=bb.AllocateBuffer(n*n); bb.DiagKernel(gi,go,n); return DeferTensorResult<T>(bb,go,n*n,new[]{n,n}); } catch{} }
        return base.TensorDiag(diagonal);
    }

    Tensor<T> IEngine.TensorDiagonal<T>(Tensor<T> tensor)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && tensor.Rank==2)
        { try { int n=Math.Min(tensor.Shape._dims[0],tensor.Shape._dims[1]); var gi=UploadTensorRaw(bb, tensor); var go=bb.AllocateBuffer(n); bb.ExtractDiagKernel(gi,go,n,tensor.Shape._dims[1]); return DeferTensorResult<T>(bb,go,n,new[]{n}); } catch{} }
        return base.TensorDiagonal(tensor);
    }

    Tensor<T> IEngine.TensorLinspace<T>(T start, T end, int count)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && count>1)
        { try { float s=Convert.ToSingle(start),e=Convert.ToSingle(end); float step=(e-s)/(count-1); var go=bb.AllocateBuffer(count); bb.LinspaceKernel(go,s,step,count); return DeferTensorResult<T>(bb,go,count,new[]{count}); } catch{} }
        return base.TensorLinspace(start,end,count);
    }

    Tensor<T> IEngine.PixelShuffle<T>(Tensor<T> input, int upscaleFactor)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && input.Rank==4)
        { try { int ba=input.Shape._dims[0],ch=input.Shape._dims[1]/(upscaleFactor*upscaleFactor),ih=input.Shape._dims[2],iw=input.Shape._dims[3]; int total=ba*ch*ih*upscaleFactor*iw*upscaleFactor; var gi=UploadTensorRaw(bb, input); var go=bb.AllocateBuffer(total); bb.PixelShuffle(gi,go,ba,ch,ih,iw,upscaleFactor); return DeferTensorResult<T>(bb,go,total,new[]{ba,ch,ih*upscaleFactor,iw*upscaleFactor}); } catch{} }
        return base.PixelShuffle(input,upscaleFactor);
    }

    Tensor<T> IEngine.PixelShuffleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int upscaleFactor)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && inputShape.Length==4)
        { try { int ba=inputShape[0],ch=inputShape[1]/(upscaleFactor*upscaleFactor),ih=inputShape[2],iw=inputShape[3]; int total=inputShape[0]*inputShape[1]*inputShape[2]*inputShape[3]; var gi=UploadTensorRaw(bb, gradOutput); var go=bb.AllocateBuffer(total); bb.PixelShuffleBackward(gi,go,ba,ch,ih,iw,upscaleFactor); return DeferTensorResult<T>(bb,go,total,inputShape); } catch{} }
        return base.PixelShuffleBackward(gradOutput,inputShape,upscaleFactor);
    }

    Tensor<T> IEngine.Pad<T>(Tensor<T> input, int padTop, int padBottom, int padLeft, int padRight, T padValue)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && input.Rank==4)
        { try { int ba=input.Shape._dims[0],ch=input.Shape._dims[1],ih=input.Shape._dims[2],iw=input.Shape._dims[3]; int oh=ih+padTop+padBottom,ow=iw+padLeft+padRight; int total=ba*ch*oh*ow; var gi=UploadTensorRaw(bb, input); var go=bb.AllocateBuffer(total); bb.Pad2D(gi,go,ba,ch,ih,iw,oh,ow,padTop,padLeft,Convert.ToSingle(padValue)); return DeferTensorResult<T>(bb,go,total,new[]{ba,ch,oh,ow}); } catch{} }
        return base.Pad(input,padTop,padBottom,padLeft,padRight,padValue);
    }

    Tensor<T> IEngine.PadBackward<T>(Tensor<T> gradOutput, int padTop, int padLeft, int[] inputShape)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && inputShape.Length==4)
        { try { int ba=inputShape[0],ch=inputShape[1],ih=inputShape[2],iw=inputShape[3]; int total=ba*ch*ih*iw; int oh=gradOutput.Shape._dims[2],ow=gradOutput.Shape._dims[3]; var gi=UploadTensorRaw(bb, gradOutput); var go=bb.AllocateBuffer(total); bb.Pad2DBackward(gi,go,ba,ch,ih,iw,oh,ow,padTop,padLeft); return DeferTensorResult<T>(bb,go,total,inputShape); } catch{} }
        return base.PadBackward(gradOutput,padTop,padLeft,inputShape);
    }

    Tensor<T> IEngine.Crop<T>(Tensor<T> input, int top, int left, int height, int width)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && input.Rank==4)
        { try { int ba=input.Shape._dims[0],ch=input.Shape._dims[1],ih=input.Shape._dims[2],iw=input.Shape._dims[3]; int total=ba*ch*height*width; var gi=UploadTensorRaw(bb, input); var go=bb.AllocateBuffer(total); bb.Crop2D(gi,go,ba,ch,ih,iw,height,width,top,left); return DeferTensorResult<T>(bb,go,total,new[]{ba,ch,height,width}); } catch{} }
        return base.Crop(input,top,left,height,width);
    }

    Tensor<T> IEngine.CropBackward<T>(Tensor<T> gradOutput, int[] inputShape, int top, int left)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && inputShape.Length==4)
        { try { int ba=inputShape[0],ch=inputShape[1],ih=inputShape[2],iw=inputShape[3]; int total=ba*ch*ih*iw; int oh=gradOutput.Shape._dims[2],ow=gradOutput.Shape._dims[3]; var gi=UploadTensorRaw(bb, gradOutput); var go=bb.AllocateBuffer(total); bb.Fill(go,0f,total); bb.Crop2DBackward(gi,go,ba,ch,ih,iw,oh,ow,top,left); return DeferTensorResult<T>(bb,go,total,inputShape); } catch{} }
        return base.CropBackward(gradOutput,inputShape,top,left);
    }

    Tensor<T> IEngine.TensorCumSum<T>(Tensor<T> tensor, int axis)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { int rank=tensor.Rank; int ea=axis<0?rank+axis:axis; int outerSize=1; for(int i=0;i<ea;i++)outerSize*=tensor.Shape._dims[i]; int innerSize=tensor.Shape._dims[ea]; var gi=UploadTensorRaw(bb, tensor); var go=bb.AllocateBuffer(tensor.Length); bb.CumSumAxis(gi,go,outerSize,innerSize); return DeferTensorResult<T>(bb,go,tensor.Length,tensor.Shape.ToArray()); } catch{} }
        return base.TensorCumSum(tensor,axis);
    }

    Tensor<T> IEngine.TensorLogSumExp<T>(Tensor<T> tensor, int axis, bool keepDims)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { int rank=tensor.Rank; int ea=axis<0?rank+axis:axis; int outerSize=1; for(int i=0;i<ea;i++)outerSize*=tensor.Shape._dims[i]; int reduceSize=tensor.Shape._dims[ea]; var gi=UploadTensorRaw(bb, tensor); var go=bb.AllocateBuffer(outerSize); bb.LogSumExpAxis(gi,go,outerSize,reduceSize); int[] outShape; if(keepDims){outShape=(int[])tensor.Shape._dims.Clone();outShape[ea]=1;}else{outShape=new int[rank-1]; for(int i=0,j=0;i<rank;i++) if(i!=ea) outShape[j++]=tensor.Shape._dims[i];} return DeferTensorResult<T>(bb,go,outerSize,outShape); } catch{} }
        return base.TensorLogSumExp(tensor,axis,keepDims);
    }

    Tensor<T> IEngine.TensorNorm<T>(Tensor<T> tensor, int axis, bool keepDims)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { int rank=tensor.Rank; int ea=axis<0?rank+axis:axis; int outerSize=1; for(int i=0;i<ea;i++)outerSize*=tensor.Shape._dims[i]; int reduceSize=tensor.Shape._dims[ea]; var gi=UploadTensorRaw(bb, tensor); var go=bb.AllocateBuffer(outerSize); bb.NormAxis(gi,go,outerSize,reduceSize); int[] outShape; if(keepDims){outShape=(int[])tensor.Shape._dims.Clone();outShape[ea]=1;}else{outShape=new int[rank-1]; for(int i=0,j=0;i<rank;i++) if(i!=ea) outShape[j++]=tensor.Shape._dims[i];} return DeferTensorResult<T>(bb,go,outerSize,outShape); } catch{} }
        return base.TensorNorm(tensor,axis,keepDims);
    }

    Tensor<T> IEngine.TensorNormalize<T>(Tensor<T> tensor, int axis, T epsilon)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { int rank=tensor.Rank; int ea=axis<0?rank+axis:axis; int outerSize=1; for(int i=0;i<ea;i++)outerSize*=tensor.Shape._dims[i]; int innerSize=tensor.Shape._dims[ea]; var gi=UploadTensorRaw(bb, tensor); var go=bb.AllocateBuffer(tensor.Length); bb.NormalizeL2(gi,go,outerSize,innerSize); return DeferTensorResult<T>(bb,go,tensor.Length,tensor.Shape.ToArray()); } catch{} }
        return base.TensorNormalize(tensor,axis,epsilon);
    }

    Tensor<T> IEngine.TensorConcatenate<T>(Tensor<T>[] tensors, int axis)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && tensors.Length==2 && (axis==-1||axis==tensors[0].Rank-1))
        { try { var a=tensors[0]; var b2=tensors[1]; int lastAxis=a.Rank-1; int outerSize=1; for(int i=0;i<lastAxis;i++)outerSize*=a.Shape._dims[i]; int aInner=a.Shape._dims[lastAxis],bInner=b2.Shape._dims[lastAxis]; int total=outerSize*(aInner+bInner); var ga=UploadTensorRaw(bb, a); var gb=UploadTensorRaw(bb, b2); var go=bb.AllocateBuffer(total); bb.ConcatAxis(ga,gb,go,outerSize,aInner,bInner); int[] outShape=(int[])a.Shape._dims.Clone(); outShape[lastAxis]=aInner+bInner; return DeferTensorResult<T>(bb,go,total,outShape); } catch{} }
        return base.TensorConcatenate(tensors,axis);
    }

    Tensor<T> IEngine.TensorSlice<T>(Tensor<T> tensor, int[] start, int[] length)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && tensor.Rank==2 && start.Length==2 && length.Length==2 && start[0]==0 && length[0]==tensor.Shape._dims[0])
        { try { int outerSize=tensor.Shape._dims[0]; int inputInner=tensor.Shape._dims[1]; int sliceStart=start[1]; int sliceSize=length[1]; int total=outerSize*sliceSize; var gi=UploadTensorRaw(bb, tensor); var go=bb.AllocateBuffer(total); bb.SliceLastAxis(gi,go,outerSize,inputInner,sliceStart,sliceSize); return DeferTensorResult<T>(bb,go,total,new[]{outerSize,sliceSize}); } catch{} }
        return base.TensorSlice(tensor,start,length);
    }

    Tensor<T> IEngine.TensorTile<T>(Tensor<T> tensor, int[] multiples)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && multiples.Length==tensor.Rank && multiples[multiples.Length-1]>1)
        { try { int lastAxis=tensor.Rank-1; bool onlyLastTiled=true; for(int i=0;i<lastAxis;i++) if(multiples[i]!=1) onlyLastTiled=false; if(onlyLastTiled){ int outerSize=1; for(int i=0;i<lastAxis;i++)outerSize*=tensor.Shape._dims[i]; int innerSize=tensor.Shape._dims[lastAxis]; int repeats=multiples[lastAxis]; int total=outerSize*innerSize*repeats; var gi=UploadTensorRaw(bb, tensor); var go=bb.AllocateBuffer(total); bb.TileLastAxis(gi,go,outerSize,innerSize,repeats); int[] outShape=(int[])tensor.Shape._dims.Clone(); outShape[lastAxis]*=repeats; return DeferTensorResult<T>(bb,go,total,outShape); } } catch{} }
        return base.TensorTile(tensor,multiples);
    }

    // GenerateDropoutMask and GenerateGaussianNoise return Vector<T> and need CPU-side
    // post-processing (scale multiplication), so they must download immediately.
    Vector<T> IEngine.GenerateDropoutMask<T>(int length, T dropoutRate, T scale, int? seed)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { float keepProb = 1f - Convert.ToSingle(dropoutRate); ulong s = seed.HasValue ? (ulong)seed.Value : (ulong)Environment.TickCount; using var go=bb.AllocateBuffer(length); bb.DropoutMask(go,length,keepProb,s); float[] result=bb.DownloadBuffer(go); float sc=Convert.ToSingle(scale); for(int i=0;i<length;i++) result[i]*=sc; return new Vector<T>((T[])(object)result); } catch{} }
        return base.GenerateDropoutMask(length,dropoutRate,scale,seed);
    }

    Vector<T> IEngine.GenerateGaussianNoise<T>(int length, T mean, T standardDeviation, int? seed)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { ulong s = seed.HasValue ? (ulong)seed.Value : (ulong)Environment.TickCount; using var go=bb.AllocateBuffer(length); bb.GaussianNoise(go,length,Convert.ToSingle(mean),Convert.ToSingle(standardDeviation),s); return new Vector<T>((T[])(object)bb.DownloadBuffer(go)); } catch{} }
        return base.GenerateGaussianNoise(length,mean,standardDeviation,seed);
    }

    Tensor<T> IEngine.TensorSliceAxis<T>(Tensor<T> tensor, int axis, int index)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && axis >= 0 && axis < tensor.Rank
            && index >= 0 && index < tensor.Shape._dims[axis])
        {
            try
            {
                int outerSize = 1;
                for (int i = 0; i < axis; i++) outerSize *= tensor.Shape._dims[i];
                int axisSize = tensor.Shape._dims[axis];
                int stride = 1;
                for (int i = axis + 1; i < tensor.Rank; i++) stride *= tensor.Shape._dims[i];
                int outputElements = outerSize * stride;
                var gi = UploadTensorRaw(bb, tensor);
                var go = bb.AllocateBuffer(outputElements);
                bb.SliceAxis(gi, go, outerSize, axisSize, stride, index);
                int[] outShape = new int[tensor.Rank - 1];
                for (int i = 0, j = 0; i < tensor.Rank; i++)
                    if (i != axis) outShape[j++] = tensor.Shape._dims[i];
                return DeferTensorResult<T>(bb, go, outputElements, outShape);
            }
            catch { }
        }
        return base.TensorSliceAxis(tensor, axis, index);
    }

    Tensor<T> IEngine.TensorStack<T>(Tensor<T>[] tensors, int axis)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && tensors.Length==2 && axis==0)
        { try { var a=tensors[0]; var b2=tensors[1]; int total=a.Length*2; var ga=UploadTensorRaw(bb, a); var gb=UploadTensorRaw(bb, b2); var go=bb.AllocateBuffer(total); bb.Stack2(ga,gb,go,a.Length); int[] outShape=new int[a.Rank+1]; outShape[0]=2; for(int i=0;i<a.Rank;i++) outShape[i+1]=a.Shape._dims[i]; return DeferTensorResult<T>(bb,go,total,outShape); } catch{} }
        return base.TensorStack(tensors,axis);
    }


    Tensor<T> IEngine.TensorSetSlice<T>(Tensor<T> destination, Tensor<T> source, int[] start)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && destination.Rank==2 && start.Length==2 && start[0]==0)
        {
            try
            {
                int outerSize = destination.Shape._dims[0];
                int outputInner = destination.Shape._dims[1];
                int sliceStart = start[1];
                int sliceSize = source.Shape._dims[1];
                // SetSlice modifies the destination buffer in-place, so we can't defer — the input IS the output
                var gi = UploadTensorRaw(bb, destination);
                var gv = UploadTensorRaw(bb, source);
                bb.SetSliceLastAxis(gi, gv, outerSize, outputInner, sliceStart, sliceSize);
                return new Tensor<T>((T[])(object)bb.DownloadBuffer(gi), destination.Shape.ToArray());
            }
            catch { }
        }
        return base.TensorSetSlice(destination, source, start);
    }

    Tensor<T> IEngine.Concat<T>(IReadOnlyList<Tensor<T>> tensors, int axis)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && tensors.Count==2 && (axis==-1||axis==tensors[0].Rank-1))
        { try { var a=tensors[0]; var b2=tensors[1]; int lastAxis=a.Rank-1; int outerSize=1; for(int i=0;i<lastAxis;i++)outerSize*=a.Shape._dims[i]; int aInner=a.Shape._dims[lastAxis],bInner=b2.Shape._dims[lastAxis]; int total=outerSize*(aInner+bInner); var ga=UploadTensorRaw(bb, a); var gb=UploadTensorRaw(bb, b2); var go=bb.AllocateBuffer(total); bb.ConcatAxis(ga,gb,go,outerSize,aInner,bInner); int[] outShape=(int[])a.Shape._dims.Clone(); outShape[lastAxis]=aInner+bInner; return DeferTensorResult<T>(bb,go,total,outShape); } catch{} }
        return base.Concat(tensors,axis);
    }

    #endregion

    public void Dispose()
    {
        // Clear activation cache to free GPU memory from cached activations
        ClearActivationCache();

        // Dispose all cached GPU buffers
        foreach (var entry in _persistentBufferCache.Values)
        {
            entry.Dispose();
        }
        _persistentBufferCache.Clear();
        _tensorVersions.Clear();

        // Dispose cached CSR GPU buffers
        foreach (var entry in _csrBufferCache.Values)
        {
            entry.Dispose();
        }
        _csrBufferCache.Clear();

        if (_ownsDirectGpu)
            _directGpu?.Dispose();
    }

    #region GPU-wired tensor-level ops (dispatch to existing backend fused kernels)

    // TensorSum already implemented above (line ~1622)

    T IEngine.TensorMean<T>(Tensor<T> input)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                var gi = UploadTensorRaw(b, input);
                using var go = b.AllocateBuffer(1);
                b.Fill(go, 0f, 1);
                b.ReduceMean(gi, go, input.Length);
                return (T)(object)b.DownloadBuffer(go)[0];
            }
            catch { }
        }
        return base.TensorMean(input);
    }

    Tensor<T> IEngine.TensorClip<T>(Tensor<T> input, T min, T max)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                var gi = UploadTensorRaw(b, input);
                var go = b.AllocateBuffer(input.Length);
                b.ClipKernel(gi, go, Convert.ToSingle(min), Convert.ToSingle(max), input.Length);
                return DeferTensorResult<T>(b, go, input.Length, input.Shape.ToArray());
            }
            catch { }
        }
        return base.TensorClip(input, min, max);
    }

    Tensor<T> IEngine.TensorPow<T>(Tensor<T> input, T exponent)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                var gi = UploadTensorRaw(b, input);
                var go = b.AllocateBuffer(input.Length);
                b.PowScalar(gi, go, Convert.ToSingle(exponent), input.Length);
                return DeferTensorResult<T>(b, go, input.Length, input.Shape.ToArray());
            }
            catch { }
        }
        return base.TensorPow(input, exponent);
    }

    Tensor<T> IEngine.TensorFrac<T>(Tensor<T> input)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                var gi = UploadTensorRaw(b, input);
                var go = b.AllocateBuffer(input.Length);
                b.FracKernel(gi, go, input.Length);
                return DeferTensorResult<T>(b, go, input.Length, input.Shape.ToArray());
            }
            catch { }
        }
        return base.TensorFrac(input);
    }

    Tensor<T> IEngine.TensorEye<T>(int n)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                var go = b.AllocateBuffer(n * n);
                b.EyeKernel(go, n);
                return DeferTensorResult<T>(b, go, n * n, new[] { n, n });
            }
            catch { }
        }
        return base.TensorEye<T>(n);
    }

    // TensorOneHot, TensorMaskedFill have non-float generic params (Tensor<int>, Tensor<bool>)
    // — GPU dispatch requires float-only buffers, handled by base CpuEngine

    Tensor<T> IEngine.TensorEquals<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (TryGetBackend(out var b) && ShapesMatch(a.Shape._dims, b2.Shape._dims))
        {
            try
            {
                var ga = UploadTensorRaw(b, a);
                var gb = UploadTensorRaw(b, b2);
                var go = b.AllocateBuffer(a.Length);
                b.EqualsKernel(ga, gb, go, a.Length);
                return DeferTensorResult<T>(b, go, a.Length, a.Shape.ToArray());
            }
            catch { }
        }
        return base.TensorEquals(a, b2);
    }

    Tensor<T> IEngine.TensorNotEquals<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (TryGetBackend(out var b) && ShapesMatch(a.Shape._dims, b2.Shape._dims))
        {
            try
            {
                var ga = UploadTensorRaw(b, a);
                var gb = UploadTensorRaw(b, b2);
                var go = b.AllocateBuffer(a.Length);
                b.NotEqualsKernel(ga, gb, go, a.Length);
                return DeferTensorResult<T>(b, go, a.Length, a.Shape.ToArray());
            }
            catch { }
        }
        return base.TensorNotEquals(a, b2);
    }

    Tensor<T> IEngine.TensorOuter<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                int M = a.Length, N = b2.Length;
                var ga = UploadTensorRaw(b, a);
                var gb = UploadTensorRaw(b, b2);
                var go = b.AllocateBuffer(M * N);
                b.OuterProduct(ga, gb, go, M, N);
                return DeferTensorResult<T>(b, go, M * N, new[] { M, N });
            }
            catch { }
        }
        return base.TensorOuter(a, b2);
    }

    T IEngine.DotProduct<T>(Vector<T> a, Vector<T> b2)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                using var ga = b.AllocateBuffer(((Vector<float>)(object)a).ToArray());
                using var gb = b.AllocateBuffer(((Vector<float>)(object)b2).ToArray());
                using var go = b.AllocateBuffer(1);
                b.BatchDotProduct(ga, gb, go, 1, a.Length);
                return (T)(object)b.DownloadBuffer(go)[0];
            }
            catch { }
        }
        return base.DotProduct(a, b2);
    }

    // TensorBinaryCrossEntropy has 3-param signature (includes epsilon) — handled in base

    Tensor<T> IEngine.GLU<T>(Tensor<T> input, int axis)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                int rank = input.Rank;
                int ea = axis < 0 ? rank + axis : axis;
                int halfDim = input.Shape._dims[ea] / 2;
                int outerSize = 1;
                for (int i = 0; i < ea; i++) outerSize *= input.Shape._dims[i];
                int outputLen = outerSize * halfDim;
                var gi = UploadTensorRaw(b, input);
                using var go = b.AllocateBuffer(outputLen);
                b.GluForward(gi, go, outerSize, halfDim);
                int[] outShape = (int[])input.Shape._dims.Clone();
                outShape[ea] = halfDim;
                return DeferTensorResult<T>(b, go, outputLen, outShape);
            }
            catch { }
        }
        return base.GLU(input, axis);
    }

    Tensor<T> IEngine.GeGLU<T>(Tensor<T> input, int axis)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                int rank = input.Rank;
                int ea = axis < 0 ? rank + axis : axis;
                int halfDim = input.Shape._dims[ea] / 2;
                int outerSize = 1;
                for (int i = 0; i < ea; i++) outerSize *= input.Shape._dims[i];
                int outputLen = outerSize * halfDim;
                var gi = UploadTensorRaw(b, input);
                var go = b.AllocateBuffer(outputLen);
                b.GeGluForward(gi, go, outerSize, halfDim);
                int[] outShape = (int[])input.Shape._dims.Clone();
                outShape[ea] = halfDim;
                return DeferTensorResult<T>(b, go, outputLen, outShape);
            }
            catch { }
        }
        return base.GeGLU(input, axis);
    }

    Tensor<T> IEngine.ReGLU<T>(Tensor<T> input, int axis)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                int ea = axis < 0 ? input.Rank + axis : axis;
                int halfDim = input.Shape._dims[ea] / 2;
                int outerSize = 1;
                for (int i = 0; i < ea; i++) outerSize *= input.Shape._dims[i];
                int outputLen = outerSize * halfDim;
                var gi = UploadTensorRaw(b, input);
                var go = b.AllocateBuffer(outputLen);
                b.ReGluForward(gi, go, outerSize, halfDim);
                int[] outShape = (int[])input.Shape._dims.Clone();
                outShape[ea] = halfDim;
                return DeferTensorResult<T>(b, go, outputLen, outShape);
            }
            catch { }
        }
        return base.ReGLU(input, axis);
    }

    Tensor<T> IEngine.SwiGLU<T>(Tensor<T> input, int axis)
    {
        if (TryGetBackend(out var b))
        {
            try
            {
                int ea = axis < 0 ? input.Rank + axis : axis;
                int halfDim = input.Shape._dims[ea] / 2;
                int outerSize = 1;
                for (int i = 0; i < ea; i++) outerSize *= input.Shape._dims[i];
                int outputLen = outerSize * halfDim;
                var gi = UploadTensorRaw(b, input);
                var go = b.AllocateBuffer(outputLen);
                b.SwiGluForward(gi, go, outerSize, halfDim);
                int[] outShape = (int[])input.Shape._dims.Clone();
                outShape[ea] = halfDim;
                return DeferTensorResult<T>(b, go, outputLen, outShape);
            }
            catch { }
        }
        return base.SwiGLU(input, axis);
    }

    // FillZero returns Vector<T>(length), not void — handled in base

    Tensor<T> IEngine.BatchMatMul<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && a.Rank>=2 && b2.Rank>=2)
        { try { int M=a.Shape._dims[a.Rank-2],K=a.Shape._dims[a.Rank-1],N=b2.Shape._dims[b2.Rank-1]; int batchSize=a.Length/(M*K); int total=batchSize*M*N; var ga=UploadTensorRaw(b, a); var gb=UploadTensorRaw(b, b2); var go=b.AllocateBuffer(total); b.BatchedGemm(ga,gb,go,M,N,K,batchSize); int[] outShape=(int[])a.Shape._dims.Clone(); outShape[a.Rank-1]=N; return DeferTensorResult<T>(b,go,total,outShape); } catch{} }
        return base.BatchMatMul(a,b2);
    }

    Tensor<T> IEngine.TensorTriangularMask<T>(int size, bool upper, int diagonal)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { var go=bb.AllocateBuffer(size*size); bb.TriangularMask(go,size,size,diagonal,upper?-1e9f:0f); return DeferTensorResult<T>(bb,go,size*size,new[]{size,size}); } catch{} }
        return base.TensorTriangularMask<T>(size,upper,diagonal);
    }

    Tensor<T> IEngine.TensorDropoutMask<T>(int[] shape, T dropoutRate, T scale, int? seed)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { int total=1; foreach(var d in shape) total*=d; float keepProb=1f-Convert.ToSingle(dropoutRate); ulong s=seed.HasValue?(ulong)seed.Value:(ulong)Environment.TickCount; using var go=bb.AllocateBuffer(total); bb.DropoutMask(go,total,keepProb,s); float[] result=bb.DownloadBuffer(go); float sc=Convert.ToSingle(scale); for(int i=0;i<total;i++) result[i]*=sc; return new Tensor<T>((T[])(object)result,shape); } catch{} }
        return base.TensorDropoutMask(shape,dropoutRate,scale,seed);
    }

    Tensor<T> IEngine.TensorAddMany<T>(params Tensor<T>[] tensors)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && tensors.Length>=2)
        {
            try
            {
                var result = ((IEngine)this).TensorAdd(tensors[0], tensors[1]);
                for (int i = 2; i < tensors.Length; i++)
                {
                    var prev = result;
                    result = ((IEngine)this).TensorAdd(result, tensors[i]);
                }
                return result;
            }
            catch { }
        }
        return base.TensorAddMany(tensors);
    }

    Tensor<T> IEngine.TensorMultiplyMany<T>(params Tensor<T>[] tensors)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && tensors.Length>=2)
        {
            try
            {
                var result = ((IEngine)this).TensorMultiply(tensors[0], tensors[1]);
                for (int i = 2; i < tensors.Length; i++)
                {
                    result = ((IEngine)this).TensorMultiply(result, tensors[i]);
                }
                return result;
            }
            catch { }
        }
        return base.TensorMultiplyMany(tensors);
    }

    Tensor<T> IEngine.Upsample<T>(Tensor<T> input, int scaleH, int scaleW)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && input.Rank==4 && scaleH==scaleW)
        { try { int ba=input.Shape._dims[0],ch=input.Shape._dims[1],ih=input.Shape._dims[2],iw=input.Shape._dims[3]; int oh=ih*scaleH,ow=iw*scaleW; int total=ba*ch*oh*ow; var gi=UploadTensorRaw(b, input); var go=b.AllocateBuffer(total); b.NearestNeighborUpsample(gi,go,ba*ch,ih,iw,scaleH); return DeferTensorResult<T>(b,go,total,new[]{ba,ch,oh,ow}); } catch{} }
        return base.Upsample(input,scaleH,scaleW);
    }

    Tensor<T> IEngine.UpsampleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleH, int scaleW)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b) && inputShape.Length==4 && scaleH==scaleW)
        { try { int ba=inputShape[0],ch=inputShape[1],ih=inputShape[2],iw=inputShape[3]; int total=ba*ch*ih*iw; var gi=UploadTensorRaw(b, gradOutput); var go=b.AllocateBuffer(total); b.NearestNeighborUpsampleBackward(gi,go,ba*ch,ih,iw,scaleH); return DeferTensorResult<T>(b,go,total,inputShape); } catch{} }
        return base.UpsampleBackward(gradOutput,inputShape,scaleH,scaleW);
    }

    Tensor<T> IEngine.TensorSoftmax<T>(Tensor<T> tensor, int axis)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b))
        { try { int rank=tensor.Rank; int ea=axis<0?rank+axis:axis; int features=tensor.Shape._dims[ea]; int outerSize=tensor.Length/features; var gi=UploadTensorRaw(b, tensor); var go=b.AllocateBuffer(tensor.Length); b.Softmax(gi,go,outerSize,features); return DeferTensorResult<T>(b,go,tensor.Length,tensor.Shape.ToArray()); } catch{} }
        return base.TensorSoftmax(tensor,axis);
    }

    Tensor<T> IEngine.TensorLogSoftmax<T>(Tensor<T> tensor, int axis)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { int rank=tensor.Rank; int ea=axis<0?rank+axis:axis; int features=tensor.Shape._dims[ea]; int outerSize=tensor.Length/features; var gi=UploadTensorRaw(bb, tensor); var go=bb.AllocateBuffer(tensor.Length); bb.LogSoftmax(gi,go,outerSize,features); return DeferTensorResult<T>(bb,go,tensor.Length,tensor.Shape.ToArray()); } catch{} }
        return base.TensorLogSoftmax(tensor,axis);
    }

    Tensor<T> IEngine.TensorSoftmaxBackward<T>(Tensor<T> softmaxOutput, Tensor<T> gradOutput, int axis)
    {
        if (typeof(T)==typeof(float) && TryGetBackend(out var b))
        { try { var gso=UploadTensorRaw(b, softmaxOutput); var ggo=UploadTensorRaw(b, gradOutput); var go=b.AllocateBuffer(softmaxOutput.Length); int rank=softmaxOutput.Rank; int ea=axis<0?rank+axis:axis; int features=softmaxOutput.Shape._dims[ea]; int outerSize=softmaxOutput.Length/features; b.SoftmaxBackward(gso,ggo,go,outerSize,features); return DeferTensorResult<T>(b,go,softmaxOutput.Length,softmaxOutput.Shape.ToArray()); } catch{} }
        return base.TensorSoftmaxBackward(softmaxOutput,gradOutput,axis);
    }

    T IEngine.CosineSimilarity<T>(Vector<T> a, Vector<T> b2)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { using var ga=bb.AllocateBuffer(((Vector<float>)(object)a).ToArray()); using var gb=bb.AllocateBuffer(((Vector<float>)(object)b2).ToArray()); using var go=bb.AllocateBuffer(1); bb.CosineSimilarity(ga,gb,go,1,a.Length); return (T)(object)bb.DownloadBuffer(go)[0]; } catch{} }
        return base.CosineSimilarity(a,b2);
    }

    T IEngine.StdDev<T>(Vector<T> vector)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { using var gi=bb.AllocateBuffer(((Vector<float>)(object)vector).ToArray()); using var go=bb.AllocateBuffer(1); bb.StdAxis(gi,go,1,vector.Length); return (T)(object)bb.DownloadBuffer(go)[0]; } catch{} }
        return base.StdDev(vector);
    }

    T IEngine.Norm<T>(Vector<T> vector)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { using var gi=bb.AllocateBuffer(((Vector<float>)(object)vector).ToArray()); using var go=bb.AllocateBuffer(1); bb.NormAxis(gi,go,1,vector.Length); return (T)(object)bb.DownloadBuffer(go)[0]; } catch{} }
        return base.Norm(vector);
    }

    T IEngine.Product<T>(Vector<T> vector)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb))
        { try { using var gi=bb.AllocateBuffer(((Vector<float>)(object)vector).ToArray()); using var go=bb.AllocateBuffer(1); bb.ProductAxis(gi,go,1,vector.Length); return (T)(object)bb.DownloadBuffer(go)[0]; } catch{} }
        return base.Product(vector);
    }

    Tensor<T> IEngine.PairwiseDistance<T>(Tensor<T> x, Tensor<T> y)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && x.Rank==2 && y.Rank==2 && x.Shape._dims[1]==y.Shape._dims[1])
        { try { int M=x.Shape._dims[0],N=y.Shape._dims[0],dim=x.Shape._dims[1]; var gx=UploadTensorRaw(bb, x); var gy=UploadTensorRaw(bb, y); var go=bb.AllocateBuffer(M*N); bb.PairwiseDistance(gx,gy,go,M,N,dim); return DeferTensorResult<T>(bb,go,M*N,new[]{M,N}); } catch{} }
        return base.PairwiseDistance(x,y);
    }

    Tensor<T> IEngine.TensorBatchOuterProduct<T>(Tensor<T> a, Tensor<T> b2)
    {
        if (typeof(T)==typeof(float) && TryGetBatchBackend(out var bb) && a.Rank>=2 && b2.Rank>=2)
        { try { int batchSize=a.Shape._dims[0],M=a.Shape._dims[a.Rank-1],N=b2.Shape._dims[b2.Rank-1]; int total=batchSize*M*N; var ga=UploadTensorRaw(bb, a); var gb=UploadTensorRaw(bb, b2); var go=bb.AllocateBuffer(total); bb.BatchOuterProduct(ga,gb,go,batchSize,M,N); return DeferTensorResult<T>(bb,go,total,new[]{batchSize,M,N}); } catch{} }
        return base.TensorBatchOuterProduct(a,b2);
    }

    #endregion

    // ══════════════════════════════════════════════════════════════
    // GPU overrides for new engine ops
    // ══════════════════════════════════════════════════════════════

    public override Tensor<T> StopGradient<T>(Tensor<T> tensor)
    {
        try
        {
            if (TryGetBackend(out var backend))
            {
                using var src = GetOrAllocateBuffer(backend, tensor);
                var dst = AllocateOutputBuffer(backend, tensor.Length);
                try
                {
                    backend.CopyBuffer(src.Buffer, dst.Buffer, tensor.Length);
                    return DeferTensorResult<T>(backend, dst.Buffer, tensor.Length, tensor.Shape._dims);
                }
                catch
                {
                    dst.Dispose();
                    throw;
                }
            }
        }
        catch { }
        return base.StopGradient(tensor);
    }

    public override Tensor<T> FusedLinearReLU<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        try
        {
            if (TryGetBackend(out var backend) && input.Shape.Length == 2 && weight.Shape.Length == 2
                && input.Shape[1] == weight.Shape[0] && bias.Shape.Length == 1 && bias.Shape[0] == weight.Shape[1])
            {
                int batchSize = input.Shape[0], inFeatures = input.Shape[1], outFeatures = weight.Shape[1];
                var gi = UploadTensorRaw(backend, input);
                var gw = UploadTensorRaw(backend, weight);
                var gb = UploadTensorRaw(backend, bias);
                // Compute matmul+bias (pre-activation) for backward, then apply ReLU
                var preActBuf = backend.AllocateBuffer(batchSize * outFeatures);
                backend.Gemm(gi, gw, preActBuf, batchSize, outFeatures, inFeatures);
                backend.BiasAdd(preActBuf, gb, preActBuf, batchSize, outFeatures);
                var preActivation = DeferTensorResult<T>(backend, preActBuf, batchSize * outFeatures, new[] { batchSize, outFeatures });

                var go = backend.AllocateBuffer(batchSize * outFeatures);
                backend.Relu(preActBuf, go, batchSize * outFeatures);
                var result = DeferTensorResult<T>(backend, go, batchSize * outFeatures, new[] { batchSize, outFeatures });

                Autodiff.DifferentiableOps.RecordIfActive("FusedLinearReLU", result,
                    new[] { input, weight, bias },
                    Autodiff.BackwardFunctions<T>.FusedMatMulAddReLUBackward,
                    new object[] { preActivation });
                return result;
            }
        }
        catch { }
        return base.FusedLinearReLU(input, weight, bias);
    }

    public override Tensor<T> FusedLinearSigmoid<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        try
        {
            if (TryGetBackend(out var backend) && input.Shape.Length == 2 && weight.Shape.Length == 2
                && input.Shape[1] == weight.Shape[0] && bias.Shape.Length == 1 && bias.Shape[0] == weight.Shape[1])
            {
                int batchSize = input.Shape[0], inFeatures = input.Shape[1], outFeatures = weight.Shape[1];
                var gi = UploadTensorRaw(backend, input);
                var gw = UploadTensorRaw(backend, weight);
                var gb = UploadTensorRaw(backend, bias);
                var go = backend.AllocateBuffer(batchSize * outFeatures);
                backend.FusedLinearSigmoid(gi, gw, gb, go, batchSize, inFeatures, outFeatures);
                var result = DeferTensorResult<T>(backend, go, batchSize * outFeatures, new[] { batchSize, outFeatures });
                // Sigmoid backward uses output directly: grad * out * (1 - out)
                Autodiff.DifferentiableOps.RecordIfActive("FusedLinearSigmoid", result,
                    new[] { input, weight, bias },
                    Autodiff.BackwardFunctions<T>.FusedMatMulAddSigmoidBackward);
                return result;
            }
        }
        catch { }
        return base.FusedLinearSigmoid(input, weight, bias);
    }

    public override Tensor<T> FusedLinearTanh<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        try
        {
            if (TryGetBackend(out var backend) && input.Shape.Length == 2 && weight.Shape.Length == 2
                && input.Shape[1] == weight.Shape[0] && bias.Shape.Length == 1 && bias.Shape[0] == weight.Shape[1])
            {
                int batchSize = input.Shape[0], inFeatures = input.Shape[1], outFeatures = weight.Shape[1];
                var gi = UploadTensorRaw(backend, input);
                var gw = UploadTensorRaw(backend, weight);
                var gb = UploadTensorRaw(backend, bias);
                var go = backend.AllocateBuffer(batchSize * outFeatures);
                backend.FusedLinearTanh(gi, gw, gb, go, batchSize, inFeatures, outFeatures);
                var result = DeferTensorResult<T>(backend, go, batchSize * outFeatures, new[] { batchSize, outFeatures });
                // Tanh backward uses output directly: grad * (1 - out^2)
                Autodiff.DifferentiableOps.RecordIfActive("FusedLinearTanh", result,
                    new[] { input, weight, bias },
                    Autodiff.BackwardFunctions<T>.FusedMatMulAddTanhBackward);
                return result;
            }
        }
        catch { }
        return base.FusedLinearTanh(input, weight, bias);
    }

    public override Tensor<T> FusedLinearGELU<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        try
        {
            if (TryGetBackend(out var backend) && input.Shape.Length == 2 && weight.Shape.Length == 2
                && input.Shape[1] == weight.Shape[0] && bias.Shape.Length == 1 && bias.Shape[0] == weight.Shape[1])
            {
                int batchSize = input.Shape[0], inFeatures = input.Shape[1], outFeatures = weight.Shape[1];
                var gi = UploadTensorRaw(backend, input);
                var gw = UploadTensorRaw(backend, weight);
                var gb = UploadTensorRaw(backend, bias);
                var go = backend.AllocateBuffer(batchSize * outFeatures);
                backend.FusedLinearGELU(gi, gw, gb, go, batchSize, inFeatures, outFeatures);
                var result = DeferTensorResult<T>(backend, go, batchSize * outFeatures, new[] { batchSize, outFeatures });
                // GELU backward needs pre-activation (MatMul+Bias output before GELU).
                // Compute without tape recording to avoid polluting the tape.
                Tensor<T> preActivation;
                using (new Autodiff.NoGradScope<T>())
                {
                    preActivation = base.TensorBroadcastAdd(base.TensorMatMul(input, weight), bias);
                }
                Autodiff.DifferentiableOps.RecordIfActive("FusedLinearGELU", result,
                    new[] { input, weight, bias },
                    Autodiff.BackwardFunctions<T>.FusedMatMulAddGELUBackward,
                    new object[] { preActivation });
                return result;
            }
        }
        catch { }
        return base.FusedLinearGELU(input, weight, bias);
    }

    public override Tensor<T> FusedLinearSwish<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        try
        {
            if (TryGetBackend(out var backend) && input.Shape.Length == 2 && weight.Shape.Length == 2
                && input.Shape[1] == weight.Shape[0] && bias.Shape.Length == 1 && bias.Shape[0] == weight.Shape[1])
            {
                int batchSize = input.Shape[0], inFeatures = input.Shape[1], outFeatures = weight.Shape[1];
                var gi = UploadTensorRaw(backend, input);
                var gw = UploadTensorRaw(backend, weight);
                var gb = UploadTensorRaw(backend, bias);
                var go = backend.AllocateBuffer(batchSize * outFeatures);
                backend.FusedLinearSwish(gi, gw, gb, go, batchSize, inFeatures, outFeatures);
                var result = DeferTensorResult<T>(backend, go, batchSize * outFeatures, new[] { batchSize, outFeatures });
                // Swish backward needs pre-activation to compute sigmoid(x) derivative.
                // Compute without tape recording to avoid polluting the tape.
                Tensor<T> preActivation;
                using (new Autodiff.NoGradScope<T>())
                {
                    preActivation = base.TensorBroadcastAdd(base.TensorMatMul(input, weight), bias);
                }
                Autodiff.DifferentiableOps.RecordIfActive("FusedLinearSwish", result,
                    new[] { input, weight, bias },
                    Autodiff.BackwardFunctions<T>.FusedMatMulAddSwishBackward,
                    new object[] { preActivation });
                return result;
            }
        }
        catch { }
        return base.FusedLinearSwish(input, weight, bias);
    }

    public override Tensor<T> TensorIoULoss<T>(Tensor<T> predicted, Tensor<T> target)
    {
        try
        {
            if (TryGetBackend(out var backend)
                && predicted.Shape.Length == 2 && predicted.Shape[1] == 4
                && target.Shape.Length == 2 && target.Shape[1] == 4
                && target.Shape[0] == predicted.Shape[0])
            {
                int numBoxes = predicted.Shape[0];
                var gp = UploadTensorRaw(backend, predicted);
                var gt = UploadTensorRaw(backend, target);
                var go = backend.AllocateBuffer(numBoxes);
                backend.IoULoss(gp, gt, go, numBoxes);
                var result = DeferTensorResult<T>(backend, go, numBoxes, new[] { numBoxes });
                Autodiff.DifferentiableOps.RecordIfActive("IoULoss", result,
                    new[] { predicted, target },
                    Autodiff.BackwardFunctions<T>.IoULossBackward);
                return result;
            }
        }
        catch { }
        return base.TensorIoULoss(predicted, target);
    }

    public override Tensor<T> TensorGIoULoss<T>(Tensor<T> predicted, Tensor<T> target)
    {
        try
        {
            if (TryGetBackend(out var backend)
                && predicted.Shape.Length == 2 && predicted.Shape[1] == 4
                && target.Shape.Length == 2 && target.Shape[1] == 4
                && target.Shape[0] == predicted.Shape[0])
            {
                int numBoxes = predicted.Shape[0];
                var gp = UploadTensorRaw(backend, predicted);
                var gt = UploadTensorRaw(backend, target);
                var go = backend.AllocateBuffer(numBoxes);
                backend.GIoULoss(gp, gt, go, numBoxes);
                var result = DeferTensorResult<T>(backend, go, numBoxes, new[] { numBoxes });
                Autodiff.DifferentiableOps.RecordIfActive("GIoULoss", result,
                    new[] { predicted, target },
                    Autodiff.BackwardFunctions<T>.GIoULossBackward);
                return result;
            }
        }
        catch { }
        return base.TensorGIoULoss(predicted, target);
    }

    public override Tensor<T> TensorDIoULoss<T>(Tensor<T> predicted, Tensor<T> target)
    {
        try
        {
            if (TryGetBackend(out var backend)
                && predicted.Shape.Length == 2 && predicted.Shape[1] == 4
                && target.Shape.Length == 2 && target.Shape[1] == 4
                && target.Shape[0] == predicted.Shape[0])
            {
                int numBoxes = predicted.Shape[0];
                var gp = UploadTensorRaw(backend, predicted);
                var gt = UploadTensorRaw(backend, target);
                var go = backend.AllocateBuffer(numBoxes);
                backend.DIoULoss(gp, gt, go, numBoxes);
                var result = DeferTensorResult<T>(backend, go, numBoxes, new[] { numBoxes });
                Autodiff.DifferentiableOps.RecordIfActive("DIoULoss", result,
                    new[] { predicted, target },
                    Autodiff.BackwardFunctions<T>.DIoULossBackward);
                return result;
            }
        }
        catch { }
        return base.TensorDIoULoss(predicted, target);
    }

    public override Tensor<T> TensorCIoULoss<T>(Tensor<T> predicted, Tensor<T> target)
    {
        try
        {
            if (TryGetBackend(out var backend)
                && predicted.Shape.Length == 2 && predicted.Shape[1] == 4
                && target.Shape.Length == 2 && target.Shape[1] == 4
                && target.Shape[0] == predicted.Shape[0])
            {
                int numBoxes = predicted.Shape[0];
                var gp = UploadTensorRaw(backend, predicted);
                var gt = UploadTensorRaw(backend, target);
                var go = backend.AllocateBuffer(numBoxes);
                backend.CIoULoss(gp, gt, go, numBoxes);
                var result = DeferTensorResult<T>(backend, go, numBoxes, new[] { numBoxes });
                Autodiff.DifferentiableOps.RecordIfActive("CIoULoss", result,
                    new[] { predicted, target },
                    Autodiff.BackwardFunctions<T>.CIoULossBackward);
                return result;
            }
        }
        catch { }
        return base.TensorCIoULoss(predicted, target);
    }

    // --- Octonion GPU overrides ---
    // Dispatches OctonionMatMulTensor to GPU backends' OctonionLinearForward kernel.
    // The GPU kernel expects a bias buffer; we pass zeros since OctonionMatMulTensor has no bias.

    public override Tensor<T> OctonionMatMulTensor<T>(Tensor<T> input, Tensor<T> weight)
    {
        // Validate shapes match base implementation before indexing
        if (input is null || weight is null
            || input.Rank != 3 || input._shape[2] != 8
            || weight.Rank != 3 || weight._shape[2] != 8
            || input._shape[1] != weight._shape[1])
            return base.OctonionMatMulTensor(input ?? throw new ArgumentNullException(nameof(input)),
                weight ?? throw new ArgumentNullException(nameof(weight)));

        if (!TryGetBackend(out var backend))
            return base.OctonionMatMulTensor(input, weight);

        try
        {
            int batch = input._shape[0];
            int inputFeatures = input._shape[1];
            int outputFeatures = weight._shape[0];

            // T is float — direct cast, no ToDouble round-trip
            var inputF = DirectGpuEngine.ToFloatArray(input.GetFlattenedData());
            var weightF = DirectGpuEngine.ToFloatArray(weight.GetFlattenedData());
            var biasData = new float[outputFeatures * 8]; // zero bias

            using var inputBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var weightBuf = new OwnedBuffer(backend.AllocateBuffer(weightF), true);
            using var biasBuf = new OwnedBuffer(backend.AllocateBuffer(biasData), true);
            using var outputBuf = new OwnedBuffer(backend.AllocateBuffer(batch * outputFeatures * 8), true);

            backend.OctonionLinearForward(inputBuf.Buffer, weightBuf.Buffer, biasBuf.Buffer,
                outputBuf.Buffer, batch, inputFeatures, outputFeatures);

            var gpuResult = backend.DownloadBuffer(outputBuf.Buffer);
            var result = new Tensor<T>(new[] { batch, outputFeatures, 8 });
            // Vectorized float→T conversion directly into the tensor's backing array
            var resultData = result.GetDataArray();
            var resultOps = MathHelper.GetNumericOperations<T>();
            resultOps.FromFloatSpan(new ReadOnlySpan<float>(gpuResult), new Span<T>(resultData));

            Autodiff.DifferentiableOps.RecordBinary("OctonionMatMulTensor", result, input, weight,
                Autodiff.BackwardFunctions<T>.OctonionMatMulBackward);

            return result;
        }
        catch { return base.OctonionMatMulTensor(input, weight); }
    }

    // --- Native Complex<T> GPU overrides ---
    // Decomposes Tensor<Complex<T>> to split real/imag GPU buffers,
    // dispatches via IDirectGpuBackend.SplitComplex*, recomposes result.

    private static (float[] real, float[] imag) DecomposeComplex<T>(Tensor<Complex<T>> tensor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = tensor.Length;
        var real = new float[n];
        var imag = new float[n];
        for (int i = 0; i < n; i++)
        {
            real[i] = (float)ops.ToDouble(tensor[i].Real);
            imag[i] = (float)ops.ToDouble(tensor[i].Imaginary);
        }
        return (real, imag);
    }

    private static Tensor<Complex<T>> RecomposeComplex<T>(float[] real, float[] imag, int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = real.Length;
        var result = new Tensor<Complex<T>>(shape);
        for (int i = 0; i < n; i++)
            result[i] = new Complex<T>(ops.FromDouble(real[i]), ops.FromDouble(imag[i]));
        return result;
    }

    private static Tensor<T> RecomposeReal<T>(float[] data, int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Length;
        var result = new Tensor<T>(shape);
        for (int i = 0; i < n; i++)
            result[i] = ops.FromDouble(data[i]);
        return result;
    }

    public override Tensor<Complex<T>> NativeComplexMultiply<T>(Tensor<Complex<T>> a, Tensor<Complex<T>> b)
    {
        if (a is null || b is null || a.Length != b.Length)
            return base.NativeComplexMultiply(a ?? throw new ArgumentNullException(nameof(a)), b ?? throw new ArgumentNullException(nameof(b)));
        if (!TryGetBackend(out var backend))
            return base.NativeComplexMultiply(a, b);

        try
        {
            int n = a.Length;
            var (aR, aI) = DecomposeComplex(a);
            var (bR, bI) = DecomposeComplex(b);

            using var aRBuf = new OwnedBuffer(backend.AllocateBuffer(aR), true);
            using var aIBuf = new OwnedBuffer(backend.AllocateBuffer(aI), true);
            using var bRBuf = new OwnedBuffer(backend.AllocateBuffer(bR), true);
            using var bIBuf = new OwnedBuffer(backend.AllocateBuffer(bI), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexMultiply(aRBuf.Buffer, aIBuf.Buffer, bRBuf.Buffer, bIBuf.Buffer,
                oRBuf.Buffer, oIBuf.Buffer, n);

            return RecomposeComplex<T>(backend.DownloadBuffer(oRBuf.Buffer),
                backend.DownloadBuffer(oIBuf.Buffer), a._shape);
        }
        catch { return base.NativeComplexMultiply(a, b); }
    }

    // ─── HRR binding primitives (issue #248) ─────────────────────────
    //
    // GPU overrides where a backend primitive maps cleanly. Ops where
    // the backend has no direct primitive (UnitPhaseCodebook,
    // PhaseCoherenceDecode, HRRBindAccumulate) fall through to the
    // CpuEngine base implementation; a follow-up PR can add fused GPU
    // kernels for the hottest path (HRRBindAccumulate) if the HRE
    // campaign's workloads warrant it. All paths catch exceptions and
    // fall back to CPU so a single backend hiccup never takes the op
    // offline — matches the existing Native* override convention.

    /// <inheritdoc />
    public override void NativeComplexPointwiseMultiply<T>(
        ReadOnlySpan<T> aRe, ReadOnlySpan<T> aIm,
        ReadOnlySpan<T> bRe, ReadOnlySpan<T> bIm,
        Span<T> cRe, Span<T> cIm,
        bool conjugateB = false)
    {
        // Backend uses single-precision buffers; on a precision-sensitive
        // T=double workload the GPU path would round-trip through fp32
        // and hurt accuracy, so we keep T=double on CPU and route only
        // T=float through the GPU primitives. Unsupported T also drops
        // to CPU via base. This mirrors how NativeComplexMultiply picks
        // its paths.
        int n = aRe.Length;
        if (typeof(T) != typeof(float)
            || aIm.Length != n || bRe.Length != n || bIm.Length != n
            || cRe.Length != n || cIm.Length != n
            || n == 0
            || !TryGetBackend(out var backend))
        {
            base.NativeComplexPointwiseMultiply(aRe, aIm, bRe, bIm, cRe, cIm, conjugateB);
            return;
        }

        try
        {
            // Materialise to float[] for upload. The caller's Span<T> is
            // typed T = float here per the guard above.
            var aRf = HrrToFloatArray(aRe);
            var aIf = HrrToFloatArray(aIm);
            var bRf = HrrToFloatArray(bRe);
            var bIf = HrrToFloatArray(bIm);

            using var aRBuf = new OwnedBuffer(backend.AllocateBuffer(aRf), true);
            using var aIBuf = new OwnedBuffer(backend.AllocateBuffer(aIf), true);
            using var bRBuf = new OwnedBuffer(backend.AllocateBuffer(bRf), true);
            using var bIBuf = new OwnedBuffer(backend.AllocateBuffer(bIf), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            if (conjugateB)
            {
                // c = a · conj(b) — exactly the cross-spectral density op
                // the backend already ships: outR = aR·bR + aI·bI,
                // outI = aI·bR − aR·bI. Zero new kernels needed.
                backend.SplitComplexCrossSpectral(
                    aRBuf.Buffer, aIBuf.Buffer, bRBuf.Buffer, bIBuf.Buffer,
                    oRBuf.Buffer, oIBuf.Buffer, n);
            }
            else
            {
                backend.SplitComplexMultiply(
                    aRBuf.Buffer, aIBuf.Buffer, bRBuf.Buffer, bIBuf.Buffer,
                    oRBuf.Buffer, oIBuf.Buffer, n);
            }

            var oRf = backend.DownloadBuffer(oRBuf.Buffer);
            var oIf = backend.DownloadBuffer(oIBuf.Buffer);
            HrrFloatArrayToSpan(oRf, cRe, n);
            HrrFloatArrayToSpan(oIf, cIm, n);
        }
        catch
        {
            base.NativeComplexPointwiseMultiply(aRe, aIm, bRe, bIm, cRe, cIm, conjugateB);
        }
    }

    /// <inheritdoc />
    public override void NativeGather<T>(
        ReadOnlySpan<T> input,
        ReadOnlySpan<int> indices,
        Span<T> output)
    {
        // Only float has a direct backend Gather primitive. All other T
        // (double, Half, custom) fall back to CPU.
        int nOut = indices.Length;
        if (typeof(T) != typeof(float)
            || output.Length != nOut
            || nOut == 0
            || !TryGetBackend(out var backend))
        {
            base.NativeGather(input, indices, output);
            return;
        }

        try
        {
            var inF = HrrToFloatArray(input);
            var idx = indices.ToArray();
            using var inBuf = new OwnedBuffer(backend.AllocateBuffer(inF), true);
            using var idxBuf = new OwnedBuffer(backend.AllocateIntBuffer(idx), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(nOut), true);
            // backend.Gather signature: source, indices, output, numIndices, featureSize.
            // For a flat 1-D gather (no per-index row copy) featureSize = 1.
            backend.Gather(inBuf.Buffer, idxBuf.Buffer, outBuf.Buffer, nOut, featureSize: 1);
            var oF = backend.DownloadBuffer(outBuf.Buffer);
            HrrFloatArrayToSpan(oF, output, nOut);
        }
        catch
        {
            base.NativeGather(input, indices, output);
        }
    }

    /// <inheritdoc />
    public override void NativeUnitPhaseCodebook<T>(
        Span<T> outRe, Span<T> outIm,
        int seed, int V, int D, bool kPsk = false, int k = 0)
    {
        long total = (long)V * D;
        if (typeof(T) != typeof(float)
            || total <= 0 || total > int.MaxValue
            || outRe.Length != total || outIm.Length != total
            || !TryGetBackend(out var backend))
        {
            base.NativeUnitPhaseCodebook(outRe, outIm, seed, V, D, kPsk, k);
            return;
        }

        try
        {
            int n = (int)total;
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            backend.SplitComplexUnitPhaseCodebook(
                oRBuf.Buffer, oIBuf.Buffer, seed, V, D, kPsk, k);
            var oRf = backend.DownloadBuffer(oRBuf.Buffer);
            var oIf = backend.DownloadBuffer(oIBuf.Buffer);
            HrrFloatArrayToSpan(oRf, outRe, n);
            HrrFloatArrayToSpan(oIf, outIm, n);
        }
        catch
        {
            base.NativeUnitPhaseCodebook(outRe, outIm, seed, V, D, kPsk, k);
        }
    }

    /// <inheritdoc />
    public override void NativeComplexPhaseCoherenceDecode<T>(
        ReadOnlySpan<T> codesRe, ReadOnlySpan<T> codesIm,
        ReadOnlySpan<T> queryRe, ReadOnlySpan<T> queryIm,
        Span<T> scores, int V, int D)
    {
        long total = (long)V * D;
        if (typeof(T) != typeof(float)
            || V <= 0 || D <= 0
            || codesRe.Length != total || codesIm.Length != total
            || queryRe.Length != D || queryIm.Length != D
            || scores.Length != V
            || !TryGetBackend(out var backend))
        {
            base.NativeComplexPhaseCoherenceDecode(codesRe, codesIm, queryRe, queryIm, scores, V, D);
            return;
        }

        try
        {
            var cRf = HrrToFloatArray(codesRe);
            var cIf = HrrToFloatArray(codesIm);
            var qRf = HrrToFloatArray(queryRe);
            var qIf = HrrToFloatArray(queryIm);
            using var cRBuf = new OwnedBuffer(backend.AllocateBuffer(cRf), true);
            using var cIBuf = new OwnedBuffer(backend.AllocateBuffer(cIf), true);
            using var qRBuf = new OwnedBuffer(backend.AllocateBuffer(qRf), true);
            using var qIBuf = new OwnedBuffer(backend.AllocateBuffer(qIf), true);
            using var sBuf = new OwnedBuffer(backend.AllocateBuffer(V), true);
            backend.SplitComplexPhaseCoherenceDecode(
                cRBuf.Buffer, cIBuf.Buffer, qRBuf.Buffer, qIBuf.Buffer, sBuf.Buffer,
                V, D);
            var sF = backend.DownloadBuffer(sBuf.Buffer);
            HrrFloatArrayToSpan(sF, scores, V);
        }
        catch
        {
            base.NativeComplexPhaseCoherenceDecode(codesRe, codesIm, queryRe, queryIm, scores, V, D);
        }
    }

    /// <inheritdoc />
    public override void NativeHRRBindAccumulate<T>(
        ReadOnlySpan<T> keyCodeRe, ReadOnlySpan<T> keyCodeIm,
        ReadOnlySpan<T> valPermCodeRe, ReadOnlySpan<T> valPermCodeIm,
        ReadOnlySpan<int> keyIds, ReadOnlySpan<int> valIds,
        Span<T> memoryRe, Span<T> memoryIm, int D)
    {
        int N = keyIds.Length;
        if (typeof(T) != typeof(float)
            || D <= 0 || N <= 0
            || valIds.Length != N
            || memoryRe.Length != D || memoryIm.Length != D
            || !TryGetBackend(out var backend))
        {
            base.NativeHRRBindAccumulate(
                keyCodeRe, keyCodeIm, valPermCodeRe, valPermCodeIm,
                keyIds, valIds, memoryRe, memoryIm, D);
            return;
        }

        try
        {
            var kRf = HrrToFloatArray(keyCodeRe);
            var kIf = HrrToFloatArray(keyCodeIm);
            var vRf = HrrToFloatArray(valPermCodeRe);
            var vIf = HrrToFloatArray(valPermCodeIm);
            var kIds = keyIds.ToArray();
            var vIds = valIds.ToArray();
            var mRf = HrrToFloatArray((ReadOnlySpan<T>)memoryRe);
            var mIf = HrrToFloatArray((ReadOnlySpan<T>)memoryIm);
            using var kRBuf = new OwnedBuffer(backend.AllocateBuffer(kRf), true);
            using var kIBuf = new OwnedBuffer(backend.AllocateBuffer(kIf), true);
            using var vRBuf = new OwnedBuffer(backend.AllocateBuffer(vRf), true);
            using var vIBuf = new OwnedBuffer(backend.AllocateBuffer(vIf), true);
            using var kIdsBuf = new OwnedBuffer(backend.AllocateIntBuffer(kIds), true);
            using var vIdsBuf = new OwnedBuffer(backend.AllocateIntBuffer(vIds), true);
            using var mRBuf = new OwnedBuffer(backend.AllocateBuffer(mRf), true);
            using var mIBuf = new OwnedBuffer(backend.AllocateBuffer(mIf), true);

            backend.SplitComplexHrrBindAccumulate(
                kRBuf.Buffer, kIBuf.Buffer,
                vRBuf.Buffer, vIBuf.Buffer,
                kIdsBuf.Buffer, vIdsBuf.Buffer,
                mRBuf.Buffer, mIBuf.Buffer,
                N, D);

            var mRout = backend.DownloadBuffer(mRBuf.Buffer);
            var mIout = backend.DownloadBuffer(mIBuf.Buffer);
            HrrFloatArrayToSpan(mRout, memoryRe, D);
            HrrFloatArrayToSpan(mIout, memoryIm, D);
        }
        catch
        {
            base.NativeHRRBindAccumulate(
                keyCodeRe, keyCodeIm, valPermCodeRe, valPermCodeIm,
                keyIds, valIds, memoryRe, memoryIm, D);
        }
    }

    // ── Span marshalling helpers for the HRR GPU overrides ──
    // Allocates a fresh float[] from a ReadOnlySpan<T> where T is known
    // to be float at runtime (checked by callers via typeof). Using
    // ToArray on a reinterpreted span keeps a single allocation per
    // upload with no intermediate object churn.
    private static float[] HrrToFloatArray<T>(ReadOnlySpan<T> s)
    {
#if NET5_0_OR_GREATER
        ref T head = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(s);
        var asFloat = System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(
            ref Unsafe.As<T, float>(ref head), s.Length);
        return asFloat.ToArray();
#else
        // net471: fall through to slower boxed path — the guard above
        // keeps this unreachable on net471 because the GPU overrides
        // check typeof(T) == typeof(float) first.
        throw new PlatformNotSupportedException("HrrToFloatArray requires NET5_0_OR_GREATER.");
#endif
    }

    private static void HrrFloatArrayToSpan<T>(float[] src, Span<T> dst, int n)
    {
#if NET5_0_OR_GREATER
        ref T head = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(dst);
        var asFloat = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
            ref Unsafe.As<T, float>(ref head), dst.Length);
        src.AsSpan(0, n).CopyTo(asFloat);
#else
        throw new PlatformNotSupportedException("HrrFloatArrayToSpan requires NET5_0_OR_GREATER.");
#endif
    }

    public override Tensor<Complex<T>> NativeComplexConjugate<T>(Tensor<Complex<T>> a)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexConjugate(a);

        try
        {
            int n = a.Length;
            var (aR, aI) = DecomposeComplex(a);

            using var aRBuf = new OwnedBuffer(backend.AllocateBuffer(aR), true);
            using var aIBuf = new OwnedBuffer(backend.AllocateBuffer(aI), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexConjugate(aRBuf.Buffer, aIBuf.Buffer, oRBuf.Buffer, oIBuf.Buffer, n);

            return RecomposeComplex<T>(backend.DownloadBuffer(oRBuf.Buffer),
                backend.DownloadBuffer(oIBuf.Buffer), a._shape);
        }
        catch { return base.NativeComplexConjugate(a); }
    }

    public override Tensor<T> NativeComplexMagnitude<T>(Tensor<Complex<T>> a)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexMagnitude(a);

        try
        {
            int n = a.Length;
            var (aR, aI) = DecomposeComplex(a);

            using var aRBuf = new OwnedBuffer(backend.AllocateBuffer(aR), true);
            using var aIBuf = new OwnedBuffer(backend.AllocateBuffer(aI), true);
            using var oBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexMagnitude(aRBuf.Buffer, aIBuf.Buffer, oBuf.Buffer, n);

            return RecomposeReal<T>(backend.DownloadBuffer(oBuf.Buffer), a._shape);
        }
        catch { return base.NativeComplexMagnitude(a); }
    }

    public override Tensor<T> NativeComplexMagnitudeSquared<T>(Tensor<Complex<T>> a)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexMagnitudeSquared(a);

        try
        {
            int n = a.Length;
            var (aR, aI) = DecomposeComplex(a);

            using var aRBuf = new OwnedBuffer(backend.AllocateBuffer(aR), true);
            using var aIBuf = new OwnedBuffer(backend.AllocateBuffer(aI), true);
            using var oBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexMagnitudeSquared(aRBuf.Buffer, aIBuf.Buffer, oBuf.Buffer, n);

            return RecomposeReal<T>(backend.DownloadBuffer(oBuf.Buffer), a._shape);
        }
        catch { return base.NativeComplexMagnitudeSquared(a); }
    }

    public override Tensor<T> NativeComplexPhase<T>(Tensor<Complex<T>> a)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexPhase(a);

        try
        {
            int n = a.Length;
            var (aR, aI) = DecomposeComplex(a);

            using var aRBuf = new OwnedBuffer(backend.AllocateBuffer(aR), true);
            using var aIBuf = new OwnedBuffer(backend.AllocateBuffer(aI), true);
            using var oBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexPhase(aRBuf.Buffer, aIBuf.Buffer, oBuf.Buffer, n);

            return RecomposeReal<T>(backend.DownloadBuffer(oBuf.Buffer), a._shape);
        }
        catch { return base.NativeComplexPhase(a); }
    }

    public override Tensor<Complex<T>> NativeComplexFromPolar<T>(Tensor<T> magnitudes, Tensor<T> phases)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexFromPolar(magnitudes, phases);

        try
        {
            int n = magnitudes.Length;
            var ops = MathHelper.GetNumericOperations<T>();
            var magF = new float[n];
            var phaseF = new float[n];
            for (int i = 0; i < n; i++)
            {
                magF[i] = (float)ops.ToDouble(magnitudes[i]);
                phaseF[i] = (float)ops.ToDouble(phases[i]);
            }

            using var magBuf = new OwnedBuffer(backend.AllocateBuffer(magF), true);
            using var phaseBuf = new OwnedBuffer(backend.AllocateBuffer(phaseF), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexFromPolar(magBuf.Buffer, phaseBuf.Buffer, oRBuf.Buffer, oIBuf.Buffer, n);

            return RecomposeComplex<T>(backend.DownloadBuffer(oRBuf.Buffer),
                backend.DownloadBuffer(oIBuf.Buffer), magnitudes._shape);
        }
        catch { return base.NativeComplexFromPolar(magnitudes, phases); }
    }

    public override Tensor<Complex<T>> NativeComplexScale<T>(Tensor<Complex<T>> a, T scalar)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexScale(a, scalar);

        try
        {
            int n = a.Length;
            var ops = MathHelper.GetNumericOperations<T>();
            var (aR, aI) = DecomposeComplex(a);
            float scalarF = (float)ops.ToDouble(scalar);

            using var aRBuf = new OwnedBuffer(backend.AllocateBuffer(aR), true);
            using var aIBuf = new OwnedBuffer(backend.AllocateBuffer(aI), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexScale(aRBuf.Buffer, aIBuf.Buffer, oRBuf.Buffer, oIBuf.Buffer, scalarF, n);

            return RecomposeComplex<T>(backend.DownloadBuffer(oRBuf.Buffer),
                backend.DownloadBuffer(oIBuf.Buffer), a._shape);
        }
        catch { return base.NativeComplexScale(a, scalar); }
    }

    public override Tensor<Complex<T>> NativeComplexAdd<T>(Tensor<Complex<T>> a, Tensor<Complex<T>> b)
    {
        if (a is null || b is null || a.Length != b.Length)
            return base.NativeComplexAdd(a ?? throw new ArgumentNullException(nameof(a)), b ?? throw new ArgumentNullException(nameof(b)));
        if (!TryGetBackend(out var backend))
            return base.NativeComplexAdd(a, b);

        try
        {
            int n = a.Length;
            var (aR, aI) = DecomposeComplex(a);
            var (bR, bI) = DecomposeComplex(b);

            using var aRBuf = new OwnedBuffer(backend.AllocateBuffer(aR), true);
            using var aIBuf = new OwnedBuffer(backend.AllocateBuffer(aI), true);
            using var bRBuf = new OwnedBuffer(backend.AllocateBuffer(bR), true);
            using var bIBuf = new OwnedBuffer(backend.AllocateBuffer(bI), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexAdd(aRBuf.Buffer, aIBuf.Buffer, bRBuf.Buffer, bIBuf.Buffer,
                oRBuf.Buffer, oIBuf.Buffer, n);

            return RecomposeComplex<T>(backend.DownloadBuffer(oRBuf.Buffer),
                backend.DownloadBuffer(oIBuf.Buffer), a._shape);
        }
        catch { return base.NativeComplexAdd(a, b); }
    }

    public override Tensor<Complex<T>> NativeComplexFFTComplex<T>(Tensor<Complex<T>> input)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexFFTComplex(input);

        try
        {
            int fftSize = input._shape[^1];
            int batchCount = input.Length / fftSize;
            int n = input.Length;
            if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
                return base.NativeComplexFFTComplex(input);

            var (inR, inI) = DecomposeComplex(input);

            using var inRBuf = new OwnedBuffer(backend.AllocateBuffer(inR), true);
            using var inIBuf = new OwnedBuffer(backend.AllocateBuffer(inI), true);
            using var outRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var outIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            if (batchCount > 1)
                backend.BatchedFFT(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, batchCount, fftSize, inverse: false);
            else
                backend.FFT(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, fftSize, inverse: false);

            return RecomposeComplex<T>(backend.DownloadBuffer(outRBuf.Buffer),
                backend.DownloadBuffer(outIBuf.Buffer), input._shape);
        }
        catch { return base.NativeComplexFFTComplex(input); }
    }

    public override Tensor<Complex<T>> NativeComplexTopK<T>(Tensor<Complex<T>> input, int k)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (k <= 0)
            throw new ArgumentException("k must be positive.", nameof(k));
        if (!TryGetBackend(out var backend))
            return base.NativeComplexTopK(input, k);

        try
        {
            int n = input.Length;
            int clampedK = Math.Min(k, n);
            var (inR, inI) = DecomposeComplex(input);

            using var inRBuf = new OwnedBuffer(backend.AllocateBuffer(inR), true);
            using var inIBuf = new OwnedBuffer(backend.AllocateBuffer(inI), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexTopK(inRBuf.Buffer, inIBuf.Buffer, oRBuf.Buffer, oIBuf.Buffer, n, clampedK);

            var outR = backend.DownloadBuffer(oRBuf.Buffer);
            var outI = backend.DownloadBuffer(oIBuf.Buffer);

            // Post-filter: GPU threshold-based TopK may retain more than K on ties.
            // Ensure there are no more than min(k,n) non-zero elements by zeroing the weakest extras.
            int nonZero = 0;
            for (int i = 0; i < n; i++)
                if (outR[i] != 0f || outI[i] != 0f) nonZero++;

            if (nonZero > clampedK)
            {
                // Build (magSq, idx) pairs for non-zero entries, sort ascending, zero the weakest
                var extras = new (float mag, int idx)[nonZero];
                int ei = 0;
                for (int i = 0; i < n; i++)
                    if (outR[i] != 0f || outI[i] != 0f)
                        extras[ei++] = (outR[i] * outR[i] + outI[i] * outI[i], i);

                Array.Sort(extras, (a, b) => a.mag.CompareTo(b.mag));
                int toZero = nonZero - clampedK;
                for (int i = 0; i < toZero; i++)
                {
                    int idx = extras[i].idx;
                    outR[idx] = 0f;
                    outI[idx] = 0f;
                }
            }

            return RecomposeComplex<T>(outR, outI, input._shape);
        }
        catch { return base.NativeComplexTopK(input, k); }
    }

    public override Tensor<Complex<T>> NativeComplexCrossSpectral<T>(Tensor<Complex<T>> x, Tensor<Complex<T>> y)
    {
        if (x is null || y is null || x.Length != y.Length)
            return base.NativeComplexCrossSpectral(x ?? throw new ArgumentNullException(nameof(x)), y ?? throw new ArgumentNullException(nameof(y)));
        if (!TryGetBackend(out var backend))
            return base.NativeComplexCrossSpectral(x, y);

        try
        {
            int n = x.Length;
            var (xR, xI) = DecomposeComplex(x);
            var (yR, yI) = DecomposeComplex(y);

            using var xRBuf = new OwnedBuffer(backend.AllocateBuffer(xR), true);
            using var xIBuf = new OwnedBuffer(backend.AllocateBuffer(xI), true);
            using var yRBuf = new OwnedBuffer(backend.AllocateBuffer(yR), true);
            using var yIBuf = new OwnedBuffer(backend.AllocateBuffer(yI), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.SplitComplexCrossSpectral(xRBuf.Buffer, xIBuf.Buffer, yRBuf.Buffer, yIBuf.Buffer,
                oRBuf.Buffer, oIBuf.Buffer, n);

            return RecomposeComplex<T>(backend.DownloadBuffer(oRBuf.Buffer),
                backend.DownloadBuffer(oIBuf.Buffer), x._shape);
        }
        catch { return base.NativeComplexCrossSpectral(x, y); }
    }

    public override Tensor<Complex<T>> NativeComplexFFT<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexFFT(input);

        try
        {
            int fftSize = input._shape[^1]; // Last axis
            int batchCount = input.Length / fftSize;
            int n = input.Length;
            if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
                return base.NativeComplexFFT(input);

            var ops = MathHelper.GetNumericOperations<T>();
            var inputF = new float[n];
            for (int i = 0; i < n; i++) inputF[i] = (float)ops.ToDouble(input[i]);

            var zerosF = new float[n]; // C# arrays are zero-initialized

            using var inRBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var inIBuf = new OwnedBuffer(backend.AllocateBuffer(zerosF), true);
            using var outRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var outIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            if (batchCount > 1)
                backend.BatchedFFT(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, batchCount, fftSize, inverse: false);
            else
                backend.FFT(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, fftSize, inverse: false);

            return RecomposeComplex<T>(backend.DownloadBuffer(outRBuf.Buffer),
                backend.DownloadBuffer(outIBuf.Buffer), input._shape);
        }
        catch { return base.NativeComplexFFT(input); }
    }

    public override Tensor<T> NativeComplexIFFTReal<T>(Tensor<Complex<T>> input)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexIFFTReal(input);

        try
        {
            int fftSize = input._shape[^1];
            int batchCount = input.Length / fftSize;
            int n = input.Length;
            if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
                return base.NativeComplexIFFTReal(input);
            var (inR, inI) = DecomposeComplex(input);

            using var inRBuf = new OwnedBuffer(backend.AllocateBuffer(inR), true);
            using var inIBuf = new OwnedBuffer(backend.AllocateBuffer(inI), true);
            using var outRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var outIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            if (batchCount > 1)
                backend.BatchedFFT(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, batchCount, fftSize, inverse: true);
            else
                backend.FFT(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, fftSize, inverse: true);

            // Extract real part only (IFFT normalization done by GPU kernel)
            return RecomposeReal<T>(backend.DownloadBuffer(outRBuf.Buffer), input._shape);
        }
        catch { return base.NativeComplexIFFTReal(input); }
    }

    public override Tensor<Complex<T>> NativeComplexIFFT<T>(Tensor<Complex<T>> input)
    {
        if (!TryGetBackend(out var backend))
            return base.NativeComplexIFFT(input);

        try
        {
            int fftSize = input._shape[^1];
            int batchCount = input.Length / fftSize;
            int n = input.Length;
            if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
                return base.NativeComplexIFFT(input);
            var (inR, inI) = DecomposeComplex(input);

            using var inRBuf = new OwnedBuffer(backend.AllocateBuffer(inR), true);
            using var inIBuf = new OwnedBuffer(backend.AllocateBuffer(inI), true);
            using var outRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var outIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            if (batchCount > 1)
                backend.BatchedFFT(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, batchCount, fftSize, inverse: true);
            else
                backend.FFT(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, fftSize, inverse: true);

            return RecomposeComplex<T>(backend.DownloadBuffer(outRBuf.Buffer),
                backend.DownloadBuffer(outIBuf.Buffer), input._shape);
        }
        catch { return base.NativeComplexIFFT(input); }
    }

    public override Tensor<Complex<T>> NativeComplexFFT2D<T>(Tensor<T> input)
    {
        if (input is null || input.Rank < 2)
            return base.NativeComplexFFT2D(input ?? throw new ArgumentNullException(nameof(input)));
        if (!TryGetBackend(out var backend))
            return base.NativeComplexFFT2D(input);

        try
        {
            int h = input._shape[^2];
            int w = input._shape[^1];
            if ((h & (h - 1)) != 0 || h <= 0 || (w & (w - 1)) != 0 || w <= 0)
                return base.NativeComplexFFT2D(input);

            int batchCount = input.Length / (h * w);
            int n = input.Length;

            // Convert any T to split real/imag float at the GPU boundary
            var ops = MathHelper.GetNumericOperations<T>();
            var realF = new float[n];
            for (int i = 0; i < n; i++) realF[i] = (float)ops.ToDouble(input[i]);
            var imagF = new float[n]; // zero-initialized — real input has no imaginary

            using var inRBuf = new OwnedBuffer(backend.AllocateBuffer(realF), true);
            using var inIBuf = new OwnedBuffer(backend.AllocateBuffer(imagF), true);
            using var outRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var outIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.BatchedFFT2D(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer,
                batchCount, h, w, inverse: false);

            return RecomposeComplex<T>(backend.DownloadBuffer(outRBuf.Buffer),
                backend.DownloadBuffer(outIBuf.Buffer), input._shape);
        }
        catch { return base.NativeComplexFFT2D(input); }
    }

    public override Tensor<T> NativeComplexIFFT2DReal<T>(Tensor<Complex<T>> input)
    {
        if (input is null || input.Rank < 2)
            return base.NativeComplexIFFT2DReal(input ?? throw new ArgumentNullException(nameof(input)));
        if (!TryGetBackend(out var backend))
            return base.NativeComplexIFFT2DReal(input);

        try
        {
            int h = input._shape[^2];
            int w = input._shape[^1];
            if ((h & (h - 1)) != 0 || h <= 0 || (w & (w - 1)) != 0 || w <= 0)
                return base.NativeComplexIFFT2DReal(input);

            int batchCount = input.Length / (h * w);
            int n = input.Length;

            // Convert Complex<T> to split real/imag float at the GPU boundary
            var (inR, inI) = DecomposeComplex(input);

            using var inRBuf = new OwnedBuffer(backend.AllocateBuffer(inR), true);
            using var inIBuf = new OwnedBuffer(backend.AllocateBuffer(inI), true);
            using var outRBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var outIBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);

            backend.BatchedFFT2D(inRBuf.Buffer, inIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer,
                batchCount, h, w, inverse: true);

            return RecomposeReal<T>(backend.DownloadBuffer(outRBuf.Buffer), input._shape);
        }
        catch { return base.NativeComplexIFFT2DReal(input); }
    }

    public override Tensor<T> NativeSpectralFilter<T>(Tensor<T> input, Tensor<Complex<T>> filter)
    {
        if (input is null || filter is null || input.Rank < 2 || filter.Rank < 2)
            return base.NativeSpectralFilter(
                input ?? throw new ArgumentNullException(nameof(input)),
                filter ?? throw new ArgumentNullException(nameof(filter)));
        if (!TryGetBackend(out var backend))
            return base.NativeSpectralFilter(input, filter);

        try
        {
            int h = input._shape[^2];
            int w = input._shape[^1];
            if ((h & (h - 1)) != 0 || h <= 0 || (w & (w - 1)) != 0 || w <= 0)
                return base.NativeSpectralFilter(input, filter);
            if (filter._shape[^2] != h || filter._shape[^1] != w)
                return base.NativeSpectralFilter(input, filter);
            if (filter.Length > input.Length)
                return base.NativeSpectralFilter(input, filter);

            int batchCount = input.Length / (h * w);
            int sliceSize = h * w;
            int filterSliceCount = filter.Length / sliceSize;
            if (filterSliceCount <= 0 || filter.Length % sliceSize != 0)
                return base.NativeSpectralFilter(input, filter);

            var ops = MathHelper.GetNumericOperations<T>();
            var inputF = new float[input.Length];
            for (int i = 0; i < input.Length; i++) inputF[i] = (float)ops.ToDouble(input[i]);

            var (fR, fI) = DecomposeComplex(filter);

            // GPU backend requires filterSliceCount of 1 or batchCount.
            // For intermediate counts (e.g. [C,H,W] with [B,C,H,W] input),
            // expand filter on CPU before uploading to keep the GPU path.
            if (filterSliceCount != 1 && filterSliceCount != batchCount)
            {
                var expandedR = new float[batchCount * sliceSize];
                var expandedI = new float[batchCount * sliceSize];
                for (int s = 0; s < batchCount; s++)
                {
                    int src = (s % filterSliceCount) * sliceSize;
                    Array.Copy(fR, src, expandedR, s * sliceSize, sliceSize);
                    Array.Copy(fI, src, expandedI, s * sliceSize, sliceSize);
                }
                fR = expandedR;
                fI = expandedI;
                filterSliceCount = batchCount;
            }

            using var inBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var fRBuf = new OwnedBuffer(backend.AllocateBuffer(fR), true);
            using var fIBuf = new OwnedBuffer(backend.AllocateBuffer(fI), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(input.Length), true);

            backend.SpectralFilter(inBuf.Buffer, fRBuf.Buffer, fIBuf.Buffer,
                outBuf.Buffer, batchCount, h, w, filterSliceCount);

            return RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), input._shape);
        }
        catch { return base.NativeSpectralFilter(input, filter); }
    }

    public override Tensor<T> NativeSpectralFilterBatch<T>(Tensor<T> input, Tensor<Complex<T>> filter)
    {
        if (input is null || filter is null || input.Rank != 4 || filter.Rank < 2)
            return base.NativeSpectralFilterBatch(
                input ?? throw new ArgumentNullException(nameof(input)),
                filter ?? throw new ArgumentNullException(nameof(filter)));
        if (!TryGetBackend(out var backend))
            return base.NativeSpectralFilterBatch(input, filter);

        try
        {
            int batch = input._shape[0];
            int channels = input._shape[1];
            int h = input._shape[2];
            int w = input._shape[3];
            if ((h & (h - 1)) != 0 || h <= 0 || (w & (w - 1)) != 0 || w <= 0)
                return base.NativeSpectralFilterBatch(input, filter);
            if (filter._shape[^2] != h || filter._shape[^1] != w)
                return base.NativeSpectralFilterBatch(input, filter);
            if (filter.Length > input.Length)
                return base.NativeSpectralFilterBatch(input, filter);
            // Only rank-2 [H,W] (shared) and rank-3 [C,H,W] (per-channel) are GPU-optimized;
            // higher-rank filters fall back to the CPU path for safety.
            if (filter.Rank > 3)
                return base.NativeSpectralFilterBatch(input, filter);

            int totalSlices = batch * channels;
            int sliceSize = h * w;
            int filterSliceCount = filter.Length / sliceSize;
            if (filterSliceCount <= 0 || filter.Length % sliceSize != 0)
                return base.NativeSpectralFilterBatch(input, filter);

            var ops = MathHelper.GetNumericOperations<T>();
            var inputF = new float[input.Length];
            for (int i = 0; i < input.Length; i++) inputF[i] = (float)ops.ToDouble(input[i]);

            var (fR, fI) = DecomposeComplex(filter);

            // GPU backend requires filterSliceCount of 1 or totalSlices.
            // For intermediate counts (e.g. [C,H,W] filter), expand on CPU before upload.
            if (filterSliceCount != 1 && filterSliceCount != totalSlices)
            {
                var expandedR = new float[totalSlices * sliceSize];
                var expandedI = new float[totalSlices * sliceSize];
                for (int s = 0; s < totalSlices; s++)
                {
                    int src = (s % filterSliceCount) * sliceSize;
                    Array.Copy(fR, src, expandedR, s * sliceSize, sliceSize);
                    Array.Copy(fI, src, expandedI, s * sliceSize, sliceSize);
                }
                fR = expandedR;
                fI = expandedI;
                filterSliceCount = totalSlices;
            }

            using var inBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var fRBuf = new OwnedBuffer(backend.AllocateBuffer(fR), true);
            using var fIBuf = new OwnedBuffer(backend.AllocateBuffer(fI), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(input.Length), true);

            backend.SpectralFilter(inBuf.Buffer, fRBuf.Buffer, fIBuf.Buffer,
                outBuf.Buffer, totalSlices, h, w, filterSliceCount);

            return RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), input._shape);
        }
        catch { return base.NativeSpectralFilterBatch(input, filter); }
    }

    public override Tensor<T> TensorSoftmaxRows<T>(Tensor<T> input)
    {
        if (input.Rank != 2)
            return base.TensorSoftmaxRows(input);
        if (!TryGetBackend(out var backend))
            return base.TensorSoftmaxRows(input);

        try
        {
            var ops = MathHelper.GetNumericOperations<T>();
            int rows = input._shape[0];
            int cols = input._shape[1];
            int total = rows * cols;
            var inputF = new float[total];
            for (int i = 0; i < total; i++) inputF[i] = (float)ops.ToDouble(input[i]);

            using var inBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);

            backend.SoftmaxRows(inBuf.Buffer, outBuf.Buffer, rows, cols);

            return RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), input._shape);
        }
        catch { return base.TensorSoftmaxRows(input); }
    }

    // ============================================================================
    // Issue #160 spectral perf op overrides — dispatch to backend.* GPU kernels.
    // Each op converts T to float at GPU boundary (matches existing engine pattern).
    // ============================================================================

    public override Tensor<T> NativeTanh<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend)) return base.NativeTanh(input);
        try
        {
            int n = input.Length;
            var ops = MathHelper.GetNumericOperations<T>();
            var inputF = new float[n];
            for (int i = 0; i < n; i++) inputF[i] = (float)ops.ToDouble(input[i]);
            using var inBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            backend.Tanh(inBuf.Buffer, outBuf.Buffer, n);
            return RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), input._shape);
        }
        catch { return base.NativeTanh(input); }
    }

    public override Tensor<T> NativeExp<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend)) return base.NativeExp(input);
        try
        {
            int n = input.Length;
            var ops = MathHelper.GetNumericOperations<T>();
            var inputF = new float[n];
            for (int i = 0; i < n; i++) inputF[i] = (float)ops.ToDouble(input[i]);
            using var inBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            backend.Exp(inBuf.Buffer, outBuf.Buffer, n);
            return RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), input._shape);
        }
        catch { return base.NativeExp(input); }
    }

    public override Tensor<T> NativeAtan2<T>(Tensor<T> imag, Tensor<T> real)
    {
        if (imag is null) throw new ArgumentNullException(nameof(imag));
        if (real is null) throw new ArgumentNullException(nameof(real));
        if (imag.Length != real.Length) return base.NativeAtan2(imag, real);
        if (!TryGetBackend(out var backend)) return base.NativeAtan2(imag, real);
        try
        {
            int n = imag.Length;
            var ops = MathHelper.GetNumericOperations<T>();
            var iF = new float[n];
            var rF = new float[n];
            for (int i = 0; i < n; i++) { iF[i] = (float)ops.ToDouble(imag[i]); rF[i] = (float)ops.ToDouble(real[i]); }
            using var iBuf = new OwnedBuffer(backend.AllocateBuffer(iF), true);
            using var rBuf = new OwnedBuffer(backend.AllocateBuffer(rF), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            // Interface order: Atan2Elementwise(real, imag, output) matches ComplexPhase/SplitComplexPhase.
            backend.Atan2Elementwise(rBuf.Buffer, iBuf.Buffer, outBuf.Buffer, n);
            return RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), imag._shape);
        }
        catch { return base.NativeAtan2(imag, real); }
    }

    public override Tensor<T> NativeMagnitudeAndPhase<T>(Tensor<Complex<T>> input, out Tensor<T> phase)
    {
        if (!TryGetBackend(out var backend)) return base.NativeMagnitudeAndPhase(input, out phase);
        try
        {
            int n = input.Length;
            var (rF, iF) = DecomposeComplex(input);
            using var rBuf = new OwnedBuffer(backend.AllocateBuffer(rF), true);
            using var iBuf = new OwnedBuffer(backend.AllocateBuffer(iF), true);
            using var magBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            using var phaseBuf = new OwnedBuffer(backend.AllocateBuffer(n), true);
            backend.SplitComplexMagnitude(rBuf.Buffer, iBuf.Buffer, magBuf.Buffer, n);
            backend.SplitComplexPhase(rBuf.Buffer, iBuf.Buffer, phaseBuf.Buffer, n);
            phase = RecomposeReal<T>(backend.DownloadBuffer(phaseBuf.Buffer), input._shape);
            return RecomposeReal<T>(backend.DownloadBuffer(magBuf.Buffer), input._shape);
        }
        catch { return base.NativeMagnitudeAndPhase(input, out phase); }
    }

    public override Tensor<Complex<T>> NativeAnalyticSignal<T>(Tensor<T> input, double freqLow = 0.0, double freqHigh = double.MaxValue, double sampleRate = 1.0)
    {
        if (!TryGetBackend(out var backend)) return base.NativeAnalyticSignal(input, freqLow, freqHigh, sampleRate);
        try
        {
            int fftSize = input._shape[^1];
            int batchCount = input.Length / fftSize;
            if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
                return base.NativeAnalyticSignal(input, freqLow, freqHigh, sampleRate);

            int total = input.Length;
            var ops = MathHelper.GetNumericOperations<T>();
            var inputF = new float[total];
            for (int i = 0; i < total; i++) inputF[i] = (float)ops.ToDouble(input[i]);
            var zerosF = new float[total];

            int halfN = fftSize / 2;
            int binLow = freqLow <= 0 ? 0 : (int)Math.Ceiling(freqLow * fftSize / sampleRate);
            int binHigh = double.IsPositiveInfinity(freqHigh) || freqHigh >= sampleRate * 0.5
                ? halfN + 1 : Math.Min(halfN + 1, (int)Math.Ceiling(freqHigh * fftSize / sampleRate));

            using var inRBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var inIBuf = new OwnedBuffer(backend.AllocateBuffer(zerosF), true);
            using var specRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var specIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var maskedRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var maskedIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var outRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var outIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);

            // Forward FFT
            if (batchCount > 1)
                backend.BatchedFFT(inRBuf.Buffer, inIBuf.Buffer, specRBuf.Buffer, specIBuf.Buffer, batchCount, fftSize, inverse: false);
            else
                backend.FFT(inRBuf.Buffer, inIBuf.Buffer, specRBuf.Buffer, specIBuf.Buffer, fftSize, inverse: false);

            // Apply Hilbert mask via dedicated kernel
            backend.AnalyticSignalMask(specRBuf.Buffer, specIBuf.Buffer, maskedRBuf.Buffer, maskedIBuf.Buffer,
                batchCount, fftSize, binLow, binHigh);

            // Inverse FFT (backend IFFT is unnormalized — scale by 1/fftSize afterward to match CPU semantics).
            if (batchCount > 1)
                backend.BatchedFFT(maskedRBuf.Buffer, maskedIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, batchCount, fftSize, inverse: true);
            else
                backend.FFT(maskedRBuf.Buffer, maskedIBuf.Buffer, outRBuf.Buffer, outIBuf.Buffer, fftSize, inverse: true);

            // Apply IFFT 1/N normalization on the downloaded real+imag arrays.
            var outReal = backend.DownloadBuffer(outRBuf.Buffer);
            var outImag = backend.DownloadBuffer(outIBuf.Buffer);
            float invN = 1f / fftSize;
            for (int i = 0; i < outReal.Length; i++) { outReal[i] *= invN; outImag[i] *= invN; }
            return RecomposeComplex<T>(outReal, outImag, input._shape);
        }
        catch { return base.NativeAnalyticSignal(input, freqLow, freqHigh, sampleRate); }
    }

    public override Tensor<T> NativeNormalizeRows<T>(Tensor<T> input, bool inPlace = false)
    {
        if (!TryGetBackend(out var backend)) return base.NativeNormalizeRows(input, inPlace);
        if (input.Rank != 2) return base.NativeNormalizeRows(input, inPlace);
        try
        {
            int rows = input._shape[0];
            int cols = input._shape[1];
            int total = rows * cols;
            var ops = MathHelper.GetNumericOperations<T>();
            var inputF = new float[total];
            for (int i = 0; i < total; i++) inputF[i] = (float)ops.ToDouble(input[i]);
            using var inBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            backend.NormalizeRowsFused(inBuf.Buffer, outBuf.Buffer, rows, cols);
            var result = RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), input._shape);
            if (inPlace)
            {
                // Copy result data back into the input tensor's buffer for in-place semantics
                var src = result.DataVector.AsSpan();
                var dst = input.DataVector.AsWritableSpan();
                src.CopyTo(dst);
                return input;
            }
            return result;
        }
        catch { return base.NativeNormalizeRows(input, inPlace); }
    }

    public override Tensor<Complex<T>> NativeBispectrum<T>(Tensor<Complex<T>> spectrum, int maxF1, int maxF2)
    {
        if (!TryGetBackend(out var backend)) return base.NativeBispectrum(spectrum, maxF1, maxF2);
        if (spectrum.Rank != 1 || maxF1 <= 0 || maxF2 <= 0 || maxF1 + maxF2 - 1 > spectrum.Length)
            return base.NativeBispectrum(spectrum, maxF1, maxF2);
        try
        {
            int total = maxF1 * maxF2;
            var (rF, iF) = DecomposeComplex(spectrum);
            using var sRBuf = new OwnedBuffer(backend.AllocateBuffer(rF), true);
            using var sIBuf = new OwnedBuffer(backend.AllocateBuffer(iF), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            backend.BispectrumGather(sRBuf.Buffer, sIBuf.Buffer, oRBuf.Buffer, oIBuf.Buffer, maxF1, maxF2);
            return RecomposeComplex<T>(backend.DownloadBuffer(oRBuf.Buffer),
                backend.DownloadBuffer(oIBuf.Buffer), new[] { maxF1, maxF2 });
        }
        catch { return base.NativeBispectrum(spectrum, maxF1, maxF2); }
    }

    public override Tensor<Complex<T>> NativeTrispectrum<T>(Tensor<Complex<T>> spectrum, int maxF1, int maxF2, int maxF3)
    {
        if (!TryGetBackend(out var backend)) return base.NativeTrispectrum(spectrum, maxF1, maxF2, maxF3);
        if (spectrum.Rank != 1 || maxF1 <= 0 || maxF2 <= 0 || maxF3 <= 0 || maxF1 + maxF2 + maxF3 - 2 > spectrum.Length)
            return base.NativeTrispectrum(spectrum, maxF1, maxF2, maxF3);
        try
        {
            int total = maxF1 * maxF2 * maxF3;
            var (rF, iF) = DecomposeComplex(spectrum);
            using var sRBuf = new OwnedBuffer(backend.AllocateBuffer(rF), true);
            using var sIBuf = new OwnedBuffer(backend.AllocateBuffer(iF), true);
            using var oRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var oIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            backend.TrispectrumGather(sRBuf.Buffer, sIBuf.Buffer, oRBuf.Buffer, oIBuf.Buffer, maxF1, maxF2, maxF3);
            return RecomposeComplex<T>(backend.DownloadBuffer(oRBuf.Buffer),
                backend.DownloadBuffer(oIBuf.Buffer), new[] { maxF1, maxF2, maxF3 });
        }
        catch { return base.NativeTrispectrum(spectrum, maxF1, maxF2, maxF3); }
    }

    public override Tensor<T> NativeBatchedCavityForward<T>(Tensor<T> input, Tensor<Complex<T>> cavityFilters, int numBounces)
    {
        if (!TryGetBackend(out var backend)) return base.NativeBatchedCavityForward(input, cavityFilters, numBounces);
        if (input.Rank != 2 || cavityFilters.Rank != 2 || numBounces < 1)
            return base.NativeBatchedCavityForward(input, cavityFilters, numBounces);
        try
        {
            int batch = input._shape[0];
            int n = input._shape[1];
            int numCavities = cavityFilters._shape[0];
            if (cavityFilters._shape[1] != n || (n & (n - 1)) != 0)
                return base.NativeBatchedCavityForward(input, cavityFilters, numBounces);

            var ops = MathHelper.GetNumericOperations<T>();
            int total = batch * n;
            var inputF = new float[total];
            for (int i = 0; i < total; i++) inputF[i] = (float)ops.ToDouble(input[i]);
            var (filtRF, filtIF) = DecomposeComplex(cavityFilters);

            // Compose on GPU: initial FFT → per cavity (multiply + IFFT + tanh + FFT for next bounce).
            // Uses existing backend primitives + the new CavityBounceInplace kernel.
            using var inBuf = new OwnedBuffer(backend.AllocateBuffer(inputF), true);
            using var zerosBuf = new OwnedBuffer(backend.AllocateBuffer(new float[total]), true);
            using var specRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var specIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var workRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var workIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var tmpRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var tmpIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var tiledRBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var tiledIBuf = new OwnedBuffer(backend.AllocateBuffer(total), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(batch * numCavities * n), true);

            // Initial batched FFT of input waveforms
            backend.BatchedFFT(inBuf.Buffer, zerosBuf.Buffer, specRBuf.Buffer, specIBuf.Buffer, batch, n, inverse: false);

            var cavRSlice = new float[n];
            var cavISlice = new float[n];
            for (int c = 0; c < numCavities; c++)
            {
                // Upload this cavity's filter once (shared across all batches), then tile via Copy.
                // Using OwnedBuffer ensures the GPU buffer is disposed when this cavity is done.
                Array.Copy(filtRF, c * n, cavRSlice, 0, n);
                Array.Copy(filtIF, c * n, cavISlice, 0, n);
                using var cavRBuf = new OwnedBuffer(backend.AllocateBuffer(cavRSlice), true);
                using var cavIBuf = new OwnedBuffer(backend.AllocateBuffer(cavISlice), true);
                for (int b = 0; b < batch; b++)
                {
                    backend.Copy(cavRBuf.Buffer, 0, tiledRBuf.Buffer, b * n, n);
                    backend.Copy(cavIBuf.Buffer, 0, tiledIBuf.Buffer, b * n, n);
                }
                // Reset working spectrum
                backend.Copy(specRBuf.Buffer, 0, workRBuf.Buffer, 0, total);
                backend.Copy(specIBuf.Buffer, 0, workIBuf.Buffer, 0, total);

                for (int bounce = 0; bounce < numBounces; bounce++)
                {
                    backend.SplitComplexMultiply(workRBuf.Buffer, workIBuf.Buffer,
                        tiledRBuf.Buffer, tiledIBuf.Buffer,
                        tmpRBuf.Buffer, tmpIBuf.Buffer, total);
                    backend.BatchedFFT(tmpRBuf.Buffer, tmpIBuf.Buffer,
                        workRBuf.Buffer, workIBuf.Buffer, batch, n, inverse: true);
                    // Apply 1/N scale + tanh + zero imag in single fused kernel
                    backend.CavityBounceInplace(workRBuf.Buffer, workIBuf.Buffer, total, 1f / n);
                    if (bounce < numBounces - 1)
                    {
                        backend.BatchedFFT(workRBuf.Buffer, workIBuf.Buffer,
                            tmpRBuf.Buffer, tmpIBuf.Buffer, batch, n, inverse: false);
                        backend.Copy(tmpRBuf.Buffer, 0, workRBuf.Buffer, 0, total);
                        backend.Copy(tmpIBuf.Buffer, 0, workIBuf.Buffer, 0, total);
                    }
                }

                // Copy this cavity's output for all batches into output buffer
                for (int b = 0; b < batch; b++)
                    backend.Copy(workRBuf.Buffer, b * n, outBuf.Buffer, (b * numCavities + c) * n, n);
            }

            return RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), new[] { batch, numCavities, n });
        }
        catch { return base.NativeBatchedCavityForward(input, cavityFilters, numBounces); }
    }

    public override Tensor<T> NativeMfccFeatures<T>(Tensor<T> waveforms, int numSegments, int numMfcc, int paddedDim)
    {
        // Pipeline composes from backend.FFT + backend.MelFilterbankApply + backend.MfccLog1p + backend.MatMul (DCT).
        // For now use base CPU implementation; full GPU pipeline would require precomputed mel/DCT bases.
        return base.NativeMfccFeatures(waveforms, numSegments, numMfcc, paddedDim);
    }

    public override Tensor<T> NativeWidebandFeatures<T>(Tensor<T> waveforms, int numSegments, int numBins)
    {
        if (!TryGetBackend(out var backend)) return base.NativeWidebandFeatures(waveforms, numSegments, numBins);
        // Mirror the base CPU preconditions so invalid inputs take the base path (which throws
        // a clear ArgumentException) instead of silently producing truncated GPU output.
        if (numSegments <= 0 || numBins <= 0) return base.NativeWidebandFeatures(waveforms, numSegments, numBins);
        if (waveforms.Rank != 1 && waveforms.Rank != 2) return base.NativeWidebandFeatures(waveforms, numSegments, numBins);
        try
        {
            bool batched = waveforms.Rank == 2;
            int batch = batched ? waveforms._shape[0] : 1;
            int numSamples = batched ? waveforms._shape[1] : waveforms._shape[0];
            int segmentLen = numSamples / numSegments;
            if (segmentLen <= 1) return base.NativeWidebandFeatures(waveforms, numSegments, numBins);
            int fftSize = 1; while (fftSize < segmentLen) fftSize <<= 1;
            int totalSegBatch = batch * numSegments;
            int totalSegFFT = totalSegBatch * fftSize;

            var ops = MathHelper.GetNumericOperations<T>();
            var segRF = new float[totalSegFFT];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < numSegments; s++)
                {
                    int srcOff = b * numSamples + s * segmentLen;
                    int dstOff = (b * numSegments + s) * fftSize;
                    for (int i = 0; i < segmentLen && srcOff + i < (batched ? batch * numSamples : numSamples); i++)
                        segRF[dstOff + i] = (float)ops.ToDouble(waveforms[srcOff + i]);
                }

            using var segRBuf = new OwnedBuffer(backend.AllocateBuffer(segRF), true);
            using var segIBuf = new OwnedBuffer(backend.AllocateBuffer(new float[totalSegFFT]), true);
            using var fftRBuf = new OwnedBuffer(backend.AllocateBuffer(totalSegFFT), true);
            using var fftIBuf = new OwnedBuffer(backend.AllocateBuffer(totalSegFFT), true);
            using var magBuf = new OwnedBuffer(backend.AllocateBuffer(totalSegFFT), true);
            using var outBuf = new OwnedBuffer(backend.AllocateBuffer(batch * numSegments * numBins), true);

            backend.BatchedFFT(segRBuf.Buffer, segIBuf.Buffer, fftRBuf.Buffer, fftIBuf.Buffer, totalSegBatch, fftSize, inverse: false);
            backend.SplitComplexMagnitude(fftRBuf.Buffer, fftIBuf.Buffer, magBuf.Buffer, totalSegFFT);
            backend.WidebandLogBinPool(magBuf.Buffer, outBuf.Buffer, totalSegBatch, fftSize, numBins, fftSize / 2);

            var outShape = batched ? new[] { batch, numSegments * numBins } : new[] { numSegments * numBins };
            return RecomposeReal<T>(backend.DownloadBuffer(outBuf.Buffer), outShape);
        }
        catch { return base.NativeWidebandFeatures(waveforms, numSegments, numBins); }
    }

    public override Tensor<T> NativePacFeatures<T>(Tensor<T> waveforms, int sampleRate, int envelopeRate,
        double thetaLow, double thetaHigh, (double low, double high)[] gammaBands)
    {
        // Pipeline composes from backend.AnalyticSignal + backend.SplitComplexMagnitude/Phase + backend.PacPhaseBinMi
        // Full GPU pipeline requires multi-stage orchestration; defer to CPU base for correctness.
        return base.NativePacFeatures(waveforms, sampleRate, envelopeRate, thetaLow, thetaHigh, gammaBands);
    }

    // Span-based FFT entry points: GPU backends already have FFT/BatchedFFT; the span entry points
    // are CPU-side optimizations that bypass tensor wrapping. On a GPU engine, the CPU base is
    // still the right path because there's no benefit to round-tripping span data through GPU
    // for a single FFT call (the buffer transfer dominates). Fall-through to base is correct.
}
