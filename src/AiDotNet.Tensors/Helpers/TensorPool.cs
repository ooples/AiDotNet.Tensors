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
    /// When true, every <see cref="Rent{T}(int[])"/> /
    /// <see cref="RentUninitialized{T}(int[])"/> call allocates a
    /// fresh <c>T[]</c> of EXACTLY <c>product(shape)</c> elements —
    /// no ArrayPool bucket padding, no thread-local cache reuse with
    /// over-sized buffers. Two tensors with byte-equal logical
    /// content always have backing arrays of identical length and
    /// content, eliminating the cross-construction-path divergence
    /// that issue #318 reports.
    ///
    /// <para>Default <c>false</c> — backwards-compatible perf
    /// behaviour. Enable when training-time and inference-time
    /// determinism contracts (state_dict round-trip, deterministic
    /// Clone) outweigh the GC-pressure cost of skipping the pool.
    /// Can also be enabled via the
    /// <c>AIDOTNET_FORCE_FRESH_ALLOCATIONS=1</c> environment
    /// variable so consumer test rigs can flip it without code
    /// changes.</para>
    ///
    /// <para>Cost when enabled: every rent allocates an exact-sized
    /// array instead of using the ArrayPool / thread-local cache.
    /// <c>Rent</c> still zeroes its values; <c>RentUninitialized</c>
    /// may skip clearing reference-free values on supported runtimes.
    /// For training loops with many small
    /// allocations this turns into Gen-0 GC pressure — the same
    /// regression the pool was designed to avoid. Don't enable
    /// globally unless you have a measured determinism need.</para>
    /// </summary>
    /// <remarks>
    /// The setter changes the process-wide default. Code that needs fresh allocations for one operation (tracing a
    /// compiled step, a determinism check) should use <see cref="ForceFreshAllocationsScope"/> instead: it applies to
    /// the calling thread only, so other threads renting tensors at the same time keep their pooling and arena reuse.
    /// </remarks>
    public static bool ForceFreshAllocations
    {
        get
        {
            var local = _threadForceFresh;
            return local == 0 ? _forceFreshDefault : local > 0;
        }
        set => _forceFreshDefault = value;
    }

    private static bool _forceFreshDefault = IsEnvTrue("AIDOTNET_FORCE_FRESH_ALLOCATIONS");

    // 0: follow the process default; 1: force fresh; -1: force pooling. Per thread, set by ForceFreshAllocationsScope.
    [ThreadStatic] private static int _threadForceFresh;

    /// <summary>
    /// Overrides <see cref="ForceFreshAllocations"/> for the calling thread until the returned scope is disposed.
    /// Scopes nest; disposing restores the previous value.
    /// </summary>
    public static IDisposable ForceFreshAllocationsScope(bool forceFresh)
    {
        var previous = _threadForceFresh;
        _threadForceFresh = forceFresh ? 1 : -1;
        return new FreshScope(previous);
    }

    private sealed class FreshScope : IDisposable
    {
        private readonly int _previous;
        private bool _disposed;
        public FreshScope(int previous) => _previous = previous;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _threadForceFresh = _previous;
        }
    }

    /// <summary>
    /// Bytes of IDLE cross-arena buffers one thread may keep: buffers whose (type, size) bucket no arena has rented or
    /// returned for more than two arena lifetimes. Active buckets (a training loop's per-step shapes) are never
    /// counted or evicted. Within the budget idle buckets survive, so a process that alternates models (an
    /// interleaved latency harness, a service hosting several models) reuses each model's buffers instead of
    /// allocating and zeroing them again every call; beyond it the least recently touched buckets are dropped first.
    /// 0 restores pure generation eviction. Default 256 MiB, or <c>AIDOTNET_ARENA_IDLE_RETENTION_MB</c>.
    /// </summary>
    public static long ArenaIdleRetentionBytes { get; set; } = EnvMegabytes("AIDOTNET_ARENA_IDLE_RETENTION_MB", 256);

    private static long EnvMegabytes(string name, long defaultMb)
    {
        var text = Environment.GetEnvironmentVariable(name);
        long mb = long.TryParse(text, out var v) && v >= 0 ? v : defaultMb;
        // Saturate rather than wrap: a huge setting means "no practical budget", never a negative one.
        return mb > long.MaxValue / (1024L * 1024L) ? long.MaxValue : mb * 1024L * 1024L;
    }

    /// <summary>
    /// Creates a zero-initialized tensor with the given shape.
    /// Large tensors use ArrayPool to reduce GC pressure; small-medium tensors
    /// use standard CLR allocation. All paths return zeroed memory.
    /// </summary>
    public static Tensor<T> Rent<T>(int[] shape)
    {
        if (!Enabled)
            return new Tensor<T>(shape);

        // ForceFreshAllocations is honoured inside TensorAllocator.Rent
        // — this keeps allocator-side bookkeeping (MemoryProfiler hooks)
        // consistent and gives one source of truth for the policy.
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

        // ForceFreshAllocations is honoured inside
        // TensorAllocator.RentUninitialized — same single source of
        // truth as the Rent path above.
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
