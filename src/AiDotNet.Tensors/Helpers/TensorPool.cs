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
    /// <para>Cost when enabled: every <c>Rent</c> goes through
    /// <c>new T[totalSize]</c> instead of the ArrayPool / thread-
    /// local cache path. For training loops with many small
    /// allocations this turns into Gen-0 GC pressure — the same
    /// regression the pool was designed to avoid. Don't enable
    /// globally unless you have a measured determinism need.</para>
    /// </summary>
    public static bool ForceFreshAllocations { get; set; } = IsEnvTrue("AIDOTNET_FORCE_FRESH_ALLOCATIONS");

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
