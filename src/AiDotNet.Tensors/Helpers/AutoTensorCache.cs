using System.Collections.Concurrent;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Thread-local auto-caching tensor pool for zero-allocation eager operations.
/// When CpuEngine needs an output tensor, it checks this cache first.
/// If a matching-shape buffer exists, it's reused instead of allocating.
///
/// Hardware-adaptive: auto-detects available memory and CPU cache sizes
/// to set optimal caching limits. Configurable via CachePolicy for
/// different deployment scenarios.
///
/// Thread-static to avoid contention. Max buffers per shape scales
/// with available memory to cover common input/output reuse patterns.
/// </summary>
internal static class AutoTensorCache
{
    /// <summary>Whether auto-caching is enabled (default: true).</summary>
    internal static bool Enabled { get; set; } = true;

    /// <summary>
    /// Cache policy controlling memory usage vs performance tradeoff.
    /// </summary>
    internal enum CachePolicy
    {
        /// <summary>Auto-detect based on hardware (default).</summary>
        Auto,
        /// <summary>Aggressive caching — uses more memory for maximum reuse.</summary>
        Aggressive,
        /// <summary>Conservative caching — minimal memory footprint.</summary>
        Conservative,
        /// <summary>Balanced — moderate caching suitable for most workloads.</summary>
        Balanced
    }

    /// <summary>Gets or sets the cache policy. Default: Auto (hardware-detected).</summary>
    internal static CachePolicy Policy { get; set; } = CachePolicy.Auto;

    [ThreadStatic]
    private static ConcurrentDictionary<long, ConcurrentQueue<object>>? _pools;

    // Hardware-detected limits (computed once)
    private static readonly int _maxPerShape;
    private static readonly long _maxElementsPerTensor;
    private static readonly long _maxTotalBudgetBytes;

    static AutoTensorCache()
    {
        // Detect available memory and set limits accordingly
        long availableMemory = GetAvailableMemoryBytes();
        int processorCount = Environment.ProcessorCount;

        // Budget: min(1GB, available_ram / 8) per thread
        // With thread-static pools, each thread gets its own budget
        _maxTotalBudgetBytes = Math.Min(1024L * 1024 * 1024, availableMemory / 8);

        // Max tensor size to cache: tensors that fit in L3-equivalent (~16MB)
        // are most beneficial to cache. Larger tensors are memory-bound anyway.
        // For systems with lots of RAM, allow up to 64MB tensors.
        if (availableMemory > 32L * 1024 * 1024 * 1024) // >32GB RAM
            _maxElementsPerTensor = 16_000_000; // ~64MB float tensors
        else if (availableMemory > 8L * 1024 * 1024 * 1024) // >8GB RAM
            _maxElementsPerTensor = 8_000_000; // ~32MB float tensors
        else
            _maxElementsPerTensor = 4_000_000; // ~16MB float tensors

        // Per-shape: keep more cached copies on systems with more cores
        // (more concurrent operations = more reuse opportunities)
        _maxPerShape = processorCount >= 16 ? 4 : processorCount >= 8 ? 3 : 2;
    }

    /// <summary>Max cached tensors per shape key (hardware-adaptive).</summary>
    internal static int MaxPerShape => Policy switch
    {
        CachePolicy.Aggressive => _maxPerShape + 2,
        CachePolicy.Conservative => 1,
        CachePolicy.Balanced => 2,
        _ => _maxPerShape // Auto
    };

    /// <summary>Max elements per cached tensor (hardware-adaptive).</summary>
    internal static long MaxElementsPerTensor => Policy switch
    {
        CachePolicy.Aggressive => _maxElementsPerTensor * 2,
        CachePolicy.Conservative => _maxElementsPerTensor / 2,
        CachePolicy.Balanced => _maxElementsPerTensor,
        _ => _maxElementsPerTensor // Auto
    };

    /// <summary>
    /// Gets a cached tensor with the given shape, or allocates a new one.
    /// The returned tensor's data is UNINITIALIZED — caller must overwrite.
    /// </summary>
    internal static Tensor<T> RentOrAllocate<T>(int[] shape)
    {
        if (!Enabled)
            return TensorAllocator.RentUninitialized<T>(shape);

        var pools = _pools ??= new ConcurrentDictionary<long, ConcurrentQueue<object>>();
        long key = ComputeKey<T>(shape);

        if (pools.TryGetValue(key, out var pool) && pool.TryDequeue(out var obj))
        {
            var tensor = (Tensor<T>)obj;
            // Verify shape matches (hash collision guard)
            if (ShapesMatch(tensor._shape, shape))
                return tensor;
            // Shape mismatch (collision) — return it back and allocate fresh
            pool.Enqueue(obj);
        }

        return TensorAllocator.RentUninitialized<T>(shape);
    }

    /// <summary>
    /// Returns a tensor to the cache for reuse by future ops.
    /// Only caches contiguous tensors within the hardware-adaptive size limit.
    /// </summary>
    internal static void Return<T>(Tensor<T> tensor)
    {
        if (!Enabled || tensor == null || !tensor.IsContiguous)
            return;

        if (tensor.Length > MaxElementsPerTensor)
            return;

        var pools = _pools ??= new ConcurrentDictionary<long, ConcurrentQueue<object>>();
        long key = ComputeKey<T>(tensor._shape);
        var pool = pools.GetOrAdd(key, _ => new ConcurrentQueue<object>());

        if (pool.Count < MaxPerShape)
            pool.Enqueue(tensor);
    }

    /// <summary>Clears all cached tensors (useful for memory pressure or testing).</summary>
    internal static void Clear()
    {
        _pools?.Clear();
    }

    /// <summary>Compute a fast hash key from shape + type.</summary>
    private static long ComputeKey<T>(int[] shape)
    {
        // FNV-1a hash for shape dimensions + type hash
        long hash = unchecked((long)0xcbf29ce484222325L);
        hash ^= typeof(T).GetHashCode();
        hash *= 0x100000001b3L;
        for (int i = 0; i < shape.Length; i++)
        {
            hash ^= shape[i];
            hash *= 0x100000001b3L;
        }
        hash ^= shape.Length;
        hash *= 0x100000001b3L;
        return hash;
    }

    private static bool ShapesMatch(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    private static long GetAvailableMemoryBytes()
    {
        try
        {
#if NET5_0_OR_GREATER
            var gcInfo = GC.GetGCMemoryInfo();
            return gcInfo.TotalAvailableMemoryBytes;
#else
            // net471 fallback: estimate from GC
            return GC.GetTotalMemory(false) * 16; // rough estimate
#endif
        }
        catch
        {
            return 8L * 1024 * 1024 * 1024; // Default: assume 8GB
        }
    }
}
