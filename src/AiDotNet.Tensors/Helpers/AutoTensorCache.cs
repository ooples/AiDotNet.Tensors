using System.Collections.Concurrent;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Thread-local auto-caching tensor pool for zero-allocation eager operations.
/// When CpuEngine needs an output tensor, it checks this cache first.
/// If a matching-shape buffer exists, it's reused instead of allocating.
///
/// This makes eager tensor ops nearly as fast as compiled plans by
/// eliminating the 400KB allocation that dominates per-op cost.
///
/// Thread-static to avoid contention. Max 2 buffers per shape to
/// limit memory while covering common input/output reuse patterns.
/// </summary>
internal static class AutoTensorCache
{
    /// <summary>Whether auto-caching is enabled (default: true).</summary>
    internal static bool Enabled { get; set; } = true;

    [ThreadStatic]
    private static ConcurrentDictionary<long, ConcurrentQueue<object>>? _pools;

    /// <summary>Max cached tensors per shape key.</summary>
    private const int MaxPerShape = 2;

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
    /// Only caches contiguous tensors with reasonable sizes.
    /// </summary>
    internal static void Return<T>(Tensor<T> tensor)
    {
        if (!Enabled || tensor == null || !tensor.IsContiguous)
            return;

        // Don't cache very large tensors (>16MB) — they'd hold too much memory
        if (tensor.Length > 4_000_000)
            return;

        var pools = _pools ??= new ConcurrentDictionary<long, ConcurrentQueue<object>>();
        long key = ComputeKey<T>(tensor._shape);
        var pool = pools.GetOrAdd(key, _ => new ConcurrentQueue<object>());

        if (pool.Count < MaxPerShape)
            pool.Enqueue(tensor);
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
}
