using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Caches compiled computation graphs keyed by input shape + graph structure hash.
/// Avoids recompilation when the same model is called with the same input dimensions.
/// </summary>
/// <remarks>
/// <para>
/// Similar to torch.compile's cache: the first call with a given shape captures
/// and compiles the graph. Subsequent calls with the same shape reuse the compiled plan.
/// When input shapes change (e.g., different batch sizes), a new graph is compiled
/// and cached separately.
/// </para>
/// </remarks>
public sealed class CompiledGraphCache
{
    private readonly ConcurrentDictionary<string, CacheEntry> _cache = new();
    private long _hits;
    private long _misses;

    /// <summary>Number of cached compiled graphs.</summary>
    public int CachedGraphCount => _cache.Count;

    /// <summary>Cache hit count.</summary>
    public long Hits => _hits;

    /// <summary>Cache miss count.</summary>
    public long Misses => _misses;

    /// <summary>Hit ratio (0.0 to 1.0).</summary>
    public double HitRatio => _hits + _misses > 0 ? (double)_hits / (_hits + _misses) : 0;

    /// <summary>
    /// Gets a compiled memory plan for the given graph, or null if not cached.
    /// </summary>
    /// <param name="graph">The computation graph to look up.</param>
    /// <param name="inputShapes">Shapes of external input tensors.</param>
    /// <returns>Cached memory plan, or null if not found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MemoryPlanner.MemoryPlan? TryGet(ComputationGraph graph, int[][] inputShapes)
    {
        string key = ComputeKey(graph, inputShapes);
        if (_cache.TryGetValue(key, out var entry))
        {
            Interlocked.Increment(ref _hits);
            return entry.Plan;
        }

        Interlocked.Increment(ref _misses);
        return null;
    }

    /// <summary>
    /// Gets or creates a compiled memory plan for the given graph.
    /// Thread-safe: only one thread compiles if multiple threads miss simultaneously.
    /// </summary>
    /// <param name="graph">The finalized computation graph.</param>
    /// <param name="inputShapes">Shapes of external input tensors.</param>
    /// <returns>The compiled memory plan (cached or freshly compiled).</returns>
    public MemoryPlanner.MemoryPlan GetOrCompile(ComputationGraph graph, int[][] inputShapes)
    {
        string key = ComputeKey(graph, inputShapes);

        if (_cache.TryGetValue(key, out var existing))
        {
            Interlocked.Increment(ref _hits);
            return existing.Plan;
        }

        Interlocked.Increment(ref _misses);

        // Compile the graph
        var plan = graph.Optimize();
        var entry = new CacheEntry(plan, DateTime.UtcNow);
        _cache.TryAdd(key, entry);

        return plan;
    }

    /// <summary>
    /// Evicts all cached graphs. Call when the model architecture changes.
    /// </summary>
    public void Clear()
    {
        _cache.Clear();
    }

    /// <summary>
    /// Evicts cached graphs that haven't been used since the given cutoff.
    /// </summary>
    /// <param name="cutoff">Evict entries created before this time.</param>
    /// <returns>Number of entries evicted.</returns>
    public int EvictOlderThan(DateTime cutoff)
    {
        int evicted = 0;
        foreach (var kvp in _cache)
        {
            if (kvp.Value.CreatedAt < cutoff)
            {
                if (_cache.TryRemove(kvp.Key, out _))
                    evicted++;
            }
        }
        return evicted;
    }

    /// <summary>
    /// Computes a cache key from graph structure + input shapes.
    /// </summary>
    private static string ComputeKey(ComputationGraph graph, int[][] inputShapes)
    {
        var sb = new StringBuilder(256);

        // Include input shapes in the key
        foreach (var shape in inputShapes)
        {
            sb.Append('I');
            foreach (int dim in shape)
            {
                sb.Append(dim);
                sb.Append(',');
            }
            sb.Append('|');
        }

        // Include graph structure: op types and output shapes
        foreach (var node in graph.Nodes)
        {
            if (node.IsInput) continue;

            sb.Append((int)node.Type);
            sb.Append(':');
            foreach (int dim in node.OutputShape)
            {
                sb.Append(dim);
                sb.Append(',');
            }
            foreach (int inp in node.InputIds)
            {
                sb.Append(inp);
                sb.Append('+');
            }
            sb.Append(';');
        }

        // Hash the string for a compact key
#if NET5_0_OR_GREATER
        var bytes = Encoding.UTF8.GetBytes(sb.ToString());
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash);
#else
        using var sha = SHA256.Create();
        var bytes = Encoding.UTF8.GetBytes(sb.ToString());
        var hash = sha.ComputeHash(bytes);
        return BitConverter.ToString(hash).Replace("-", "");
#endif
    }

    private sealed class CacheEntry
    {
        public MemoryPlanner.MemoryPlan Plan { get; }
        public DateTime CreatedAt { get; }

        public CacheEntry(MemoryPlanner.MemoryPlan plan, DateTime createdAt)
        {
            Plan = plan;
            CreatedAt = createdAt;
        }
    }
}
