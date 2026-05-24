namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Diagnostic counters for the <see cref="BlasManaged"/> kernel. Returned by
/// <c>BlasManaged.GetStats()</c>; useful for measuring autotune cache hit
/// rate, JIT emission cost, and weight-pack cache effectiveness in benchmarks
/// and inference servers.
/// </summary>
public struct BlasManagedStats
{
    /// <summary>Number of GEMM calls that found a cached autotune choice for their shape.</summary>
    public long AutotuneHits;
    /// <summary>Number of GEMM calls that ran the first-call benchmark to pick a strategy.</summary>
    public long AutotuneMisses;
    /// <summary>Number of shape-specialized microkernels emitted via DynamicMethod (Phase J).</summary>
    public long JitEmissions;
    /// <summary>Number of GEMM calls that used a JIT-emitted microkernel from cache.</summary>
    public long JitCacheHits;
    /// <summary>Number of <see cref="WeightPackHandle"/> lookups served from cache without re-packing.</summary>
    public long PackCacheHits;
    /// <summary>Number of <see cref="WeightPackHandle"/> lookups that required re-packing (first call or post-MarkDirty).</summary>
    public long PackCacheMisses;
    /// <summary>Approximate bytes currently held by the weight pre-pack cache (Layer 3 of the allocator).</summary>
    public long PackCacheBytes;
}
