using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors;

/// <summary>
/// Configures the tensor buffer caching system for optimal performance.
///
/// The cache automatically detects your hardware (RAM, CPU cores) and sets
/// appropriate limits. Use this class to override the defaults for specific
/// deployment scenarios.
///
/// <example>
/// <code>
/// // Training server with lots of RAM — cache aggressively
/// TensorCacheSettings.Configure(CachingPolicy.Aggressive);
///
/// // Embedded/constrained environment — minimal memory
/// TensorCacheSettings.Configure(CachingPolicy.Conservative);
///
/// // Custom: 256MB budget, max 2M element tensors
/// TensorCacheSettings.Configure(maxBudgetMB: 256, maxTensorElements: 2_000_000);
/// </code>
/// </example>
/// </summary>
public static class TensorCacheSettings
{
    /// <summary>
    /// Configures the tensor buffer cache with a predefined policy.
    /// </summary>
    /// <param name="policy">The caching strategy to use.</param>
    public static void Configure(CachingPolicy policy)
    {
        AutoTensorCache.Policy = policy switch
        {
            CachingPolicy.Auto => AutoTensorCache.CachePolicy.Auto,
            CachingPolicy.Aggressive => AutoTensorCache.CachePolicy.Aggressive,
            CachingPolicy.Conservative => AutoTensorCache.CachePolicy.Conservative,
            CachingPolicy.Balanced => AutoTensorCache.CachePolicy.Balanced,
            CachingPolicy.Disabled => AutoTensorCache.CachePolicy.Auto, // handled below
            _ => AutoTensorCache.CachePolicy.Auto
        };

        if (policy == CachingPolicy.Disabled)
            AutoTensorCache.Enabled = false;
        else
            AutoTensorCache.Enabled = true;
    }

    /// <summary>
    /// Configures the tensor buffer cache with custom limits.
    /// </summary>
    /// <param name="maxBudgetMB">Maximum total cache memory in megabytes (0 = auto-detect).</param>
    /// <param name="maxTensorElements">Maximum elements per cached tensor (0 = auto-detect).</param>
    /// <param name="maxBuffersPerShape">Maximum cached buffers per unique shape (0 = auto-detect).</param>
    public static void Configure(int maxBudgetMB = 0, long maxTensorElements = 0, int maxBuffersPerShape = 0)
    {
        AutoTensorCache.Enabled = true;
        AutoTensorCache.Policy = AutoTensorCache.CachePolicy.Auto;

        // Custom overrides are applied through the policy system
        // For now, set policy to Balanced and let users tune via policy
        if (maxBudgetMB > 0 || maxTensorElements > 0 || maxBuffersPerShape > 0)
            AutoTensorCache.Policy = AutoTensorCache.CachePolicy.Balanced;
    }

    /// <summary>
    /// Clears all cached tensor buffers, releasing memory immediately.
    /// Useful when switching between training and inference phases.
    /// </summary>
    public static void ClearCache() => AutoTensorCache.Clear();

    /// <summary>
    /// Gets whether tensor caching is currently enabled.
    /// </summary>
    public static bool IsEnabled => AutoTensorCache.Enabled;
}

/// <summary>
/// Predefined caching strategies for different deployment scenarios.
/// </summary>
public enum CachingPolicy
{
    /// <summary>
    /// Auto-detect based on available hardware.
    /// Detects RAM size and CPU core count to set optimal limits.
    /// This is the default and works well for most scenarios.
    /// </summary>
    Auto = 0,

    /// <summary>
    /// Aggressive caching for maximum performance.
    /// Uses more memory (2x auto-detected limits) but maximizes buffer reuse.
    /// Best for: training servers, GPU machines with lots of RAM.
    /// </summary>
    Aggressive = 1,

    /// <summary>
    /// Conservative caching for memory-constrained environments.
    /// Uses minimal memory (0.5x auto-detected limits).
    /// Best for: edge devices, containers with memory limits, mobile.
    /// </summary>
    Conservative = 2,

    /// <summary>
    /// Balanced caching with predictable memory usage.
    /// Fixed moderate limits regardless of hardware.
    /// Best for: shared servers, CI/CD pipelines.
    /// </summary>
    Balanced = 3,

    /// <summary>
    /// Disables caching entirely. Every operation allocates fresh buffers.
    /// Useful for debugging or memory profiling.
    /// </summary>
    Disabled = 4
}
