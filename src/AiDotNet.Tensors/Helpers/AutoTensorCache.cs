using System.Collections.Concurrent;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Thread-local auto-caching tensor pool for zero-allocation eager operations.
/// When CpuEngine needs an output tensor, it checks this cache first.
/// If a matching-shape buffer exists, it's reused instead of allocating.
///
/// Hardware-adaptive: detects L1/L2/L3 cache sizes and available RAM
/// to set optimal caching thresholds:
///   - Tensors fitting in L2: cached aggressively (4 copies per shape)
///   - Tensors fitting in L3: cached moderately (2 copies per shape)
///   - Tensors exceeding L3 but under RAM budget: cached conservatively (1 copy)
///   - Tensors exceeding RAM budget: not cached
///
/// Thread-static to avoid contention. Configurable via CachePolicy.
/// </summary>
internal static class AutoTensorCache
{
    /// <summary>Whether auto-caching is enabled (default: true).</summary>
    internal static bool Enabled { get; set; } = true;

    internal enum CachePolicy
    {
        Auto,
        Aggressive,
        Conservative,
        Balanced
    }

    internal static CachePolicy Policy { get; set; } = CachePolicy.Auto;

    [ThreadStatic]
    private static ConcurrentDictionary<long, ConcurrentQueue<object>>? _pools;

    // Track all thread-local pools so Clear() can flush all threads, not just the caller's
    private static readonly List<WeakReference<ConcurrentDictionary<long, ConcurrentQueue<object>>>> _allPools = new();

    // ═══ Hardware-detected cache hierarchy ═══
    // These are computed once at startup from actual CPU cache sizes.

    /// <summary>L1 data cache size per core in bytes (typically 32-48KB).</summary>
    internal static readonly int L1CacheBytes;

    /// <summary>L2 cache size per core in bytes (typically 256KB-1MB).</summary>
    internal static readonly int L2CacheBytes;

    /// <summary>L3 cache size (shared) in bytes (typically 4-64MB).</summary>
    internal static readonly long L3CacheBytes;

    /// <summary>L4 cache / HBM size in bytes (0 if not present; 64MB-4GB on server platforms).</summary>
    internal static readonly long L4CacheBytes;

    /// <summary>Total available RAM in bytes.</summary>
    internal static readonly long AvailableRamBytes;

    // Derived limits per cache tier
    private static readonly int _maxPerShapeL2;   // copies for tensors fitting in L2
    private static readonly int _maxPerShapeL3;   // copies for tensors fitting in L3
    private static readonly int _maxPerShapeL4;   // copies for tensors fitting in L4/HBM
    private static readonly int _maxPerShapeRam;  // copies for tensors in RAM only
    private static readonly long _maxElementsL2;  // elements that fit in L2 (per core)
    private static readonly long _maxElementsL3;  // elements that fit in L3 (shared)
    private static readonly long _maxElementsL4;  // elements that fit in L4/HBM
    private static readonly long _maxElementsRam; // absolute max elements to cache

    // User-configurable overrides (0 = use auto-detected defaults)
    internal static long MaxElementsOverride { get; set; }
    internal static int MaxBuffersPerShapeOverride { get; set; }

    static AutoTensorCache()
    {
        // Detect hardware cache hierarchy
        var (l1, l2, l3) = DetectCacheSizes();
        L1CacheBytes = l1;
        L2CacheBytes = l2;
        L3CacheBytes = l3;
        AvailableRamBytes = GetAvailableMemoryBytes();

        // Detect L4/HBM: some server platforms (Sapphire Rapids, etc.) have
        // 64MB-4GB of on-package HBM acting as L4 cache
        L4CacheBytes = DetectL4Cache(AvailableRamBytes);

        int coreCount = Environment.ProcessorCount;

        // L2 tier: tensors fitting in a single core's L2
        // Highest reuse benefit — keep most copies
        _maxElementsL2 = L2CacheBytes / sizeof(float);  // e.g. 512KB → 128K floats
        _maxPerShapeL2 = Math.Min(6, coreCount >= 16 ? 4 : 3);

        // L3 tier: tensors fitting in shared L3
        // Good reuse but shared across cores
        _maxElementsL3 = L3CacheBytes / sizeof(float);  // e.g. 16MB → 4M floats
        _maxPerShapeL3 = 2;

        // L4/HBM tier: tensors fitting in on-package memory
        // Still faster than DRAM — worth caching
        _maxElementsL4 = L4CacheBytes > 0 ? L4CacheBytes / sizeof(float) : _maxElementsL3;
        _maxPerShapeL4 = L4CacheBytes > 0 ? 2 : 1;

        // RAM tier: tensors too large for any cache
        // Budget: min(1GB, RAM/8) per thread
        long ramBudget = Math.Min(1024L * 1024 * 1024, AvailableRamBytes / 8);
        _maxElementsRam = ramBudget / sizeof(float);
        _maxPerShapeRam = 1;
    }

    /// <summary>
    /// Gets the max cached tensors per shape, adaptive to tensor size and cache tier.
    /// </summary>
    internal static int GetMaxPerShape(long tensorElements)
    {
        if (MaxBuffersPerShapeOverride > 0) return MaxBuffersPerShapeOverride;

        int baseMax;
        if (tensorElements <= _maxElementsL2)
            baseMax = _maxPerShapeL2;       // Fits in L2 -> aggressive
        else if (tensorElements <= _maxElementsL3)
            baseMax = _maxPerShapeL3;       // Fits in L3 -> moderate
        else if (tensorElements <= _maxElementsL4)
            baseMax = _maxPerShapeL4;       // Fits in L4/HBM -> still worth caching
        else
            baseMax = _maxPerShapeRam;      // RAM only -> minimal

        return Policy switch
        {
            CachePolicy.Aggressive => baseMax + 2,
            CachePolicy.Conservative => Math.Max(1, baseMax / 2),
            CachePolicy.Balanced => 2,
            _ => baseMax // Auto
        };
    }

    /// <summary>
    /// Gets the max element count for cached tensors, based on cache hierarchy.
    /// </summary>
    internal static long MaxElementsPerTensor
    {
        get
        {
            if (MaxElementsOverride > 0) return MaxElementsOverride;
            return Policy switch
            {
                CachePolicy.Aggressive => _maxElementsRam,
                CachePolicy.Conservative => _maxElementsL3,
                CachePolicy.Balanced => _maxElementsL3,
                _ => _maxElementsRam // Auto: cache up to RAM budget
            };
        }
    }

    /// <summary>
    /// Gets a cached tensor with the given shape, or allocates a new one.
    /// The returned tensor's data is UNINITIALIZED — caller must overwrite.
    /// </summary>
    internal static Tensor<T> RentOrAllocate<T>(int[] shape)
    {
        if (!Enabled)
            return TensorAllocator.RentUninitialized<T>(shape);

        var pools = _pools;
        if (pools is null)
        {
            pools = new ConcurrentDictionary<long, ConcurrentQueue<object>>();
            _pools = pools;
            lock (_allPools) { _allPools.Add(new WeakReference<ConcurrentDictionary<long, ConcurrentQueue<object>>>(pools)); }
        }
        long key = ComputeKey<T>(shape);

        if (pools.TryGetValue(key, out var pool) && pool.TryDequeue(out var obj))
        {
            var tensor = (Tensor<T>)obj;
            if (ShapesMatch(tensor._shape, shape))
            {
                // Clear mutable state from previous use to prevent stale metadata
                tensor.Grad = null;
                tensor.LazySource = null;
                tensor.GradFn = null;
                return tensor;
            }
            pool.Enqueue(obj);
        }

        return TensorAllocator.RentUninitialized<T>(shape);
    }

    /// <summary>
    /// Returns a tensor to the cache for reuse by future ops.
    /// Cache tier (L2/L3/RAM) determines how many copies are kept.
    /// </summary>
    internal static void Return<T>(Tensor<T> tensor)
    {
        if (!Enabled || tensor == null || !tensor.IsContiguous)
            return;

        long elements = tensor.Length;
        if (elements > MaxElementsPerTensor)
            return;

        var pools = _pools;
        if (pools is null)
        {
            pools = new ConcurrentDictionary<long, ConcurrentQueue<object>>();
            _pools = pools;
            lock (_allPools) { _allPools.Add(new WeakReference<ConcurrentDictionary<long, ConcurrentQueue<object>>>(pools)); }
        }
        long key = ComputeKey<T>(tensor._shape);
        var pool = pools.GetOrAdd(key, _ => new ConcurrentQueue<object>());

        int maxCopies = GetMaxPerShape(elements);
        if (pool.Count < maxCopies)
            pool.Enqueue(tensor);
    }

    /// <summary>Clears all cached tensors across all threads, releasing memory immediately.</summary>
    internal static void Clear()
    {
        _pools?.Clear();
        lock (_allPools)
        {
            for (int i = _allPools.Count - 1; i >= 0; i--)
            {
                if (_allPools[i].TryGetTarget(out var pool))
                    pool.Clear();
                else
                    _allPools.RemoveAt(i); // Remove GC'd references
            }
        }
    }

    /// <summary>Compute a fast hash key from shape + type.</summary>
    private static long ComputeKey<T>(int[] shape)
    {
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

    /// <summary>
    /// Detects CPU cache sizes (L1, L2, L3) using runtime introspection.
    /// Falls back to common defaults if detection fails.
    /// </summary>
    private static (int l1, int l2, long l3) DetectCacheSizes()
    {
        // Defaults based on modern CPUs (2020+):
        // L1: 32KB data per core, L2: 512KB per core, L3: 16MB shared
        int defaultL1 = 32 * 1024;
        int defaultL2 = 512 * 1024;
        long defaultL3 = 16L * 1024 * 1024;

        try
        {
#if NET8_0_OR_GREATER
            // .NET 8+ has System.Runtime.Intrinsics.X86.X86Base.CpuId for cache detection
            // But the simplest reliable approach is environment-based heuristics
            if (System.Runtime.Intrinsics.X86.X86Base.IsSupported)
            {
                // ProcessorCount includes hyperthreads — estimate physical cores
                int logicalCores = Environment.ProcessorCount;
                int physicalCores = Math.Max(1, logicalCores / 2); // SMT/HT gives 2x

                // Scale L3 by physical cores (each core contributes ~2MB L3 slice)
                // AMD Zen 2/3: ~4MB L3 per core (CCX-based, 16MB per 4-core CCX)
                // AMD Zen 4: ~4MB L3 per core (32MB per CCD)
                // Intel 12th+: ~1.5-2.5MB L3 per core
                long l3Estimate = physicalCores * 4L * 1024 * 1024; // ~4MB per physical core
                l3Estimate = Math.Min(l3Estimate, 128L * 1024 * 1024); // Cap at 128MB

                // L1/L2 per core:
                // AMD Zen 2/3: L1=32KB, L2=512KB
                // AMD Zen 4: L1=32KB, L2=1MB
                // Intel 12th+: L1=48KB, L2=1.25MB
                // Intel 13th+: L1=48KB, L2=2MB (P-cores)
                // Default to AMD Zen 2/3 (most common for compute workloads)
                int l1 = 32 * 1024;   // 32KB — safe conservative default
                int l2 = 512 * 1024;  // 512KB — AMD Zen 2/3 default

                // Detect if Intel (has wider L2)
                // Simple heuristic: check if SSE4.2 + no FMA3 fallback needed
                // This isn't reliable, so just use physical cores as proxy:
                // High core count (24+) = likely server/Intel with bigger L2
                if (physicalCores >= 24)
                {
                    l1 = 48 * 1024;    // Intel P-core style
                    l2 = 2048 * 1024;  // Intel 13th gen+ P-core L2
                }
                else if (physicalCores >= 12)
                {
                    l2 = 1024 * 1024;  // Could be AMD Zen 4 or Intel 12th
                }

                return (l1, l2, l3Estimate);
            }
#endif
#if NET5_0_OR_GREATER
            // Try /sys/devices/system/cpu on Linux for exact cache sizes
            if (OperatingSystem.IsLinux())
            {
                return DetectCacheSizesLinux() ?? (defaultL1, defaultL2, defaultL3);
            }
#endif
        }
        catch
        {
            // Detection failed — use defaults
        }

        return (defaultL1, defaultL2, defaultL3);
    }

    private static (int l1, int l2, long l3)? DetectCacheSizesLinux()
    {
        try
        {
            // Read from /sys/devices/system/cpu/cpu0/cache/
            int l1 = 32 * 1024, l2 = 512 * 1024;
            long l3 = 16L * 1024 * 1024;

            string basePath = "/sys/devices/system/cpu/cpu0/cache";
            if (System.IO.Directory.Exists(basePath))
            {
                foreach (var indexDir in System.IO.Directory.GetDirectories(basePath))
                {
                    string levelFile = System.IO.Path.Combine(indexDir, "level");
                    string sizeFile = System.IO.Path.Combine(indexDir, "size");
                    string typeFile = System.IO.Path.Combine(indexDir, "type");

                    if (!System.IO.File.Exists(levelFile) || !System.IO.File.Exists(sizeFile))
                        continue;

                    string level = System.IO.File.ReadAllText(levelFile).Trim();
                    string sizeStr = System.IO.File.ReadAllText(sizeFile).Trim();
                    string type = System.IO.File.Exists(typeFile) ? System.IO.File.ReadAllText(typeFile).Trim() : "";

                    long sizeBytes = ParseCacheSize(sizeStr);
                    if (sizeBytes <= 0) continue;

                    if (level == "1" && type == "Data")
                        l1 = (int)sizeBytes;
                    else if (level == "2")
                        l2 = (int)sizeBytes;
                    else if (level == "3")
                        l3 = sizeBytes;
                }
                return (l1, l2, l3);
            }
            return null;
        }
        catch { return null; }
    }

    private static long ParseCacheSize(string sizeStr)
    {
        // Parse "32K", "512K", "16384K", "16M"
        if (string.IsNullOrEmpty(sizeStr)) return 0;
        sizeStr = sizeStr.Trim();
        if (sizeStr.EndsWith("K", StringComparison.OrdinalIgnoreCase))
        {
            if (long.TryParse(sizeStr.Substring(0, sizeStr.Length - 1), out long kb))
                return kb * 1024;
        }
        else if (sizeStr.EndsWith("M", StringComparison.OrdinalIgnoreCase))
        {
            if (long.TryParse(sizeStr.Substring(0, sizeStr.Length - 1), out long mb))
                return mb * 1024 * 1024;
        }
        else if (long.TryParse(sizeStr, out long bytes))
        {
            return bytes;
        }
        return 0;
    }

    /// <summary>
    /// Detects L4/HBM cache. Server platforms like Intel Sapphire Rapids
    /// have 64MB-4GB of on-package HBM. Detected via core count + RAM heuristics.
    /// </summary>
    private static long DetectL4Cache(long totalRam)
    {
        // L4/HBM detection heuristics:
        // - Very high core count (64+) with massive RAM (512GB+) → likely server with HBM
        // - 32+ cores with 128GB+ → possible HBM
        // For now, return 0 since we can't reliably detect HBM from managed code.
        // On Linux, /sys/devices/system/cpu/cpu0/cache/index4 would indicate L4.
        int cores = Environment.ProcessorCount;
#if NET5_0_OR_GREATER
        if (OperatingSystem.IsLinux())
        {
            try
            {
                string l4Path = "/sys/devices/system/cpu/cpu0/cache/index4";
                if (System.IO.Directory.Exists(l4Path))
                {
                    string sizeFile = System.IO.Path.Combine(l4Path, "size");
                    if (System.IO.File.Exists(sizeFile))
                    {
                        string sizeStr = System.IO.File.ReadAllText(sizeFile).Trim();
                        long bytes = ParseCacheSize(sizeStr);
                        if (bytes > 0) return bytes;
                    }
                }
            }
            catch { }
        }
#endif
        // Heuristic: very high-end server with 64+ cores likely has HBM
        if (cores >= 64 && totalRam > 256L * 1024 * 1024 * 1024)
            return 256L * 1024 * 1024; // Assume 256MB HBM as conservative estimate

        return 0; // No L4 detected
    }

    private static long GetAvailableMemoryBytes()
    {
        try
        {
#if NET5_0_OR_GREATER
            var gcInfo = GC.GetGCMemoryInfo();
            return gcInfo.TotalAvailableMemoryBytes;
#else
            return GC.GetTotalMemory(false) * 16; // rough estimate for net471
#endif
        }
        catch
        {
            return 8L * 1024 * 1024 * 1024; // Default: 8GB
        }
    }
}
