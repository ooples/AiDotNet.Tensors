using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Internal registry of tunable kernels that <see cref="AutotuneCache"/>
/// knows how to benchmark. Each entry pairs a <see cref="KernelId"/> with
/// a variant enumerator and a benchmark function — the data the
/// <see cref="AutotuneCache.WarmupCommonKernelsAsync"/> entry point needs
/// to turn "warm up everything relevant" into a concrete set of
/// benchmark jobs without the caller having to know library internals.
///
/// <para>Internal-only so only Tensors itself can register entries;
/// downstream consumers can't inject kernels they don't own, which keeps
/// the catalog semantics meaningful ("common kernels shipped with
/// Tensors").</para>
///
/// <para>Populated at type-initialisation time by the modules that own
/// each kernel family (GEMM, Conv, SDPA). An empty catalog is a valid
/// runtime state — the warmup APIs return
/// <see cref="AutotuneWarmupReport"/> with <c>KernelsWarmed = 0</c>.</para>
/// </summary>
internal static class AutotuneKernelCatalog
{
    // ConcurrentDictionary keyed on KernelId gives us atomic register +
    // idempotent re-registration (duplicate keys overwrite, last-in wins).
    // Register is called from static constructors; order across type
    // initializers is non-deterministic, but later overwrites mean the
    // last-registered variant-set/benchmarker is what Warmup sees. In
    // practice each kernel id is registered once.
    private static readonly ConcurrentDictionary<KernelId, AutotuneCatalogEntry> _entries = new();

    /// <summary>Adds <paramref name="entry"/> to the registry, overwriting any
    /// existing entry for the same <see cref="AutotuneCatalogEntry.Id"/>.</summary>
    public static void Register(AutotuneCatalogEntry entry)
    {
        if (entry is null) throw new ArgumentNullException(nameof(entry));
        _entries[entry.Id] = entry;
    }

    /// <summary>Snapshot of every registered entry. Order is unspecified.</summary>
    public static IReadOnlyList<AutotuneCatalogEntry> Entries => _entries.Values.ToArray();

    /// <summary>Entries whose <see cref="KernelId.Category"/> matches
    /// <paramref name="category"/> (ordinal, case-sensitive).</summary>
    public static IReadOnlyList<AutotuneCatalogEntry> EntriesForCategory(string category)
    {
        if (category is null) throw new ArgumentNullException(nameof(category));
        return _entries.Values
            .Where(e => string.Equals(e.Id.Category, category, StringComparison.Ordinal))
            .ToArray();
    }

    /// <summary>Test-hook: erases every registered entry. Tests that need a
    /// clean catalog call this in Arrange.</summary>
    internal static void Clear() => _entries.Clear();
}

/// <summary>
/// A tunable kernel family in the catalog. Variants are the candidate
/// implementations (e.g. <c>"blas"</c>, <c>"simd"</c>); BenchmarkVariant
/// times one variant at a shape and returns GFLOPS (higher is better).
/// </summary>
internal sealed class AutotuneCatalogEntry
{
    public KernelId Id { get; }

    /// <summary>Variant names to evaluate at a given shape. Same shape may
    /// expose different variant sets (e.g. tensor-core kernels only for
    /// shapes divisible by 8 on Ampere+).</summary>
    public Func<ShapeProfile, IEnumerable<string>> Variants { get; }

    /// <summary>Runs the benchmark for one variant at one shape, returns
    /// measured GFLOPS. Return 0 or negative to signal "not applicable" —
    /// that variant is skipped for this shape.</summary>
    public Func<ShapeProfile, string, CancellationToken, Task<double>> BenchmarkVariant { get; }

    public AutotuneCatalogEntry(
        KernelId id,
        Func<ShapeProfile, IEnumerable<string>> variants,
        Func<ShapeProfile, string, CancellationToken, Task<double>> benchmarkVariant)
    {
        // KernelId is a readonly record struct (value type) so it can never be
        // null — but its Category/Name strings can be. Catalog consumers build
        // cache filenames from those strings, and a null here would corrupt
        // the cache path. Fail fast at construction instead of handing the
        // problem to the filesystem layer.
        if (id.Category is null)
            throw new ArgumentException("KernelId.Category must not be null.", nameof(id));
        if (id.Name is null)
            throw new ArgumentException("KernelId.Name must not be null.", nameof(id));
        Id = id;
        Variants = variants ?? throw new ArgumentNullException(nameof(variants));
        BenchmarkVariant = benchmarkVariant ?? throw new ArgumentNullException(nameof(benchmarkVariant));
    }
}

/// <summary>
/// Summary of a <see cref="AutotuneCache.WarmupCommonKernelsAsync"/> call.
/// </summary>
/// <param name="KernelsWarmed">Number of distinct catalog entries that
/// ran at least one fresh benchmark during the call. Does not include
/// entries that were already in the cache.</param>
/// <param name="ShapesPerKernel">Number of representative shapes supplied
/// to the warmup.</param>
/// <param name="Duration">Wall time for the whole warmup.</param>
/// <param name="BestGflopsByKernel">The winning variant's measured GFLOPS
/// per (KernelId.ToFileStem, shape) pair — useful for logging and CI
/// perf dashboards.</param>
public sealed record AutotuneWarmupReport(
    int KernelsWarmed,
    int ShapesPerKernel,
    TimeSpan Duration,
    Dictionary<string, double> BestGflopsByKernel);
