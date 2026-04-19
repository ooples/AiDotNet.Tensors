using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// xUnit collection that pins <see cref="AutotuneWarmupTests"/> to serial
/// execution. Each test mutates process-wide shared state
/// (<c>AIDOTNET_AUTOTUNE_CACHE_PATH</c> env var + the static
/// <see cref="AutotuneKernelCatalog"/>); parallel workers would otherwise
/// race on both. Mirrors the <see cref="AutotuneCacheTests"/> pattern.
/// </summary>
[CollectionDefinition("AutotuneWarmupTests", DisableParallelization = true)]
public sealed class AutotuneWarmupTestsCollection { }

/// <summary>
/// Tests for issue #200 — <see cref="AutotuneCache.WarmupCommonKernelsAsync"/>
/// and <see cref="AutotuneCache.WarmupCategoryAsync"/>. Uses a synthetic
/// <see cref="AutotuneCatalogEntry"/> so tests are deterministic and don't
/// depend on host-specific BLAS.
/// </summary>
[Collection("AutotuneWarmupTests")]
public sealed class AutotuneWarmupTests : IDisposable
{
    private const string EnvVar = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string _tempRoot;
    private readonly string? _originalEnv;

    public AutotuneWarmupTests()
    {
        // Snapshot the prior env var so Dispose can restore it — otherwise a
        // CI runner that preset the var loses its setting after the first
        // test runs. Save-and-restore matches AutotuneCacheTests.
        _originalEnv = Environment.GetEnvironmentVariable(EnvVar);
        AutotuneKernelCatalog.Clear();
        _tempRoot = Path.Combine(
            Path.GetTempPath(),
            "aidotnet-autotune-test-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(EnvVar, _tempRoot);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EnvVar, _originalEnv);
        AutotuneKernelCatalog.Clear();
        try
        {
            if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true);
        }
        catch
        {
            // Best-effort cleanup — OS may hold a handle briefly after Dispose.
        }
    }

    [Fact]
    public async Task WarmupCommonKernels_EmptyCatalog_ReturnsNoOpReport()
    {
        var report = await AutotuneCache.WarmupCommonKernelsAsync();
        Assert.Equal(0, report.KernelsWarmed);
        Assert.True(report.ShapesPerKernel > 0, "default shape set should be non-empty");
        Assert.Empty(report.BestGflopsByKernel);
    }

    [Fact]
    public async Task WarmupCommonKernels_SingleEntry_PopulatesCacheForEveryShape()
    {
        var id = new KernelId("unit-test", "stub-gemm");
        // Variants: "fast" (100 GFLOPS) and "slow" (50 GFLOPS) — fast must win.
        AutotuneKernelCatalog.Register(new AutotuneCatalogEntry(
            id,
            variants: _ => new[] { "fast", "slow" },
            benchmarkVariant: (shape, variant, ct) =>
                Task.FromResult(variant == "fast" ? 100.0 : 50.0)));

        var shapes = new[] { new[] { 2, 4 }, new[] { 8, 16 } };
        var report = await AutotuneCache.WarmupCommonKernelsAsync(shapes);

        Assert.Equal(1, report.KernelsWarmed);
        Assert.Equal(2, report.ShapesPerKernel);

        foreach (var s in shapes)
        {
            var choice = AutotuneCache.Lookup(id, new ShapeProfile(s));
            Assert.NotNull(choice);
            Assert.Equal("fast", choice!.Variant);
            Assert.Equal(100.0, choice.MeasuredGflops);
        }
    }

    [Fact]
    public async Task WarmupCommonKernels_SecondRun_SkipsAlreadyCachedEntries()
    {
        int benchmarkCalls = 0;
        var id = new KernelId("unit-test", "counted");
        AutotuneKernelCatalog.Register(new AutotuneCatalogEntry(
            id,
            variants: _ => new[] { "only" },
            benchmarkVariant: (shape, variant, ct) =>
            {
                Interlocked.Increment(ref benchmarkCalls);
                return Task.FromResult(75.0);
            }));

        var shapes = new[] { new[] { 1, 1 } };
        await AutotuneCache.WarmupCommonKernelsAsync(shapes);
        int afterFirst = benchmarkCalls;
        await AutotuneCache.WarmupCommonKernelsAsync(shapes);

        Assert.Equal(1, afterFirst);
        Assert.Equal(afterFirst, benchmarkCalls); // no new benchmark on round 2
    }

    [Fact]
    public async Task WarmupCategoryAsync_FiltersByCategory()
    {
        var idGemm = new KernelId("gemm", "stub");
        var idConv = new KernelId("conv", "stub");
        bool gemmBenched = false, convBenched = false;
        AutotuneKernelCatalog.Register(new AutotuneCatalogEntry(
            idGemm, _ => new[] { "v" },
            (s, v, ct) => { gemmBenched = true; return Task.FromResult(10.0); }));
        AutotuneKernelCatalog.Register(new AutotuneCatalogEntry(
            idConv, _ => new[] { "v" },
            (s, v, ct) => { convBenched = true; return Task.FromResult(10.0); }));

        var shapes = new[] { new[] { 2, 2 } };
        var report = await AutotuneCache.WarmupCategoryAsync("gemm", shapes);

        Assert.True(gemmBenched, "gemm category entry should run");
        Assert.False(convBenched, "conv category entry should be skipped");
        Assert.Equal(1, report.KernelsWarmed);
    }

    [Fact]
    public async Task Warmup_SkipsVariantsReportingZeroOrNegativeGflops()
    {
        // A variant signaling "not applicable here" returns 0 or negative.
        // The warmup should not pick it even if it's the only candidate
        // (shape with no applicable variants stays uncached).
        var id = new KernelId("unit-test", "all-na");
        AutotuneKernelCatalog.Register(new AutotuneCatalogEntry(
            id,
            variants: _ => new[] { "na-1", "na-2" },
            benchmarkVariant: (s, v, ct) => Task.FromResult(0.0)));

        var shapes = new[] { new[] { 2, 2 } };
        var report = await AutotuneCache.WarmupCommonKernelsAsync(shapes);

        Assert.Equal(0, report.KernelsWarmed);
        Assert.Null(AutotuneCache.Lookup(id, new ShapeProfile(shapes[0])));
    }

    [Fact]
    public async Task Warmup_Progress_ReportsOnceForEachShape()
    {
        var id = new KernelId("unit-test", "progress");
        AutotuneKernelCatalog.Register(new AutotuneCatalogEntry(
            id, _ => new[] { "only" }, (s, v, ct) => Task.FromResult(42.0)));

        var lines = new List<string>();
        // Synchronous IProgress impl — Progress<T> posts to the captured
        // SyncContext asynchronously, which makes assertion racy. Direct
        // callback completes before the warmup's await yields, so the
        // count is deterministic.
        var progress = new SyncProgress(lines);
        var shapes = new[] { new[] { 1, 1 }, new[] { 2, 2 }, new[] { 3, 3 } };
        await AutotuneCache.WarmupCommonKernelsAsync(shapes, progress);

        Assert.Equal(shapes.Length, lines.Count);
        foreach (var line in lines)
            Assert.Contains("warmed", line, StringComparison.Ordinal);
    }

    private sealed class SyncProgress : IProgress<string>
    {
        private readonly List<string> _out;
        public SyncProgress(List<string> sink) { _out = sink; }
        public void Report(string value) { lock (_out) _out.Add(value); }
    }

    [Fact]
    public async Task Warmup_Cancellation_PropagatesBeforeRunningAllEntries()
    {
        var idA = new KernelId("unit-test", "a");
        int calls = 0;
        AutotuneKernelCatalog.Register(new AutotuneCatalogEntry(
            idA,
            _ => new[] { "v" },
            (s, v, ct) =>
            {
                Interlocked.Increment(ref calls);
                ct.ThrowIfCancellationRequested();
                return Task.FromResult(1.0);
            }));

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
            await AutotuneCache.WarmupCommonKernelsAsync(new[] { new[] { 1, 1 } }, ct: cts.Token));
    }
}
