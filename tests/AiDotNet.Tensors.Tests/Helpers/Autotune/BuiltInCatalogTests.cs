using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// Issue #200 acceptance spec: "After WarmupCommonKernelsAsync completes,
/// AutotuneCache.Lookup(id, shape) returns a non-null KernelChoice for
/// every common kernel at every supplied shape."
///
/// <para>These tests make the acceptance criterion load-bearing: the
/// built-in catalog registers real tunable entries (GEMM variant
/// select), warmup actually benchmarks and stores, second run is a
/// fast no-op on cache hit.</para>
/// </summary>
public class BuiltInCatalogTests : IDisposable
{
    private readonly string _cacheDir;
    private readonly string? _prevEnv;

    public BuiltInCatalogTests()
    {
        _prevEnv = Environment.GetEnvironmentVariable("AIDOTNET_AUTOTUNE_CACHE_PATH");
        _cacheDir = Path.Combine(Path.GetTempPath(), "aidotnet-autotune-builtin-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable("AIDOTNET_AUTOTUNE_CACHE_PATH", _cacheDir);
        AutotuneKernelCatalog.Clear();
        BuiltInCatalog.ResetRegistrationForTests();
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_AUTOTUNE_CACHE_PATH", _prevEnv);
        try { if (Directory.Exists(_cacheDir)) Directory.Delete(_cacheDir, recursive: true); } catch { }
        AutotuneKernelCatalog.Clear();
        BuiltInCatalog.ResetRegistrationForTests();
    }

    [Fact]
    public async Task WarmupCommonKernels_Populates_SGEMM_ForEverySuppliedShape()
    {
        // Tiny shapes so the benchmark completes fast.
        var shapes = new[] { new[] { 16, 16, 16 }, new[] { 32, 16, 32 } };
        var report = await AutotuneCache.WarmupCommonKernelsAsync(shapes);

        Assert.True(report.KernelsWarmed >= 1,
            "Built-in catalog should register SGEMM; WarmupCommonKernelsAsync must benchmark it.");

        // Acceptance: every common kernel, every supplied shape → non-null Lookup.
        foreach (var s in shapes)
        {
            var choice = AutotuneCache.Lookup(BuiltInCatalog.SGEMM, new ShapeProfile(s));
            Assert.NotNull(choice);
            Assert.False(string.IsNullOrEmpty(choice!.Variant));
            Assert.True(choice.MeasuredGflops > 0,
                $"Expected positive GFLOPS for {BuiltInCatalog.SGEMM.ToFileStem()}@{string.Join('x', s)}");
        }
    }

    [Fact]
    public async Task WarmupCommonKernels_SecondRun_IsFastNoOp()
    {
        // Issue #200 acceptance: "On a fresh process, the cache is restored
        // from DefaultCachePath transparently; a second WarmupCommonKernelsAsync
        // is a no-op (fast return)."
        var shapes = new[] { new[] { 16, 16, 16 } };

        var firstReport = await AutotuneCache.WarmupCommonKernelsAsync(shapes);
        Assert.True(firstReport.KernelsWarmed >= 1);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var secondReport = await AutotuneCache.WarmupCommonKernelsAsync(shapes);
        sw.Stop();

        // Second run should take a fraction of the first. We don't compare
        // absolute times (CI noise) — just verify no fresh benchmark ran.
        Assert.Equal(0, secondReport.KernelsWarmed);
    }

    [Fact]
    public async Task WarmupCategory_Gemm_PicksUpSgemm()
    {
        var shapes = new[] { new[] { 16, 16, 16 } };
        var report = await AutotuneCache.WarmupCategoryAsync("gemm", shapes);
        Assert.Equal(1, report.KernelsWarmed);
        Assert.NotNull(AutotuneCache.Lookup(BuiltInCatalog.SGEMM, new ShapeProfile(shapes[0])));
    }

    [Fact]
    public async Task WarmupCategory_UnknownCategory_ReturnsEmpty()
    {
        var report = await AutotuneCache.WarmupCategoryAsync(
            "nonexistent", new[] { new[] { 2, 2, 2 } });
        Assert.Equal(0, report.KernelsWarmed);
    }

    [Fact]
    public async Task WarmupCommonKernels_DefaultShapes_PopulatesAtLeastOne()
    {
        // No shapes supplied → default representative shapes are used.
        var report = await AutotuneCache.WarmupCommonKernelsAsync();
        Assert.True(report.KernelsWarmed >= 1);
        Assert.True(report.ShapesPerKernel > 0);
        Assert.NotEmpty(report.BestGflopsByKernel);
    }
}
