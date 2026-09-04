using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// xUnit collection that pins <see cref="BuiltInCatalogTests"/> to serial
/// execution. Each test mutates process-wide shared state
/// (<c>AIDOTNET_AUTOTUNE_CACHE_PATH</c> env var, the static
/// <see cref="AutotuneKernelCatalog"/>, and the one-shot
/// <see cref="BuiltInCatalog"/> registration latch); parallel xUnit workers
/// would race on all three. Mirrors the <see cref="AutotuneCacheTests"/>
/// and <see cref="AutotuneWarmupTests"/> patterns.
/// </summary>
[CollectionDefinition("BuiltInCatalogTests", DisableParallelization = true)]
public sealed class BuiltInCatalogTestsCollection { }

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
[Collection("BuiltInCatalogTests")]
public sealed class BuiltInCatalogTests : IDisposable
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
                $"Expected positive GFLOPS for {BuiltInCatalog.SGEMM.ToFileStem()}@{string.Join("x", s)}");
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
    public async Task ParallelGemmBenchmark_NeverMutatesTheProcessWideParallelPolicy()
    {
        bool previous = SimdGemm.UseParallelGemm;
        try
        {
            SimdGemm.UseParallelGemm = false;
            BuiltInCatalog.EnsureRegistered();
            AutotuneCatalogEntry entry = Assert.Single(
                AutotuneKernelCatalog.Entries,
                candidate => candidate.Id == BuiltInCatalog.SGEMM);
            Task<double> benchmark = Task.Run(async () =>
                await entry.BenchmarkVariant(
                    new ShapeProfile(256, 256, 256),
                    "parallel",
                    CancellationToken.None));

            while (!benchmark.IsCompleted)
            {
                Assert.False(Volatile.Read(ref SimdGemm.UseParallelGemm));
                Thread.Yield();
            }

            Assert.True(await benchmark > 0);
            Assert.False(SimdGemm.UseParallelGemm);
        }
        finally
        {
            SimdGemm.UseParallelGemm = previous;
        }
    }

    [Theory]
    [InlineData("gemm", "corrupt")]
    [InlineData("sparse_mm", "corrupt")]
    public async Task BuiltInBenchmark_RejectsUnknownFiniteVariant(
        string category,
        string variant)
    {
        BuiltInCatalog.EnsureRegistered();
        AutotuneCatalogEntry entry = Assert.Single(AutotuneKernelCatalog.EntriesForCategory(category));

        await Assert.ThrowsAsync<ArgumentException>(() => entry.BenchmarkVariant(
            new ShapeProfile(16, 16, 16, 500),
            variant,
            CancellationToken.None));
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
