using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// Device-agnostic tests for <see cref="GpuFirstRunAutotuner"/>. The on-device
/// measurement is injected as a delegate, so the full "tune the first time this
/// GPU is seen, then reuse" orchestration — including the poison/launch-fail
/// rejection that Phase 2 relies on — is verified without a GPU. Shares the
/// <c>AutotuneCacheTests</c> collection because these tests redirect the
/// process-wide <c>AIDOTNET_AUTOTUNE_CACHE_PATH</c> root.
/// </summary>
[Collection("AutotuneCacheTests")]
public sealed class GpuFirstRunAutotunerTests : IDisposable
{
    private const string EnvVar = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string _tempRoot;
    private readonly string? _originalEnv;

    public GpuFirstRunAutotunerTests()
    {
        _originalEnv = Environment.GetEnvironmentVariable(EnvVar);
        _tempRoot = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            "aidotnet-gpu-autotune-test-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(EnvVar, _tempRoot);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EnvVar, _originalEnv);
        try
        {
            if (System.IO.Directory.Exists(_tempRoot))
                System.IO.Directory.Delete(_tempRoot, recursive: true);
        }
        catch { /* best effort */ }
    }

    private static AutotuneCandidate Tile(int t) => new(
        "tile-" + t.ToString(System.Globalization.CultureInfo.InvariantCulture),
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["Tile"] = t.ToString(System.Globalization.CultureInfo.InvariantCulture)
        });

    private static IReadOnlyList<AutotuneCandidate> Tiles(params int[] tiles)
    {
        var list = new List<AutotuneCandidate>(tiles.Length);
        foreach (int t in tiles) list.Add(Tile(t));
        return list;
    }

    [Fact]
    public void GpuKernelId_FoldsFingerprintIntoName_AndKeysPerDevice()
    {
        KernelId a = GpuFirstRunAutotuner.GpuKernelId("conv2d", "tiled-1x1", "gpu-AAA-sm86-drv550");
        KernelId b = GpuFirstRunAutotuner.GpuKernelId("conv2d", "tiled-1x1", "gpu-BBB-sm89-drv555");

        Assert.Equal("conv2d", a.Category);
        Assert.Equal("tiled-1x1@gpu-AAA-sm86-drv550", a.Name);
        // Two distinct cards must never share a cache stem.
        Assert.NotEqual(a.ToFileStem(), b.ToFileStem());
    }

    [Fact]
    public void GpuKernelId_RejectsMissingFingerprint()
    {
        Assert.Throws<ArgumentException>(() =>
            GpuFirstRunAutotuner.GpuKernelId("conv2d", "tiled", ""));
    }

    [Fact]
    public void Resolve_SweepsPicksHighestGflops_ThenCachesAndReuses()
    {
        KernelId id = GpuFirstRunAutotuner.GpuKernelId("conv2d", "tiled-1x1", "gpu-sweep-sm86");
        var shape = new ShapeProfile(32, 64, 64, 3136);
        IReadOnlyList<AutotuneCandidate> candidates = Tiles(8, 16, 32);

        int firstRunCalls = 0;
        AutotuneResolution first = GpuFirstRunAutotuner.Resolve(
            id, shape, candidates,
            c => { firstRunCalls++; return c.Variant == "tile-16" ? 900.0 : 400.0; },
            autotuneEnabled: true);

        Assert.Equal("tile-16", first.Variant);
        Assert.True(first.Measured);
        Assert.False(first.FromCache);
        Assert.Equal(900.0, first.MeasuredGflops, 3);
        Assert.Equal(3, firstRunCalls); // all three benchmarked

        // Second resolve must be a pure cache hit — the benchmark must never run.
        AutotuneResolution second = GpuFirstRunAutotuner.Resolve(
            id, shape, candidates,
            _ => throw new InvalidOperationException("benchmark must not run on a cache hit"),
            autotuneEnabled: true);

        Assert.Equal("tile-16", second.Variant);
        Assert.True(second.FromCache);
        Assert.False(second.Measured);
        Assert.Equal("16", second.Parameters["Tile"]);
    }

    [Fact]
    public void Resolve_SkipsUnlaunchableAndNonPositiveCandidates()
    {
        KernelId id = GpuFirstRunAutotuner.GpuKernelId("conv2d", "poison", "gpu-poison-sm86");
        var shape = new ShapeProfile(16, 128, 128, 784);

        // tile-32 "launch-fails" (throws, e.g. shared memory over budget);
        // tile-8 reports a non-positive score; only tile-16 is a valid winner.
        AutotuneResolution r = GpuFirstRunAutotuner.Resolve(
            id, shape, Tiles(8, 16, 32),
            c => c.Variant switch
            {
                "tile-32" => throw new InvalidOperationException("too much shared memory"),
                "tile-8" => 0.0,
                _ => 750.0
            },
            autotuneEnabled: true);

        Assert.Equal("tile-16", r.Variant);
        Assert.True(r.Measured);
    }

    [Fact]
    public void Resolve_AllCandidatesFail_FallsBackToFirst_AndDoesNotCache()
    {
        KernelId id = GpuFirstRunAutotuner.GpuKernelId("conv2d", "allfail", "gpu-allfail-sm86");
        var shape = new ShapeProfile(8, 256, 256, 196);
        IReadOnlyList<AutotuneCandidate> candidates = Tiles(8, 16);

        AutotuneResolution r = GpuFirstRunAutotuner.Resolve(
            id, shape, candidates,
            _ => throw new InvalidOperationException("every launch fails"),
            autotuneEnabled: true);

        Assert.Equal("tile-8", r.Variant); // first candidate, safe default
        Assert.False(r.Measured);
        Assert.Null(AutotuneCache.Lookup(id, shape)); // nothing persisted
    }

    [Fact]
    public void Resolve_Disabled_UsesFirstCandidate_AndDoesNotMeasureOrCache()
    {
        KernelId id = GpuFirstRunAutotuner.GpuKernelId("conv2d", "disabled", "gpu-disabled-sm86");
        var shape = new ShapeProfile(1, 64, 64, 256);

        AutotuneResolution r = GpuFirstRunAutotuner.Resolve(
            id, shape, Tiles(8, 16, 32),
            _ => throw new InvalidOperationException("must not benchmark when disabled"),
            autotuneEnabled: false);

        Assert.Equal("tile-8", r.Variant);
        Assert.False(r.Measured);
        Assert.Null(AutotuneCache.Lookup(id, shape));
    }

    [Fact]
    public void Resolve_StaleCachedVariantNoLongerOffered_Retunes()
    {
        KernelId id = GpuFirstRunAutotuner.GpuKernelId("conv2d", "stale", "gpu-stale-sm86");
        var shape = new ShapeProfile(32, 64, 64, 3136);

        // Seed the cache with a variant that is no longer a candidate.
        AutotuneCache.Store(id, shape, new KernelChoice
        {
            Variant = "tile-64",
            Parameters = new Dictionary<string, string>(StringComparer.Ordinal) { ["Tile"] = "64" },
            MeasuredGflops = 1234.0
        });

        AutotuneResolution r = GpuFirstRunAutotuner.Resolve(
            id, shape, Tiles(8, 16),
            c => c.Variant == "tile-16" ? 500.0 : 200.0,
            autotuneEnabled: true);

        Assert.Equal("tile-16", r.Variant); // ignored the stale cached tile-64
        Assert.True(r.Measured);
    }

    [Fact]
    public void Resolve_SingleCandidate_UsesItWithoutMeasuring()
    {
        KernelId id = GpuFirstRunAutotuner.GpuKernelId("conv2d", "single", "gpu-single-sm86");
        var shape = new ShapeProfile(1, 64, 64, 256);

        AutotuneResolution r = GpuFirstRunAutotuner.Resolve(
            id, shape, Tiles(16),
            _ => throw new InvalidOperationException("no sweep for a single candidate"),
            autotuneEnabled: true);

        Assert.Equal("tile-16", r.Variant);
        Assert.False(r.Measured);
    }
}
