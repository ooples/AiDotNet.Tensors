using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// In-memory <see cref="IGpuTuningExchange"/> for tests: records published
/// profiles and serves fetches from them, so the whole Phase-2 client flow runs
/// without a network or Supabase.
/// </summary>
internal sealed class InMemoryGpuTuningExchange : IGpuTuningExchange
{
    private readonly List<GpuTuningProfile> _rows = new();
    public bool IsEnabled { get; set; } = true;
    public int FetchCount { get; private set; }
    public List<GpuTuningProfile> Published { get; } = new();

    public void Seed(GpuTuningProfile profile) => _rows.Add(profile);

    public IReadOnlyList<GpuTuningProfile> Fetch(
        string modelKey, string category, string kernelName, string shapeKey)
    {
        FetchCount++;
        return _rows.Where(r =>
                r.ModelKey == modelKey && r.Category == category &&
                r.KernelName == kernelName && r.ShapeKey == shapeKey)
            .OrderByDescending(r => r.MeasuredGflops)
            .ToList();
    }

    public void Publish(GpuTuningProfile profile)
    {
        Published.Add(profile);
        _rows.Add(profile);
    }
}

/// <summary>
/// GPU-free tests for <see cref="CommunityAutotune"/> — the Phase-2
/// "download-as-candidate, re-verify on-device" trust model. Shares the
/// <c>AutotuneCacheTests</c> collection because it redirects the cache root.
/// </summary>
[Collection("AutotuneCacheTests")]
public sealed class CommunityAutotuneTests : IDisposable
{
    private const string EnvVar = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string _tempRoot;
    private readonly string? _originalEnv;

    public CommunityAutotuneTests()
    {
        _originalEnv = Environment.GetEnvironmentVariable(EnvVar);
        _tempRoot = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "aidotnet-community-test-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(EnvVar, _tempRoot);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EnvVar, _originalEnv);
        try { if (System.IO.Directory.Exists(_tempRoot)) System.IO.Directory.Delete(_tempRoot, true); }
        catch { /* best effort */ }
    }

    private static readonly GpuDeviceFingerprint Card =
        GpuDeviceFingerprint.FromCuda("NVIDIA GeForce RTX 3080", "uuidComm", 8, 6, 12030);

    private const string Category = ConvTileAutotune.Category;
    private const string Kernel = ConvTileAutotune.TiledOneByOneName;

    private static ShapeProfile Shape() => ConvTileAutotune.Shape(32, 64, 64, 3136);
    private static IReadOnlyList<AutotuneCandidate> Local() => ConvTileAutotune.Candidates(64, 64, 3136);

    private GpuTuningProfile Community(string variant, int tile, double gflops) => new()
    {
        ModelKey = Card.ModelKey,
        Vendor = Card.Vendor,
        Model = Card.Model,
        Architecture = Card.Architecture,
        DriverVersion = Card.DriverVersion,
        Category = Category,
        KernelName = Kernel,
        ShapeKey = Shape().ToFileStem(),
        Variant = variant,
        Parameters = new Dictionary<string, string>(StringComparer.Ordinal) { ["Tile"] = tile.ToString() },
        MeasuredGflops = gflops
    };

    [Fact]
    public void GoodCommunityConfig_WinsOnReverify_IsUsed_AndPublished()
    {
        var exchange = new InMemoryGpuTuningExchange();
        // A community peer reports tile-32 is great on this model.
        exchange.Seed(Community("tile-32", 32, 1500.0));

        // On THIS device the sweep confirms tile-32 really is fastest.
        AutotuneResolution r = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            c => c.Variant == "tile-32" ? 1400.0 : 500.0,
            autotuneEnabled: true);

        Assert.True(r.Measured);
        Assert.True(ConvTileAutotune.TryGetTile(r, out int tile));
        Assert.Equal(32, tile);              // community config re-verified and selected
        Assert.Single(exchange.Published);   // our own measurement corroborates it
        Assert.Equal("tile-32", exchange.Published[0].Variant);
        Assert.Equal(Card.ModelKey, exchange.Published[0].ModelKey);
    }

    [Fact]
    public void PoisonedCommunityConfig_ThatLaunchFails_Loses_LocalWinnerSelected()
    {
        var exchange = new InMemoryGpuTuningExchange();
        // A malicious/over-reported config claims a huge number but cannot launch here.
        exchange.Seed(Community("tile-999", 999, 999999.0));

        AutotuneResolution r = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            c => c.Variant == "tile-999"
                ? throw new InvalidOperationException("shared-mem over budget")
                : (c.Variant == "tile-16" ? 800.0 : 400.0),
            autotuneEnabled: true);

        Assert.True(r.Measured);
        Assert.True(ConvTileAutotune.TryGetTile(r, out int tile));
        Assert.Equal(16, tile); // poison lost; a real local candidate won
    }

    [Fact]
    public void Disabled_ExchangeNeverConsulted_AndDefaultIsLocal()
    {
        var exchange = new InMemoryGpuTuningExchange();
        exchange.Seed(Community("tile-32", 32, 9999.0));

        AutotuneResolution r = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            _ => throw new InvalidOperationException("must not benchmark when disabled"),
            autotuneEnabled: false);

        Assert.False(r.Measured);
        Assert.Equal(0, exchange.FetchCount);          // network never touched on the disabled path
        Assert.Equal("tile-16", r.Variant);            // local default, not the community tile-32
        Assert.Empty(exchange.Published);
    }

    [Fact]
    public void NullExchange_BehavesAsLocalOnly()
    {
        AutotuneResolution r = CommunityAutotune.Resolve(
            NullGpuTuningExchange.Instance, Category, Kernel, Card, Shape(), Local(),
            c => c.Variant == "tile-16" ? 700.0 : 300.0,
            autotuneEnabled: true);

        Assert.True(r.Measured);
        Assert.Equal("tile-16", r.Variant);
    }

    [Fact]
    public void MergeCommunityCandidates_PutsLocalFirst_Dedups_AndCaps()
    {
        IReadOnlyList<AutotuneCandidate> local = Local(); // tile-16, tile-32, tile-8
        var community = new List<GpuTuningProfile>
        {
            Community("tile-32", 32, 100.0),   // duplicate of a local variant -> dropped
            Community("tile-11", 11, 300.0),   // new, highest gflops -> first community add
            Community("tile-7", 7, 200.0),     // new
            Community("tile-5", 5, 150.0),     // new but beyond the cap of 2
        };

        IReadOnlyList<AutotuneCandidate> merged =
            CommunityAutotune.MergeCommunityCandidates(local, community, maxCommunity: 2);

        // Local candidates preserved and first (candidates[0] stays the local default).
        Assert.Equal("tile-16", merged[0].Variant);
        Assert.Equal(local.Count + 2, merged.Count);           // exactly 2 community adds
        var variants = merged.Select(c => c.Variant).ToList();
        Assert.Contains("tile-11", variants);                  // best-reported new one included
        Assert.Contains("tile-7", variants);
        Assert.DoesNotContain("tile-5", variants);             // capped
        Assert.Equal(1, variants.Count(v => v == "tile-32"));  // duplicate not doubled
    }

    [Fact]
    public void CacheHit_SkipsFetchAndPublish()
    {
        var exchange = new InMemoryGpuTuningExchange();

        // First call sweeps and caches a local winner.
        CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            c => c.Variant == "tile-16" ? 900.0 : 300.0, autotuneEnabled: true);
        int fetchesAfterFirst = exchange.FetchCount;
        int publishedAfterFirst = exchange.Published.Count;

        // Second call is a cache hit: no fetch, no publish, no benchmark.
        AutotuneResolution second = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            _ => throw new InvalidOperationException("cache hit must not benchmark"),
            autotuneEnabled: true);

        Assert.True(second.FromCache);
        Assert.Equal(fetchesAfterFirst, exchange.FetchCount);      // no extra fetch
        Assert.Equal(publishedAfterFirst, exchange.Published.Count); // no extra publish
    }
}
