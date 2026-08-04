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
/// local-allowlist plus on-device remeasurement trust model. Shares the
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
    private static bool AllowAnyCandidate(AutotuneCandidate _) => true;

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
    public void GoodCommunityConfig_WinsOnRemeasure_IsUsed_AndPublished()
    {
        var exchange = new InMemoryGpuTuningExchange();
        // tile-11 is absent from Local(), so selecting it proves the community
        // fetch-and-merge path actually participated in the sweep.
        exchange.Seed(Community("tile-11", 11, 1500.0));
        bool communityWasBenchmarked = false;

        // On THIS device the sweep confirms tile-11 really is fastest.
        AutotuneResolution r = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            c =>
            {
                if (c.Variant == "tile-11") communityWasBenchmarked = true;
                return c.Variant == "tile-11" ? 1400.0 : 500.0;
            },
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.True(r.Measured);
        Assert.True(exchange.FetchCount > 0);
        Assert.True(communityWasBenchmarked);
        Assert.True(ConvTileAutotune.TryGetTile(r, out int tile));
        Assert.Equal(11, tile);              // community config re-measured and selected
        Assert.Single(exchange.Published);   // our own measurement corroborates it
        Assert.Equal("tile-11", exchange.Published[0].Variant);
        Assert.Equal(Card.ModelKey, exchange.Published[0].ModelKey);
    }

    [Fact]
    public void PoisonedCommunityConfig_ThatLaunchFails_Loses_LocalWinnerSelected()
    {
        var exchange = new InMemoryGpuTuningExchange();
        // A malicious/over-reported config claims a huge number but cannot launch here.
        exchange.Seed(Community("tile-999", 999, 999999.0));
        bool poisonedWasBenchmarked = false;

        AutotuneResolution r = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            c =>
            {
                if (c.Variant == "tile-999")
                {
                    poisonedWasBenchmarked = true;
                    throw new InvalidOperationException("shared-mem over budget");
                }
                return c.Variant == "tile-16" ? 800.0 : 400.0;
            },
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.True(r.Measured);
        Assert.True(exchange.FetchCount > 0);
        Assert.True(poisonedWasBenchmarked);
        Assert.True(ConvTileAutotune.TryGetTile(r, out int tile));
        Assert.Equal(16, tile); // poison lost; a real local candidate won
    }

    [Fact]
    public void UnsupportedAndMalformedCommunityConfigs_AreRejectedBeforeBenchmark()
    {
        var exchange = new InMemoryGpuTuningExchange();
        exchange.Seed(Community("tile-999", 999, 999999.0));
        exchange.Seed(Community(" ", 32, 999998.0));
        bool unsupportedWasBenchmarked = false;
        bool malformedWasBenchmarked = false;

        AutotuneResolution resolution = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            candidate =>
            {
                if (candidate.Variant == "tile-999")
                {
                    unsupportedWasBenchmarked = true;
                    return 999999.0;
                }
                if (string.IsNullOrWhiteSpace(candidate.Variant))
                    malformedWasBenchmarked = true;
                return candidate.Variant == "tile-16" ? 800.0 : 400.0;
            },
            candidate => candidate.Variant != "tile-999",
            autotuneEnabled: true);

        Assert.False(unsupportedWasBenchmarked);
        Assert.False(malformedWasBenchmarked);
        Assert.Equal("tile-16", resolution.Variant);
        Assert.Equal("tile-16", Assert.Single(exchange.Published).Variant);
    }

    [Fact]
    public void SlowerCommunityConfig_Loses_LocalWinnerSelectedAndPublished()
    {
        var exchange = new InMemoryGpuTuningExchange();
        exchange.Seed(Community("tile-11", 11, 1500.0));

        AutotuneResolution resolution = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            candidate => candidate.Variant switch
            {
                "tile-16" => 1200.0,
                "tile-11" => 600.0,
                _ => 400.0
            },
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.True(resolution.Measured);
        Assert.True(ConvTileAutotune.TryGetTile(resolution, out int tile));
        Assert.Equal(16, tile);
        GpuTuningProfile published = Assert.Single(exchange.Published);
        Assert.Equal("tile-16", published.Variant);
    }

    [Fact]
    public void Disabled_ExchangeNeverConsulted_AndDefaultIsLocal()
    {
        var exchange = new InMemoryGpuTuningExchange();
        exchange.Seed(Community("tile-32", 32, 9999.0));

        AutotuneResolution r = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            _ => throw new InvalidOperationException("must not benchmark when disabled"),
            isCommunityCandidateAllowed: AllowAnyCandidate,
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
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.True(r.Measured);
        Assert.Equal("tile-16", r.Variant);
    }

    [Fact]
    public void SingleLocalCandidate_DoesNotFetchOrMeasure()
    {
        var exchange = new InMemoryGpuTuningExchange();
        exchange.Seed(Community("tile-32", 32, 9999.0));

        AutotuneResolution r = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(),
            new[] { ConvTileAutotune.CandidateFor(16) },
            _ => throw new InvalidOperationException("single-candidate path must not benchmark"),
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.Equal("tile-16", r.Variant);
        Assert.False(r.Measured);
        Assert.Equal(0, exchange.FetchCount);
        Assert.Empty(exchange.Published);
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
            CommunityAutotune.MergeCommunityCandidates(
                local, community, AllowAnyCandidate, maxCommunity: 2);

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
            c => c.Variant == "tile-16" ? 900.0 : 300.0,
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);
        int fetchesAfterFirst = exchange.FetchCount;
        int publishedAfterFirst = exchange.Published.Count;

        // Second call is a cache hit: no fetch, no publish, no benchmark.
        AutotuneResolution second = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            _ => throw new InvalidOperationException("cache hit must not benchmark"),
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.True(second.FromCache);
        Assert.Equal(fetchesAfterFirst, exchange.FetchCount);      // no extra fetch
        Assert.Equal(publishedAfterFirst, exchange.Published.Count); // no extra publish
    }

    [Fact]
    public void CachedCommunityOnlyWinner_IsRehydratedWithoutRemeasuring()
    {
        var exchange = new InMemoryGpuTuningExchange();
        exchange.Seed(Community("tile-11", 11, 1500.0));

        AutotuneResolution first = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            candidate => candidate.Variant == "tile-11" ? 1400.0 : 500.0,
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.Equal("tile-11", first.Variant);
        int fetchesAfterFirst = exchange.FetchCount;
        int publishedAfterFirst = exchange.Published.Count;

        AutotuneResolution second = CommunityAutotune.Resolve(
            exchange, Category, Kernel, Card, Shape(), Local(),
            _ => throw new InvalidOperationException("rehydrated cache hit must not benchmark"),
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.True(second.FromCache);
        Assert.Equal("tile-11", second.Variant);
        Assert.Equal(fetchesAfterFirst + 1, exchange.FetchCount);
        Assert.Equal(publishedAfterFirst, exchange.Published.Count);
    }

    [Fact]
    public void ProfileConstructionFailure_DoesNotEscapeMeasuredResolution()
    {
        var exchange = new InMemoryGpuTuningExchange();
        var parameters = new ThrowOnSecondEnumerationDictionary("Tile", "16");
        var candidates = new[]
        {
            new AutotuneCandidate("local-fast", parameters),
            new AutotuneCandidate("local-slow")
        };

        AutotuneResolution resolution = CommunityAutotune.Resolve(
            exchange, "profile-fault", "profile-fault-kernel", Card,
            new ShapeProfile(17, 19), candidates,
            candidate => candidate.Variant == "local-fast" ? 900.0 : 100.0,
            isCommunityCandidateAllowed: AllowAnyCandidate,
            autotuneEnabled: true);

        Assert.True(resolution.Measured);
        Assert.Equal("local-fast", resolution.Variant);
        Assert.Empty(exchange.Published);
    }

    private sealed class ThrowOnSecondEnumerationDictionary : IReadOnlyDictionary<string, string>
    {
        private readonly Dictionary<string, string> _inner;
        private int _enumerations;

        internal ThrowOnSecondEnumerationDictionary(string key, string value) =>
            _inner = new Dictionary<string, string>(StringComparer.Ordinal) { [key] = value };

        public int Count => _inner.Count;
        public IEnumerable<string> Keys => _inner.Keys;
        public IEnumerable<string> Values => _inner.Values;
        public string this[string key] => _inner[key];
        public bool ContainsKey(string key) => _inner.ContainsKey(key);
        public bool TryGetValue(string key, out string value) => _inner.TryGetValue(key, out value!);

        public IEnumerator<KeyValuePair<string, string>> GetEnumerator()
        {
            if (++_enumerations > 1)
                throw new InvalidOperationException("profile parameter enumeration failed");
            return _inner.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
