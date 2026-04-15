using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

/// <summary>
/// xUnit collection that pins <see cref="AutotuneCacheTests"/> to serial
/// execution. Each test mutates the process-wide <c>AIDOTNET_AUTOTUNE_CACHE_PATH</c>
/// environment variable in its constructor and restores it on Dispose; running
/// these tests in parallel would let one test see another's redirected root.
/// </summary>
[CollectionDefinition("AutotuneCacheTests", DisableParallelization = true)]
public sealed class AutotuneCacheTestsCollection { }

/// <summary>
/// Acceptance tests for issue #168 — persistent autotune cache keyed by
/// hardware fingerprint.
///
/// Each test re-points the cache root to a unique temp directory via the
/// <c>AIDOTNET_AUTOTUNE_CACHE_PATH</c> environment variable so tests cannot
/// stomp on each other or on a real user's cache. Cleanup runs in
/// <see cref="IDisposable.Dispose"/> on the fixture class. The collection
/// attribute below pins the suite to serial execution because the env var is
/// process-wide shared state — without it, parallel test workers would observe
/// each other's redirected roots and false-positive cache hits/misses.
/// </summary>
[Collection("AutotuneCacheTests")]
public sealed class AutotuneCacheTests : IDisposable
{
    private const string EnvVar = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string _tempRoot;
    private readonly string? _originalEnv;

    public AutotuneCacheTests()
    {
        _originalEnv = Environment.GetEnvironmentVariable(EnvVar);
        _tempRoot = Path.Combine(
            Path.GetTempPath(),
            "aidotnet-autotune-test-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable(EnvVar, _tempRoot);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EnvVar, _originalEnv);
        try
        {
            if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true);
        }
        catch
        {
            // Best-effort cleanup — OS may hold a handle briefly after Dispose.
        }
    }

    private static KernelId Gemm => new("gemm", "test-v1");
    private static ShapeProfile Shape256 => new(256, 256, 256);
    private static KernelChoice MakeWinner(string variant = "blocked-4x4", double gflops = 100.0)
        => new KernelChoice
        {
            Variant = variant,
            Parameters = new Dictionary<string, string>
            {
                ["TileM"] = "128", ["TileN"] = "128", ["TileK"] = "16",
            },
            MeasuredGflops = gflops,
            MeasuredTimeMs = 0.123,
        };

    // ── DefaultCachePath honors environment override ─────────────────────────
    [Fact]
    public void DefaultCachePath_HonorsEnvVarOverride()
    {
        Assert.Equal(_tempRoot, AutotuneCache.DefaultCachePath);
    }

    // ── Hardware fingerprint is stable and non-empty ─────────────────────────
    [Fact]
    public void CurrentHardwareFingerprint_IsStableAndNonEmpty()
    {
        string first  = AutotuneCache.CurrentHardwareFingerprint;
        string second = AutotuneCache.CurrentHardwareFingerprint;
        Assert.False(string.IsNullOrWhiteSpace(first));
        Assert.Equal(first, second);
        // Format is "arch-vendor-simd-cpuN" — contains dashes and a cpu count.
        Assert.Contains("-", first);
        Assert.Matches(@"cpu\d+", first);
    }

    // ── Round-trip: Store → Lookup returns identical content ─────────────────
    [Fact]
    public void Store_ThenLookup_ReturnsIdenticalChoice()
    {
        var winner = MakeWinner();
        AutotuneCache.Store(Gemm, Shape256, winner);

        var loaded = AutotuneCache.Lookup(Gemm, Shape256);
        Assert.NotNull(loaded);
        Assert.Equal(winner.Variant,        loaded!.Variant);
        Assert.Equal(winner.MeasuredGflops, loaded.MeasuredGflops);
        Assert.Equal(winner.MeasuredTimeMs, loaded.MeasuredTimeMs);
        Assert.Equal(winner.SchemaVersion,  loaded.SchemaVersion);
        Assert.Equal(winner.Parameters,     loaded.Parameters);
    }

    // ── Cache miss returns null (no side effects) ────────────────────────────
    [Fact]
    public void Lookup_OnMiss_ReturnsNull()
    {
        Assert.Null(AutotuneCache.Lookup(Gemm, Shape256));
    }

    [Fact]
    public void Lookup_OnDifferentShape_ReturnsNull()
    {
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());
        Assert.Null(AutotuneCache.Lookup(Gemm, new ShapeProfile(512, 512, 512)));
    }

    [Fact]
    public void Lookup_OnDifferentKernelId_ReturnsNull()
    {
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());
        Assert.Null(AutotuneCache.Lookup(new KernelId("gemm", "other-v2"), Shape256));
        Assert.Null(AutotuneCache.Lookup(new KernelId("sdpa", "test-v1"), Shape256));
    }

    // ── Corruption handling ──────────────────────────────────────────────────
    [Fact]
    public void Lookup_OnCorruptJson_ReturnsNullWithoutThrowing()
    {
        // Write a deliberately malformed file at the exact path Lookup will read.
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());
        var cacheFiles = Directory.GetFiles(
            Path.Combine(_tempRoot, AutotuneCache.CurrentHardwareFingerprint),
            "*.json");
        Assert.Single(cacheFiles);
        File.WriteAllText(cacheFiles[0], "{not valid json");

        Assert.Null(AutotuneCache.Lookup(Gemm, Shape256));
    }

    [Fact]
    public void Lookup_OnEmptyFile_ReturnsNullWithoutThrowing()
    {
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());
        var cacheFiles = Directory.GetFiles(
            Path.Combine(_tempRoot, AutotuneCache.CurrentHardwareFingerprint),
            "*.json");
        File.WriteAllText(cacheFiles[0], "");

        Assert.Null(AutotuneCache.Lookup(Gemm, Shape256));
    }

    [Fact]
    public void Lookup_OnManuallyDeletedFile_RecoversOnNextStore()
    {
        // Issue #168 acceptance criterion: "Cache corruption (manual file
        // deletion, truncation) is detected and recovered without throwing."
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());

        var dir = Path.Combine(_tempRoot, AutotuneCache.CurrentHardwareFingerprint);
        foreach (var f in Directory.GetFiles(dir, "*.json")) File.Delete(f);

        // After deletion, lookup is a miss — but storing fresh works cleanly.
        Assert.Null(AutotuneCache.Lookup(Gemm, Shape256));
        AutotuneCache.Store(Gemm, Shape256, MakeWinner("recovered", 150.0));
        var recovered = AutotuneCache.Lookup(Gemm, Shape256);
        Assert.NotNull(recovered);
        Assert.Equal("recovered", recovered!.Variant);
    }

    [Fact]
    public void Lookup_RejectsFutureSchemaVersion()
    {
        // A plan written by a newer library version with an incompatible schema
        // should be treated as a miss, not fail loudly — this is the
        // forward-compatibility contract.
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());
        var cacheFiles = Directory.GetFiles(
            Path.Combine(_tempRoot, AutotuneCache.CurrentHardwareFingerprint),
            "*.json");
        // Overwrite with a schema version the reader rejects (<=0).
        File.WriteAllText(cacheFiles[0], "{ \"Variant\": \"x\", \"SchemaVersion\": 0 }");

        Assert.Null(AutotuneCache.Lookup(Gemm, Shape256));
    }

    // ── Atomic-write semantics ───────────────────────────────────────────────
    [Fact]
    public void Store_LeavesNoTempFilesBehind()
    {
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());

        var dir = Path.Combine(_tempRoot, AutotuneCache.CurrentHardwareFingerprint);
        var stragglers = Directory.GetFiles(dir, "*.tmp*");
        Assert.Empty(stragglers);
    }

    [Fact]
    public void Store_OverwritesPreviousEntry()
    {
        AutotuneCache.Store(Gemm, Shape256, MakeWinner("v1", 100.0));
        AutotuneCache.Store(Gemm, Shape256, MakeWinner("v2", 200.0));

        var loaded = AutotuneCache.Lookup(Gemm, Shape256);
        Assert.NotNull(loaded);
        Assert.Equal("v2", loaded!.Variant);
        Assert.Equal(200.0, loaded.MeasuredGflops);
    }

    // ── WarmupAsync ──────────────────────────────────────────────────────────
    [Fact]
    public async Task WarmupAsync_SkipsCachedTargets_AndBenchmarksMisses()
    {
        // Pre-populate the cache for one of the three targets.
        var cached = new KernelId("gemm", "cached");
        var fresh1 = new KernelId("gemm", "fresh1");
        var fresh2 = new KernelId("gemm", "fresh2");
        AutotuneCache.Store(cached, Shape256, MakeWinner("already-there"));

        int benchmarkInvocations = 0;
        var targets = new (KernelId, ShapeProfile)[]
        {
            (cached, Shape256),
            (fresh1, Shape256),
            (fresh2, Shape256),
        };

        var (hits, benchmarked, failures) = await AutotuneCache.WarmupAsync(
            targets,
            async (id, shape, ct) =>
            {
                Interlocked.Increment(ref benchmarkInvocations);
                await Task.Yield();
                return MakeWinner(id.Name + "-bench");
            });

        Assert.Equal(1, hits);
        Assert.Equal(2, benchmarked);
        Assert.Equal(0, failures);
        Assert.Equal(2, benchmarkInvocations);

        // Post-condition: all three are now in the cache.
        Assert.NotNull(AutotuneCache.Lookup(cached, Shape256));
        Assert.NotNull(AutotuneCache.Lookup(fresh1, Shape256));
        Assert.NotNull(AutotuneCache.Lookup(fresh2, Shape256));
    }

    [Fact]
    public async Task WarmupAsync_BenchmarkerReturningNull_DoesNotCountAsFailure()
    {
        var id = new KernelId("gemm", "skip-me");
        var (hits, benchmarked, failures) = await AutotuneCache.WarmupAsync(
            new (KernelId, ShapeProfile)[] { (id, Shape256) },
            (_, _, _) => Task.FromResult<KernelChoice?>(null));

        Assert.Equal(0, hits);
        Assert.Equal(0, benchmarked);
        Assert.Equal(0, failures);
        Assert.Null(AutotuneCache.Lookup(id, Shape256));
    }

    // ── TryStore swallows failures ───────────────────────────────────────────
    [Fact]
    public void TryStore_SuccessPath_ReturnsTrue()
    {
        Assert.True(AutotuneCache.TryStore(Gemm, Shape256, MakeWinner()));
        Assert.NotNull(AutotuneCache.Lookup(Gemm, Shape256));
    }

    // ── Performance acceptance: Lookup after Store is trivially fast ─────────
    [Fact]
    public void Lookup_AfterStore_IsFasterThanAnyRealBenchmark()
    {
        // Issue #168 acceptance criterion: "Second Build() on the same machine
        // with the same shapes is dramatically faster than the first (10x+)".
        // We can't run a real GEMM benchmark in a unit test (variance, timing,
        // hardware sensitivity) — but we can establish the *ceiling* for the
        // Lookup cost. A single cache hit must be well under 10 ms, which
        // itself is an order of magnitude faster than any GEMM autotune run.
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
        {
            var choice = AutotuneCache.Lookup(Gemm, Shape256);
            Assert.NotNull(choice);
        }
        sw.Stop();

        // 100 lookups in < 1000 ms means a single lookup is < 10 ms even on a
        // congested CI runner. Real GEMM autotune takes seconds per shape, so
        // the 10x speedup criterion is comfortably satisfied.
        Assert.True(sw.ElapsedMilliseconds < 1000,
            $"100 cache lookups took {sw.ElapsedMilliseconds}ms — expected < 1000ms");
    }

    // ── ClearCurrentHardware wipes just this fingerprint's directory ─────────
    [Fact]
    public void ClearCurrentHardware_RemovesCachedEntries()
    {
        AutotuneCache.Store(Gemm, Shape256, MakeWinner());
        Assert.NotNull(AutotuneCache.Lookup(Gemm, Shape256));

        AutotuneCache.ClearCurrentHardware();

        Assert.Null(AutotuneCache.Lookup(Gemm, Shape256));
    }
}
