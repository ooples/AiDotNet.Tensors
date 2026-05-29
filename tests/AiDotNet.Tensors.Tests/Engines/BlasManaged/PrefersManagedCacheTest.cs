using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-F3 (#374) task: verifies <see cref="PrefersManagedCache"/> measures both
/// paths on cache miss, caches the winner, and routes <see cref="BlasProvider.TryGemmEx"/>
/// through the cached decision when <see cref="BlasManagedLib.AutotuneRouting"/> is true.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class PrefersManagedCacheTest
{
    public PrefersManagedCacheTest()
    {
        // Sub-F4: redirect DiskPath to a per-test-run temp file so the cache
        // doesn't read/write the user's ~/.aidotnet/autotune/ during tests.
        PrefersManagedCache.DiskPath = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            $"prefers-managed-cache-test-{Guid.NewGuid():N}.json");
        PrefersManagedCache.Clear();
        // Force a clean baseline for the direct-cache tests in this class. NOTE: the
        // PRODUCT default of AutotuneRouting is now `true` (see BlasManaged.AutotuneRouting
        // — managed deterministic-parallel + non-deterministic best-of is the intended
        // CPU GEMM behavior); this ctor resets it to false so per-test setup is explicit.
        BlasManagedLib.AutotuneRouting = false;
        BlasManagedLib.PreferManaged = false;
    }

    [Fact]
    public void Ctor_ResetsAutotuneRouting_ForTestIsolation()
    {
        // The ctor forces AutotuneRouting=false so the direct PrefersManaged tests
        // start from a known state. (The product default is true — asserting that here
        // is impossible because the ctor has already mutated the static.)
        Assert.False(BlasManagedLib.AutotuneRouting);
    }

    [Fact]
    public void First_Call_Populates_Cache()
    {
        PrefersManagedCache.Clear();
        Assert.Equal(0, PrefersManagedCache.Count);

        // Trigger measurement.
        bool result = PrefersManagedCache.PrefersManaged(64, 64, 64, false, false, typeof(float));

        Assert.Equal(1, PrefersManagedCache.Count);
        // Result is true or false depending on which path was faster — we don't
        // assert direction, just that it stabilizes.
        bool result2 = PrefersManagedCache.PrefersManaged(64, 64, 64, false, false, typeof(float));
        Assert.Equal(result, result2);
        Assert.Equal(1, PrefersManagedCache.Count);  // still one entry — cache hit
    }

    [Fact]
    public void Different_Shapes_Get_Different_Cache_Entries()
    {
        PrefersManagedCache.Clear();
        PrefersManagedCache.PrefersManaged(32, 32, 32, false, false, typeof(float));
        PrefersManagedCache.PrefersManaged(64, 64, 64, false, false, typeof(float));
        PrefersManagedCache.PrefersManaged(32, 32, 32, false, false, typeof(double));  // FP64 differs by dtype
        Assert.Equal(3, PrefersManagedCache.Count);
    }

    [Fact]
    public void TryGemmEx_With_AutotuneRouting_Returns_Correct_Output_FP32()
    {
        PrefersManagedCache.Clear();
        BlasManagedLib.AutotuneRouting = true;
        // Autotune routing is a NON-deterministic-mode feature: deterministic mode
        // (the default) forces managed dispatch BEFORE the timing-based autotune is
        // consulted (BlasProvider.ShouldRouteManaged), because a measurement-chosen
        // kernel would not be bit-reproducible. Drop into non-deterministic mode so
        // the autotune cache is actually exercised.
        bool beforeDet = BlasProvider.IsDeterministicMode;
        BlasProvider.SetDeterministicMode(false);
        try
        {
            const int M = 64, N = 64, K = 64;
            var rng = new Random(42);
            var a = new float[M * K];
            var b = new float[K * N];
            var c = new float[M * N];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            bool ok = BlasProvider.TryGemmEx(M, N, K, a, 0, K, false, b, 0, N, false, c, 0, N);
            Assert.True(ok);

            // Output is non-zero (kernel actually ran via cache-routed path).
            bool anyNonZero = false;
            for (int i = 0; i < c.Length; i++) if (c[i] != 0) { anyNonZero = true; break; }
            Assert.True(anyNonZero);

            // Cache was populated.
            Assert.True(PrefersManagedCache.Count >= 1);
        }
        finally
        {
            BlasManagedLib.AutotuneRouting = false;
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }

    [Fact]
    public void Default_Behavior_Unchanged_When_AutotuneRouting_False()
    {
        BlasManagedLib.AutotuneRouting = false;
        BlasManagedLib.PreferManaged = false;
        PrefersManagedCache.Clear();
        // Must be non-deterministic: deterministic mode forces managed dispatch
        // regardless of AutotuneRouting (ShouldRouteManaged), so "autotune off → native
        // path" only holds when determinism isn't forcing managed.
        bool beforeDet = BlasProvider.IsDeterministicMode;
        BlasProvider.SetDeterministicMode(false);
        try
        {
            const int M = 32, N = 32, K = 32;
            var rng = new Random(42);
            var a = new float[M * K];
            var b = new float[K * N];
            var c = new float[M * N];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            bool ok = BlasProvider.TryGemmEx(M, N, K, a, 0, K, false, b, 0, N, false, c, 0, N);

            // With autotune off (and non-deterministic), cache must NOT be touched.
            Assert.Equal(0, PrefersManagedCache.Count);
            // ok matches native availability (same as pre-F3 behavior).
            Assert.Equal(BlasProvider.IsAvailable, ok);
        }
        finally
        {
            BlasProvider.SetDeterministicMode(beforeDet);
        }
    }

    [Fact]
    public void F4_Cache_Round_Trips_Via_Disk()
    {
        // Per-test isolated path (constructor sets one already).
        string path = PrefersManagedCache.DiskPath!;

        // Populate cache.
        PrefersManagedCache.PrefersManaged(64, 64, 64, false, false, typeof(float));
        Assert.Equal(1, PrefersManagedCache.Count);

        // File written.
        Assert.True(System.IO.File.Exists(path), $"DiskPath {path} should exist after measurement");

        // Clear in-memory + re-load.
        PrefersManagedCache.Clear();
        Assert.Equal(0, PrefersManagedCache.Count);
        PrefersManagedCache.LoadFromDisk();
        Assert.Equal(1, PrefersManagedCache.Count);

        // Cleanup.
        try { System.IO.File.Delete(path); } catch { }
    }

    [Fact]
    public void F4_Different_Hardware_Fingerprint_Discards_Disk_Cache()
    {
        string path = PrefersManagedCache.DiskPath!;

        // Manually write a cache file with a fake fingerprint.
        System.IO.File.WriteAllText(path,
            "{\"SchemaVersion\":\"1\",\"HardwareFingerprint\":\"definitely-not-this-host-xyz\"," +
            "\"Entries\":[{\"M\":64,\"N\":64,\"K\":64,\"TransA\":false,\"TransB\":false,\"Dtype\":1,\"PrefersManaged\":true}]}");

        PrefersManagedCache.Clear();
        PrefersManagedCache.LoadFromDisk();

        // Wrong fingerprint → cache should be empty (entries discarded).
        Assert.Equal(0, PrefersManagedCache.Count);

        try { System.IO.File.Delete(path); } catch { }
    }

    [Fact]
    public void PreferManaged_True_Bypasses_Autotune()
    {
        BlasManagedLib.PreferManaged = true;
        BlasManagedLib.AutotuneRouting = true;
        PrefersManagedCache.Clear();
        try
        {
            const int M = 32, N = 32, K = 32;
            var rng = new Random(42);
            var a = new float[M * K];
            var b = new float[K * N];
            var c = new float[M * N];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            bool ok = BlasProvider.TryGemmEx(M, N, K, a, 0, K, false, b, 0, N, false, c, 0, N);
            Assert.True(ok);

            // PreferManaged is checked BEFORE autotune; cache should still be empty.
            Assert.Equal(0, PrefersManagedCache.Count);
        }
        finally
        {
            BlasManagedLib.PreferManaged = false;
            BlasManagedLib.AutotuneRouting = false;
        }
    }
}
