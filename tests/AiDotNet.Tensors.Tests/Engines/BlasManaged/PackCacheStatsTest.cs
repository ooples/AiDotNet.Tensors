using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-E (#373) acceptance — the pack-cache hit/miss counters must increment
/// correctly when a pre-pack handle is reused, and <see cref="WeightPackHandle.MarkDirty"/>
/// must force a re-pack on the next call.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]  // Serialize with other stats-counter tests.
public class PackCacheStatsTest
{
    [Fact]
    public void Reusing_PackedB_Across_Calls_Increments_PackCacheHits()
    {
        const int M = 64, N = 128, K = 128;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.ClearCaches();
        var handle = BlasManagedLib.PrePackB<float>(b, N, false, K, N);
        try
        {
            var opts = new BlasOptions<float> { PackedB = handle };
            var statsBefore = BlasManagedLib.GetStats();

            // Drive 3 GEMMs with the same handle — every consume site inside
            // each call should record a hit, none should miss (handle is
            // current the whole time).
            for (int it = 0; it < 3; it++)
            {
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K, opts);
            }

            var statsAfter = BlasManagedLib.GetStats();
            long hits = statsAfter.PackCacheHits - statsBefore.PackCacheHits;
            long misses = statsAfter.PackCacheMisses - statsBefore.PackCacheMisses;

            Assert.True(hits > 0,
                $"Expected ≥1 PackCacheHit across 3 calls with reused handle, got {hits} (misses={misses})");
            Assert.Equal(0, misses);
        }
        finally
        {
            handle.Dispose();
        }
    }

    [Fact]
    public void MarkDirty_Forces_RePack_On_Next_Call()
    {
        const int M = 64, N = 128, K = 128;
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.ClearCaches();
        var handle = BlasManagedLib.PrePackB<float>(b, N, false, K, N);
        try
        {
            var opts = new BlasOptions<float> { PackedB = handle };

            // Warm: first call after PrePack — cache is current, every consume = hit.
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K, opts);
            var statsAfterWarm = BlasManagedLib.GetStats();
            long hitsBeforeDirty = statsAfterWarm.PackCacheHits;
            long missesBeforeDirty = statsAfterWarm.PackCacheMisses;

            // Mutate weights and signal cache-dirty.
            for (int i = 0; i < b.Length; i++) b[i] *= 1.5f;
            handle.MarkDirty();

            // Next call should NOT take the cached path — IsCacheCurrent==false
            // forces the consume to fall through to live pack. Every per-tile
            // consume site that found the handle non-current = miss.
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, c, N, M, N, K, opts);
            var statsAfterDirty = BlasManagedLib.GetStats();

            long hitsDelta = statsAfterDirty.PackCacheHits - hitsBeforeDirty;
            long missesDelta = statsAfterDirty.PackCacheMisses - missesBeforeDirty;

            Assert.Equal(0, hitsDelta);
            Assert.True(missesDelta > 0,
                $"MarkDirty should force misses on next call; got hits={hitsDelta}, misses={missesDelta}");
        }
        finally
        {
            handle.Dispose();
        }
    }

    [Fact]
    public void PrePackB_Allocation_Records_PackCacheBytes()
    {
        const int K = 64, N = 64;
        var b = new float[K * N];
        for (int i = 0; i < b.Length; i++) b[i] = i * 0.001f;

        BlasManagedLib.ClearCaches();
        var statsBefore = BlasManagedLib.GetStats();
        var handle = BlasManagedLib.PrePackB<float>(b, N, false, K, N);
        try
        {
            var statsAfter = BlasManagedLib.GetStats();
            long delta = statsAfter.PackCacheBytes - statsBefore.PackCacheBytes;
            Assert.True(delta > 0, $"PrePackB should record positive bytes delta; got {delta}");
        }
        finally
        {
            handle.Dispose();
        }
    }
}
