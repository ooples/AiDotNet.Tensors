using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Tests that depend on the global <c>BlasManagedStatsTracker</c> counters
/// being in a known state. xUnit's default behavior runs test classes in
/// parallel, which races with these globals. The <c>[Collection]</c>
/// attribute serializes all tests in the same collection name, eliminating
/// the race.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class StatsCounterTests
{
    // ── H5: BlasManagedStatsTracker + GetStats/ClearCaches ──────────────────

    [Fact]
    public void BlasManaged_GetStats_TracksAutotuneHitsAndMisses()
    {
        BlasManagedLib.ClearCaches();  // Reset counters.
        var initial = BlasManagedLib.GetStats();
        Assert.Equal(0L, initial.AutotuneHits);
        Assert.Equal(0L, initial.AutotuneMisses);

        // Trigger an autotune call with a unique prime-shaped call (likely unseen).
        int m = 9967, n = 9967, k = 9967;

        AutotuneDispatcher.Decide<double>(
            m, n, k,
            transA: false, transB: false,
            mr: 8, nr: 16,
            procs: 8,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.Auto);

        var afterFirst = BlasManagedLib.GetStats();
        // Either hit or miss happened; one of the counters incremented.
        Assert.True(afterFirst.AutotuneHits + afterFirst.AutotuneMisses >= 1);

        // Second call to same shape — should be a cache hit now.
        AutotuneDispatcher.Decide<double>(
            m, n, k,
            transA: false, transB: false,
            mr: 8, nr: 16,
            procs: 8,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.Auto);

        var afterSecond = BlasManagedLib.GetStats();
        // Total calls = 2; hits + misses ≥ 2.
        Assert.True(afterSecond.AutotuneHits + afterSecond.AutotuneMisses >= 2);
        // And the second call should have been a hit.
        Assert.True(afterSecond.AutotuneHits > afterFirst.AutotuneHits);
    }

    [Fact]
    public void BlasManaged_GetStats_DisableAutotune_DoesNotIncrementCounters()
    {
        BlasManagedLib.ClearCaches();
        var initial = BlasManagedLib.GetStats();

        AutotuneDispatcher.Decide<double>(
            m: 64, n: 64, k: 64,
            transA: false, transB: false,
            mr: 8, nr: 16,
            procs: 4,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.DisableAutotune);

        var after = BlasManagedLib.GetStats();
        Assert.Equal(initial.AutotuneHits, after.AutotuneHits);
        Assert.Equal(initial.AutotuneMisses, after.AutotuneMisses);
    }

    [Fact]
    public void BlasManaged_ClearCaches_ResetsCounters()
    {
        // Trigger an autotune call.
        AutotuneDispatcher.Decide<double>(
            m: 64, n: 64, k: 64,
            transA: false, transB: false,
            mr: 8, nr: 16,
            procs: 4,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.Auto);
        var before = BlasManagedLib.GetStats();
        Assert.True(before.AutotuneHits + before.AutotuneMisses >= 1);

        BlasManagedLib.ClearCaches();
        var after = BlasManagedLib.GetStats();
        Assert.Equal(0L, after.AutotuneHits);
        Assert.Equal(0L, after.AutotuneMisses);
    }

    [Fact]
    public void AutotuneDispatcher_CallSameShape10Times_LaterCallsHitCache()
    {
        BlasManagedLib.ClearCaches();
        var initial = BlasManagedLib.GetStats();
        Assert.Equal(0L, initial.AutotuneHits);
        Assert.Equal(0L, initial.AutotuneMisses);

        // Use a unique shape (primes) so the first call is guaranteed-likely a miss.
        // (If the test ran before on this dev machine, the on-disk cache may already
        // have this entry; in that case all 10 calls are hits — still a valid result.)
        int m = 1009, n = 1013, k = 1019;  // distinct primes
        int mr = 8, nr = 16;

        for (int i = 0; i < 10; i++)
        {
            AutotuneDispatcher.Decide<double>(
                m, n, k,
                transA: false, transB: false,
                mr: mr, nr: nr,
                procs: 8,
                isDeterministic: false,
                hasEpilogue: false,
                packingMode: PackingMode.Auto);
        }

        var after = BlasManagedLib.GetStats();

        // Total dispatches = hits + misses = 10.
        Assert.Equal(10L, after.AutotuneHits + after.AutotuneMisses);
        // At most 1 miss (the first call). On a cached-from-prior-run scenario, 0 misses.
        Assert.True(after.AutotuneMisses <= 1, $"Expected at most 1 miss, got {after.AutotuneMisses}");
        // At least 9 hits.
        Assert.True(after.AutotuneHits >= 9, $"Expected at least 9 hits, got {after.AutotuneHits}");
    }

    [Fact]
    public void JittedKernelCache_HitIncrementsStats()
    {
        BlasManagedLib.ClearCaches();
        var key = new KernelKey { M = 4, N = 4, K = 4, ElemType = typeof(double) };

        // Capture baseline BEFORE Store so that both Store (emission) and
        // TryGetJittedKernel (cache hit) produce visible deltas.
        var before = BlasManagedLib.GetStats();
        JittedKernelCache.Store(key, (Action)(() => { }));  // increments JitEmissions
        JittedKernelCache.TryGetJittedKernel(key);          // increments JitCacheHits
        var after = BlasManagedLib.GetStats();

        Assert.True(after.JitCacheHits > before.JitCacheHits);
        Assert.True(after.JitEmissions > before.JitEmissions);  // Store counted as emission.
    }

    [Fact]
    public void BlasManaged_ClearCaches_ClearsJitCache()
    {
        var key = new KernelKey { M = 4, N = 4, K = 4, ElemType = typeof(double) };
        JittedKernelCache.Store(key, (Action)(() => { }));
        Assert.True(JittedKernelCache.Count > 0);

        BlasManagedLib.ClearCaches();
        Assert.Equal(0, JittedKernelCache.Count);
    }
}
