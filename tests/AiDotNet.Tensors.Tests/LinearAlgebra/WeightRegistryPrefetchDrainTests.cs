// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Covers <c>WeightRegistry.DrainInFlightPrefetches</c> and its use inside
/// <see cref="WeightRegistry.Reset"/> / <see cref="WeightRegistry.Configure"/>.
///
/// <para>Prefetch workers are fire-and-forget (<see cref="WeightRegistry.PrefetchAsync{T}"/>):
/// they capture the pool ref and run <c>Rehydrate</c> on the ThreadPool, releasing their
/// in-flight marker + semaphore permit in a <c>finally</c>. Tearing the pool down
/// (Reset/Configure) while a worker is still in flight is the hazard the drain guards: the
/// worker must not be left paging bytes against a pool we are about to dispose, and — worse —
/// the handle counter restarts when the pool is recreated, so a straggler in-flight marker for
/// (reused) handle 0 would suppress the SUCCESSOR pool's prefetch of its own handle 0. The drain
/// quiesces all workers before teardown; Reset additionally clears the in-flight set so no stale
/// marker survives a timed-out drain.</para>
///
/// <para><c>[Collection("WeightRegistry")]</c> serializes against the other classes that mutate
/// the process-wide <see cref="WeightRegistry"/> singleton.</para>
/// </summary>
[Collection("WeightRegistry")]
public class WeightRegistryPrefetchDrainTests : IDisposable
{
    private readonly string _backingDir;

    public WeightRegistryPrefetchDrainTests()
    {
        _backingDir = Path.Combine(Path.GetTempPath(), "aidotnet-wr-drain-test-" + Guid.NewGuid().ToString("N"));
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32, // tight, so a filler evicts our weight and prefetch has real work
            StreamingBackingStorePath = _backingDir,
        });
    }

    public void Dispose()
    {
        WeightRegistry.Reset();
        if (Directory.Exists(_backingDir))
        {
            try { Directory.Delete(_backingDir, recursive: true); } catch { /* best-effort */ }
        }
    }

    private static Tensor<float> NewStreamingWeight(params float[] values)
    {
        var t = new Tensor<float>(values, new[] { values.Length });
        t.Lifetime = WeightLifetime.Streaming;
        return t;
    }

    [Fact]
    public void DrainInFlightPrefetches_WhenIdle_ReturnsTrue_AndKeepsPoolUsable()
    {
        // With no worker in flight the drain acquires + releases every permit and reports quiesced.
        Assert.True(WeightRegistry.DrainInFlightPrefetches(timeoutMs: 5000));

        // Critically, the permits must be RELEASED again — otherwise the prefetch subsystem would be
        // dead after one drain. Prove it by issuing a real prefetch afterwards and seeing it run.
        var t = NewStreamingWeight(1f, 2f, 3f, 4f);
        WeightRegistry.RegisterWeight(t);
        var filler = NewStreamingWeight(new float[16]);
        WeightRegistry.RegisterWeight(filler);            // evicts t
        Assert.False(WeightRegistry.IsResidentInPool(t));
        WeightRegistry.UnregisterWeight(filler);          // free budget so the prefetch can land

        Assert.True(WeightRegistry.PrefetchAsyncForTesting(t).Wait(TimeSpan.FromSeconds(5)));
        Assert.True(WeightRegistry.IsResidentInPool(t),
            "Prefetch after a drain must still run — the drain left the semaphore permits acquired.");
    }

    [Fact]
    public async Task DrainInFlightPrefetches_WaitsForInFlightWorker_ToComplete()
    {
        // Behavioral proof that the drain BLOCKS until the worker's Rehydrate finishes, rather than
        // returning early. Issue a prefetch (worker queued, doing a disk read), then drain WITHOUT
        // awaiting the task. If the drain genuinely waits, the bytes are guaranteed resident the
        // instant it returns true. If it returned early, residency would be a race.
        var t = NewStreamingWeight(5f, 6f, 7f, 8f);
        WeightRegistry.RegisterWeight(t);
        var filler = NewStreamingWeight(new float[16]);
        WeightRegistry.RegisterWeight(filler);            // evicts t
        Assert.False(WeightRegistry.IsResidentInPool(t));
        WeightRegistry.UnregisterWeight(filler);

        var workerTask = WeightRegistry.PrefetchAsyncForTesting(t);

        bool drained = WeightRegistry.DrainInFlightPrefetches(timeoutMs: 5000);

        Assert.True(drained, "Drain should report full quiescence within the timeout.");
        Assert.True(WeightRegistry.IsResidentInPool(t),
            "Drain returned before the in-flight prefetch worker finished its Rehydrate — it did not wait.");

        await workerTask; // already complete; observe it so nothing is left dangling
    }

    [Fact]
    public void ResetDuringActivePrefetch_LeavesSuccessorPoolUsable_NoStaleSuppression()
    {
        // The race the drain exists to close: Reset() tears the pool down while a prefetch worker may
        // still be in flight, and the successor pool restarts handle ids from 0 — so a straggler
        // in-flight marker for handle 0 would suppress the successor's own handle-0 prefetch (dedup
        // sees it as "already in flight") AND a worker could page into a pool being disposed.
        //
        // Stress phase: repeatedly register handle 0, fire a prefetch FIRE-AND-FORGET (production
        // shape — never awaited), and immediately Reset. Each Reset must drain whatever it left in
        // flight without throwing or hanging, and clear the in-flight set so nothing carries over.
        const int rounds = 60;
        for (int round = 0; round < rounds; round++)
        {
            WeightRegistry.Reset(); // drains + disposes the PRIOR round's pool (with its in-flight worker)
            WeightRegistry.Configure(new GpuOffloadOptions
            {
                StreamingPoolMaxResidentBytes = 32,
                StreamingBackingStorePath = _backingDir,
            });

            var t = NewStreamingWeight(1f, 2f, 3f, 4f);
            WeightRegistry.RegisterWeight(t);             // handle 0 in this fresh pool
            var filler = NewStreamingWeight(new float[16]);
            WeightRegistry.RegisterWeight(filler);        // evicts t
            WeightRegistry.UnregisterWeight(filler);

            // Fire-and-forget — may still be running its Rehydrate when the next iteration's Reset
            // disposes this pool. That is exactly the teardown-vs-prefetch race under test.
            _ = WeightRegistry.PrefetchAsyncForTesting(t);
        }

        // Final deterministic round: prove the successor pool is fully usable after all that churn.
        // A stale in-flight marker for handle 0 (i.e. a Reset that failed to drain+clear) would make
        // this AWAITED prefetch dedup into a no-op, leaving t non-resident → the assert below fails.
        WeightRegistry.Reset();
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingPoolMaxResidentBytes = 32,
            StreamingBackingStorePath = _backingDir,
        });
        var verify = NewStreamingWeight(11f, 22f, 33f, 44f);
        WeightRegistry.RegisterWeight(verify);            // handle 0 again
        var verifyFiller = NewStreamingWeight(new float[16]);
        WeightRegistry.RegisterWeight(verifyFiller);      // evicts verify
        Assert.False(WeightRegistry.IsResidentInPool(verify), "setup: verify should be evicted");
        WeightRegistry.UnregisterWeight(verifyFiller);

        // Await THIS prefetch (the only one in flight for handle 0) — it does the real Rehydrate.
        Assert.True(WeightRegistry.PrefetchAsyncForTesting(verify).Wait(TimeSpan.FromSeconds(5)),
            "final prefetch did not complete — a stale in-flight marker likely suppressed handle 0.");
        Assert.True(WeightRegistry.IsResidentInPool(verify),
            "handle 0 not resident after prefetch — Reset failed to drain/clear, leaving the successor unusable.");

        WeightRegistry.Materialize(verify);
        var span = verify.DataVector.AsSpan();
        Assert.Equal(11f, span[0]);
        Assert.Equal(44f, span[3]);
    }
}
