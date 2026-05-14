using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Issue #319 regression: verifies that work-aware
/// <see cref="CpuParallelSettings.LightweightParallel(int, long, Action{int})"/>
/// honors <see cref="PersistentParallelExecutor.DefaultSerialGrainSize"/>.
/// Small total-work runs inline on the calling thread (no
/// PersistentParallelExecutor dispatch, no <c>LowLevelLifoSemaphore</c>
/// wakeup); above-threshold work fans out to the persistent worker pool.
///
/// PR #343 review note: tests originally exercised
/// <c>ParallelForOrSerial</c>, but that helper dispatches to raw
/// <c>System.Threading.Tasks.Parallel.For</c> above grain size — which
/// still parks workers on <c>LowLevelLifoSemaphore</c>. The CpuEngine
/// call sites this PR converts ALL route through <c>LightweightParallel</c>
/// (PersistentParallelExecutor), so the regression tests cover that
/// helper instead. Above-grain detection uses a
/// <see cref="ManualResetEventSlim"/> probe — same deterministic
/// pattern as <c>PersistentExecutorGrainSizeTests</c> — rather than a
/// <c>SpinWait</c> race that could complete before the scheduler runs
/// a second thread.
/// </summary>
public class GrainSizeDispatchTests
{
    [Fact]
    public void LightweightParallel_BelowGrainSize_RunsInlineOnCallingThread()
    {
        int callerThread = Thread.CurrentThread.ManagedThreadId;
        var observed = new int[16];

        // Total work explicitly below DefaultSerialGrainSize (32 K).
        long totalWork = 1024;
        CpuParallelSettings.LightweightParallel(16, totalWork, c =>
        {
            observed[c] = Thread.CurrentThread.ManagedThreadId;
        });

        // Every iteration must have run on the caller thread — inline path.
        for (int c = 0; c < observed.Length; c++)
            Assert.Equal(callerThread, observed[c]);
    }

    [Fact]
    public void LightweightParallel_AtOrAboveGrainSize_FansOutToWorkers()
    {
        // Skip when the host is single-core: no workers to dispatch to.
        if (Environment.ProcessorCount <= 1)
            return;

        int callerThread = Thread.CurrentThread.ManagedThreadId;
        int numChunks = Math.Max(2, Environment.ProcessorCount);
        long totalWork = (long)PersistentParallelExecutor.DefaultSerialGrainSize * 4;
        var observed = new int[numChunks];

        // Synchronization-based probe (mirrors PersistentExecutorGrainSizeTests):
        // caller-thread iterations Wait briefly for any worker to signal,
        // worker-thread iterations Set the event. If no worker ever runs
        // (the bug we're catching), the Wait times out and we fail with a
        // clean diagnostic. Replaces the prior SpinWait(2000) which could
        // complete before the scheduler woke another thread on a busy CI box.
        using var workerSeen = new ManualResetEventSlim(false);
        CpuParallelSettings.LightweightParallel(numChunks, totalWork, c =>
        {
            int tid = Thread.CurrentThread.ManagedThreadId;
            observed[c] = tid;
            if (tid != callerThread)
                workerSeen.Set();
            else
                workerSeen.Wait(TimeSpan.FromMilliseconds(200));
        });

        bool sawWorker = false;
        for (int c = 0; c < numChunks; c++)
            if (observed[c] != callerThread) { sawWorker = true; break; }
        Assert.True(sawWorker && workerSeen.IsSet,
            $"All {numChunks} chunks ran on caller thread {callerThread} despite "
            + $"totalWork ({totalWork}) >> grain size "
            + $"({PersistentParallelExecutor.DefaultSerialGrainSize}). "
            + $"LightweightParallel should have dispatched to the worker pool. "
            + $"workerSeen.IsSet={workerSeen.IsSet}.");
    }

    [Fact]
    public void LightweightParallel_EmptyRange_NoOp()
    {
        int callCount = 0;
        CpuParallelSettings.LightweightParallel(0, totalWork: 1_000_000, _ =>
            Interlocked.Increment(ref callCount));
        Assert.Equal(0, callCount);
    }

    [Fact]
    public void LightweightParallel_ExactlyAtGrainSize_AllowsParallel()
    {
        // Exact-threshold edge case: totalWork == DefaultSerialGrainSize
        // is NOT below the threshold, so parallel dispatch is allowed.
        long totalWork = PersistentParallelExecutor.DefaultSerialGrainSize;
        int callCount = 0;
        CpuParallelSettings.LightweightParallel(4, totalWork, _ =>
            Interlocked.Increment(ref callCount));
        Assert.Equal(4, callCount);
    }
}
