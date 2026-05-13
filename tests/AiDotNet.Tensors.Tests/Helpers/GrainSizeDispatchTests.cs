using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Issue #319 regression: verifies that work-aware
/// <see cref="CpuParallelSettings.ParallelForOrSerial"/> honors the
/// <see cref="PersistentParallelExecutor.DefaultSerialGrainSize"/>
/// threshold — small total-work runs inline on the calling thread (no
/// PersistentParallelExecutor dispatch, no LowLevelLifoSemaphore wakeup).
///
/// Pre-fix, hot ViT-Base ops fan-out via raw <c>Parallel.For</c> regardless
/// of work size, paying ~50µs of dispatch overhead per call even when the
/// per-iteration body is microseconds.
/// </summary>
public class GrainSizeDispatchTests
{
    [Fact]
    public void ParallelForOrSerial_BelowGrainSize_RunsInlineOnCallingThread()
    {
        int callerThread = System.Threading.Thread.CurrentThread.ManagedThreadId;
        var observedThreads = new System.Collections.Concurrent.ConcurrentBag<int>();

        // Total work explicitly below DefaultSerialGrainSize (32 K).
        long totalWork = 1024;
        CpuParallelSettings.ParallelForOrSerial(0, 16, totalWork, _ =>
        {
            observedThreads.Add(System.Threading.Thread.CurrentThread.ManagedThreadId);
        });

        Assert.Equal(16, observedThreads.Count);
        // Every iteration must have run on the caller thread — inline path.
        foreach (var tid in observedThreads)
            Assert.Equal(callerThread, tid);
    }

    [Fact]
    public void ParallelForOrSerial_AtOrAboveGrainSize_FansOutToWorkers()
    {
        // Skip when the host is single-core: ParallelForOrSerial collapses
        // to inline regardless of work size when MaxDegreeOfParallelism <= 1.
        if (CpuParallelSettings.MaxDegreeOfParallelism <= 1)
            return;

        int callerThread = System.Threading.Thread.CurrentThread.ManagedThreadId;
        var observedThreads = new System.Collections.Concurrent.ConcurrentBag<int>();

        // Total work well above DefaultSerialGrainSize (32 K).
        long totalWork = 1_000_000;
        CpuParallelSettings.ParallelForOrSerial(0, 64, totalWork, _ =>
        {
            // Modest body so the parallel dispatch has time to spin workers up.
            System.Threading.Thread.SpinWait(2000);
            observedThreads.Add(System.Threading.Thread.CurrentThread.ManagedThreadId);
        });

        Assert.Equal(64, observedThreads.Count);
        // At least one iteration must have run off-thread.
        bool anyOffThread = false;
        foreach (var tid in observedThreads)
            if (tid != callerThread) { anyOffThread = true; break; }
        Assert.True(anyOffThread,
            "ParallelForOrSerial with work > grain size must dispatch off the caller thread.");
    }

    [Fact]
    public void ParallelForOrSerial_EmptyRange_NoOp()
    {
        int callCount = 0;
        CpuParallelSettings.ParallelForOrSerial(0, 0, totalWork: 1_000_000, _ => callCount++);
        Assert.Equal(0, callCount);
    }

    [Fact]
    public void ParallelForOrSerial_ExactlyAtGrainSize_AllowsParallel()
    {
        // Exact-threshold edge case: totalWork == DefaultSerialGrainSize
        // is NOT below the threshold, so parallel dispatch is allowed
        // (matches the < comparison in PersistentParallelExecutor.Execute).
        long totalWork = PersistentParallelExecutor.DefaultSerialGrainSize;
        int callCount = 0;
        CpuParallelSettings.ParallelForOrSerial(0, 4, totalWork, _ =>
        {
            System.Threading.Interlocked.Increment(ref callCount);
        });
        Assert.Equal(4, callCount);
    }
}
