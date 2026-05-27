using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Coverage for <see cref="CpuParallelSettings.ParallelForOrSerial(int,int,long,System.Action{int})"/>
/// and its localInit/localFinally overload, focused on the
/// <see cref="CpuParallelSettings.DeterministicReductions"/> dispatch gate added
/// for bit-reproducible training (PR #443). Verifies that (1) large total-work
/// fans out to worker threads when the flag is false, and (2) the same large
/// work runs entirely on the calling thread when the flag is true. Every test
/// saves and restores the global flag so it can't leak into other tests.
/// </summary>
public class ParallelForOrSerialDispatchTests
{
    [Fact]
    public void ParallelForOrSerial_LargeWork_DeterministicFalse_FansOutToWorkers()
    {
        // No worker pool to fan out to on a single-core host.
        if (Environment.ProcessorCount <= 1)
            return;

        bool savedDeterministic = CpuParallelSettings.DeterministicReductions;
        int savedMaxDegree = CpuParallelSettings.MaxDegreeOfParallelism;
        ThreadPool.GetMinThreads(out int savedMinWorker, out int savedMinIo);
        try
        {
            CpuParallelSettings.DeterministicReductions = false;
            CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
            // Pre-spawn worker threads so fan-out doesn't hinge on a cold ThreadPool
            // warming up — the dominant flake source on a shared 4-vCPU CI runner
            // where sibling xUnit tests saturate the pool (#447 / #451 / #455).
            ThreadPool.SetMinThreads(Math.Max(savedMinWorker, Environment.ProcessorCount), savedMinIo);

            int callerThread = Thread.CurrentThread.ManagedThreadId;
            // totalWork >> grain forces ParallelForOrSerial down the PARALLEL branch
            // (Parallel.For). But TPL is still ALLOWED to inline that parallel branch
            // entirely on the caller when the pool is momentarily starved — that is a
            // TPL scheduling choice, NOT a ParallelForOrSerial regression. So we retry
            // the worker-probe: a genuine serial-dispatch regression (wrong branch
            // taken) fails EVERY attempt; a transient starvation passes one. Each
            // attempt blocks ONE caller iteration briefly (Interlocked-designated
            // waiter — the rest continue so a real regression fails fast) to give TPL
            // a window to recruit a worker that Sets the event.
            int numChunks = Math.Max(16, Environment.ProcessorCount * 4);
            long totalWork = (long)PersistentParallelExecutor.DefaultSerialGrainSize * 4;
            const int attempts = 8;
            bool sawWorker = false;

            for (int attempt = 0; attempt < attempts && !sawWorker; attempt++)
            {
                var observed = new int[numChunks];
                using var workerSeen = new ManualResetEventSlim(false);
                int callerWaiterClaimed = 0;
                CpuParallelSettings.ParallelForOrSerial(0, numChunks, totalWork, c =>
                {
                    int tid = Thread.CurrentThread.ManagedThreadId;
                    observed[c] = tid;
                    if (tid != callerThread)
                        workerSeen.Set();
                    else if (Interlocked.Exchange(ref callerWaiterClaimed, 1) == 0)
                        workerSeen.Wait(TimeSpan.FromSeconds(2));
                });

                for (int c = 0; c < numChunks; c++)
                    if (observed[c] != callerThread) { sawWorker = true; break; }
            }

            Assert.True(sawWorker,
                $"totalWork ({totalWork}) >> grain size "
                + $"({PersistentParallelExecutor.DefaultSerialGrainSize}) with DeterministicReductions=false "
                + $"should have dispatched to the worker pool, but across {attempts} attempts all "
                + $"{numChunks} chunks ran on caller thread {callerThread} — ParallelForOrSerial took the "
                + "serial branch when it should have parallelized.");
        }
        finally
        {
            CpuParallelSettings.DeterministicReductions = savedDeterministic;
            CpuParallelSettings.MaxDegreeOfParallelism = savedMaxDegree;
            ThreadPool.SetMinThreads(savedMinWorker, savedMinIo);
        }
    }

    [Fact]
    public void ParallelForOrSerial_DeterministicReductions_RunsOnCallerThread()
    {
        bool savedDeterministic = CpuParallelSettings.DeterministicReductions;
        try
        {
            CpuParallelSettings.DeterministicReductions = true;

            int callerThread = Thread.CurrentThread.ManagedThreadId;
            int numChunks = Math.Max(4, Environment.ProcessorCount * 2);
            // Well above the grain size — so serial dispatch can ONLY be due to
            // the DeterministicReductions flag, not the work-size threshold.
            long totalWork = (long)PersistentParallelExecutor.DefaultSerialGrainSize * 8;
            var observed = new int[numChunks];

            CpuParallelSettings.ParallelForOrSerial(0, numChunks, totalWork, c =>
            {
                observed[c] = Thread.CurrentThread.ManagedThreadId;
            });

            for (int c = 0; c < numChunks; c++)
                Assert.Equal(callerThread, observed[c]);
        }
        finally
        {
            CpuParallelSettings.DeterministicReductions = savedDeterministic;
        }
    }

    [Fact]
    public void ParallelForOrSerial_Generic_DeterministicReductions_RunsOnCallerThread()
    {
        bool savedDeterministic = CpuParallelSettings.DeterministicReductions;
        try
        {
            CpuParallelSettings.DeterministicReductions = true;

            int callerThread = Thread.CurrentThread.ManagedThreadId;
            int numChunks = Math.Max(4, Environment.ProcessorCount * 2);
            long totalWork = (long)PersistentParallelExecutor.DefaultSerialGrainSize * 8;
            var observed = new int[numChunks];
            long sum = 0;

            CpuParallelSettings.ParallelForOrSerial(
                0, numChunks, totalWork,
                localInit: () => 0L,
                body: (i, _, local) =>
                {
                    observed[i] = Thread.CurrentThread.ManagedThreadId;
                    return local + i;
                },
                localFinally: local => Interlocked.Add(ref sum, local));

            for (int c = 0; c < numChunks; c++)
                Assert.Equal(callerThread, observed[c]);
            // localFinally must still run once and aggregate every iteration.
            Assert.Equal((long)numChunks * (numChunks - 1) / 2, sum);
        }
        finally
        {
            CpuParallelSettings.DeterministicReductions = savedDeterministic;
        }
    }
}
