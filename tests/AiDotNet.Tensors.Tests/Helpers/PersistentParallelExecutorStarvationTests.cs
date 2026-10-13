using System;
using System.Diagnostics;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// #1248: a woken worker that does not get a core must not stall a plain dispatch. Each participant used to own a
/// fixed strided chunk set (p, p + stride, ...), so the dispatch waited for every woken worker to run its set; ONNX
/// Runtime's spinning pool starved one of ours and a ~40 µs MatMul took 21.8 ms. Plain dispatches now claim chunks
/// from a shared counter.
/// </summary>
[Collection("CompilationGlobalState")]
public class PersistentParallelExecutorStarvationTests
{
    private static void SkipUnlessPool()
    {
        Skip.If(!CpuParallelSettings.UseCooperativePool, "The cooperative pool is disabled (AIDOTNET_COOP_POOL=0).");
        Skip.If(PersistentParallelExecutor.Instance.WorkerCount < 1 || CpuParallelSettings.MaxDegreeOfParallelism < 2,
            "Needs at least one worker and a degree of parallelism of 2.");
    }

    // The worker's first chunk blocks until every OTHER chunk has finished, which stands in for a worker the OS
    // never schedules. With fixed strided sets the worker owns half the chunks, so they can never finish without it
    // and the wait times out; with claiming the dispatching thread takes them.
    [SkippableFact]
    public void PlainDispatch_CompletesWithoutTheStalledWorker()
    {
        SkipUnlessPool();
        const int chunks = 64;
        int dispatcher = Environment.CurrentManagedThreadId;
        int finished = 0;
        int stalled = 0;
        bool waitTimedOut = false;
        var ran = new int[chunks];

        PersistentParallelExecutor.Instance.Execute(chunks, maxDop: 2, c =>
        {
            Interlocked.Increment(ref ran[c]);
            if (Environment.CurrentManagedThreadId != dispatcher && Interlocked.CompareExchange(ref stalled, 1, 0) == 0)
            {
                var sw = Stopwatch.StartNew();
                while (Volatile.Read(ref finished) < chunks - 1)
                {
                    if (sw.Elapsed > TimeSpan.FromSeconds(10)) { waitTimedOut = true; break; }
                    Thread.Sleep(1);
                }
            }
            Interlocked.Increment(ref finished);
        });

        Assert.False(waitTimedOut, "The other chunks waited on the stalled worker instead of being claimed by the caller.");
        Assert.Equal(chunks, finished);
        for (int c = 0; c < chunks; c++) Assert.Equal(1, ran[c]);
    }

    // Many back-to-back dispatches with random stalls: every chunk runs exactly once per dispatch, and a late
    // worker never re-runs or steals from a later dispatch.
    [SkippableFact]
    public void PlainDispatch_EveryChunkRunsExactlyOnce_UnderStalls()
    {
        SkipUnlessPool();
        var rng = new Random(1248);
        for (int d = 0; d < 300; d++)
        {
            int chunks = 2 + rng.Next(40);
            int sleepChunk = rng.Next(chunks);
            var ran = new int[chunks];
            PersistentParallelExecutor.Instance.Execute(chunks, c =>
            {
                Interlocked.Increment(ref ran[c]);
                if (c == sleepChunk && (d & 7) == 0) Thread.Sleep(1);
            });
            for (int c = 0; c < chunks; c++) Assert.True(ran[c] == 1, $"dispatch {d}: chunk {c} ran {ran[c]} times");
        }
    }

    // Thread-local dispatches keep fixed per-participant sets: each chunk once, one local per participant.
    [SkippableFact]
    public void ThreadLocalDispatch_EachChunkOnce_OneLocalPerParticipant()
    {
        SkipUnlessPool();
        for (int d = 0; d < 200; d++)
        {
            const int chunks = 37;
            var ran = new int[chunks];
            int inits = 0, finals = 0;
            PersistentParallelExecutor.Instance.Execute<int[]>(chunks, maxDop: 0,
                localInit: () => { Interlocked.Increment(ref inits); return new int[1]; },
                body: (c, local) => { local[0]++; Interlocked.Increment(ref ran[c]); },
                localFinally: local => Interlocked.Increment(ref finals));
            for (int c = 0; c < chunks; c++) Assert.Equal(1, ran[c]);
            Assert.Equal(inits, finals);
            Assert.InRange(inits, 1, PersistentParallelExecutor.Instance.MaxParticipants);
        }
    }
}