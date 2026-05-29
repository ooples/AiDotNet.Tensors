// Copyright (c) AiDotNet. All rights reserved.
// Phase 2 guard for CooperativeGemmScheduler — the shared work-sharing pool that
// lets concurrent GEMMs interleave their chunks instead of serializing whole
// dispatches (StreamingWorkerPool #492) or blocking (PersistentParallelExecutor).
//
// These pin the correctness properties the scheduler must hold before it can
// replace the legacy pools:
//   1. every chunk of a dispatch runs exactly once;
//   2. a chunk exception is re-thrown to the dispatching caller;
//   3. a nested dispatch (chunk re-enters Dispatch) runs serially — no deadlock;
//   4. MANY concurrent dispatches from many threads all complete with correct,
//      non-cross-contaminated results and no deadlock (the property the legacy
//      pools can't provide in parallel).

using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.BlasManaged.Pool;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class CooperativeSchedulerConcurrencyTests
{
    // Force the parallel path: DefaultSerialGrainSize gates the (chunks,totalWork)
    // overload, but the (chunks,action) overload only runs serial for numChunks==1
    // or nested calls — so multi-chunk dispatches here exercise the real pool.

    [Fact]
    public void Dispatch_RunsEveryChunkExactlyOnce()
    {
        const int chunks = 64;
        var counts = new int[chunks];
        CooperativeGemmScheduler.Dispatch(chunks, c => Interlocked.Increment(ref counts[c]));
        for (int c = 0; c < chunks; c++)
            Assert.Equal(1, counts[c]);
    }

    [Fact]
    public void Dispatch_RethrowsChunkException()
    {
        var ex = Assert.ThrowsAny<Exception>(() =>
            CooperativeGemmScheduler.Dispatch(32, c =>
            {
                if (c == 17) throw new InvalidOperationException("boom-17");
            }));
        Assert.Contains("boom-17", ex.Message);
    }

    [Fact]
    public void NestedDispatch_RunsSerially_NoDeadlock()
    {
        // A chunk that itself dispatches must not deadlock on the pool: the nested
        // dispatch runs serially on the worker thread (the _inScheduler guard).
        const int outer = 8, inner = 8;
        var inner_counts = new int[outer * inner];
        CooperativeGemmScheduler.Dispatch(outer, o =>
        {
            CooperativeGemmScheduler.Dispatch(inner, i =>
                Interlocked.Increment(ref inner_counts[o * inner + i]));
        });
        for (int i = 0; i < inner_counts.Length; i++)
            Assert.Equal(1, inner_counts[i]);
    }

    [Fact]
    public void Pool_IsFixedSize_DoesNotGrowUnderConcurrentLoad()
    {
        // No-oversubscription guarantee (plan 2.8): the cooperative pool is a FIXED set
        // of workers and callers PARTICIPATE rather than the pool spawning a thread per
        // dispatch/chunk. So heavy concurrent dispatch must NOT grow the worker count —
        // that's what keeps active worker threads ≈ cores regardless of caller count.
        CooperativeGemmScheduler.Dispatch(8, _ => { }); // trigger pool init
        int initial = CooperativeGemmScheduler.WorkerCount;
        Assert.InRange(initial, 1, Environment.ProcessorCount);

        Parallel.For(0, Environment.ProcessorCount * 3, _ =>
        {
            for (int i = 0; i < 25; i++)
                CooperativeGemmScheduler.Dispatch(48, c => { var _ = c * c; });
        });

        Assert.Equal(initial, CooperativeGemmScheduler.WorkerCount); // fixed — no per-dispatch threads
    }

    [Fact]
    public void ConcurrentDispatches_AllComplete_CorrectAndUncontaminated()
    {
        // The headline property: many threads each run their own dispatch
        // concurrently; every dispatch must complete with its OWN chunks written
        // (no cross-job contamination) and the whole thing must not deadlock.
        int threads = Math.Max(8, Environment.ProcessorCount * 2);
        const int chunksPerJob = 50;
        const int itersPerThread = 40;
        var failures = new ConcurrentBag<string>();

        using var startGate = new Barrier(threads + 1);
        var workers = new Task[threads];
        for (int t = 0; t < threads; t++)
        {
            int tid = t;
            workers[t] = Task.Factory.StartNew(() =>
            {
                startGate.SignalAndWait();
                for (int it = 0; it < itersPerThread; it++)
                {
                    // Each job writes a value derived from (tid,it) into its own
                    // buffer, one element per chunk. Cross-job contamination (a chunk
                    // from another job writing here) or a missed chunk fails the check.
                    var buf = new int[chunksPerJob];
                    int stamp = unchecked(tid * 100003 + it * 31 + 1);
                    CooperativeGemmScheduler.Dispatch(chunksPerJob, c => buf[c] = stamp + c);
                    for (int c = 0; c < chunksPerJob; c++)
                        if (buf[c] != stamp + c)
                        {
                            failures.Add($"tid {tid} it {it} chunk {c}: {buf[c]} != {stamp + c}");
                            break;
                        }
                }
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }

        startGate.SignalAndWait();
        // WaitAll with a generous timeout so a deadlock fails the test instead of
        // hanging the run forever.
        bool finished = Task.WaitAll(workers, TimeSpan.FromSeconds(120));
        Assert.True(finished, "Concurrent dispatches did not complete within 120s — possible deadlock.");
        Assert.True(failures.IsEmpty,
            $"{failures.Count} concurrent dispatches were incorrect/contaminated. " +
            $"First: {(failures.IsEmpty ? "" : System.Linq.Enumerable.First(failures))}");
    }
}
