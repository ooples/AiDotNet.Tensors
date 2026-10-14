using System;
using System.Diagnostics;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// #1248: in a plain dispatch each participant runs its own chunk set through a cursor, and the calling thread, after a
/// short grace period, runs any chunk a woken worker has not started; the dispatch completes when every chunk has
/// finished, so a worker that never gets a core no longer holds it. That lets a worker wake late, after its dispatch
/// ended or into the next one; these pin that it still runs every chunk exactly once, never joins a dispatch twice,
/// never runs a thread-local dispatch's chunks as a slot that dispatch did not wake, and that a chunk's exception
/// still reaches the caller.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class PersistentExecutorClaimedChunksTests
{
    [Fact]
    public void AlternatingPlainAndThreadLocalDispatches_RunEveryChunkExactlyOnce()
    {
        int priorDop = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Math.Min(16, Environment.ProcessorCount));
            var rng = new Random(1248);
            for (int it = 0; it < 3000; it++)
            {
                int chunks = rng.Next(2, 64);
                if (it % 3 == 2)
                {
                    // Thread-local: per-participant partial sums of the chunk indices.
                    long total = 0;
                    var hits = new int[chunks];
                    PersistentParallelExecutor.Instance.Execute(chunks, 0,
                        () => new long[1],
                        (c, acc) => { Interlocked.Increment(ref hits[c]); acc[0] += c; },
                        acc => Interlocked.Add(ref total, acc[0]));
                    Assert.Equal((long)chunks * (chunks - 1) / 2, total);
                    for (int c = 0; c < chunks; c++)
                        Assert.True(hits[c] == 1, $"thread-local dispatch {it}: chunk {c} ran {hits[c]} times");
                }
                else
                {
                    var hits = new int[chunks];
                    // Uneven chunk costs, so claiming order varies from dispatch to dispatch.
                    PersistentParallelExecutor.Instance.Execute(chunks, c =>
                    {
                        if ((c * 7 + it) % 5 == 0) Thread.SpinWait(400);
                        Interlocked.Increment(ref hits[c]);
                    });
                    for (int c = 0; c < chunks; c++)
                        Assert.True(hits[c] == 1, $"plain dispatch {it}: chunk {c} ran {hits[c]} times");
                }
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = priorDop;
        }
    }

    [Fact]
    public void ASlowWorker_DoesNotKeepItsUnstartedChunks()
    {
        // The first worker thread to run a chunk takes 20 ms on every chunk it runs -- the shape of a worker that keeps
        // losing its core. With fixed sets it ran all 8 of its 32 chunks (~160 ms); now the calling thread, after a
        // short grace period, runs the ones it has not started, so the dispatch costs about one slow chunk.
        int priorDop = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 4;
            int caller = Environment.CurrentManagedThreadId;
            int slow = 0;
            var hits = new int[32];
            var sw = Stopwatch.StartNew();
            PersistentParallelExecutor.Instance.Execute(32, c =>
            {
                int me = Environment.CurrentManagedThreadId;
                if (me != caller)
                {
                    Interlocked.CompareExchange(ref slow, me, 0);
                    if (Volatile.Read(ref slow) == me) Thread.Sleep(20);
                }
                Interlocked.Increment(ref hits[c]);
            });
            sw.Stop();
            Assert.All(hits, h => Assert.Equal(1, h));
            Assert.True(sw.ElapsedMilliseconds < 100, $"dispatch took {sw.ElapsedMilliseconds} ms (fixed sets: ~160 ms)");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = priorDop;
        }
    }

    [Fact]
    public void ChunkException_ReachesTheCaller_AndTheNextDispatchIsClean()
    {
        int priorDop = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Math.Min(8, Environment.ProcessorCount));
            for (int it = 0; it < 200; it++)
            {
                int bad = it % 16;
                var ex = Assert.Throws<InvalidOperationException>(() =>
                    PersistentParallelExecutor.Instance.Execute(16, c =>
                    {
                        if (c == bad) throw new InvalidOperationException("chunk " + c);
                    }));
                Assert.Equal("chunk " + bad, ex.Message);

                var hits = new int[16];
                PersistentParallelExecutor.Instance.Execute(16, c => Interlocked.Increment(ref hits[c]));
                Assert.All(hits, h => Assert.Equal(1, h));
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = priorDop;
        }
    }
}
