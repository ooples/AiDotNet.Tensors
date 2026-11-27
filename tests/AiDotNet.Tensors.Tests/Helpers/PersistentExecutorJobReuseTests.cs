using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The executor re-arms one plain-mode Job for every plain dispatch instead of allocating one. A worker can still hold
/// that Job from an earlier dispatch when it is re-armed (it woke late, or the OS descheduled it), so these hold
/// workers at the two points where that happens -- between reading the published Job and taking its view, and between
/// the view and claiming chunks -- and check that across many rapid dispatches every chunk runs exactly once, with its
/// own dispatch's body, never after that dispatch returned; that exceptions still reach the right caller; and that a
/// plain dispatch allocates nothing on the dispatching thread.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class PersistentExecutorJobReuseTests
{
    // One dispatch's record: which chunks ran, and whether any ran after Execute returned.
    private sealed class Dispatch
    {
        public readonly int[] Hits;
        public volatile bool Returned;
        public int LateRuns;
        public Dispatch(int chunks) => Hits = new int[chunks];

        public void Run(int c)
        {
            if (Returned) Interlocked.Increment(ref LateRuns);
            Interlocked.Increment(ref Hits[c]);
        }
    }

    private static void WithPool(int degree, Action body)
    {
        int priorDop = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = degree;
            RunWithDeadline(body, TimeSpan.FromSeconds(120));
        }
        finally
        {
            PersistentParallelExecutor.TestWorkerStartDelay = null;
            PersistentParallelExecutor.TestWorkerViewDelay = null;
            CpuParallelSettings.MaxDegreeOfParallelism = priorDop;
            // Let any held worker wake, find its dispatch over, and go back to waiting before other tests dispatch.
            Thread.Sleep(100);
        }
    }

    // A broken re-arm can leave a dispatch's pending count above zero for good; fail with a message instead of
    // hanging the test run. The body runs on its own background thread; its exceptions are rethrown here.
    private static void RunWithDeadline(Action body, TimeSpan deadline)
    {
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try { body(); }
            catch (Exception ex) { failure = ex; }
        }) { IsBackground = true };
        thread.Start();
        if (!thread.Join(deadline))
            Assert.Fail($"The dispatches did not finish within {deadline.TotalSeconds:F0} s: a dispatch never completed.");
        if (failure is not null)
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
    }

    // Deterministic per-thread noise: mostly nothing, sometimes a spin, rarely a sleep that outlasts many dispatches.
    private static void Jitter(ref uint state, int slot)
    {
        state = state * 1664525u + 1013904223u + (uint)slot;
        uint r = state >> 24;
        if (r < 200) return;
        if (r < 250) Thread.SpinWait(200);
        else if (r < 255) Thread.Sleep(0);
        else if ((state & 3) == 0) Thread.Sleep(1);
    }

    [Fact]
    public void ManyRapidDispatches_WithDelayedWorkers_RunEveryChunkOnceWithItsOwnBody()
    {
        WithPool(Math.Max(3, Math.Min(16, Environment.ProcessorCount)), () =>
        {
            var noise = new ThreadLocal<uint>(() => (uint)Environment.CurrentManagedThreadId * 2654435761u);
            PersistentParallelExecutor.TestWorkerViewDelay = slot =>
            {
                uint s = noise.Value;
                Jitter(ref s, slot);
                noise.Value = s;
            };
            PersistentParallelExecutor.TestWorkerStartDelay = slot =>
            {
                uint s = noise.Value;
                Jitter(ref s, slot + 101);
                noise.Value = s;
            };

            var rng = new Random(1873);
            const int dispatches = 3_000;
            var all = new Dispatch[dispatches];
            for (int d = 0; d < dispatches; d++)
            {
                int chunks = rng.Next(2, 48);
                if (d % 7 == 6)
                {
                    // A thread-local dispatch in between: it publishes its own Job, and the plain Job stays idle.
                    long total = 0;
                    PersistentParallelExecutor.Instance.Execute(chunks, 0,
                        () => new long[1], (c, acc) => acc[0] += c, acc => Interlocked.Add(ref total, acc[0]));
                    Assert.Equal((long)chunks * (chunks - 1) / 2, total);
                }
                var record = all[d] = new Dispatch(chunks);
                PersistentParallelExecutor.Instance.Execute(chunks, record.Run);
                record.Returned = true;
                for (int c = 0; c < chunks; c++)
                    Assert.True(record.Hits[c] == 1, $"dispatch {d}: chunk {c} ran {record.Hits[c]} times");
            }

            // Workers held by the delays must not have run an old dispatch's body afterwards.
            Thread.Sleep(50);
            for (int d = 0; d < dispatches; d++)
            {
                Assert.True(all[d].LateRuns == 0, $"dispatch {d}: {all[d].LateRuns} chunk(s) ran after it returned");
                Assert.True(Array.TrueForAll(all[d].Hits, h => h == 1), $"dispatch {d}: a chunk ran again later");
            }
        });
    }

    [Fact]
    public void AWorkerHeldWithTheJobAcrossManyReArms_NeverRunsAStaleOrTornDispatch()
    {
        // Worker slot 0 reads the published Job, then is held 30 ms -- hundreds of dispatches re-arm that same Job
        // meanwhile. When it resumes, its view is either rejected, of a finished dispatch (whose cursors are
        // exhausted), or of the live one; never a mix, and never a chunk twice.
        WithPool(4, () =>
        {
            int held = 0;
            PersistentParallelExecutor.TestWorkerViewDelay = slot =>
            {
                if (slot == 0 && Interlocked.Exchange(ref held, 1) == 0) Thread.Sleep(30);
            };
            var deadline = DateTime.UtcNow.AddMilliseconds(120);
            int d = 0;
            var records = new System.Collections.Generic.List<Dispatch>();
            while (DateTime.UtcNow < deadline || d < 500)
            {
                var record = new Dispatch(4 + d % 13);
                records.Add(record);
                PersistentParallelExecutor.Instance.Execute(record.Hits.Length, record.Run);
                record.Returned = true;
                Assert.All(record.Hits, h => Assert.Equal(1, h));
                if (d % 50 == 0) Volatile.Write(ref held, 0); // hold a worker again
                d++;
            }
            Thread.Sleep(60);
            foreach (var r in records)
            {
                Assert.Equal(0, r.LateRuns);
                Assert.All(r.Hits, h => Assert.Equal(1, h));
            }
        });
    }

    [Fact]
    public void Exceptions_ReachTheirOwnCaller_UnderDelayedWorkers()
    {
        WithPool(Math.Max(3, Math.Min(8, Environment.ProcessorCount)), () =>
        {
            var noise = new ThreadLocal<uint>(() => (uint)Environment.CurrentManagedThreadId * 40503u);
            PersistentParallelExecutor.TestWorkerViewDelay = slot =>
            {
                uint s = noise.Value;
                Jitter(ref s, slot);
                noise.Value = s;
            };
            for (int it = 0; it < 1000; it++)
            {
                int chunks = 8 + it % 9;
                int bad = it % chunks;
                var record = new Dispatch(chunks);
                var ex = Assert.Throws<InvalidOperationException>(() =>
                    PersistentParallelExecutor.Instance.Execute(chunks, c =>
                    {
                        record.Run(c);
                        if (c == bad) throw new InvalidOperationException("dispatch " + it + " chunk " + c);
                    }));
                record.Returned = true;
                Assert.Equal("dispatch " + it + " chunk " + bad, ex.Message);
                // The failing dispatch still ran every other chunk once.
                Assert.All(record.Hits, h => Assert.Equal(1, h));

                var clean = new Dispatch(chunks);
                PersistentParallelExecutor.Instance.Execute(chunks, clean.Run);
                clean.Returned = true;
                Assert.All(clean.Hits, h => Assert.Equal(1, h));
                Assert.Equal(0, record.LateRuns);
            }
        });
    }

#if NETCOREAPP
    [Fact]
    public void APlainDispatch_AllocatesNothingOnTheDispatchingThread()
    {
        WithPool(Math.Max(2, Math.Min(16, Environment.ProcessorCount)), () =>
        {
            var hits = new int[32];
            Action<int> body = c => Interlocked.Increment(ref hits[c]);
            for (int i = 0; i < 200; i++) PersistentParallelExecutor.Instance.Execute(32, body); // warm up
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 1000; i++) PersistentParallelExecutor.Instance.Execute(32, body);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.Equal(0, allocated);
            Assert.All(hits, h => Assert.Equal(1200, h));
        });
    }
#endif
}
