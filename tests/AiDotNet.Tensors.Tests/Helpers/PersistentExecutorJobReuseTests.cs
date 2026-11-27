using System;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The executor re-arms one plain-mode Job for every plain dispatch instead of allocating one. A worker can still hold
/// that Job from an earlier dispatch when it is re-armed (it woke late, or the OS descheduled it), so these hold
/// workers at the two points where that happens -- between reading the published Job and taking its view, and between
/// the view and claiming chunks -- and check that across many rapid dispatches every chunk runs exactly once, with its
/// own dispatch's body, never after that dispatch returned; that a held worker never joins a narrower dispatch it is
/// not part of; that exceptions (and an interrupt of the dispatching thread) still reach the right caller only once
/// the dispatch is over; and that a plain dispatch allocates nothing on the dispatching thread.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class PersistentExecutorJobReuseTests
{
    private static readonly TimeSpan Bound = TimeSpan.FromSeconds(30);

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

    /// <summary>
    /// Counts workers between the view-delay hook and the end of that dispatch attempt, so a test can wait until every
    /// held or delayed worker is done (and would have run any stale chunk) instead of sleeping for a fixed time.
    /// </summary>
    private sealed class InFlight
    {
        private int _count;
        private readonly ThreadLocal<bool> _entered = new();

        public void Enter()
        {
            if (_entered.Value) return;
            _entered.Value = true;
            Interlocked.Increment(ref _count);
        }

        public void Done()
        {
            if (!_entered.Value) return;
            _entered.Value = false;
            Interlocked.Decrement(ref _count);
        }

        public void WaitIdle() =>
            Assert.True(SpinWait.SpinUntil(() => Volatile.Read(ref _count) == 0, Bound),
                "a delayed worker did not finish its dispatch attempt");
    }

    private static void WithPool(int degree, Action<InFlight> body)
    {
        int priorDop = CpuParallelSettings.MaxDegreeOfParallelism;
        bool priorPool = CpuParallelSettings.UseCooperativePool;
        var inFlight = new InFlight();
        try
        {
            // The hooks under test live in the cooperative pool; AIDOTNET_COOP_POOL=0 would route every dispatch
            // through Parallel.For and these tests would check nothing.
            CpuParallelSettings.UseCooperativePool = true;
            CpuParallelSettings.MaxDegreeOfParallelism = degree;
            PersistentParallelExecutor.TestWorkerAttemptDone = _ => inFlight.Done();
            RunWithDeadline(() =>
            {
                body(inFlight);
                // Every worker a hook held or delayed is done before the next test dispatches.
                PersistentParallelExecutor.TestWorkerViewDelay = null;
                PersistentParallelExecutor.TestWorkerStartDelay = null;
                inFlight.WaitIdle();
            }, TimeSpan.FromSeconds(120));
        }
        finally
        {
            PersistentParallelExecutor.TestWorkerStartDelay = null;
            PersistentParallelExecutor.TestWorkerViewDelay = null;
            PersistentParallelExecutor.TestWorkerAttemptDone = null;
            CpuParallelSettings.MaxDegreeOfParallelism = priorDop;
            CpuParallelSettings.UseCooperativePool = priorPool;
        }
    }

    // A broken re-arm can leave a dispatch's pending count above zero for good. The body runs on its own background
    // thread; its exceptions are rethrown here. A body still running at the deadline may hold the singleton
    // executor's lock, so every later dispatch in this process would wait on it and the run would hang with the
    // failure unreported: end the process with the reason instead.
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
        {
            string reason = $"{nameof(PersistentExecutorJobReuseTests)}: the dispatches did not finish within " +
                $"{deadline.TotalSeconds:F0} s: a dispatch never completed and still holds the executor.";
            Console.Error.WriteLine(reason);
            Environment.FailFast(reason);
        }
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
        WithPool(Math.Max(3, Math.Min(16, Environment.ProcessorCount)), inFlight =>
        {
            var noise = new ThreadLocal<uint>(() => (uint)Environment.CurrentManagedThreadId * 2654435761u);
            PersistentParallelExecutor.TestWorkerViewDelay = (slot, _) =>
            {
                inFlight.Enter();
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

            // Workers held by the delays must not have run an old dispatch's body afterwards: wait until each one
            // has finished the attempt it was delayed in, then check.
            PersistentParallelExecutor.TestWorkerViewDelay = null;
            PersistentParallelExecutor.TestWorkerStartDelay = null;
            inFlight.WaitIdle();
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
        // Worker slot 0 reads the published plain Job of a live dispatch, then is held while 100 later dispatches
        // re-arm that same Job. It is released either after them (the Job is cleared) or in the middle of the last
        // one (the Job is live with another dispatch). Its view is then rejected, of a finished dispatch (whose
        // cursors are exhausted), or of the live one; never a mix, and never a chunk twice.
        WithPool(4, inFlight =>
        {
            int arm = 0;
            var holding = new ManualResetEventSlim();
            var release = new ManualResetEventSlim();
            var heldDone = new ManualResetEventSlim();
            var isHeld = new ThreadLocal<bool>();
            PersistentParallelExecutor.TestWorkerViewDelay = (slot, published) =>
            {
                inFlight.Enter();
                // Only a worker that read a published Job: one woken late into an ended dispatch reads none.
                if (slot != 0 || !published || Interlocked.CompareExchange(ref arm, 0, 1) != 1) return;
                isHeld.Value = true;
                holding.Set();
                release.Wait(Bound);
            };
            PersistentParallelExecutor.TestWorkerAttemptDone = _ =>
            {
                if (isHeld.Value)
                {
                    isHeld.Value = false;
                    heldDone.Set();
                }
                inFlight.Done();
            };

            var records = new List<Dispatch>();
            try
            {
                for (int round = 0; round < 6; round++)
                {
                    holding.Reset();
                    release.Reset();
                    heldDone.Reset();
                    Volatile.Write(ref arm, 1);

                    // The first dispatch stays live until slot 0 holds its Job: its own chunk 0 waits for that.
                    var first = new Dispatch(8);
                    records.Add(first);
                    PersistentParallelExecutor.Instance.Execute(first.Hits.Length, c =>
                    {
                        if (c == 0 && !holding.Wait(Bound))
                            throw new TimeoutException("worker slot 0 never read the first dispatch's Job");
                        first.Run(c);
                    });
                    first.Returned = true;
                    Assert.All(first.Hits, h => Assert.Equal(1, h));

                    bool releaseMidDispatch = round % 2 == 1;
                    for (int d = 0; d < 100; d++)
                    {
                        var record = new Dispatch(4 + (round * 100 + d) % 13);
                        records.Add(record);
                        bool last = d == 99;
                        PersistentParallelExecutor.Instance.Execute(record.Hits.Length, c =>
                        {
                            if (c == 0 && last && releaseMidDispatch)
                            {
                                // Resume the held worker while this re-armed dispatch is live, and keep it live
                                // until that worker is done with it.
                                release.Set();
                                if (!heldDone.Wait(Bound))
                                    throw new TimeoutException("the held worker did not finish its attempt");
                            }
                            record.Run(c);
                        });
                        record.Returned = true;
                        Assert.All(record.Hits, h => Assert.Equal(1, h));
                    }
                    if (!releaseMidDispatch)
                        Assert.False(heldDone.IsSet, "the held worker finished before it was released");

                    release.Set();
                    Assert.True(heldDone.Wait(Bound), "the held worker did not finish its attempt");
                }
            }
            finally
            {
                release.Set();
            }

            inFlight.WaitIdle();
            foreach (var r in records)
            {
                Assert.Equal(0, r.LateRuns);
                Assert.All(r.Hits, h => Assert.Equal(1, h));
            }
        });
    }

    [Fact]
    public void AWorkerHeldFromAWiderDispatch_DoesNotJoinANarrowerOne()
    {
        // Slot 1 is woken for A (three participants) and held after reading the plain Job. A finishes; B re-arms the
        // Job with a cap of two participants (the caller and slot 0) and eager stealing. Released while B is live,
        // slot 1 views B; it is outside B's team, so it must neither run nor steal B's chunks.
        WithPool(3, inFlight =>
        {
            var priorWarm = PersistentParallelExecutor.WarmWindow;
            int arm = 0, heldThread = 0;
            var holding = new ManualResetEventSlim();
            var release = new ManualResetEventSlim();
            var heldDone = new ManualResetEventSlim();
            var isHeld = new ThreadLocal<bool>();
            try
            {
                // Parked workers: every dispatch steals eagerly, as the reported schedule needs.
                PersistentParallelExecutor.WarmWindow = TimeSpan.Zero;
                PersistentParallelExecutor.TestWorkerViewDelay = (slot, published) =>
                {
                    inFlight.Enter();
                    if (slot != 1 || !published || Interlocked.CompareExchange(ref arm, 0, 1) != 1) return;
                    isHeld.Value = true;
                    Volatile.Write(ref heldThread, Environment.CurrentManagedThreadId);
                    holding.Set();
                    release.Wait(Bound);
                };
                PersistentParallelExecutor.TestWorkerAttemptDone = _ =>
                {
                    if (isHeld.Value)
                    {
                        isHeld.Value = false;
                        heldDone.Set();
                    }
                    inFlight.Done();
                };

                Volatile.Write(ref arm, 1);
                var a = new Dispatch(3);
                PersistentParallelExecutor.Instance.Execute(3, 3, c =>
                {
                    if (c == 0 && !holding.Wait(Bound))
                        throw new TimeoutException("worker slot 1 never read A's Job");
                    a.Run(c);
                });
                a.Returned = true;
                Assert.All(a.Hits, h => Assert.Equal(1, h));

                var b = new Dispatch(8);
                int ranOnHeldWorker = 0;
                PersistentParallelExecutor.Instance.Execute(8, 2, c =>
                {
                    if (Environment.CurrentManagedThreadId == Volatile.Read(ref heldThread))
                        Interlocked.Increment(ref ranOnHeldWorker);
                    // The two team members' first chunks stay busy until the held worker is done with B, so the
                    // rest of B is unclaimed while it looks.
                    if (c == 0)
                    {
                        release.Set();
                        if (!heldDone.Wait(Bound))
                            throw new TimeoutException("the held worker did not finish its attempt");
                    }
                    else if (c == 1)
                    {
                        heldDone.Wait(Bound);
                    }
                    b.Run(c);
                });
                b.Returned = true;
                Assert.All(b.Hits, h => Assert.Equal(1, h));
                Assert.True(ranOnHeldWorker == 0,
                    $"{ranOnHeldWorker} chunk(s) of the two-participant dispatch ran on a worker outside its team");

                inFlight.WaitIdle();
                Assert.Equal(0, a.LateRuns);
                Assert.Equal(0, b.LateRuns);
            }
            finally
            {
                release.Set();
                PersistentParallelExecutor.WarmWindow = priorWarm;
            }
        });
    }

    [Fact]
    public void ANullBody_IsRejectedBeforeTheJobIsArmed()
    {
        WithPool(Math.Max(2, Math.Min(8, Environment.ProcessorCount)), _ =>
        {
            // Armed, a null body reads as a cleared Job: no one claims a chunk and the dispatch never completes.
            Assert.Throws<ArgumentNullException>(() => PersistentParallelExecutor.Instance.Execute(2, 1, null!));
            Assert.Throws<ArgumentNullException>(() => PersistentParallelExecutor.Instance.Execute(16, null!));
            // A zero-chunk call stays a no-op.
            PersistentParallelExecutor.Instance.Execute(0, null!);

            var record = new Dispatch(16);
            PersistentParallelExecutor.Instance.Execute(16, record.Run);
            Assert.All(record.Hits, h => Assert.Equal(1, h));
        });
    }

    [Fact]
    public void AnInterruptedDispatcher_ReportsTheInterruptOnlyOnceItsDispatchHasDrained()
    {
        // The dispatching thread is interrupted while it waits for a chunk running on a worker. It must not leave
        // Execute (and release the executor) until that chunk is done: the next dispatch re-arms the same Job, and the
        // straggler's late completion would count against the new dispatch.
        WithPool(2, _ =>
        {
            var chunk1Started = new ManualResetEventSlim();
            var chunk0Done = new ManualResetEventSlim();
            var releaseChunk1 = new ManualResetEventSlim();
            int chunk1Finished = 0;
            bool chunk1FinishedWhenThrown = false;
            Exception? caught = null;
            var dispatcher = new Thread(() =>
            {
                try
                {
                    PersistentParallelExecutor.Instance.Execute(2, 2, c =>
                    {
                        if (c == 0)
                        {
                            // Chunk 1 can only run on the worker while the caller is in here.
                            if (!chunk1Started.Wait(Bound))
                                throw new TimeoutException("the worker never started chunk 1");
                            chunk0Done.Set();
                        }
                        else
                        {
                            chunk1Started.Set();
                            releaseChunk1.Wait(Bound);
                            Volatile.Write(ref chunk1Finished, 1);
                        }
                    });
                }
                catch (Exception ex)
                {
                    chunk1FinishedWhenThrown = Volatile.Read(ref chunk1Finished) == 1;
                    caught = ex;
                }
            }) { IsBackground = true };

            try
            {
                dispatcher.Start();
                Assert.True(chunk0Done.Wait(Bound), "the dispatch did not reach its completion wait");
                dispatcher.Interrupt();
                // Unfixed, the interrupt ends the wait at once; fixed, Execute keeps waiting for chunk 1.
                Assert.False(dispatcher.Join(500), "Execute returned while a chunk of its dispatch was still running");
            }
            finally
            {
                releaseChunk1.Set();
            }
            Assert.True(dispatcher.Join(Bound), "Execute did not return after its last chunk finished");
            Assert.IsType<ThreadInterruptedException>(caught);
            Assert.True(chunk1FinishedWhenThrown, "the interrupt was reported before the dispatch drained");

            // The executor is usable afterwards, and the next dispatches complete exactly.
            for (int i = 0; i < 50; i++)
            {
                var record = new Dispatch(16);
                PersistentParallelExecutor.Instance.Execute(16, record.Run);
                record.Returned = true;
                Assert.All(record.Hits, h => Assert.Equal(1, h));
                Assert.Equal(0, record.LateRuns);
            }
        });
    }

    [Fact]
    public void Exceptions_ReachTheirOwnCaller_UnderDelayedWorkers()
    {
        WithPool(Math.Max(3, Math.Min(8, Environment.ProcessorCount)), inFlight =>
        {
            var noise = new ThreadLocal<uint>(() => (uint)Environment.CurrentManagedThreadId * 40503u);
            PersistentParallelExecutor.TestWorkerViewDelay = (slot, _) =>
            {
                inFlight.Enter();
                uint s = noise.Value;
                Jitter(ref s, slot);
                noise.Value = s;
            };
            var records = new List<Dispatch>();
            for (int it = 0; it < 1000; it++)
            {
                int chunks = 8 + it % 9;
                int bad = it % chunks;
                var record = new Dispatch(chunks);
                records.Add(record);
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
                records.Add(clean);
                PersistentParallelExecutor.Instance.Execute(chunks, clean.Run);
                clean.Returned = true;
                Assert.All(clean.Hits, h => Assert.Equal(1, h));
                Assert.Equal(0, record.LateRuns);
            }

            PersistentParallelExecutor.TestWorkerViewDelay = null;
            inFlight.WaitIdle();
            foreach (var r in records)
            {
                Assert.Equal(0, r.LateRuns);
                Assert.All(r.Hits, h => Assert.Equal(1, h));
            }
        });
    }

#if NETCOREAPP
    [Fact]
    public void APlainDispatch_AllocatesNothingOnTheDispatchingThread()
    {
        WithPool(Math.Max(2, Math.Min(16, Environment.ProcessorCount)), _ =>
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
