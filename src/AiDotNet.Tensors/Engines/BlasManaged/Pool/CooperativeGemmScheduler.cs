using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged.Pool;

/// <summary>
/// Cooperative work-sharing scheduler for concurrent GEMM dispatch — the Phase 2
/// "exceed PyTorch on concurrent CPU" primitive.
///
/// <para>
/// The existing pools treat one dispatch as exclusive: <see cref="StreamingWorkerPool"/>
/// runs a contended second caller serially on itself (#492), and
/// <c>PersistentParallelExecutor</c> blocks a second caller on a lock. So N GEMMs from
/// N threads either serialize whole dispatches or oversubscribe (N caller threads PLUS
/// the pool's workers). PyTorch's shared intra-op pool has the same oversubscription
/// problem under concurrent inference.
/// </para>
///
/// <para>
/// This scheduler instead has ONE process-wide pool of <c>cores-1</c> parked workers
/// draining ONE shared queue. <see cref="Dispatch(int, Action{int})"/> enqueues its
/// chunks under a per-dispatch countdown and then the CALLING thread PARTICIPATES —
/// it drains the shared queue (any job's chunks) until its own job's countdown hits
/// zero, then returns. Effects:
/// <list type="bullet">
///   <item>Concurrent dispatches interleave their chunks on the shared workers — no
///   whole-GEMM serialization.</item>
///   <item>The caller does work instead of blocking idle, and workers PARK when the
///   queue is empty — so the count of threads actively burning CPU stays ≈ cores
///   regardless of how many callers are in flight (no oversubscription).</item>
/// </list>
/// </para>
///
/// <para>
/// Correctness rests on the GEMM strategies' invariant (Phase 0): chunks write
/// DISJOINT output regions, so interleaving chunks from different concurrent GEMMs is
/// safe. A chunk that re-enters <see cref="Dispatch(int, Action{int})"/> (nested
/// parallelism) runs serially — the <c>[ThreadStatic]</c> in-scheduler flag prevents
/// pool re-entry deadlock, mirroring the existing pools' nested guard.
/// </para>
///
/// <para>
/// Disabled by default (<see cref="Enabled"/>) until the concurrency benchmark proves
/// parity-or-better; the strategies keep using the existing pools until then.
/// </para>
/// </summary>
internal static class CooperativeGemmScheduler
{
    /// <summary>Master switch. When false, callers use the legacy pools. Default off
    /// until the Phase 2 concurrency benchmark proves this scheduler wins.</summary>
    internal static bool Enabled { get; set; }

    private readonly struct WorkItem
    {
        public readonly Job Job;
        public readonly int Chunk;
        public WorkItem(Job job, int chunk) { Job = job; Chunk = chunk; }
    }

    private sealed class Job
    {
        public readonly Action<int> Action;
        public int Remaining;            // Interlocked-decremented as chunks finish.
        public Exception? FirstException; // First chunk exception (CompareExchange).

        // Spin-then-park join (issue #589). Once the participating caller exhausts its
        // spin budget waiting for the LAST chunks (held by other threads), it parks on
        // Done instead of busy-spinning a core — that spin otherwise stole ~47% of a
        // core from a conv kernel running on the parallel path, halving conv throughput.
        // Single waiter (the dispatch caller), so Done is created only by the caller and
        // only when it actually needs to park (the common case completes while draining).
        // ParkPending=1 tells the thread that decrements Remaining to 0 to wake the caller.
        public int ParkPending;
        public ManualResetEventSlim? Done;

        public Job(Action<int> action, int remaining) { Action = action; Remaining = remaining; }
    }

    // MPMC queue of pending chunks across ALL in-flight dispatches.
    private static readonly ConcurrentQueue<WorkItem> _queue = new();
    // Counts pending items so parked workers can wake. Over-signaling only causes a
    // spurious wake (worker finds the queue empty and re-waits) — never lost work,
    // because completion is tracked by Job.Remaining, not by this semaphore.
    private static readonly SemaphoreSlim _workAvailable = new(0);

    // True while THIS thread is executing a chunk (worker or participating caller).
    // A nested Dispatch from inside a chunk runs serially to avoid pool re-entry.
    [ThreadStatic] private static bool _inScheduler;

    // ---- #653 Phase 0: opt-in core-utilization gauge ----
    // Counts how many fanned-out chunks are executing concurrently on the pool
    // (worker threads + the participating caller). It directly measures the #642
    // symptom — "worker pool parked, receiving no work" reads ~0 here; a saturated
    // op reads ~maxdop. OFF by default: the per-chunk Interlocked pair is skipped
    // entirely unless a ParallelUtilizationMeter has set MeasureUtilization, so there
    // is zero added cost on the production dispatch path. RunSerial (below-grain inline
    // work) is intentionally NOT counted — this gauge reports pool fan-out only, and
    // the serial orchestrator time is captured by the probe's speedup/N efficiency.
    internal static volatile bool MeasureUtilization;
    private static int s_activeWorkers;
    /// <summary>Current count of chunks executing concurrently on the cooperative pool.</summary>
    internal static int ActiveWorkers => Volatile.Read(ref s_activeWorkers);
    /// <summary>Zero the gauge before a measurement window.</summary>
    internal static void ResetUtilization() => Volatile.Write(ref s_activeWorkers, 0);

    private static int _initialized; // 0 = not started, 1 = workers spawned
    private static readonly object _initLock = new();
    private static int _numWorkers;

    /// <summary>Number of persistent worker threads (0 until first dispatch). The pool
    /// is fixed-size — callers PARTICIPATE rather than the pool spawning threads per
    /// dispatch — so this never grows under load (the no-oversubscription guarantee).
    /// Test/diagnostic accessor.</summary>
    internal static int WorkerCount => Volatile.Read(ref _initialized) == 1 ? _numWorkers : 0;

    private static void EnsureInitialized()
    {
        if (Volatile.Read(ref _initialized) == 1) return;
        lock (_initLock)
        {
            if (_initialized == 1) return;
            // Worker count. Default cores-1 (the caller participates as the cores-th).
            // PR #531 tail study: at full subscription the workers + participating caller +
            // .NET runtime threads exceed the logical cores, so the OS preempts active
            // workers and a stolen chunk stalls the caller's join — the p95 tail. Tunable
            // to leave runtime headroom (AIDOTNET_COOP_WORKERS).
            _numWorkers = AiDotNet.Tensors.Helpers.CpuParallelSettings.WorkerPoolThreads;   // capped (was ProcessorCount-1)
            if (int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_COOP_WORKERS"), out var wOverride) && wOverride >= 1)
                _numWorkers = Math.Min(wOverride, Math.Max(1, Environment.ProcessorCount - 1));
            for (int i = 0; i < _numWorkers; i++)
            {
                var t = new Thread(WorkerLoop)
                {
                    IsBackground = true,
                    Name = $"AiDotNet-CoopGemm-{i}",
                };
                t.Start();
            }
            Volatile.Write(ref _initialized, 1);
        }
    }

    private static void WorkerLoop()
    {
        while (true)
        {
            _workAvailable.Wait();
            // A permit may correspond to an item a participating caller already took;
            // TryDequeue-empty just means re-wait. Drain greedily while items remain.
            while (_queue.TryDequeue(out var item))
            {
                Execute(item);
                // Each successfully dequeued item consumed one logical permit; keep
                // draining without re-waiting. Remaining permits for items this loop
                // drains are reconciled by the spurious-wake-tolerant Wait above.
                if (!_queue.IsEmpty)
                {
                    // Best-effort: consume a permit for the next item we're about to
                    // take so the semaphore count tracks the queue (bounded drift only).
                    _workAvailable.Wait(0);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Execute(in WorkItem item)
    {
        bool prev = _inScheduler;
        _inScheduler = true;
        bool counted = MeasureUtilization;
        if (counted) Interlocked.Increment(ref s_activeWorkers);
        try
        {
            item.Job.Action(item.Chunk);
        }
        catch (Exception ex)
        {
            Interlocked.CompareExchange(ref item.Job.FirstException, ex, null);
        }
        finally
        {
            _inScheduler = prev;
            if (counted) Interlocked.Decrement(ref s_activeWorkers);
            if (Interlocked.Decrement(ref item.Job.Remaining) == 0)
                SignalIfParked(item.Job);
        }
    }

    /// <summary>Default serial-fallback grain size (matches the other pools).</summary>
    public static int DefaultSerialGrainSize { get; set; } = 32 * 1024;

    /// <summary>
    /// Grain-size-gated dispatch: below <see cref="DefaultSerialGrainSize"/> total work,
    /// runs all chunks inline on the caller (no pool traffic).
    /// </summary>
    public static void Dispatch(int numChunks, long totalWork, Action<int> action)
    {
        if (numChunks <= 0) return;
        if (totalWork < DefaultSerialGrainSize)
        {
            RunSerial(numChunks, action);
            return;
        }
        Dispatch(numChunks, action);
    }

    /// <summary>
    /// Run <paramref name="action"/> over <paramref name="numChunks"/> chunks on the
    /// shared cooperative pool. Returns when all of THIS dispatch's chunks complete
    /// (the caller participates meanwhile). Re-throws the first chunk exception.
    /// </summary>
    public static void Dispatch(int numChunks, Action<int> action)
    {
        if (numChunks <= 0) return;
        // Run serial on the caller when: a single chunk; a nested call (this thread is
        // already running a chunk — pool re-entry would risk deadlock); or parallelism
        // is pinned off via CpuParallelSettings.MaxDegreeOfParallelism <= 1 (honor the
        // class-level contract the legacy pools and ParallelForOrSerial also respect).
        if (numChunks == 1 || _inScheduler || CpuParallelSettings.MaxDegreeOfParallelism <= 1)
        {
            RunSerial(numChunks, action);
            return;
        }

        EnsureInitialized();

        var job = new Job(action, numChunks);
        for (int c = 0; c < numChunks; c++)
            _queue.Enqueue(new WorkItem(job, c));
        _workAvailable.Release(numChunks);

        // Caller participates: drain the shared queue (any job's chunks) until OUR job
        // is done. Marked in-scheduler so a nested Dispatch from a chunk runs serial.
        bool prev = _inScheduler;
        _inScheduler = true;
        try
        {
            int spin = 0;
            while (Volatile.Read(ref job.Remaining) > 0)
            {
                if (_queue.TryDequeue(out var item))
                {
                    // We took an item a worker permit was issued for; consume the
                    // matching permit so a parked worker doesn't wake to an empty queue.
                    _workAvailable.Wait(0);
                    ExecuteAlreadyInScheduler(item);
                    spin = 0;
                }
                else
                {
                    // Queue empty but our job isn't done → other threads hold our
                    // remaining chunks; we cannot help. Spin briefly to cover the
                    // sub-microsecond completion gap, then PARK on the job's done
                    // event instead of an unbounded SpinWait/Yield loop. The old loop
                    // burned a whole core busy-waiting — under conv-dominated workloads
                    // that core contends with the FusedConvHelper conv running on the
                    // parallel path, roughly halving conv throughput (issue #589).
                    if (spin++ < CoopJoinSpinBeforePark)
                    {
                        Thread.SpinWait(8);
                    }
                    else
                    {
                        var done = job.Done ?? EnsureDoneEvent(job);
                        Volatile.Write(ref job.ParkPending, 1);
                        // Publish ParkPending=1 before re-reading Remaining. Without this
                        // fence the thread completing the last chunk could read
                        // ParkPending==0 and skip the wake while we then read a stale
                        // Remaining>0 and park forever (lost wakeup).
                        Interlocked.MemoryBarrier();
                        if (Volatile.Read(ref job.Remaining) > 0)
                            done.Wait();
                        Volatile.Write(ref job.ParkPending, 0);
                        spin = 0;
                    }
                }
            }
        }
        finally
        {
            _inScheduler = prev;
        }

        var ex = job.FirstException;
        if (ex is not null) throw ex;
    }

    // Execute when the caller has ALREADY set _inScheduler (avoids save/restore churn
    // in the hot participate loop). Still counts down + captures exceptions.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExecuteAlreadyInScheduler(in WorkItem item)
    {
        bool counted = MeasureUtilization;
        if (counted) Interlocked.Increment(ref s_activeWorkers);
        try
        {
            item.Job.Action(item.Chunk);
        }
        catch (Exception ex)
        {
            Interlocked.CompareExchange(ref item.Job.FirstException, ex, null);
        }
        finally
        {
            if (counted) Interlocked.Decrement(ref s_activeWorkers);
            if (Interlocked.Decrement(ref item.Job.Remaining) == 0)
                SignalIfParked(item.Job);
        }
    }

    /// <summary>
    /// SpinWait(8) iterations the participating caller burns waiting for the last
    /// in-flight chunks before it parks on the job's done event (issue #589). 1000
    /// preserves the original sub-microsecond busy-wait latency for fast completions;
    /// only the previously-unbounded yield tail is replaced by a park.
    /// </summary>
    private const int CoopJoinSpinBeforePark = 1000;

    /// <summary>
    /// Lazily creates the dispatch caller's park event. Only the caller creates it
    /// (single creator), and it is published before any <c>ParkPending = 1</c> store,
    /// so a thread that observes <c>ParkPending == 1</c> always sees a non-null Done.
    /// </summary>
    private static ManualResetEventSlim EnsureDoneEvent(Job job)
    {
        var ev = new ManualResetEventSlim(false);
        Volatile.Write(ref job.Done, ev);
        return ev;
    }

    /// <summary>
    /// Wakes the participating caller if it has parked on this job. Called by the thread
    /// that decrements <see cref="Job.Remaining"/> to 0 — that Interlocked decrement is a
    /// full fence, so this read of ParkPending sees the caller's store; if the caller has
    /// committed to parking we set its event, otherwise its post-store re-read of
    /// Remaining (now 0) makes it skip the park.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SignalIfParked(Job job)
    {
        if (Volatile.Read(ref job.ParkPending) == 1)
            job.Done?.Set();
    }

    private static void RunSerial(int numChunks, Action<int> action)
    {
        Exception? first = null;
        for (int c = 0; c < numChunks; c++)
        {
            try { action(c); }
            catch (Exception ex) { first ??= ex; }
        }
        if (first is not null) throw first;
    }
}
