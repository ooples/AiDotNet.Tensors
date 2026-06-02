using System;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged.Pool;

/// <summary>
/// Lock-free spin-then-park worker pool specialised for the Streaming GEMM
/// strategy. Designed for sub-µs dispatch overhead at GEMM call rates
/// (~1 dispatch / 10–50 µs). See spec section 3.
///
/// <para>
/// Each worker owns a cache-line-aligned <see cref="WorkerSlot"/>. Dispatch
/// writes the action + assigned chunk range and increments the slot's seq
/// counter; the worker observes the counter change via spin-then-park and
/// runs its range. Completion is tracked by a single shared counter that
/// workers decrement; the dispatcher spin-waits on it reaching zero.
/// </para>
/// </summary>
internal static class StreamingWorkerPool
{
    [ThreadStatic] private static bool _isExecuting;

    private static readonly int _numWorkers = Math.Max(1, Environment.ProcessorCount - 1);
    private static readonly WorkerSlot[] _slots = new WorkerSlot[_numWorkers];
    private static readonly Thread[] _workers = new Thread[_numWorkers];

    // Decremented by each worker when its chunk range finishes; the dispatcher
    // spin-waits until it observes zero.
    private static int _remaining;

    // First worker exception is captured here; re-thrown on the dispatcher.
    private static volatile Exception? _firstException;

    private static int _initialized; // 0 = not started; 1 = workers spawned
    private static readonly object _initLock = new();

    // The pool has ONE shared set of worker slots (_slots) plus _remaining /
    // _firstException, so only a single dispatch can drive it at a time. The
    // [ThreadStatic] _isExecuting flag only guards NESTED (same-thread) re-entry;
    // it does NOT stop a DIFFERENT thread from calling Dispatch concurrently. Two
    // concurrent GEMMs would otherwise both overwrite _slots and _remaining, so the
    // workers run a mix of the two actions and write into the wrong output buffers
    // (observed as large nondeterministic GEMM result drift under a parallel test
    // suite). Dispatch acquires this lock; a concurrent caller that can't acquire it
    // runs its chunks serially on itself instead of racing the shared state.
    private static readonly object _dispatchLock = new();

    private static void EnsureInitialized()
    {
        if (Volatile.Read(ref _initialized) == 1) return;
        lock (_initLock)
        {
            if (_initialized == 1) return;
            for (int i = 0; i < _numWorkers; i++)
            {
                _slots[i].ParkEvent = new ManualResetEventSlim(false);
                int slot = i;
                _workers[i] = new Thread(() => WorkerLoop(slot))
                {
                    IsBackground = true,
                    Name = $"AiDotNet-StreamPool-{i}",
                };
                _workers[i].Start();
            }
            Volatile.Write(ref _initialized, 1);
        }
    }

    // PROTOTYPE (option 2 — low-latency hot pool): keep-alive spin window before a
    // worker parks. Small ops dispatched back-to-back (e.g. a transformer's many
    // per-layer attention/FFN/layernorm ops, or a training step's 7 GEMMs) otherwise
    // park between each dispatch and pay the full ManualResetEventSlim wakeup latency —
    // profiled as the dominant cost (>70% of samples in semaphore/Monitor wait, <1%
    // in the AVX kernel) for small-op inference. Holding workers hot across the burst
    // collapses that wakeup latency. Trade-off: more CPU burned while genuinely idle,
    // so it's tunable and defaults to the original window (1000 spin / 5000 yield).
    // Multiplier via AIDOTNET_POOL_SPIN (e.g. 20 keeps workers hot ~20x longer); 0/1 =
    // original behaviour. Validate on a quiet rig with the AB suite before changing
    // the default — see PR description.
    private static readonly int _spinMul =
        int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_POOL_SPIN"), out var sm) && sm > 1 ? sm : 1;
    private static readonly int _spinThresh = 1000 * _spinMul;
    private static readonly int _yieldThresh = 5000 * _spinMul;

    private static void WorkerLoop(int slot)
    {
        long lastSeq = 0;
        while (true)
        {
            // Spin-then-park wait for a new dispatch generation.
            int spinCount = 0;
            while (Volatile.Read(ref _slots[slot].Seq) == lastSeq)
            {
                if (spinCount < _spinThresh)
                {
                    Thread.SpinWait(1);
                    spinCount++;
                }
                else if (spinCount < _yieldThresh)
                {
                    Thread.Yield();
                    spinCount++;
                }
                else
                {
                    Volatile.Write(ref _slots[slot].ParkPending, 1);
                    // FULL FENCE before re-reading Seq. This is a Dekker-style mutual
                    // signal: the worker writes ParkPending then reads Seq, while the
                    // dispatcher writes Seq (Interlocked.Increment — a full fence) then
                    // reads ParkPending. Volatile.Write/Read alone do NOT order a store
                    // followed by a load (x86 TSO permits store-load reordering), so
                    // without this barrier the worker could read a stale Seq==lastSeq and
                    // park while the dispatcher reads a stale ParkPending==0 and skips the
                    // wake — a lost wakeup that hangs the dispatcher's spin-wait on
                    // _remaining forever (intermittent deadlock under concurrent dispatch).
                    // The barrier makes ParkPending=1 globally visible before the Seq read,
                    // pairing with the dispatcher's interlocked Seq increment.
                    Thread.MemoryBarrier();
                    // Re-check under park to avoid lost wakeup.
                    if (Volatile.Read(ref _slots[slot].Seq) == lastSeq)
                        _slots[slot].ParkEvent!.Wait();
                    _slots[slot].ParkEvent!.Reset();
                    Volatile.Write(ref _slots[slot].ParkPending, 0);
                    spinCount = 0;
                }
            }

            lastSeq = Volatile.Read(ref _slots[slot].Seq);
            var body = _slots[slot].Body;
            int chunkStart = _slots[slot].ChunkStart;
            int chunkEnd = _slots[slot].ChunkEnd;

            // Set the reentrancy guard on the worker thread so any nested
            // Dispatch from inside the body falls back to serial — workers are
            // already busy with the outer dispatch, so re-entering would
            // deadlock the pool.
            _isExecuting = true;
            try
            {
                for (int c = chunkStart; c < chunkEnd; c++)
                    body!(c);
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref _firstException, ex, null);
            }
            finally
            {
                _isExecuting = false;
            }

            Interlocked.Decrement(ref _remaining);
        }
    }

    /// <summary>
    /// Default serial-fallback grain size (matches <c>PersistentParallelExecutor.DefaultSerialGrainSize</c>).
    /// Below this total-work threshold, <see cref="Dispatch(int, long, Action{int})"/>
    /// runs all chunks on the caller (no worker wakeup, no completion wait).
    /// </summary>
    public static int DefaultSerialGrainSize { get; set; } = 32 * 1024;

    /// <summary>
    /// Dispatch with PyTorch-style serial-fallback: when <paramref name="totalWork"/>
    /// is below <see cref="DefaultSerialGrainSize"/>, all chunks run on the caller.
    /// </summary>
    public static void Dispatch(int numChunks, long totalWork, Action<int> action)
    {
        if (numChunks <= 0) return;
        if (totalWork < DefaultSerialGrainSize)
        {
            Exception? first = null;
            for (int c = 0; c < numChunks; c++)
            {
                try { action(c); }
                catch (Exception ex) { first ??= ex; }
            }
            if (first is not null) throw first;
            return;
        }
        Dispatch(numChunks, action);
    }

    /// <summary>
    /// Run <paramref name="action"/> across <paramref name="numChunks"/>
    /// chunks. Workers run a leading prefix; the caller runs the tail
    /// in-line for overlap. Returns when all chunks complete. Re-throws
    /// the first worker (or caller) exception.
    /// </summary>
    public static void Dispatch(int numChunks, Action<int> action)
    {
        if (numChunks <= 0) return;
        // Phase 2 migration seam: when the cooperative scheduler is enabled, route
        // there instead. Unlike this pool (one shared _slots set → a concurrent
        // second caller falls back to fully-serial, #492), the cooperative scheduler
        // lets concurrent dispatches interleave their chunks on a shared work queue —
        // so concurrent GEMMs stay parallel. Default-off until the concurrency
        // benchmark proves it; flip CooperativeGemmScheduler.Enabled to A/B test.
        if (CooperativeGemmScheduler.Enabled)
        {
            CooperativeGemmScheduler.Dispatch(numChunks, action);
            return;
        }
        // Run serially on the caller when: a single chunk; this thread is already
        // driving a dispatch (nested — avoids ThreadPool starvation); or ANOTHER
        // thread is currently driving the shared worker slots. The last case is the
        // concurrency fix: the pool has one shared _slots set, so a second concurrent
        // dispatch must fall back to serial rather than overwrite the in-flight
        // dispatch's slots (which corrupted both callers' GEMM results).
        // Monitor.TryEnter is non-blocking, so concurrent GEMMs never deadlock or
        // stall — at most one runs parallel while the rest run serial.
        if (numChunks == 1 || _isExecuting || !Monitor.TryEnter(_dispatchLock))
        {
            for (int c = 0; c < numChunks; c++) action(c);
            return;
        }

        EnsureInitialized();
        _isExecuting = true;
        try
        {
            // Caller runs the LAST 1/(N+1) of chunks; workers split the leading
            // portion. With N workers + 1 caller, each gets ≈ numChunks/(N+1)
            // chunks. When numChunks < (N+1), some workers get an empty range.
            int callerStart = (int)((long)_numWorkers * numChunks / (_numWorkers + 1));
            if (callerStart < 1) callerStart = numChunks; // Tiny dispatch — caller does it all.

            int activeWorkers = 0;
            for (int w = 0; w < _numWorkers; w++)
            {
                int wStart = (int)((long)w * callerStart / _numWorkers);
                int wEnd = (int)((long)(w + 1) * callerStart / _numWorkers);
                _slots[w].Body = action;
                _slots[w].ChunkStart = wStart;
                _slots[w].ChunkEnd = wEnd;
                if (wEnd > wStart) activeWorkers++;
            }

            _firstException = null;
            Volatile.Write(ref _remaining, activeWorkers);

            // Wake only workers with non-empty ranges. Workers with empty
            // ranges (wEnd == wStart) stay parked — incrementing their seq
            // would wake them just to decrement _remaining and break the
            // termination count.
            for (int w = 0; w < _numWorkers; w++)
            {
                if (_slots[w].ChunkEnd <= _slots[w].ChunkStart) continue;
                Interlocked.Increment(ref _slots[w].Seq);
                if (Volatile.Read(ref _slots[w].ParkPending) == 1)
                    _slots[w].ParkEvent!.Set();
            }

            // Caller runs chunks [callerStart..numChunks) in-line.
            Exception? callerException = null;
            for (int c = callerStart; c < numChunks; c++)
            {
                try { action(c); }
                catch (Exception ex) { callerException ??= ex; }
            }

            // Spin-wait for workers to finish. Workers with empty ranges don't
            // decrement _remaining (we only counted activeWorkers above), so
            // the wait terminates as soon as all non-empty ranges complete.
            int waitSpin = 0;
            while (Volatile.Read(ref _remaining) > 0)
            {
                if (waitSpin < 1000) { Thread.SpinWait(1); waitSpin++; }
                else { Thread.Yield(); waitSpin = 0; }
            }

            var pooledEx = _firstException ?? callerException;
            if (pooledEx is not null) throw pooledEx;
        }
        finally
        {
            _isExecuting = false;
            Monitor.Exit(_dispatchLock);
        }
    }
}
