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

    private static void WorkerLoop(int slot)
    {
        long lastSeq = 0;
        while (true)
        {
            // Spin-then-park wait for a new dispatch generation.
            int spinCount = 0;
            while (Volatile.Read(ref _slots[slot].Seq) == lastSeq)
            {
                if (spinCount < 1000)
                {
                    Thread.SpinWait(1);
                    spinCount++;
                }
                else if (spinCount < 5000)
                {
                    Thread.Yield();
                    spinCount++;
                }
                else
                {
                    Volatile.Write(ref _slots[slot].ParkPending, 1);
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
    /// Run <paramref name="action"/> across <paramref name="numChunks"/>
    /// chunks. Workers run a leading prefix; the caller runs the tail
    /// in-line for overlap. Returns when all chunks complete. Re-throws
    /// the first worker (or caller) exception.
    /// </summary>
    public static void Dispatch(int numChunks, Action<int> action)
    {
        if (numChunks <= 0) return;
        if (numChunks == 1 || _isExecuting)
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
        finally { _isExecuting = false; }
    }
}
