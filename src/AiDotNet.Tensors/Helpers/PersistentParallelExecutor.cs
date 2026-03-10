using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Pre-spawned worker thread pool for near-zero dispatch overhead parallel execution.
/// Mimics libtorch's OpenMP pattern: threads idle between calls, wake instantly on signal.
/// Eliminates per-call ThreadPool queuing, CountdownEvent allocation, and closure overhead.
/// </summary>
internal sealed class PersistentParallelExecutor
{
    private static readonly Lazy<PersistentParallelExecutor> LazyInstance =
        new(() => new PersistentParallelExecutor(), LazyThreadSafetyMode.ExecutionAndPublication);

    internal static PersistentParallelExecutor Instance => LazyInstance.Value;

    private readonly int _numWorkers;
    private readonly Thread[] _workers;

    // Per-worker signaling: workers wait on these to receive work
    private readonly ManualResetEventSlim[] _workReady;

    // Shared work state — written by dispatcher, read by workers
    private volatile Action<int>? _action;
    private volatile int _numChunks;

    // Completion tracking — workers decrement, dispatcher waits for zero
    private int _remaining;

    // Completion signal for the dispatcher
    private readonly ManualResetEventSlim _allDone = new(false);

    // Serialize concurrent Execute calls
    private readonly object _executeLock = new();

    private PersistentParallelExecutor()
    {
        _numWorkers = Math.Max(1, Environment.ProcessorCount - 1);
        _workers = new Thread[_numWorkers];
        _workReady = new ManualResetEventSlim[_numWorkers];

        for (int i = 0; i < _numWorkers; i++)
        {
            _workReady[i] = new ManualResetEventSlim(false);
            int workerSlot = i;
            _workers[i] = new Thread(() => WorkerLoop(workerSlot))
            {
                IsBackground = true,
                Name = $"AiDotNet-Worker-{i}"
            };
            _workers[i].Start();
        }
    }

    // Captured worker exception (first one wins)
    private volatile Exception? _workerException;

    private void WorkerLoop(int slot)
    {
        while (true)
        {
            // Wait for work signal (very low latency — ManualResetEventSlim spins briefly then blocks)
            _workReady[slot].Wait();
            _workReady[slot].Reset();

            // Execute all assigned chunks for this worker slot.
            // Chunks are assigned round-robin across all participants (workers + main thread).
            // Total participants = _numWorkers + 1 (main). Worker slot N gets chunks (slot+1), (slot+1+stride), etc.
            int stride = _numWorkers + 1;
            int chunkId = slot + 1;
            while (chunkId < _numChunks)
            {
                try
                {
                    _action!(chunkId);
                }
                catch (Exception ex)
                {
                    // Capture first exception — will be re-thrown on the caller thread
                    Interlocked.CompareExchange(ref _workerException, ex, null);
                }
                chunkId += stride;
            }

            // Signal completion
            if (Interlocked.Decrement(ref _remaining) == 0)
            {
                _allDone.Set();
            }
        }
    }

    /// <summary>
    /// Execute an action in parallel with near-zero dispatch overhead.
    /// The main thread executes chunk 0, worker threads execute chunks 1..numChunks-1.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Execute(int numChunks, Action<int> action)
    {
        if (numChunks <= 0)
            return;
        if (numChunks == 1)
        {
            action(0);
            return;
        }

        lock (_executeLock)
        {
            int workersNeeded = Math.Min(numChunks - 1, _numWorkers);

            // Setup shared state
            _action = action;
            _numChunks = numChunks;
            _remaining = workersNeeded;
            _workerException = null;
            _allDone.Reset();

            // Wake workers (they're already spinning/blocked on ManualResetEventSlim)
            for (int i = 0; i < workersNeeded; i++)
            {
                _workReady[i].Set();
            }

            // Main thread does chunk 0 and any overflow chunks beyond worker count
            // (chunks assigned round-robin: main thread gets 0, _numWorkers+1, 2*_numWorkers+1, ...)
            Exception? mainException = null;
            int mainChunk = 0;
            while (mainChunk < numChunks)
            {
                try
                {
                    action(mainChunk);
                }
                catch (Exception ex)
                {
                    mainException ??= ex;
                }
                mainChunk += _numWorkers + 1;
            }

            // Wait for all workers to finish
            if (Volatile.Read(ref _remaining) > 0)
            {
                _allDone.Wait();
            }

            _action = null;

            // Re-throw first captured exception (worker or main thread)
            var workerEx = _workerException;
            if (mainException is not null)
                throw mainException;
            if (workerEx is not null)
                throw workerEx;
        }
    }
}
