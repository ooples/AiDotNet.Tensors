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

    // Per-worker MRES spin count before a parked worker truly blocks.
    //
    // History: #475 set this to 2047 (the MRES max) to keep the resident workers hot so
    // back-to-back small GEMMs in INFERENCE find them already spinning (no per-dispatch wake
    // latency). But under TRAINING on a many-core box the pool is woken by a near-continuous
    // stream of tiny forward/backward GEMM dispatches, and each of the ~32 workers re-arming
    // its MRES spins ~2047 iterations EVERY dispatch — profiled on an N-BEATS train (128
    // logical cores) as ~15 avg-cores of PURE SPIN (TotalProcessorTime/wall), i.e. ~12 cores
    // burned doing nothing while wall barely changed. Dropping the spin to a small bridge
    // value parks idle workers almost immediately: the SAME train fell from ~15 to ~3.8
    // avg-cores at unchanged wall (the beneficial ~4-way fan-out still happens; only the idle
    // spin is removed), freeing the rest of the box for concurrent work. 32 still bridges the
    // sub-µs gap between two GEMMs issued back-to-back in one op without holding a core for
    // tens of µs of idle spin. Env override: AIDOTNET_PPE_SPINCOUNT (0 = park immediately).
    private static readonly int _mresSpinCount =
        int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_PPE_SPINCOUNT"), out var sc) && sc >= 0 ? sc : 32;

    private readonly int _numWorkers;
    private readonly Thread[] _workers;

    // Per-worker signaling: workers wait on these to receive work
    private readonly ManualResetEventSlim[] _workReady;

    // Shared work state — written by dispatcher, read by workers
    private volatile Action<int>? _action;
    private volatile int _numChunks;

    // Per-dispatch participant stride (= active workers + 1). Workers walk chunks
    // slot+1, slot+1+stride, … so the stride MUST equal the number of participants
    // (main + woken workers) or chunks assigned to non-woken slots are dropped.
    // Set per Execute so a low MaxDegreeOfParallelism wakes fewer workers without
    // losing chunks. Volatile: published before _workReady[i].Set() wakes a worker.
    private volatile int _stride = 1;

    // Completion tracking — workers decrement, dispatcher waits for zero
    private int _remaining;

    // Completion signal for the dispatcher
    private readonly ManualResetEventSlim _allDone = new(false);

    // Serialize concurrent Execute calls
    private readonly object _executeLock = new();

    private PersistentParallelExecutor()
    {
        // Size the parked pool to the MACHINE width (cores-1, ceiling 32), NOT
        // CpuParallelSettings.WorkerPoolThreads. WorkerPoolThreads folds in the CURRENT
        // MaxDegreeOfParallelism, and this pool initializes exactly once — so a transient
        // low MaxDoP at the first dispatch (a DOP-pinned warmup/probe, or a consumer that
        // lowers MaxDoP before its first parallel op) would PERMANENTLY cap the pool and
        // kill many-core scaling for the process's life (the same singleton-init bug fixed
        // in CooperativeGemmScheduler). Per-dispatch concurrency is bounded per call below
        // (effWorkers honors the current MaxDegreeOfParallelism), so a machine-width parked
        // pool honors MaxDoP per op without the permanent cap; parked workers cost no CPU.
        const int ceiling = 32;
        _numWorkers = Math.Max(1, Math.Min(Environment.ProcessorCount - 1, ceiling));
        _workers = new Thread[_numWorkers];
        _workReady = new ManualResetEventSlim[_numWorkers];

        for (int i = 0; i < _numWorkers; i++)
        {
            // spinCount 2047 (the ManualResetEventSlim max): keep workers busy-
            // spinning before they park, so back-to-back small GEMMs (inference)
            // find them already hot — no kernel wake per Execute. #475: the
            // default low spin count made each dispatch pay ~µs wake latency per
            // worker, which dominates sub-millisecond tile work (1.1-1.7× scaling).
            // libtorch/OpenMP busy-wait for the same reason. Workers still PARK
            // after the spin window when the workload truly ends.
            _workReady[i] = new ManualResetEventSlim(false, spinCount: _mresSpinCount);
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

    // Warm-pool keep-alive: after finishing an op, busy-spin checking for the next
    // dispatch for up to this many SpinWait rounds before falling back to the
    // blocking Wait(). The MRES spinCount (2047, ~tens of µs) is too short to bridge
    // the inter-op gaps in a forward pass (LayerNorm→MHA→FFN→…), so workers block and
    // pay the full wakeup latency on EVERY op — profiled as the dominant small-op cost
    // (a transformer LayerNorm measured 192 µs parked vs ~65 µs of actual work). A
    // longer warm window keeps workers hot ACROSS the forward pass so back-to-back ops
    // find them spinning (≈ free dispatch). Trade-off: CPU burned during genuinely-idle
    // gaps; bounded by the spin count, then it parks. Env-tunable; 0 = original (park
    // immediately via MRES spin only). Each SpinWait(32) ≈ ~1 µs, so 256 ≈ ~256 µs warm.
    private static readonly int _warmSpins =
        int.TryParse(System.Environment.GetEnvironmentVariable("AIDOTNET_PPE_WARMSPIN"), out var ws) && ws > 0 ? ws : 0;

    private void WorkerLoop(int slot)
    {
        while (true)
        {
            // Warm-spin for the next dispatch before blocking (keeps the pool hot across
            // a forward pass so small back-to-back ops skip the park/wakeup latency).
            if (_warmSpins > 0)
            {
                for (int s = 0; s < _warmSpins && !_workReady[slot].IsSet; s++)
                    System.Threading.Thread.SpinWait(32);
            }
            // Wait for work signal (returns immediately if a warm-spin already saw it).
            _workReady[slot].Wait();
            _workReady[slot].Reset();

            // Execute all assigned chunks for this worker slot.
            // Chunks are assigned round-robin across all participants (workers + main thread).
            // Total participants = _numWorkers + 1 (main). Worker slot N gets chunks (slot+1), (slot+1+stride), etc.
            // Set reentrancy guard on worker thread so nested Execute() from
            // callbacks falls back to sequential instead of deadlocking on _executeLock.
            _isExecuting = true;
            int stride = _stride;   // per-dispatch participant count (main + active workers)
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
    [ThreadStatic]
    private static bool _isExecuting;

    /// <summary>
    /// Default serial-fallback grain size — total elementwise work
    /// below this threshold runs inline on the calling thread (no
    /// worker dispatch). 32K elements matches PyTorch's
    /// <c>at::internal::GRAIN_SIZE</c> default. Tuned for fp32 work;
    /// fp64-heavy hot loops can override per-call via the
    /// <see cref="Execute(int, long, Action{int})"/> overload.
    ///
    /// <para>Issue #313 background: profiling ViT-Base CPU training
    /// found ~90% of wall time in <c>LowLevelLifoSemaphore.WaitForSignal</c>
    /// and <c>ManualResetEventSlim.Wait</c> — workers being dispatched
    /// to per-channel work that's smaller than the dispatch overhead
    /// itself. Below the grain size, serial inline beats parallel
    /// dispatch + signal + join by a wide margin.</para>
    /// </summary>
    public static int DefaultSerialGrainSize { get; set; } = 32 * 1024;

    /// <summary>
    /// Parallel execute with PyTorch-style serial-fallback below
    /// <paramref name="totalWork"/> &lt; <see cref="DefaultSerialGrainSize"/>.
    /// Use this overload from any kernel that knows its inner-loop
    /// size — pass the rough element-FMA count and we'll skip the
    /// worker pool entirely when the work is too small to justify
    /// the dispatch.
    /// </summary>
    /// <param name="numChunks">Chunk count for the parallel path.</param>
    /// <param name="totalWork">Total elementwise work the
    /// <paramref name="action"/> body will perform across all chunks
    /// combined. Compared against <see cref="DefaultSerialGrainSize"/>
    /// to decide between serial inline and parallel dispatch.</param>
    /// <param name="action">Per-chunk callback.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Execute(int numChunks, long totalWork, Action<int> action)
    {
        if (numChunks <= 0) return;
        if (totalWork < DefaultSerialGrainSize)
        {
            // Below grain size — run inline on the calling thread.
            // No worker wakeup, no completion-event wait, no
            // _executeLock contention. Workers stay parked.
            //
            // Match the parallel path's exception semantics: capture
            // the first thrown exception, finish the remaining chunks
            // (the parallel path already runs main + workers to
            // completion before the dispatcher re-throws), then
            // re-throw at the end. Without this, a kernel that
            // crosses the grain-size threshold mid-run (dynamic batch
            // shrinking, varying inner shapes) would observe a
            // behavior change between the two paths: chunks 1+
            // skipped on the serial path, run on the parallel path.
            Exception? firstException = null;
            for (int c = 0; c < numChunks; c++)
            {
                try { action(c); }
                catch (Exception ex) { firstException ??= ex; }
            }
            if (firstException is not null) throw firstException;
            return;
        }
        Execute(numChunks, action);
    }

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

        // Detect reentrancy (nested Execute from callback) — fall back to sequential
        if (_isExecuting)
        {
            for (int c = 0; c < numChunks; c++)
                action(c);
            return;
        }

        lock (_executeLock)
        {
            _isExecuting = true;
            try
            {
                // Honor the CURRENT MaxDegreeOfParallelism per dispatch: wake at most
                // MaxDoP-1 workers (the caller participates as the MaxDoP-th). The parked pool
                // is machine-width, but a low MaxDoP must not oversubscribe — and the stride
                // below is set to the participant count so the un-woken slots' chunks aren't
                // dropped.
                int maxDeg = CpuParallelSettings.MaxDegreeOfParallelism;
                int effWorkers = Math.Min(_numWorkers, Math.Max(0, maxDeg - 1));
                int workersNeeded = Math.Min(numChunks - 1, effWorkers);
                int stride = workersNeeded + 1;   // participants = main + woken workers

                // Setup shared state
                _action = action;
                _numChunks = numChunks;
                _stride = stride;                 // publish BEFORE waking any worker
                _remaining = workersNeeded;
                _workerException = null;
                _allDone.Reset();

                // Wake workers (they're already spinning/blocked on ManualResetEventSlim)
                for (int i = 0; i < workersNeeded; i++)
                {
                    _workReady[i].Set();
                }

                // Main thread does chunk 0 and any overflow chunks (round-robin by stride:
                // main gets 0, stride, 2*stride, …; worker slot i gets i+1, i+1+stride, …).
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
                    mainChunk += stride;
                }

                // Wait for the woken workers to finish (each decrements _remaining; the last
                // sets _allDone). Skip when no workers were woken (workersNeeded == 0, e.g.
                // MaxDoP==1): the main thread already ran every chunk (stride == 1), and
                // _allDone would never be set — waiting would hang.
                if (workersNeeded > 0)
                    _allDone.Wait();

                _action = null;

                // Re-throw first captured exception (worker or main thread)
                var workerEx = _workerException;
                if (mainException is not null)
                    throw mainException;
                if (workerEx is not null)
                    throw workerEx;
            }
            finally
            {
                _isExecuting = false;
            }
        }
    }
}
