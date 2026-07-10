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
    // Clamped to ManualResetEventSlim's valid 0..2047 range — its ctor throws
    // ArgumentOutOfRangeException for spinCount > 2047 (the count is stored in an 11-bit
    // field, max (1 << 11) - 1). An unparseable OR out-of-range value falls back to the
    // default 32 so a bad env value can never fail executor init and take the whole pool down.
    private static readonly int _mresSpinCount =
        int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_PPE_SPINCOUNT"), out var sc)
            && sc >= 0 && sc <= 2047 ? sc : 32;

    private readonly int _numWorkers;
    private readonly Thread[] _workers;

    // Per-worker signaling: workers wait on these to receive work
    private readonly ManualResetEventSlim[] _workReady;

    // Per-dispatch work — ONE immutable Job published atomically via _job before any worker is woken.
    // A worker reads _job once and takes everything (body, chunk count, stride) from that single
    // reference, so it can never observe a partially-updated or cross-dispatch-contaminated field set
    // — the class of race that made the earlier shared-mutable-field design fragile under load.
    // Cleared to null after each dispatch so the body's captured references (e.g. the large im2col /
    // GEMM arrays a thread-local body closes over) are released for GC between dispatches.
    private volatile Job? _job;

    /// <summary>
    /// Immutable snapshot of one dispatch. Either <see cref="Action"/> (plain mode) OR the
    /// <see cref="LocalInit"/>/<see cref="LocalBody"/>/<see cref="LocalFinally"/> trio (thread-local
    /// mode — each participant builds ONE TLocal, boxed to object, and reuses it across its strided
    /// chunks) is set. Chunks are distributed round-robin across participants by <see cref="Stride"/>
    /// (= active workers + 1): participant p runs p, p+Stride, … &lt; <see cref="NumChunks"/>.
    /// </summary>
    private sealed class Job
    {
        public readonly Action<int>? Action;
        public readonly Func<object?>? LocalInit;
        public readonly Action<int, object?>? LocalBody;
        public readonly Action<object?>? LocalFinally;
        public readonly int NumChunks;
        public readonly int Stride;

        public Job(Action<int>? action, Func<object?>? localInit, Action<int, object?>? localBody,
            Action<object?>? localFinally, int numChunks, int stride)
        {
            Action = action;
            LocalInit = localInit;
            LocalBody = localBody;
            LocalFinally = localFinally;
            NumChunks = numChunks;
            Stride = stride;
        }
    }

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

    // Adaptive keep-warm window (Stopwatch ticks). After finishing an op, a worker
    // stays hot — cooperatively spinning for the next dispatch — for as long as the
    // pool has been dispatched-to within this window, then parks (blocks) so a
    // genuinely-idle pool burns no CPU.
    //
    // Why time-, not spin-count-based: the MRES spinCount and the old fixed SpinWait
    // budget (#475 lowered it to 32 to stop ~12 idle cores burning during training)
    // are far too short to bridge the inter-op gaps in a forward pass
    // (LayerNorm→MHA→FFN→…), so a parked worker paid the full OS wakeup latency on
    // EVERY op — profiled as the dominant small-op cost (a transformer LayerNorm
    // measured 192 µs parked vs ~65 µs of actual work), and the head-to-head bench
    // showed a ~70 µs wakeup floor on cold medium-work dispatches. A window keyed on
    // *dispatch recency* keeps workers hot across a hot loop (training / a forward
    // pass issue dispatches every few µs, so the window never lapses → near-free
    // dispatch) yet lets them park the moment the loop ends.
    //
    // Why this no longer re-introduces the #475 idle burn / oversubscription: the
    // spin is COOPERATIVE — it yields the core (Thread.Yield) on a fixed cadence, so
    // a spinning worker releases its core to the dispatcher / other runnable work
    // instead of monopolizing it. A fixed busy-SpinWait budget (the naive keep-warm)
    // instead blew up 7× at high worker counts because ~31 workers all held cores and
    // starved the dispatcher; the periodic yield removes that (verified: 64-chunk/32-
    // worker dispatch stays ~equal to Parallel.For instead of the 1.8× regression the
    // busy-spin showed). Env override AIDOTNET_PPE_WARMWINDOW_US sets the window in
    // microseconds; 0 disables (park immediately). Default 200 µs.
    private static readonly long _warmWindowTicks = ComputeWarmWindowTicks();

    private static long ComputeWarmWindowTicks()
    {
        long micros = 200;
        if (int.TryParse(System.Environment.GetEnvironmentVariable("AIDOTNET_PPE_WARMWINDOW_US"), out var us) && us >= 0)
            micros = us;
        // ticks = seconds * frequency = (micros / 1e6) * Stopwatch.Frequency
        return (long)(micros * (System.Diagnostics.Stopwatch.Frequency / 1_000_000.0));
    }

    // Timestamp (Stopwatch ticks) of the most recent dispatch. Workers read this to
    // decide whether to keep warm-spinning or park. Written on every Execute.
    private long _lastDispatchTicks;

    // Worker count of the most recent dispatch. Warm-spinning only pays off when the
    // dispatch leaves SPARE cores: a worker that busy-spins on a core the dispatcher
    // (or another about-to-run worker) needs just steals it, so a dispatch that
    // already saturates the machine (workersNeeded ≈ _numWorkers) must PARK its
    // workers instead — spinning there caused the 7× oversubscription blow-up in the
    // 64-chunk/32-core bench. Workers read this (from the previous, same-shape dispatch
    // in a hot loop) to gate the warm-spin; it self-corrects within one op on a shape
    // change. Written on every Execute.
    private int _lastWorkersNeeded;

    /// <summary>
    /// Runs one participant's strided chunk set: chunks <paramref name="firstChunk"/>,
    /// firstChunk+Stride, … &lt; <c>job.NumChunks</c>. In normal mode calls <c>job.Action</c> per chunk;
    /// in thread-local mode (<c>job.LocalBody</c> set) creates ONE TLocal via <c>job.LocalInit</c> for
    /// the whole run and disposes it once via <c>job.LocalFinally</c>. Returns the first exception
    /// thrown by any chunk (or localInit/localFinally), matching Parallel.For's "finish the rest,
    /// re-throw first" semantics. Shared by the main thread and every worker (each passed the SAME
    /// immutable <paramref name="job"/>) so both paths behave identically.
    /// </summary>
    private static Exception? RunParticipantChunks(Job job, int firstChunk)
    {
        Exception? first = null;
        int numChunks = job.NumChunks, stride = job.Stride;
        var localBody = job.LocalBody;
        if (localBody is null)
        {
            var action = job.Action!;
            for (int c = firstChunk; c < numChunks; c += stride)
            {
                try { action(c); }
                catch (Exception ex) { first ??= ex; }
            }
            return first;
        }

        // Thread-local mode: one TLocal for this participant's whole strided run.
        object? tlocal = null;
        bool inited = false;
        try
        {
            tlocal = job.LocalInit!();
            inited = true;
            for (int c = firstChunk; c < numChunks; c += stride)
            {
                try { localBody(c, tlocal); }
                catch (Exception ex) { first ??= ex; }
            }
        }
        catch (Exception ex) { first ??= ex; }   // localInit threw
        finally
        {
            if (inited)
            {
                try { job.LocalFinally!(tlocal); }
                catch (Exception ex) { first ??= ex; }
            }
        }
        return first;
    }

    private void WorkerLoop(int slot)
    {
        while (true)
        {
            // Adaptive cooperative warm-spin: stay hot while the pool is actively
            // being dispatched to (recency window), but yield the core periodically so
            // we never oversubscribe the dispatcher, and give up to a blocking park
            // once the pool goes idle past the window.
            long warm = _warmWindowTicks;
            // Only warm-spin when the last dispatch left spare cores (workersNeeded <
            // _numWorkers). When a dispatch saturates the machine, spinning steals the
            // core the dispatcher needs → oversubscription; park instead so the wakeup
            // overlaps the (already large, since saturating dispatches are big-work) op.
            if (warm > 0 && System.Threading.Volatile.Read(ref _lastWorkersNeeded) < _numWorkers && !_workReady[slot].IsSet)
            {
                int spins = 0;
                while (!_workReady[slot].IsSet)
                {
                    // Tight, responsive spin on EVERY iteration so an imminent dispatch is
                    // caught in ~tens of ns (checking IsSet each pass). Yielding on every
                    // pass — the naive "cooperative" version — deschedules a hot worker out
                    // from under the very dispatch it is waiting for, which measured SLOWER
                    // than parking on realistic inter-op-gap loops (118 µs vs 80 µs/op).
                    System.Threading.Thread.SpinWait(32);
                    if ((++spins & 0x1FF) == 0)
                    {
                        // Only every ~512 passes (~tens of µs): a single cooperative yield as
                        // a pressure-relief valve for the pathological all-workers-spinning-on-
                        // a-saturated-machine case (this is what kept the busy-spin from the
                        // 7× oversubscription blow-up), plus the idle-window check that parks
                        // the worker once the pool goes quiet. Infrequent enough that it never
                        // costs a hot worker its responsiveness.
                        System.Threading.Thread.Yield();
                        if (System.Diagnostics.Stopwatch.GetTimestamp() - System.Threading.Volatile.Read(ref _lastDispatchTicks) >= warm)
                            break;
                    }
                }
            }
            // Wait for work signal (returns immediately if the warm-spin already saw it).
            _workReady[slot].Wait();
            _workReady[slot].Reset();

            // Execute all assigned chunks for this worker slot.
            // Chunks are assigned round-robin across all participants (workers + main thread).
            // Total participants = _numWorkers + 1 (main). Worker slot N gets chunks (slot+1), (slot+1+stride), etc.
            // Set reentrancy guard on worker thread so nested Execute() from
            // callbacks falls back to sequential instead of deadlocking on _executeLock.
            _isExecuting = true;
            // Read the whole dispatch atomically: _job was published (volatile) BEFORE this worker's
            // _workReady was Set, so this single acquiring read gives a fully-consistent Job — no
            // chance of a half-updated body/stride/chunk-count from a concurrent or prior dispatch.
            var job = _job!;
            // Run this worker's strided chunk set (normal or thread-local mode). Capture the
            // first exception — re-thrown on the caller thread.
            var workerEx = RunParticipantChunks(job, slot + 1);
            if (workerEx is not null)
                Interlocked.CompareExchange(ref _workerException, workerEx, null);

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
    public void Execute(int numChunks, Action<int> action) => Execute(numChunks, 0, action);

    /// <summary>
    /// Per-dispatch max-degree-of-parallelism variant. <paramref name="maxDop"/> &lt;= 0 uses the
    /// global <see cref="CpuParallelSettings.MaxDegreeOfParallelism"/>; a positive value caps THIS
    /// dispatch's participant count (bounded by the global, so the process-wide "pin to 1" off-switch
    /// still wins). The drop-in for a <c>Parallel.For</c> whose <c>ParallelOptions.MaxDegreeOfParallelism</c>
    /// is set per call (e.g. the SpMM row loop's thread pin).
    /// </summary>
    public void Execute(int numChunks, int maxDop, Action<int> action)
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

        // Single global A/B / kill-switch (AIDOTNET_COOP_POOL=0), in ONE place for every
        // call site (PR #762 consolidation): route the whole dispatch to the .NET ThreadPool's
        // Parallel.For instead of the persistent worker pool. Disjoint [0,numChunks) chunks ⇒
        // bit-identical to the pool path. This is the ONLY remaining Parallel.For dispatch on the
        // CPU compute path — kept as the benchmark baseline (ParallelPrimitiveBench --ab-parallel)
        // and the operational rollback if the pool ever misbehaves on an unusual host.
        if (!CpuParallelSettings.UseCooperativePool)
        {
            int gm = CpuParallelSettings.MaxDegreeOfParallelism;
            int md = maxDop > 0 ? Math.Min(maxDop, gm) : gm;
            System.Threading.Tasks.Parallel.For(0, numChunks,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, md) },
                action);
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
                // dropped. A positive per-call maxDop caps this further (bounded by the global).
                int globalMaxDeg = CpuParallelSettings.MaxDegreeOfParallelism;
                int maxDeg = maxDop > 0 ? Math.Min(maxDop, globalMaxDeg) : globalMaxDeg;
                int effWorkers = Math.Min(_numWorkers, Math.Max(0, maxDeg - 1));
                int workersNeeded = Math.Min(numChunks - 1, effWorkers);
                int stride = workersNeeded + 1;   // participants = main + woken workers

                // Publish dispatch time + worker count so warm-spinning workers know the
                // pool is active and whether this dispatch left spare cores to spin on
                // (adaptive keep-warm, gated on spare capacity).
                System.Threading.Volatile.Write(ref _lastWorkersNeeded, workersNeeded);
                System.Threading.Volatile.Write(ref _lastDispatchTicks, System.Diagnostics.Stopwatch.GetTimestamp());

                // Completion state, then publish the whole dispatch as ONE immutable Job. The
                // _job volatile write (and the _workReady Set() release below) publishes _remaining
                // too, so a woken worker reads a fully-consistent dispatch.
                _remaining = workersNeeded;
                _workerException = null;
                _allDone.Reset();
                _job = new Job(action, null, null, null, numChunks, stride); // publish BEFORE waking any worker
                var job = _job;

                // Wake workers (they're already spinning/blocked on ManualResetEventSlim)
                for (int i = 0; i < workersNeeded; i++)
                {
                    _workReady[i].Set();
                }

                // Main thread runs its strided chunk set (chunk 0, stride, 2*stride, …) — the
                // same RunParticipantChunks path every worker takes, so normal and thread-local
                // modes behave identically across all participants.
                Exception? mainException = RunParticipantChunks(job, 0);

                // Wait for the woken workers to finish (each decrements _remaining; the last
                // sets _allDone). Skip when no workers were woken (workersNeeded == 0, e.g.
                // MaxDoP==1): the main thread already ran every chunk (stride == 1), and
                // _allDone would never be set — waiting would hang.
                if (workersNeeded > 0)
                    _allDone.Wait();

                _job = null; // release the body's captured references for GC between dispatches

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

    /// <summary>
    /// Thread-local-state variant — the persistent-pool equivalent of
    /// <see cref="System.Threading.Tasks.Parallel.For{TLocal}(int,int,Func{TLocal},Func{int,System.Threading.Tasks.ParallelLoopState,TLocal,TLocal},Action{TLocal})"/>.
    /// Each participant (main + each woken worker) calls <paramref name="localInit"/> ONCE, reuses that
    /// TLocal across the strided run of chunks it owns (<paramref name="body"/>(chunk, tlocal)), then
    /// calls <paramref name="localFinally"/> ONCE — so a per-worker rented buffer is rented/returned
    /// once per participant, not per chunk. <paramref name="maxDop"/> &lt;= 0 uses the global cap.
    /// </summary>
    public void Execute<TLocal>(int numChunks, int maxDop,
        Func<TLocal> localInit, Action<int, TLocal> body, Action<TLocal> localFinally)
    {
        if (numChunks <= 0) return;
        if (localInit is null) throw new ArgumentNullException(nameof(localInit));
        if (body is null) throw new ArgumentNullException(nameof(body));
        if (localFinally is null) throw new ArgumentNullException(nameof(localFinally));

        // Single-chunk or reentrant/nested: run inline on the calling thread, still honoring the
        // one-TLocal lifecycle. (Reentrancy would otherwise deadlock on _executeLock.)
        if (numChunks == 1 || _isExecuting)
        {
            TLocal tl = localInit();
            Exception? first = null;
            try
            {
                for (int c = 0; c < numChunks; c++)
                {
                    try { body(c, tl); }
                    catch (Exception ex) { first ??= ex; }
                }
            }
            finally
            {
                try { localFinally(tl); }
                catch (Exception ex) { first ??= ex; }
            }
            if (first is not null) throw first;
            return;
        }

        // Single global A/B / kill-switch (AIDOTNET_COOP_POOL=0) — one place for the thread-local
        // path too. Parallel.For's TLocal overload preserves the one-local-per-task + merge-once
        // semantics (localFinally runs per task, same as the pool's per-participant merge).
        if (!CpuParallelSettings.UseCooperativePool)
        {
            int gm = CpuParallelSettings.MaxDegreeOfParallelism;
            int md = maxDop > 0 ? Math.Min(maxDop, gm) : gm;
            System.Threading.Tasks.Parallel.For(0, numChunks,
                new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, md) },
                localInit,
                (c, state, tl) => { body(c, tl); return tl; },
                localFinally);
            return;
        }

        lock (_executeLock)
        {
            _isExecuting = true;
            try
            {
                int globalMaxDeg = CpuParallelSettings.MaxDegreeOfParallelism;
                int maxDeg = maxDop > 0 ? Math.Min(maxDop, globalMaxDeg) : globalMaxDeg;
                int effWorkers = Math.Min(_numWorkers, Math.Max(0, maxDeg - 1));
                int workersNeeded = Math.Min(numChunks - 1, effWorkers);
                int stride = workersNeeded + 1;

                System.Threading.Volatile.Write(ref _lastWorkersNeeded, workersNeeded);
                System.Threading.Volatile.Write(ref _lastDispatchTicks, System.Diagnostics.Stopwatch.GetTimestamp());

                // Completion state, then publish the whole thread-local dispatch as ONE Job (the
                // boxed TLocal lifecycle lives inside it), atomically, BEFORE waking any worker.
                _remaining = workersNeeded;
                _workerException = null;
                _allDone.Reset();
                _job = new Job(
                    action: null,
                    localInit: () => localInit(),
                    localBody: (c, o) => body(c, (TLocal)o!),
                    localFinally: o => localFinally((TLocal)o!),
                    numChunks: numChunks,
                    stride: stride);
                var job = _job;

                for (int i = 0; i < workersNeeded; i++)
                    _workReady[i].Set();

                Exception? mainException = RunParticipantChunks(job, 0);

                if (workersNeeded > 0)
                    _allDone.Wait();

                _job = null; // release the body's captured references for GC between dispatches

                var workerEx = _workerException;
                if (mainException is not null) throw mainException;
                if (workerEx is not null) throw workerEx;
            }
            finally
            {
                _isExecuting = false;
            }
        }
    }

    /// <summary>
    /// Reducing thread-local variant — the persistent-pool equivalent of
    /// <see cref="System.Threading.Tasks.Parallel.For{TLocal}(int,int,Func{TLocal},Func{int,System.Threading.Tasks.ParallelLoopState,TLocal,TLocal},Action{TLocal})"/>
    /// where the body <b>returns</b> the updated accumulator (functional reduction over a value- or
    /// reference-type <typeparamref name="TLocal"/>) rather than mutating it in place. Each participant
    /// seeds one accumulator via <paramref name="localInit"/>, threads it through its strided chunk set
    /// (<c>local = reduceBody(chunk, local)</c>), then merges once via <paramref name="localFinally"/>.
    ///
    /// <para>Built on the in-place <see cref="Execute{TLocal}(int,int,Func{TLocal},Action{int,TLocal},Action{TLocal})"/>
    /// overload by holding the accumulator in a single-element <c>TLocal[]</c>, so it reuses the exact
    /// same (validated) Job dispatch/join machinery — the array is the per-participant reference the
    /// Action overload mutates, and its element is the threaded accumulator. (A one-element array is
    /// used rather than <see cref="StrongBox{T}"/> because array element access is typed
    /// <c>TLocal</c>, not <c>TLocal?</c>, so it needs no null-suppression on the reduce/merge calls.)</para>
    /// </summary>
    public void Execute<TLocal>(int numChunks, int maxDop,
        Func<TLocal> localInit, Func<int, TLocal, TLocal> reduceBody, Action<TLocal> localFinally)
    {
        if (localInit is null) throw new ArgumentNullException(nameof(localInit));
        if (reduceBody is null) throw new ArgumentNullException(nameof(reduceBody));
        if (localFinally is null) throw new ArgumentNullException(nameof(localFinally));

        Execute<TLocal[]>(
            numChunks,
            maxDop,
            localInit: () => new[] { localInit() },
            body: (chunk, slot) => slot[0] = reduceBody(chunk, slot[0]),
            localFinally: slot => localFinally(slot[0]));
    }
}
