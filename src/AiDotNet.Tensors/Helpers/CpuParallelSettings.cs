using System;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides configuration settings for CPU parallel operations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This static class holds settings that control how
/// operations are parallelized across CPU cores. The default values are
/// optimized for most modern systems.
/// </remarks>
public static class CpuParallelSettings
{
    /// <summary>
    /// Gets or sets the maximum degree of parallelism for CPU operations.
    /// </summary>
    /// <remarks>
    /// Default is Environment.ProcessorCount, which uses all available cores.
    /// Set to 1 to disable parallelism.
    /// </remarks>
    public static int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// When true, the grain-size-gated <see cref="ParallelForOrSerial(int,int,long,System.Action{int})"/>
    /// helpers run serially regardless of work size. This is for the order-dependent reduction/
    /// accumulation kernels routed through these helpers, whose multi-threaded partial-sum
    /// combination is non-deterministic in floating point (thread-completion order). GEMM/conv use
    /// their own parallel paths and are unaffected, so enabling this keeps those fast while making
    /// reductions bit-reproducible. Default false (full speed); test harnesses that assert on exact
    /// training trajectories set it true alongside <see cref="BlasProvider.SetDeterministicMode"/>.
    /// </summary>
    public static bool DeterministicReductions { get; set; }

    /// <summary>
    /// Gets or sets whether SIMD (Single Instruction, Multiple Data) operations are enabled.
    /// </summary>
    /// <remarks>
    /// SIMD allows processing multiple data elements with a single instruction,
    /// significantly speeding up vector and matrix operations on supported hardware.
    /// </remarks>
    public static bool EnableSimd { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum array length before parallelization is applied.
    /// </summary>
    /// <remarks>
    /// For small arrays, the overhead of parallelization may exceed the benefits.
    /// Operations on arrays smaller than this threshold will run sequentially.
    /// </remarks>
    public static int ParallelThreshold { get; set; } = 50000;

    /// <summary>
    /// Gets or sets whether AVX2 hardware gather instructions are used for strided memory access.
    /// </summary>
    /// <remarks>
    /// AVX2 VGATHERDPS/VPGATHERDD can gather 8 floats in a single instruction using an index vector.
    /// This provides significant speedup for wavelet transforms, FFT butterfly patterns, and
    /// interleaved channel separation. However, some AMD processors (pre-Zen 3) have slow
    /// gather implementations that can be slower than scalar loops. Set to false to force
    /// scalar fallback on such hardware.
    /// </remarks>
    public static bool EnableAvx2Gather { get; set; } = true;

    /// <summary>
    /// The minimum chunk size for parallel operations.
    /// </summary>
    /// <remarks>
    /// This ensures each parallel task processes at least this many elements
    /// to avoid excessive task creation overhead.
    /// </remarks>
    public const int MinChunkSize = 8192;

    /// <summary>
    /// Thread-local flag: true while the calling thread is executing a body
    /// dispatched by one of this class's parallel loops. Nested parallel calls
    /// inspect it and run serially.
    /// </summary>
    /// <remarks>
    /// <b>Why this exists (nested-parallelism deadlock):</b> several kernels run
    /// an outer <see cref="ParallelForOrSerial(int,int,long,Action{int})"/> and,
    /// inside each iteration, call another parallel primitive — e.g. the float
    /// <c>ScaledDotProductAttention</c> parallelizes over batch·heads and each head
    /// runs a (multi-threaded) BlasManaged GEMM. Without this guard the outer
    /// workers occupy every ThreadPool thread and then block waiting on the inner
    /// loop's tasks, which can't get a thread to run on. The pool only recovers by
    /// slowly injecting new threads (~1/500ms), so the op crawls for minutes. The
    /// double SDPA path avoided this by hand-calling a sequential inner kernel; this
    /// guard generalizes that: once inside a parallel region, nested parallel loops
    /// collapse to serial automatically.
    /// </remarks>
    [ThreadStatic]
    private static bool _inParallelRegion;

    /// <summary>
    /// True when the calling thread is already running inside a parallel region
    /// dispatched by this class. Raw <c>Parallel.For</c> sites (e.g. the BlasManaged
    /// GEMM strategies) consult this to fall back to a serial loop when nested,
    /// preventing ThreadPool starvation. See <see cref="EnterParallelRegion"/>.
    /// </summary>
    public static bool IsInParallelRegion => _inParallelRegion;

    /// <summary>
    /// Marks the calling thread as inside a parallel region until the returned
    /// scope is disposed, restoring the prior value on dispose. Wrap the body of
    /// any <c>Parallel.For</c> worker with this so nested parallel calls serialize.
    /// </summary>
    internal static ParallelRegionScope EnterParallelRegion() => new ParallelRegionScope(true);

    /// <summary>
    /// RAII scope toggling <see cref="IsInParallelRegion"/>. Struct (no allocation);
    /// restores the previous value on <see cref="Dispose"/> so it nests correctly.
    /// </summary>
    internal readonly struct ParallelRegionScope : IDisposable
    {
        private readonly bool _previous;
        internal ParallelRegionScope(bool entering)
        {
            _previous = _inParallelRegion;
            _inParallelRegion = entering;
        }
        public void Dispose() => _inParallelRegion = _previous;
    }

    /// <summary>
    /// Executes a parallel for loop with chunked iterations.
    /// </summary>
    /// <param name="length">Total number of elements to process.</param>
    /// <param name="minChunkSize">Minimum elements per chunk.</param>
    /// <param name="action">Action to execute for each chunk (start index, count).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method divides work into chunks and processes
    /// them in parallel across available CPU cores for better performance.
    /// </remarks>
    public static void ParallelForChunks(int length, int minChunkSize, Action<int, int> action)
    {
        if (length <= 0)
            return;

        if (action is null)
            throw new ArgumentNullException(nameof(action));

        int maxDegree = MaxDegreeOfParallelism;
        // Nested call (already inside a parallel region) → serial, to avoid
        // ThreadPool starvation. See _inParallelRegion.
        if (maxDegree <= 1 || length <= minChunkSize || _inParallelRegion)
        {
            // Single-threaded execution
            action(0, length);
            return;
        }

        // Calculate number of chunks based on length and min chunk size
        int numChunks = Math.Min(maxDegree, (length + minChunkSize - 1) / minChunkSize);
        if (numChunks <= 1)
        {
            action(0, length);
            return;
        }

        int chunkSize = (length + numChunks - 1) / numChunks;

        Parallel.For(0, numChunks, new ParallelOptions { MaxDegreeOfParallelism = maxDegree }, i =>
        {
            using var _region = EnterParallelRegion();
            int start = i * chunkSize;
            int count = Math.Min(chunkSize, length - start);
            if (count > 0)
            {
                action(start, count);
            }
        });
    }

    /// <summary>
    /// Runs <paramref name="body"/> for each <c>p</c> in <c>[0, count)</c> across
    /// threads — but serially when the calling thread is already inside a parallel
    /// region (<see cref="IsInParallelRegion"/>) or parallelism is pinned off.
    /// Each parallel worker marks itself in-region so its own nested loops serialize.
    /// </summary>
    /// <remarks>
    /// For the BlasManaged GEMM strategies, which partition a fixed work count
    /// (<c>procs</c>) and always want all of it parallel at the top level, but must
    /// collapse to serial when invoked from inside another parallel loop (e.g. the
    /// per-head GEMM in float ScaledDotProductAttention) to avoid ThreadPool
    /// starvation. Unlike <see cref="ParallelForOrSerial(int,int,long,Action{int})"/>
    /// there is no grain-size gate — the caller has already decided this work is
    /// worth parallelizing.
    /// </remarks>
    /// <param name="count">Number of partitions / iterations.</param>
    /// <param name="body">Body invoked with each partition index.</param>
    public static void ParallelForRegion(int count, Action<int> body)
    {
        if (count <= 0) return;
        if (body is null) throw new ArgumentNullException(nameof(body));
        if (count == 1 || MaxDegreeOfParallelism <= 1 || _inParallelRegion)
        {
            for (int p = 0; p < count; p++) body(p);
            return;
        }
        Parallel.For(0, count,
            new ParallelOptions { MaxDegreeOfParallelism = MaxDegreeOfParallelism },
            p =>
            {
                using var _region = EnterParallelRegion();
                body(p);
            });
    }

    /// <summary>
    /// High-performance parallel execution using pre-spawned worker threads.
    /// Near-zero dispatch overhead — threads are already idle and wake instantly.
    /// Mimics libtorch's OpenMP thread pool pattern for maximum throughput.
    /// </summary>
    /// <param name="numChunks">Number of chunks to process in parallel.</param>
    /// <param name="action">Action receiving chunk index (0..numChunks-1).</param>
    public static void LightweightParallel(int numChunks, Action<int> action)
    {
        PersistentParallelExecutor.Instance.Execute(numChunks, action);
    }

    /// <summary>
    /// Grain-size-aware variant: forwards to
    /// <see cref="PersistentParallelExecutor.Execute(int, long, Action{int})"/>
    /// so callers that know their op's elementwise work can engage
    /// PyTorch-style serial-fallback below
    /// <see cref="PersistentParallelExecutor.DefaultSerialGrainSize"/>.
    /// Issue #319 follow-up to PR #316: kernels that pass a real
    /// totalWork value skip dispatch overhead on small ops.
    /// </summary>
    /// <param name="numChunks">Number of chunks to process in parallel.</param>
    /// <param name="totalWork">Total elementwise work the action will
    /// perform across all chunks combined.</param>
    /// <param name="action">Action receiving chunk index (0..numChunks-1).</param>
    public static void LightweightParallel(int numChunks, long totalWork, Action<int> action)
    {
        PersistentParallelExecutor.Instance.Execute(numChunks, totalWork, action);
    }

    /// <summary>
    /// Grain-size-aware drop-in for <see cref="System.Threading.Tasks.Parallel.For(int, int, Action{int})"/>.
    /// When <paramref name="totalWork"/> is below
    /// <see cref="PersistentParallelExecutor.DefaultSerialGrainSize"/>,
    /// runs the body inline on the calling thread — no ThreadPool
    /// dispatch, no <c>LowLevelLifoSemaphore</c> wait. Above the
    /// threshold, falls through to <c>Parallel.For</c>.
    ///
    /// <para>Issue #319 background: the original profile showed 42.59%
    /// of ViT-Base CPU train wall-clock in
    /// <c>LowLevelLifoSemaphore.WaitForSignal</c> — that's the .NET
    /// ThreadPool primitive used by every <c>Parallel.For</c> call.
    /// Hundreds of small-op call sites in <c>CpuEngine</c> dispatch
    /// to <c>Parallel.For</c> unconditionally even when the work is
    /// smaller than the dispatch overhead itself. This helper is the
    /// migration vehicle: each call site swaps
    /// <c>Parallel.For(0, n, body)</c> for
    /// <c>ParallelForOrSerial(0, n, totalWork, body)</c> and the
    /// JIT eliminates the dispatch on workloads below threshold.</para>
    /// </summary>
    /// <param name="fromInclusive">First index, inclusive.</param>
    /// <param name="toExclusive">One past last index, exclusive.</param>
    /// <param name="totalWork">Total elementwise work the body will
    /// perform across all iterations combined.</param>
    /// <param name="body">Iteration body — same shape as
    /// <c>Parallel.For</c>'s <c>Action&lt;int&gt;</c>.</param>
    /// <param name="deterministicSafe">When <see langword="true"/>, this loop is
    /// exempt from the <see cref="DeterministicReductions"/> serial-forcing gate: the
    /// caller guarantees its parallelism is bit-reproducible regardless of thread count
    /// (disjoint-write output tiles where each element's reduction is done by a single
    /// thread in fixed order — e.g. GEMM M/N-tile splits, see
    /// <c>DeterministicParallelGemmContractTests</c>). Order-dependent reduction/
    /// accumulation kernels leave this <see langword="false"/> so deterministic mode
    /// serializes them for reproducibility.</param>
    public static void ParallelForOrSerial(int fromInclusive, int toExclusive, long totalWork, Action<int> body, bool deterministicSafe = false)
    {
        if (toExclusive <= fromInclusive) return;
        // Honor the class-level MaxDegreeOfParallelism contract: if the user has
        // pinned to 1 thread, run serial regardless of work size. Same
        // pattern as ParallelForChunks (line 84) and the legacy LightweightParallel
        // code path. Snapshot BOTH gating values once so a concurrent setter
        // mid-call can't toggle us between the serial and parallel paths.
        int maxDegree = MaxDegreeOfParallelism;
        // DeterministicReductions forces order-dependent reductions serial for
        // bit-reproducibility; deterministicSafe callers stay parallel (they're already
        // reproducible across thread counts) so deterministic mode doesn't lose GEMM
        // parallelism — the whole point of Lever A (deterministic AND parallel).
        bool deterministic = DeterministicReductions && !deterministicSafe;
        // _inParallelRegion: a nested call (this thread is already a parallel
        // worker) runs serial to avoid ThreadPool starvation. See its docs.
        if (maxDegree <= 1 || deterministic || _inParallelRegion
            || totalWork < PersistentParallelExecutor.DefaultSerialGrainSize)
        {
            // Match Parallel.For's exception semantics: capture
            // first thrown exception, finish remaining iterations,
            // re-throw at end. Parallel.For uses
            // AggregateException — for serial-mode we re-throw the
            // raw first exception (consistent with the
            // PersistentParallelExecutor.Execute serial path).
            Exception? firstException = null;
            for (int i = fromInclusive; i < toExclusive; i++)
            {
                try { body(i); }
                catch (Exception ex) { firstException ??= ex; }
            }
            if (firstException is not null) throw firstException;
            return;
        }
        System.Threading.Interlocked.Increment(ref s_parallelForInvocations);

        // PR #531: route the general parallel-op path off raw Parallel.For (the .NET
        // ThreadPool — high dispatch latency + per-call task/range allocation, and the
        // LowLevelLifoSemaphore park/wakeup that dominated the small-op trace) onto the
        // low-latency cooperative pool. The scheduler's fixed worker set + caller-
        // participation gives cheaper dispatch (15x less allocation, lower median/p95 —
        // PR #531 bench) and no oversubscription under concurrent inference; it serializes
        // concurrent callers' chunks cooperatively rather than spawning per-call threads.
        // Disjoint-iteration safety is the SAME contract Parallel.For already requires of
        // its callers (this is the Action overload — each iteration writes its own output,
        // no cross-iteration reduction — so chunking is bit-identical to Parallel.For).
        // Gated by its OWN switch, decoupled from CooperativeGemmScheduler.Enabled (which
        // still gates the GEMM strategies pending their concurrency benchmark). Default-on
        // (validated across the full test suite); set false to fall back to Parallel.For.
        if (UseCooperativePool)
        {
            int count = toExclusive - fromInclusive;
            // Scale the chunk count with the work, not blindly to maxDegree. Waking all
            // cores-1 workers (plus the participating caller) for a small memory-bound op
            // oversubscribes the box — the OS preempts active workers and a stolen chunk
            // stalls the caller's join, which is the entire p95 tail (PR #531 sweep: at the
            // 64K-relu shape, 8 active threads → p95 29.6us beating Parallel.For's 32.6us,
            // while 31 → 224us). One chunk per ~16K work units gives each chunk enough to
            // amortize dispatch, caps active threads for small ops, and still fans a large
            // compute-bound op out to maxDegree. The pool keeps cores-1 workers parked; only
            // `chunks` of them wake per dispatch.
            const long workPerChunk = 8 * 1024;
            int byWork = (int)Math.Min(count, Math.Max(1, totalWork / workPerChunk));
            int chunks = Math.Min(maxDegree, byWork);
            int from = fromInclusive;
            Engines.BlasManaged.Pool.CooperativeGemmScheduler.Dispatch(chunks, chunk =>
            {
                using var _region = EnterParallelRegion();
                int cs = from + (int)((long)chunk * count / chunks);
                int ce = from + (int)((long)(chunk + 1) * count / chunks);
                for (int i = cs; i < ce; i++) body(i);
            });
            return;
        }

        System.Threading.Tasks.Parallel.For(
            fromInclusive,
            toExclusive,
            new ParallelOptions { MaxDegreeOfParallelism = maxDegree },
            i =>
            {
                // Mark this worker as inside a parallel region so any nested
                // parallel loop it triggers (e.g. a per-iteration GEMM) serializes.
                using var _region = EnterParallelRegion();
                body(i);
            });
    }

    /// <summary>
    /// PR #531: route <see cref="ParallelForOrSerial(int,int,long,Action{int},bool)"/>'s parallel
    /// path through the low-latency <c>CooperativeGemmScheduler</c> instead of
    /// <see cref="System.Threading.Tasks.Parallel.For(int,int,Action{int})"/>. Cheaper dispatch
    /// (≈15× less per-call allocation, lower median/p95 — the cooperative pool's fixed worker set
    /// + caller-participation avoids the .NET ThreadPool's per-call task/range state and
    /// park/wakeup) with no oversubscription under concurrent inference.
    ///
    /// <para>Default <c>true</c>. This is the Action overload only — disjoint-write iterations,
    /// so the result is bit-identical to <c>Parallel.For</c> regardless of chunk scheduling (the
    /// reduction-style <c>TLocal</c> overload is unaffected and keeps using <c>Parallel.For</c>).
    /// Decoupled from <c>CooperativeGemmScheduler.Enabled</c>, which separately gates the GEMM
    /// strategies. Set <c>false</c> to fall back to <c>Parallel.For</c> (e.g. to avoid the
    /// cooperative pool's dedicated worker threads in a thread-count-sensitive host).</para>
    /// </summary>
    public static bool UseCooperativePool { get; set; } = true;

    // PR #531 diagnostic: how often the general op path actually dispatches through
    // System.Threading.Tasks.Parallel.For (the .NET ThreadPool — the LowLevelLifoSemaphore
    // in the small-op trace) versus the low-latency StreamingWorkerPool (which is wired only
    // into StreamingStrategy). Counts the Action overload's parallel/serial branches.
    internal static long s_parallelForInvocations;
    internal static long ParallelForStatsSnapshot() => System.Threading.Volatile.Read(ref s_parallelForInvocations);
    internal static void ResetParallelForStats() { s_parallelForInvocations = 0; }

    /// <summary>
    /// Grain-size-aware drop-in for the localInit/localFinally overload of
    /// <see cref="System.Threading.Tasks.Parallel.For{TLocal}(int, int, Func{TLocal}, Func{int, System.Threading.Tasks.ParallelLoopState, TLocal, TLocal}, Action{TLocal})"/>.
    /// When <paramref name="totalWork"/> is below
    /// <see cref="PersistentParallelExecutor.DefaultSerialGrainSize"/>,
    /// runs the body inline on the calling thread with a single
    /// thread-local accumulator that's seeded by <paramref name="localInit"/>
    /// before the loop and finalised by <paramref name="localFinally"/> after.
    /// Above the threshold, dispatches to <c>Parallel.For</c>'s per-task-local
    /// overload unchanged.
    ///
    /// <para>Issue #319 follow-up: closes the gap for kernels that build
    /// per-thread accumulators (cross-entropy loss, BatchNorm reductions,
    /// random-fill primitives) — these stayed on raw <c>Parallel.For</c>
    /// because the existing helper only had the <c>Action&lt;int&gt;</c>
    /// signature. The serial-path <paramref name="localFinally"/> still runs
    /// once, so callers that aggregate per-task results into a shared field
    /// behave identically (a single thread accumulates everything inline).</para>
    /// </summary>
    /// <typeparam name="TLocal">Type of the per-task-local accumulator.</typeparam>
    /// <param name="fromInclusive">First index, inclusive.</param>
    /// <param name="toExclusive">One past last index, exclusive.</param>
    /// <param name="totalWork">Total elementwise work the body will
    /// perform across all iterations combined.</param>
    /// <param name="localInit">Factory producing the initial per-task local.</param>
    /// <param name="body">Iteration body — same shape as
    /// <c>Parallel.For</c>'s <c>Func&lt;int, ParallelLoopState, TLocal, TLocal&gt;</c>.</param>
    /// <param name="localFinally">Action invoked once per task with the final
    /// per-task local — typically merges the local into a shared accumulator.</param>
    public static void ParallelForOrSerial<TLocal>(
        int fromInclusive,
        int toExclusive,
        long totalWork,
        Func<TLocal> localInit,
        Func<int, System.Threading.Tasks.ParallelLoopState, TLocal, TLocal> body,
        Action<TLocal> localFinally)
    {
        if (toExclusive <= fromInclusive) return;
        // Honor the class-level MaxDegreeOfParallelism contract: pinning to 1
        // forces serial regardless of work size (same as the Action<int>
        // overload above). Snapshot both gating values once for consistency.
        int maxDegree = MaxDegreeOfParallelism;
        bool deterministic = DeterministicReductions;
        // _inParallelRegion: nested call → serial (avoids ThreadPool starvation).
        if (maxDegree <= 1 || deterministic || _inParallelRegion
            || totalWork < PersistentParallelExecutor.DefaultSerialGrainSize)
        {
            // Serial fast path: one local accumulator, no
            // ParallelLoopState (Stop/Break never fire serial-side).
            // Match Parallel.For's exception semantics: capture
            // first thrown exception, finish remaining iterations,
            // re-throw at end via raw exception (consistent with
            // PersistentParallelExecutor.Execute serial path).
            //
            // ParallelLoopState is sealed with no public constructor,
            // so we pass null. The body must tolerate it — same
            // contract that PersistentParallelExecutor's serial path
            // already imposes on its non-localInit body shape.
            TLocal local = localInit();
            Exception? firstException = null;
            for (int i = fromInclusive; i < toExclusive; i++)
            {
                try { local = body(i, null!, local); }
                catch (Exception ex) { firstException ??= ex; }
            }
            try { localFinally(local); }
            catch (Exception ex) { firstException ??= ex; }
            if (firstException is not null) throw firstException;
            return;
        }
        System.Threading.Tasks.Parallel.For(
            fromInclusive,
            toExclusive,
            new ParallelOptions { MaxDegreeOfParallelism = maxDegree },
            localInit,
            (i, state, local) =>
            {
                // Mark this worker as inside a parallel region for the body call
                // so nested parallel loops serialize (restored after each call).
                using var _region = EnterParallelRegion();
                return body(i, state, local);
            },
            localFinally);
    }
}
