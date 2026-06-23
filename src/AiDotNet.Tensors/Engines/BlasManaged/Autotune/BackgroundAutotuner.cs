using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Bounded-LRU shape-sighting tracker (#375 G4). Gates background measurement to a
/// shape's 2nd+ sighting (one-shot shapes never sweep) and de-dups so concurrent
/// callers enqueue a shape once while its measurement is in flight. Shape identity
/// includes the transpose flags — a transB call is a different tuning target than
/// the non-transposed shape, and the shapes reaching strategy selection are typically
/// transposed (Sub-S handles non-transposed aligned GEMM).
/// </summary>
internal sealed class SightingTracker
{
    internal readonly record struct ShapeId(int M, int N, int K, bool Fp64, bool TransA, bool TransB);

    private readonly int _capacity;
    private readonly object _lock = new();
    private readonly LinkedList<ShapeId> _lru = new();
    private readonly Dictionary<ShapeId, (LinkedListNode<ShapeId> node, int count, bool inFlight)> _map = new();

    public SightingTracker(int capacity = 4096)
    {
        if (capacity < 1)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity,
                "Capacity must be >= 1 (a non-positive cap would evict from an empty LRU and dereference null).");
        _capacity = capacity;
    }

    /// <summary>Record a sighting; return true iff this shape should be measured now
    /// (2nd+ sighting and not already in flight).</summary>
    public bool RecordAndShouldMeasure(ShapeId id)
    {
        lock (_lock)
        {
            if (_map.TryGetValue(id, out var e))
            {
                _lru.Remove(e.node);
                _lru.AddFirst(e.node);
                int newCount = e.count + 1;
                bool measure = newCount >= 2 && !e.inFlight;
                _map[id] = (e.node, newCount, e.inFlight || measure);
                return measure;
            }
            if (_map.Count >= _capacity)
            {
                var oldest = _lru.Last!;
                _lru.RemoveLast();
                _map.Remove(oldest.Value);
            }
            var node = new LinkedListNode<ShapeId>(id);
            _lru.AddFirst(node);
            _map[id] = (node, 1, false);
            return false;
        }
    }

    /// <summary>Clear the in-flight flag after a measurement completes.</summary>
    public void MarkDone(ShapeId id)
    {
        lock (_lock)
        {
            if (_map.TryGetValue(id, out var e))
                _map[id] = (e.node, e.count, false);
        }
    }
}

/// <summary>
/// Non-blocking background strategy autotuner (#375 Phase 3). A single below-normal
/// priority worker sweeps Streaming / PackAOnly / PackBoth on freshly-allocated scratch
/// buffers (never caller data, G5-safe), in the serving path's current BlasMode, and
/// persists the winner (strategy + blocking, KernelVersion-tagged) under the trans-aware
/// shape key. Skips large shapes (G3) — those get offline pre-warm instead.
/// </summary>
internal static class BackgroundAutotuner
{
    internal const long WorkCeiling = 8_000_000L;  // skip background measurement above this

    private static readonly BlockingCollection<SightingTracker.ShapeId> _queue =
        new(boundedCapacity: 64);
    private static readonly SightingTracker _tracker = new();
    private static int _started;

    /// <summary>
    /// Master switch for the background autotuner. Default ON (the #375 differentiator —
    /// learn the deployment's shapes without blocking). The test suite turns it OFF so its
    /// concurrent measurement load doesn't perturb timing-sensitive tests; the tests that
    /// exercise it re-enable locally inside their serialized collection.
    /// </summary>
    internal static bool Enabled { get; set; } = true;

    internal static bool ShouldMeasureSize(int m, int n, int k) => (long)m * n * k <= WorkCeiling;

    /// <summary>
    /// Called from SelectStrategy. Records the sighting and enqueues a measurement on the
    /// 2nd+ sighting of a not-too-large shape. Never blocks the caller.
    /// </summary>
    public static void Observe(int m, int n, int k, bool fp64, bool transA, bool transB)
    {
        if (!Enabled || !ShouldMeasureSize(m, n, k)) return;
        var id = new SightingTracker.ShapeId(m, n, k, fp64, transA, transB);
        if (!_tracker.RecordAndShouldMeasure(id)) return;
        EnsureStarted();
        if (!_queue.TryAdd(id)) _tracker.MarkDone(id); // queue full → drop, re-eligible later
    }

    private static void EnsureStarted()
    {
        if (Interlocked.CompareExchange(ref _started, 1, 0) != 0) return;
        var t = new Thread(WorkerLoop)
        {
            IsBackground = true,
            Priority = ThreadPriority.BelowNormal,
            Name = "AiDotNet-BlasAutotuner",
        };
        t.Start();
    }

    private static void WorkerLoop()
    {
        foreach (var id in _queue.GetConsumingEnumerable())
        {
            try { Measure(id); }
            catch { /* best-effort; table default stands */ }
            finally { _tracker.MarkDone(id); }
            Thread.Yield();
        }
    }

    /// <summary>Synchronous measurement entry point for tests + the pre-warm pipeline.</summary>
    internal static void MeasureNowForTest(SightingTracker.ShapeId id) => Measure(id);

    private static void Measure(SightingTracker.ShapeId id)
    {
        bool deterministic = BlasProvider.IsDeterministicMode;
        var shape = id.Fp64
            ? BlasManagedAutotune.EncodeShape<double>(id.M, id.N, id.K, id.TransA, id.TransB, 0, 0, false, deterministic)
            : BlasManagedAutotune.EncodeShape<float>(id.M, id.N, id.K, id.TransA, id.TransB, 0, 0, false, deterministic);

        // Candidate strategies. PackAOnly has no transB path (Gemm redirects it to PackBoth),
        // so skip it as a distinct candidate when transB to avoid a redundant PackBoth timing.
        Span<PackingMode> candidates = id.TransB
            ? stackalloc[] { PackingMode.ForceStreaming, PackingMode.ForcePackBoth }
            : stackalloc[] { PackingMode.ForceStreaming, PackingMode.ForcePackAOnly, PackingMode.ForcePackBoth };

        PackingMode best = PackingMode.ForceStreaming;
        double bestMs = double.MaxValue;
        foreach (var mode in candidates)
        {
            double ms = id.Fp64 ? TimeFp64(id, mode) : TimeFp32(id, mode);
            if (ms < bestMs) { bestMs = ms; best = mode; }
        }
        // Strategy + (heuristic) blocking stored together (G11). Blocking refinement reuses
        // the heuristic defaults for now; the strategy is the measured winner.
        BlasManagedAutotune.StoreStrategy(shape, best, ParallelismAxis.M,
            mc: 64, nc: 64, kc: 64, threadCount: Environment.ProcessorCount, BlasKernelVersion.Current);
    }

    private static double TimeFp32(SightingTracker.ShapeId id, PackingMode mode)
    {
        int aRows = id.TransA ? id.K : id.M, aCols = id.TransA ? id.M : id.K;
        int bRows = id.TransB ? id.N : id.K, bCols = id.TransB ? id.K : id.N;
        var a = new float[aRows * aCols];
        var b = new float[bRows * bCols];
        var c = new float[id.M * id.N];
        var rng = new Random(17);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var opts = new BlasOptions<float> { PackingMode = mode };
        for (int w = 0; w < 3; w++) BlasManaged.Gemm<float>(a, aCols, id.TransA, b, bCols, id.TransB, c, id.N, id.M, id.N, id.K, opts);
        return MinOfN<float>(a, aCols, id.TransA, b, bCols, id.TransB, c, id.M, id.N, id.K, in opts);
    }

    /// <summary>
    /// #653: per-run MIN over N timings, NOT the sum/mean of a run-batch. The background sweep
    /// runs concurrently with foreground work (BelowNormal, but not exclusive), so any single
    /// run can be preempted/contended; summing 10 runs lets one spike inflate a strategy and
    /// pick the wrong winner. The MIN captures the contention-free slice — the strategy's true
    /// cost — and is what makes the learned choice match a clean micro-benchmark.
    /// </summary>
    private static double MinOfN<T>(
        T[] a, int aCols, bool transA, T[] b, int bCols, bool transB,
        T[] c, int m, int n, int k, in BlasOptions<T> opts, int reps = 12) where T : unmanaged
    {
        double best = double.MaxValue;
        for (int i = 0; i < reps; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            BlasManaged.Gemm<T>(a, aCols, transA, b, bCols, transB, c, n, m, n, k, opts);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            if (ms < best) best = ms;
        }
        return best;
    }

    private static double TimeFp64(SightingTracker.ShapeId id, PackingMode mode)
    {
        int aRows = id.TransA ? id.K : id.M, aCols = id.TransA ? id.M : id.K;
        int bRows = id.TransB ? id.N : id.K, bCols = id.TransB ? id.K : id.N;
        var a = new double[aRows * aCols];
        var b = new double[bRows * bCols];
        var c = new double[id.M * id.N];
        var rng = new Random(17);
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        var opts = new BlasOptions<double> { PackingMode = mode };
        for (int w = 0; w < 3; w++) BlasManaged.Gemm<double>(a, aCols, id.TransA, b, bCols, id.TransB, c, id.N, id.M, id.N, id.K, opts);
        return MinOfN<double>(a, aCols, id.TransA, b, bCols, id.TransB, c, id.M, id.N, id.K, in opts);
    }
}
