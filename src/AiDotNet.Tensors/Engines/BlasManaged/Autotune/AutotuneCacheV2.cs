using System;
using System.Collections.Concurrent;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged;  // PackingMode

namespace AiDotNet.Tensors.Engines.BlasManaged.Autotune;

/// <summary>
/// Numeric element type for an autotune cache key. Mirrors the benchmark
/// catalog's <c>DType</c> but lives in src/ so the cache has no test-project
/// dependency.
/// </summary>
public enum DType
{
    /// <summary>32-bit float (FP32).</summary>
    Single,
    /// <summary>64-bit float (FP64).</summary>
    Double,
}

/// <summary>
/// Layer F (#375) — per-shape autotune cache. Stores the empirically-best
/// (<see cref="PackingMode"/> + thread count + tile sizes) per
/// (M, N, K, transA, transB, dtype). Replaces the brittle static dispatcher
/// heuristics with measured per-shape routing: unknown shapes run a one-time
/// warmup sweep, then every subsequent call of that shape hits the cached
/// winner. A re-tune trigger detects when a cached pick has gone stale (e.g.
/// thermal throttle, contention) and forces a re-sweep.
///
/// <para>
/// Instance-based (not static) to fit the facade/DI pattern — a single
/// process-wide instance lives in <see cref="BlasManaged"/>, but tests
/// construct their own. Thread-safe via <see cref="ConcurrentDictionary{TKey,TValue}"/>
/// plus per-slot locks during warmup.
/// </para>
/// </summary>
internal sealed class AutotuneCacheV2
{
    /// <summary>Cache key: a GEMM shape plus its layout + dtype.</summary>
    public readonly record struct ShapeKey(int M, int N, int K, bool TransA, bool TransB, DType Dtype);

    /// <summary>The cached empirical winner for a shape.</summary>
    public sealed class Entry
    {
        /// <summary>Winning packing strategy.</summary>
        public PackingMode Mode { get; init; }
        /// <summary>Winning thread count.</summary>
        public int NumThreads { get; init; }
        /// <summary>Macro-block M (cache-blocking parameter).</summary>
        public int Mc { get; init; }
        /// <summary>Macro-block N.</summary>
        public int Nc { get; init; }
        /// <summary>Macro-block K.</summary>
        public int Kc { get; init; }
        /// <summary>Microkernel row tile.</summary>
        public int Mr { get; init; }
        /// <summary>Microkernel col tile.</summary>
        public int Nr { get; init; }
        /// <summary>Best measured wall-time (ms) observed during warmup.</summary>
        public double MeasuredMs { get; set; }
        /// <summary>Number of warmup samples folded into this entry.</summary>
        public int SampleCount { get; set; }
        /// <summary>Consecutive observations exceeding 1.5× <see cref="MeasuredMs"/>.</summary>
        public int SlowStreak;
    }

    private sealed class WarmupSlot
    {
        public Entry? Best;
        public int SamplesTaken;
    }

    private readonly ConcurrentDictionary<ShapeKey, Entry> _cache = new();
    private readonly ConcurrentDictionary<ShapeKey, WarmupSlot> _warmup = new();

    /// <summary>Number of distinct shapes with a finalized cache entry.</summary>
    public int Count => _cache.Count;

    /// <summary>Drop all finalized entries and in-flight warmup slots, returning the
    /// cache to its empty initial state (process-global reset for tests / re-sweep).</summary>
    public void Clear()
    {
        _cache.Clear();
        _warmup.Clear();
    }

    /// <summary>Look up the cached winner for a shape. Returns false on miss.</summary>
    public bool TryGet(ShapeKey key, out Entry? entry) => _cache.TryGetValue(key, out entry);

    /// <summary>
    /// Record one warmup-sweep sample. Called once per candidate strategy for a
    /// new shape. The running minimum is kept. Call <see cref="FinalizeWarmup"/>
    /// after all candidates have been measured to publish the winner.
    /// </summary>
    public void RecordWarmupSample(ShapeKey key, PackingMode mode, int numThreads,
        int mc, int nc, int kc, int mr, int nr, double measuredMs)
    {
        var slot = _warmup.GetOrAdd(key, _ => new WarmupSlot());
        lock (slot)
        {
            slot.SamplesTaken++;
            if (slot.Best is null || measuredMs < slot.Best.MeasuredMs)
            {
                slot.Best = new Entry
                {
                    Mode = mode,
                    NumThreads = numThreads,
                    Mc = mc, Nc = nc, Kc = kc, Mr = mr, Nr = nr,
                    MeasuredMs = measuredMs,
                    SampleCount = 1,
                    SlowStreak = 0,
                };
            }
        }
    }

    /// <summary>
    /// Publish the warmup-best <see cref="Entry"/> into the read-mostly cache.
    /// Subsequent <see cref="TryGet"/> calls for this shape hit it directly.
    /// </summary>
    public void FinalizeWarmup(ShapeKey key)
    {
        if (!_warmup.TryRemove(key, out var slot)) return;
        // Read slot.Best under the same lock RecordWarmupSample uses to write it —
        // a concurrent same-shape warmup sample can still hold this slot reference
        // (it GetOrAdd'd before the TryRemove) and be mutating Best.
        Entry? best;
        lock (slot) { best = slot.Best; }
        if (best is not null) _cache[key] = best;
    }

    /// <summary>
    /// Observe an actual-call wall-time. If it exceeds the cached
    /// <see cref="Entry.MeasuredMs"/> by ≥1.5× for 10 consecutive observations,
    /// the entry is evicted and this returns true (caller should re-sweep). A
    /// single fast observation resets the streak.
    /// </summary>
    public bool ObserveAndMaybeReTune(ShapeKey key, double measuredMs)
    {
        if (!_cache.TryGetValue(key, out var entry)) return false;
        if (measuredMs > entry.MeasuredMs * 1.5)
        {
            int streak = Interlocked.Increment(ref entry.SlowStreak);
            if (streak >= 10)
            {
                _cache.TryRemove(key, out _);
                return true;
            }
        }
        else
        {
            Volatile.Write(ref entry.SlowStreak, 0);
        }
        return false;
    }
}
