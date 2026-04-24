// Copyright (c) AiDotNet. All rights reserved.
// Codegen observability — per-pass timing, autotune cache
// hit/miss stats, compiled-kernel cache occupancy. Matches the
// TelemetryConfig surface established by Wave 1 telemetry so
// downstream facade layers can thread these stats through
// PredictionModelBuilder / PredictionModelResult.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Guards;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Telemetry;

/// <summary>
/// Central telemetry sink for the codegen pipeline. Pass execution,
/// emitter success/decline counts, autotune lookups, and compiled-
/// kernel cache occupancy all feed through here; callers (tests,
/// the facade layer) query the aggregated snapshot to surface
/// diagnostics.
/// </summary>
/// <remarks>
/// <para><b>Zero cost when disabled:</b></para>
/// <para>
/// Matches the pattern used by <see cref="AnomalyModeScope"/> and
/// <see cref="SavedTensorHooks"/>: a single static
/// <see cref="IsEnabled"/> flag gates every recording method. When
/// false the caller's record-call is a single boolean branch + an
/// early return — nanosecond-scale overhead on the hot path.
/// </para>
/// <para><b>Thread-safety:</b></para>
/// <para>
/// All aggregation dictionaries are <see cref="ConcurrentDictionary{TKey, TValue}"/>
/// so recording from worker threads (Parallel.For / autotune async)
/// is safe. The per-pass timing histogram uses
/// <see cref="Interlocked.Add(ref long, long)"/> to accumulate
/// without per-sample lock acquisition.
/// </para>
/// </remarks>
public static class CodegenTelemetry
{
    private static volatile bool _enabled;

    /// <summary>
    /// Whether telemetry recording is active. Defaults to false so
    /// the production hot path pays no cost; tests or the facade
    /// enable it via <see cref="Enable"/>.
    /// </summary>
    public static bool IsEnabled => _enabled;

    /// <summary>Enables telemetry recording on all threads.</summary>
    public static void Enable() => _enabled = true;

    /// <summary>Disables telemetry recording on all threads.</summary>
    public static void Disable() => _enabled = false;

    // ─── Per-pass timing ─────────────────────────────────────────────

    private static readonly ConcurrentDictionary<string, PassTimingAccumulator> _passTimings
        = new();

    /// <summary>
    /// Starts a timing scope for a named optimization / emitter pass.
    /// Dispose the returned handle to record the elapsed ticks. Returns
    /// a no-op handle when telemetry is disabled.
    /// </summary>
    public static PassTimingScope TimePass(string passName)
    {
        if (!_enabled) return default;
        if (passName is null) throw new ArgumentNullException(nameof(passName));
        return new PassTimingScope(passName, Stopwatch.GetTimestamp());
    }

    internal static void RecordPassTiming(string passName, long elapsedTicks)
    {
        if (!_enabled) return;
        var acc = _passTimings.GetOrAdd(passName, _ => new PassTimingAccumulator());
        acc.Record(elapsedTicks);
    }

    /// <summary>
    /// Returns a snapshot of per-pass timing aggregates.
    /// </summary>
    public static IReadOnlyDictionary<string, PassTimingStats> GetPassTimings()
    {
        var result = new Dictionary<string, PassTimingStats>();
        foreach (var kv in _passTimings)
            result[kv.Key] = kv.Value.Snapshot();
        return result;
    }

    // ─── Emitter outcome counts ──────────────────────────────────────

    private static readonly ConcurrentDictionary<(CodegenTarget, string), long> _emitOutcomes
        = new();

    /// <summary>
    /// Records an emitter outcome — either "succeeded" or a specific
    /// decline reason. Exposed as aggregated counts via
    /// <see cref="GetEmitOutcomes"/>.
    /// </summary>
    public static void RecordEmitOutcome(CodegenTarget target, bool succeeded, string? declineReason = null)
    {
        if (!_enabled) return;
        string key = succeeded ? "Succeeded" : declineReason ?? "DeclinedUnspecified";
        _emitOutcomes.AddOrUpdate((target, key), 1, (_, c) => c + 1);
    }

    /// <summary>
    /// Returns a snapshot of emit-outcome counts, keyed by
    /// <c>(target, reason)</c>.
    /// </summary>
    public static IReadOnlyDictionary<(CodegenTarget Target, string Outcome), long> GetEmitOutcomes()
    {
        var result = new Dictionary<(CodegenTarget, string), long>();
        foreach (var kv in _emitOutcomes) result[kv.Key] = kv.Value;
        return result;
    }

    // ─── Autotune cache stats ────────────────────────────────────────

    private static long _autotuneHits;
    private static long _autotuneMisses;

    /// <summary>
    /// Records a cache hit. Zero-cost when disabled.
    /// </summary>
    public static void RecordAutotuneHit()
    {
        if (!_enabled) return;
        System.Threading.Interlocked.Increment(ref _autotuneHits);
    }

    /// <summary>
    /// Records a cache miss. Zero-cost when disabled.
    /// </summary>
    public static void RecordAutotuneMiss()
    {
        if (!_enabled) return;
        System.Threading.Interlocked.Increment(ref _autotuneMisses);
    }

    /// <summary>
    /// Returns the current autotune stats snapshot.
    /// </summary>
    public static AutotuneStats GetAutotuneStats()
        => new AutotuneStats(
            Hits: System.Threading.Interlocked.Read(ref _autotuneHits),
            Misses: System.Threading.Interlocked.Read(ref _autotuneMisses));

    // ─── Snapshot + reset ────────────────────────────────────────────

    /// <summary>
    /// Returns a composite snapshot — pass timings, emit outcomes,
    /// autotune stats, recompile log, compiled-kernel cache size —
    /// suitable for a single-shot dump at the end of a run.
    /// </summary>
    public static CodegenTelemetrySnapshot Snapshot()
        => new CodegenTelemetrySnapshot(
            PassTimings: GetPassTimings(),
            EmitOutcomes: GetEmitOutcomes(),
            Autotune: GetAutotuneStats(),
            RecompileLog: CodegenGuardRegistry.DumpRecompileLog(),
            CompiledKernelCacheSize: CodegenGuardRegistry.CacheEntryCount);

    /// <summary>
    /// Clears all aggregated counters. The enabled flag is left
    /// untouched — callers that want a fresh run on the same
    /// enabled state typically call this at the start.
    /// </summary>
    public static void Reset()
    {
        _passTimings.Clear();
        _emitOutcomes.Clear();
        System.Threading.Interlocked.Exchange(ref _autotuneHits, 0);
        System.Threading.Interlocked.Exchange(ref _autotuneMisses, 0);
    }
}

/// <summary>
/// Scope handle returned from <see cref="CodegenTelemetry.TimePass"/>.
/// Records the elapsed time on Dispose. Dispose on a disabled-
/// telemetry scope is a no-op.
/// </summary>
public readonly struct PassTimingScope : IDisposable
{
    private readonly string? _passName;
    private readonly long _startTicks;

    internal PassTimingScope(string passName, long startTicks)
    {
        _passName = passName;
        _startTicks = startTicks;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_passName is null) return; // Default instance — telemetry disabled.
        long elapsed = Stopwatch.GetTimestamp() - _startTicks;
        CodegenTelemetry.RecordPassTiming(_passName, elapsed);
    }
}

/// <summary>
/// Per-pass timing aggregate: call count + min / max / sum of
/// elapsed ticks. Snapshot returns immutable <see cref="PassTimingStats"/>.
/// </summary>
internal sealed class PassTimingAccumulator
{
    private long _count;
    private long _totalTicks;
    private long _minTicks = long.MaxValue;
    private long _maxTicks;
    private readonly object _lock = new();

    public void Record(long elapsedTicks)
    {
        lock (_lock)
        {
            _count++;
            _totalTicks += elapsedTicks;
            if (elapsedTicks < _minTicks) _minTicks = elapsedTicks;
            if (elapsedTicks > _maxTicks) _maxTicks = elapsedTicks;
        }
    }

    public PassTimingStats Snapshot()
    {
        lock (_lock)
        {
            return new PassTimingStats(
                CallCount: _count,
                TotalTicks: _totalTicks,
                MinTicks: _count == 0 ? 0 : _minTicks,
                MaxTicks: _maxTicks);
        }
    }
}

/// <summary>Immutable per-pass timing snapshot.</summary>
public readonly record struct PassTimingStats(
    long CallCount,
    long TotalTicks,
    long MinTicks,
    long MaxTicks)
{
    /// <summary>Mean ticks per call. Zero when no calls recorded.</summary>
    public long MeanTicks => CallCount == 0 ? 0 : TotalTicks / CallCount;

    /// <summary>Total elapsed time as a <see cref="TimeSpan"/>.</summary>
    public TimeSpan TotalElapsed => TimeSpan.FromTicks(
        (long)(TotalTicks * ((double)TimeSpan.TicksPerSecond / Stopwatch.Frequency)));
}

/// <summary>
/// Autotune hit/miss stats.
/// </summary>
public readonly record struct AutotuneStats(long Hits, long Misses)
{
    /// <summary>Total lookups (<see cref="Hits"/> + <see cref="Misses"/>).</summary>
    public long Total => Hits + Misses;

    /// <summary>Hit ratio — 0.0 when Total is 0.</summary>
    public double HitRatio => Total == 0 ? 0.0 : (double)Hits / Total;
}

/// <summary>
/// Composite snapshot of every codegen telemetry channel.
/// </summary>
public readonly record struct CodegenTelemetrySnapshot(
    IReadOnlyDictionary<string, PassTimingStats> PassTimings,
    IReadOnlyDictionary<(CodegenTarget Target, string Outcome), long> EmitOutcomes,
    AutotuneStats Autotune,
    IReadOnlyList<RecompileLogEntry> RecompileLog,
    int CompiledKernelCacheSize);
