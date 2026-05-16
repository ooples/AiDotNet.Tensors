using System;
using System.Diagnostics;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Issue #338 Phase F: forward/backward wall-time split for
/// <see cref="CompiledTrainingPlan{T}.Step"/>.
///
/// <para>Gated by env var <c>AIDOTNET_STEP_TIMING=1</c> at process start;
/// when off the timing wrappers compile to a single branch+zero, no
/// per-iter Stopwatch reads. Mirrors <c>BackwardTiming</c> (Phase A) so
/// callers can wire either / both as needed.</para>
///
/// <para>Recorded buckets: <c>"forward"</c>, <c>"backward"</c>,
/// <c>"optimizer"</c>. <see cref="DumpAndReset(Action{string})"/> emits
/// totals and percentages so Phase F can target the dominant share.</para>
/// </summary>
internal static class StepTiming
{
    private static readonly bool _enabled =
        Environment.GetEnvironmentVariable("AIDOTNET_STEP_TIMING") == "1";

    [ThreadStatic]
    private static long _forwardTicks;
    [ThreadStatic]
    private static long _backwardTicks;
    [ThreadStatic]
    private static long _optimizerTicks;
    [ThreadStatic]
    private static int _stepCount;

    internal static bool Enabled => _enabled;

    internal static void RecordForward(long ticks)
    {
        if (!_enabled) return;
        _forwardTicks += ticks;
    }

    internal static void RecordBackward(long ticks)
    {
        if (!_enabled) return;
        _backwardTicks += ticks;
    }

    internal static void RecordOptimizer(long ticks)
    {
        if (!_enabled) return;
        _optimizerTicks += ticks;
    }

    internal static void IncrementStepCount()
    {
        if (!_enabled) return;
        _stepCount++;
    }

    /// <summary>
    /// Emits the forward/backward/optimizer split via the caller-supplied
    /// writer, then resets the aggregator. Percentages are relative to
    /// total recorded ticks (forward + backward + optimizer); the
    /// remainder vs Step() wall-time (gradient-clear, loss re-seed,
    /// dispatch overhead) is uncategorized and shows up as the gap when
    /// the caller diffs against their own Stopwatch.
    /// </summary>
    internal static void DumpAndReset(Action<string> writer)
    {
        if (!_enabled) return;
        if (_stepCount == 0)
        {
            _forwardTicks = 0;
            _backwardTicks = 0;
            _optimizerTicks = 0;
            return;
        }

        long totalTicks = _forwardTicks + _backwardTicks + _optimizerTicks;
        double freqMs = 1000.0 / Stopwatch.Frequency;
        double fwdMs = _forwardTicks * freqMs;
        double bwdMs = _backwardTicks * freqMs;
        double optMs = _optimizerTicks * freqMs;
        double totMs = totalTicks * freqMs;

        double fwdPct = totalTicks > 0 ? 100.0 * _forwardTicks / totalTicks : 0.0;
        double bwdPct = totalTicks > 0 ? 100.0 * _backwardTicks / totalTicks : 0.0;
        double optPct = totalTicks > 0 ? 100.0 * _optimizerTicks / totalTicks : 0.0;

        writer($"# Phase F step-timing breakdown ({_stepCount} iters)");
        writer($"# forward_ms_per_iter={fwdMs / _stepCount:F3} ({fwdPct:F1}%)");
        writer($"# backward_ms_per_iter={bwdMs / _stepCount:F3} ({bwdPct:F1}%)");
        writer($"# optimizer_ms_per_iter={optMs / _stepCount:F3} ({optPct:F1}%)");
        writer($"# total_recorded_ms_per_iter={totMs / _stepCount:F3}");

        _forwardTicks = 0;
        _backwardTicks = 0;
        _optimizerTicks = 0;
        _stepCount = 0;
    }
}
