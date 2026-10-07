// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Per-plan census of a compiled training step: how many forward and backward actions one step replays (the
/// compiled plan's equivalent of a kernel-launch count) and what each costs, aggregated by op name.
/// </summary>
/// <remarks>
/// <para>Opt-in with <c>AIDOTNET_PLAN_CENSUS=1</c>; the report goes to <c>AIDOTNET_PLAN_CENSUS_FILE</c>, else
/// <c>%TEMP%/aidotnet_plan_census.txt</c>. When it is off the plan holds no census object and its step loops
/// are the plain ones, so the cost is one null check per step.</para>
/// <para>Each plan writes one report, after <see cref="ReportAfterSteps"/> timed steps (the first
/// <see cref="WarmupSteps"/> steps are excluded: they include JIT and first-touch costs).</para>
/// </remarks>
internal sealed class PlanCensus
{
    internal static readonly bool Enabled =
        Environment.GetEnvironmentVariable("AIDOTNET_PLAN_CENSUS") == "1";

    internal const int WarmupSteps = 3;
    internal const int ReportAfterSteps = 20;

    private static int s_planOrdinal;

    private readonly string[] _forwardNames;
    private readonly string[] _backwardNames;
    private readonly long[] _forwardTicks;
    private readonly long[] _backwardTicks;
    private readonly int _forwardStepCount;
    private long _gradZeroTicks, _optimizerTicks, _stepTicks;
    private int _steps;
    private bool _reported;

    internal PlanCensus(string[] forwardNames, string[] backwardNames, int forwardStepCount)
    {
        _forwardNames = forwardNames;
        _backwardNames = backwardNames;
        _forwardTicks = new long[forwardNames.Length];
        _backwardTicks = new long[backwardNames.Length];
        _forwardStepCount = forwardStepCount;
    }

    internal int ForwardActionCount => _forwardNames.Length;

    internal int BackwardActionCount => _backwardNames.Length;

    /// <summary>True while the current step's timings are being collected (warm, not yet reported).</summary>
    internal bool Collecting => !_reported && _steps >= WarmupSteps;

    internal void AddForward(int action, long ticks) { if (Collecting) _forwardTicks[action] += ticks; }

    internal void AddBackward(int action, long ticks) { if (Collecting) _backwardTicks[action] += ticks; }

    internal void AddGradZero(long ticks) { if (Collecting) _gradZeroTicks += ticks; }

    internal void AddOptimizer(long ticks) { if (Collecting) _optimizerTicks += ticks; }

    /// <summary>Closes one step; writes the report once enough warm steps have been timed.</summary>
    internal void EndStep(long stepTicks)
    {
        if (_reported) return;
        if (Collecting) _stepTicks += stepTicks;
        _steps++;
        if (_steps < WarmupSteps + ReportAfterSteps) return;
        _reported = true;
        try
        {
            var path = Environment.GetEnvironmentVariable("AIDOTNET_PLAN_CENSUS_FILE");
            if (string.IsNullOrEmpty(path))
                path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet_plan_census.txt");
            System.IO.File.AppendAllText(path, BuildReport());
        }
        catch
        {
            // Diagnostic only: a failed write must never break a training step.
        }
    }

    internal string BuildReport()
    {
        double usPerTick = 1_000_000.0 / Stopwatch.Frequency;
        int n = ReportAfterSteps;
        var sb = new StringBuilder();
        int ordinal = System.Threading.Interlocked.Increment(ref s_planOrdinal);
        long fwdTotal = 0, bwdTotal = 0;
        foreach (var t in _forwardTicks) fwdTotal += t;
        foreach (var t in _backwardTicks) bwdTotal += t;
        sb.AppendLine($"[plan-census] plan #{ordinal}: forward steps={_forwardStepCount} forward actions={_forwardNames.Length} "
            + $"backward actions={_backwardNames.Length} per-step actions={_forwardNames.Length + _backwardNames.Length}");
        sb.AppendLine($"  us/step: total={_stepTicks * usPerTick / n:F1} forward={fwdTotal * usPerTick / n:F1} "
            + $"gradZero={_gradZeroTicks * usPerTick / n:F1} backward={bwdTotal * usPerTick / n:F1} "
            + $"optimizer={_optimizerTicks * usPerTick / n:F1}");
        AppendByName(sb, "forward", _forwardNames, _forwardTicks, usPerTick, n);
        AppendByName(sb, "backward", _backwardNames, _backwardTicks, usPerTick, n);
        return sb.ToString();
    }

    private static void AppendByName(StringBuilder sb, string phase, string[] names, long[] ticks, double usPerTick, int n)
    {
        var count = new Dictionary<string, int>(StringComparer.Ordinal);
        var cost = new Dictionary<string, long>(StringComparer.Ordinal);
        for (int i = 0; i < names.Length; i++)
        {
            count[names[i]] = count.TryGetValue(names[i], out int c) ? c + 1 : 1;
            cost[names[i]] = (cost.TryGetValue(names[i], out long s) ? s : 0) + ticks[i];
        }
        var ordered = new List<string>(count.Keys);
        ordered.Sort((a, b) => cost[b].CompareTo(cost[a]));
        sb.AppendLine($"  {phase} by op (count, us/step):");
        foreach (var name in ordered)
            sb.AppendLine($"    {count[name],5}  {cost[name] * usPerTick / n,9:F1}  {name}");
    }
}
