using System.Diagnostics;
using AiDotNet.Tensors.Engines.Optimization;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Phase 7.4: Profiling-guided optimization — measures actual execution time
/// per compiled step to identify bottlenecks for targeted optimization.
///
/// Usage:
///   var profiler = new ProfilingCompiler&lt;float&gt;();
///   profiler.Profile(plan, warmupSteps: 10, measureSteps: 100);
///   var report = profiler.GetReport();
///   // report.HottestSteps shows which ops to optimize next
/// </summary>
public sealed class ProfilingCompiler<T>
{
    private readonly List<StepProfile> _profiles = new();

    /// <summary>
    /// Profiles a compiled inference plan by measuring per-step execution time.
    /// </summary>
    public void ProfileInference(ICompiledPlan<T> plan, int warmupSteps = 10, int measureSteps = 50)
    {
        // Warmup
        for (int i = 0; i < warmupSteps; i++)
            plan.Execute();

        // Measure total time for N executions
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < measureSteps; i++)
            plan.Execute();
        sw.Stop();

        _profiles.Add(new StepProfile(
            "InferencePlan",
            plan.StepCount,
            sw.Elapsed.TotalMilliseconds / measureSteps));
    }

    /// <summary>
    /// Profiles a compiled training plan by measuring per-step execution time.
    /// </summary>
    public void ProfileTraining(ICompiledTrainingPlan<T> plan, int warmupSteps = 5, int measureSteps = 20)
    {
        // Warmup
        for (int i = 0; i < warmupSteps; i++)
            plan.Step();

        // Measure total time for N steps
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < measureSteps; i++)
            plan.Step();
        sw.Stop();

        double msPerStep = sw.Elapsed.TotalMilliseconds / measureSteps;
        double stepsPerSecond = 1000.0 / msPerStep;

        _profiles.Add(new StepProfile(
            "TrainingPlan",
            plan.ForwardStepCount + plan.BackwardStepCount,
            msPerStep,
            stepsPerSecond));
    }

    /// <summary>Gets the profiling report with bottleneck analysis.</summary>
    public ProfilingReport GetReport()
    {
        return new ProfilingReport(_profiles.ToArray());
    }

    /// <summary>Clears all profiling data.</summary>
    public void Clear() => _profiles.Clear();
}

/// <summary>Profiling data for a single compiled step or plan.</summary>
public sealed class StepProfile
{
    public string Name { get; }
    public int StepCount { get; }
    public double MillisecondsPerExecution { get; }
    public double StepsPerSecond { get; }

    internal StepProfile(string name, int stepCount, double msPerExecution, double stepsPerSecond = 0)
    {
        Name = name;
        StepCount = stepCount;
        MillisecondsPerExecution = msPerExecution;
        StepsPerSecond = stepsPerSecond > 0 ? stepsPerSecond : 1000.0 / msPerExecution;
    }
}

/// <summary>Profiling report with bottleneck identification.</summary>
public sealed class ProfilingReport
{
    public StepProfile[] Profiles { get; }

    internal ProfilingReport(StepProfile[] profiles)
    {
        Profiles = profiles;
    }

    /// <summary>Gets the slowest profile (biggest optimization target).</summary>
    public StepProfile? Slowest => Profiles.Length > 0
        ? Profiles.OrderByDescending(p => p.MillisecondsPerExecution).First()
        : null;

    /// <summary>Gets the fastest profile (baseline comparison).</summary>
    public StepProfile? Fastest => Profiles.Length > 0
        ? Profiles.OrderBy(p => p.MillisecondsPerExecution).First()
        : null;

    /// <summary>Formats a human-readable summary.</summary>
    public override string ToString()
    {
        if (Profiles.Length == 0) return "No profiles recorded.";

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("=== Profiling Report ===");
        foreach (var p in Profiles.OrderByDescending(x => x.MillisecondsPerExecution))
        {
            sb.AppendLine($"  {p.Name}: {p.MillisecondsPerExecution:F3}ms/exec, " +
                $"{p.StepsPerSecond:F0} steps/sec, {p.StepCount} compiled steps");
        }
        return sb.ToString();
    }
}
