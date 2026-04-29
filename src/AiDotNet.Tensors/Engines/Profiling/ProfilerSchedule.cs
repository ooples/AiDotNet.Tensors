// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.Profiling;

/// <summary>
/// Step-driven phase schedule, modelled on <c>torch.profiler.schedule(wait,
/// warmup, active, repeat)</c>.
///
/// <para>
/// A trainer typically runs a loop of N iterations and calls
/// <see cref="ProfilerSession.Step"/> at the end of each. The schedule
/// classifies each step into one of four phases:
/// </para>
///
/// <list type="bullet">
///   <item><b>Wait</b>: profiler is idle (events not recorded). Lets the
///     workload reach steady state before measuring.</item>
///   <item><b>Warmup</b>: events are recorded but discarded at the end of the
///     warmup run. Gives JIT, autotune caches, and BLAS thread pools a chance
///     to warm without polluting the timeline.</item>
///   <item><b>Active</b>: events are recorded and retained — this is the
///     measured window.</item>
///   <item><b>Repeat</b>: how many wait→warmup→active cycles to run. After
///     the configured number of repeats the schedule returns
///     <see cref="ProfilerScheduleAction.Stop"/> and additional steps are
///     no-ops. <c>0</c> means "loop forever until the session is disposed."</item>
/// </list>
///
/// <para>
/// Total cycle length per repeat is <c>Wait + Warmup + Active</c> steps. With
/// <c>Wait=1</c>, <c>Warmup=1</c>, <c>Active=3</c>, <c>Repeat=2</c>, a
/// 10-step training loop spends steps 0 idle, 1 warming, 2-4 measured,
/// 5 idle, 6 warming, 7-9 measured.
/// </para>
/// </summary>
public sealed class ProfilerSchedule
{
    /// <summary>Idle steps at the start of each cycle.</summary>
    public int Wait { get; }

    /// <summary>Warmup steps after the wait — recorded but discarded.</summary>
    public int Warmup { get; }

    /// <summary>Active steps after warmup — events retained for export.</summary>
    public int Active { get; }

    /// <summary>Number of (Wait+Warmup+Active) cycles to run. <c>0</c> = forever.</summary>
    public int Repeat { get; }

    /// <summary>
    /// Constructs a schedule. All counts must be non-negative; the active
    /// window must be at least one step (a profile that records nothing
    /// would be silently useless).
    /// </summary>
    public ProfilerSchedule(int wait, int warmup, int active, int repeat)
    {
        if (wait < 0) throw new System.ArgumentOutOfRangeException(nameof(wait), "Schedule counts must be non-negative.");
        if (warmup < 0) throw new System.ArgumentOutOfRangeException(nameof(warmup), "Schedule counts must be non-negative.");
        if (active < 1) throw new System.ArgumentOutOfRangeException(nameof(active), "Schedule must have at least one Active step.");
        if (repeat < 0) throw new System.ArgumentOutOfRangeException(nameof(repeat), "Repeat must be non-negative; 0 = unbounded.");
        Wait = wait;
        Warmup = warmup;
        Active = active;
        Repeat = repeat;
    }

    /// <summary>
    /// Convenience: a never-stops schedule that records every step from the
    /// start. Equivalent to <c>new ProfilerSchedule(0, 0, int.MaxValue, 0)</c>.
    /// Useful when the caller wants a continuous trace and will dispose the
    /// session manually.
    /// </summary>
    public static ProfilerSchedule AlwaysActive { get; } = new(0, 0, int.MaxValue, 0);

    /// <summary>Maps a 0-based step index to its phase.</summary>
    public ProfilerSchedulePhase Classify(int stepIndex)
    {
        int cycle = Wait + Warmup + Active;
        if (cycle == 0) return ProfilerSchedulePhase.Stopped;

        if (Repeat > 0 && stepIndex >= cycle * Repeat)
            return ProfilerSchedulePhase.Stopped;

        int inCycle = stepIndex % cycle;
        if (inCycle < Wait) return ProfilerSchedulePhase.Wait;
        if (inCycle < Wait + Warmup) return ProfilerSchedulePhase.Warmup;
        return ProfilerSchedulePhase.Active;
    }

    /// <summary>
    /// Returns true when <paramref name="prevStep"/> ended in Active or Warmup
    /// and <paramref name="thisStep"/> begins a non-recording phase (Wait or
    /// Stopped). The session uses this edge to flush the active-window events
    /// and invoke <c>OnTraceReady</c>.
    /// </summary>
    internal bool IsTraceReadyEdge(int prevStep, int thisStep)
    {
        var prev = Classify(prevStep);
        var now = Classify(thisStep);
        bool wasRecording = prev == ProfilerSchedulePhase.Warmup || prev == ProfilerSchedulePhase.Active;
        bool nowQuiet = now == ProfilerSchedulePhase.Wait || now == ProfilerSchedulePhase.Stopped;
        return wasRecording && nowQuiet;
    }
}

/// <summary>Phase classification for a given step.</summary>
public enum ProfilerSchedulePhase
{
    /// <summary>Inside a Wait window — events not recorded.</summary>
    Wait,

    /// <summary>Inside a Warmup window — events recorded, discarded at edge.</summary>
    Warmup,

    /// <summary>Inside the measured window — events recorded and retained.</summary>
    Active,

    /// <summary>Schedule has completed all repeats. Further steps are no-ops.</summary>
    Stopped,
}

/// <summary>What the schedule's edge transition asks the session to do next.</summary>
public enum ProfilerScheduleAction
{
    /// <summary>Continue without flushing.</summary>
    Continue,

    /// <summary>Flush the active-window events and invoke <c>OnTraceReady</c>.</summary>
    Flush,

    /// <summary>Schedule complete — additional steps are no-ops.</summary>
    Stop,
}
