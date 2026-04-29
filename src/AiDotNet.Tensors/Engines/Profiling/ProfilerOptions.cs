// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.Profiling;

/// <summary>
/// Configuration for a <see cref="ProfilerSession"/>. Mirrors the
/// <c>torch.profiler.profile(activities, schedule, on_trace_ready)</c>
/// constructor surface so the mental model carries over from PyTorch.
/// </summary>
public sealed class ProfilerOptions
{
    /// <summary>Which event sources to capture. Defaults to CPU only.</summary>
    public ProfilerActivities Activities { get; set; } = ProfilerActivities.Cpu;

    /// <summary>
    /// Step schedule. <c>null</c> = always-active (every event from session
    /// start to dispose is retained). Use <see cref="ProfilerSchedule"/> for
    /// wait/warmup/active/repeat windows.
    /// </summary>
    public ProfilerSchedule? Schedule { get; set; }

    /// <summary>
    /// Callback fired at the end of each Active window (see
    /// <see cref="ProfilerSchedule"/>) and once more on dispose. The callback
    /// receives the session and can call
    /// <see cref="ProfilerSession.ExportChromeTrace(string)"/> or read
    /// <see cref="ProfilerSession.Events"/> directly.
    ///
    /// <para>Idiomatic helper: pass
    /// <see cref="Profiler.ChromeTraceHandler(string)"/> for a path-stamped
    /// auto-export.</para>
    /// </summary>
    public System.Action<ProfilerSession>? OnTraceReady { get; set; }

    /// <summary>
    /// Process ID stamped on every emitted Chrome-trace event. Defaults to the
    /// current OS process id; tests override this so generated traces compare
    /// byte-for-byte across runs.
    /// </summary>
    public int? ProcessId { get; set; }

    /// <summary>
    /// Maximum number of events the session retains in memory. When the
    /// active window holds more than this, the oldest events are dropped to
    /// keep the session bounded — protects long-running profiles from OOM.
    /// Default <c>1_000_000</c> events ≈ 200 MB at typical event sizes.
    /// </summary>
    public int MaxEvents { get; set; } = 1_000_000;
}
