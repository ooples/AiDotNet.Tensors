// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using AiDotNet.Tensors.Engines.Profiling.Trace;

namespace AiDotNet.Tensors.Engines.Profiling;

/// <summary>
/// Active profiling session — accumulates <see cref="TraceEvent"/> from every
/// thread and emits them on flush. One per <c>using var prof = Profiler.Profile(...)</c>
/// block; multiple concurrent sessions on the same thread are not supported
/// (the <see cref="Profiler.Current"/> facade is single-slot).
/// </summary>
public sealed class ProfilerSession : IDisposable
{
    private readonly ProfilerOptions _options;
    private readonly ProfilerSchedule? _schedule;
    private readonly long _sessionStartTicks;
    private readonly long _ticksPerSecond;
    private readonly ConcurrentQueue<TraceEvent> _events = new();
    private readonly int _maxEvents;
    private readonly int _processId;

    // Starts at 0 — we're "inside" step 0 from the moment the session opens.
    // Step() advances to the next step at the end of the current iteration,
    // matching the conventional "do work; profiler.Step()" loop shape.
    private int _stepIndex = 0;
    private int _eventCount;
    private bool _disposed;

    internal ProfilerSession(ProfilerOptions options)
    {
        _options = options;
        _schedule = options.Schedule;
        _sessionStartTicks = Stopwatch.GetTimestamp();
        _ticksPerSecond = Stopwatch.Frequency;
        _maxEvents = options.MaxEvents > 0 ? options.MaxEvents : 1_000_000;
        _processId = options.ProcessId ?? Process.GetCurrentProcess().Id;

        // Emit a metadata event so chrome-trace UIs label the process and the
        // initial thread. Per-thread metadata is added lazily when each
        // worker thread emits its first event (see RecordCompleteInternal).
        EnqueueEvent(TraceEvent.Metadata("process_name", _processId, threadId: 0,
            args: new Dictionary<string, string> { ["name"] = "AiDotNet.Tensors" }));
    }

    /// <summary>Configuration this session was started with.</summary>
    public ProfilerOptions Options => _options;

    /// <summary>Snapshot of events recorded so far. Thread-safe — the returned
    /// array is a copy at call time.</summary>
    public IReadOnlyList<TraceEvent> Events
    {
        get
        {
            // CopyTo gives a stable snapshot. Concurrent enqueues after the
            // copy go to the next snapshot — same semantics as PyTorch.
            var arr = _events.ToArray();
            return arr;
        }
    }

    /// <summary>Number of events held in memory right now.</summary>
    public int EventCount => System.Threading.Volatile.Read(ref _eventCount);

    /// <summary>Schedule phase the session is currently in. <see cref="ProfilerSchedulePhase.Active"/>
    /// when no schedule is configured (always-active mode).</summary>
    public ProfilerSchedulePhase CurrentPhase
        => _schedule?.Classify(_stepIndex) ?? ProfilerSchedulePhase.Active;

    /// <summary>
    /// Advances the step index and runs the schedule's edge-detection. When
    /// the schedule transitions out of an Active window, fires
    /// <see cref="ProfilerOptions.OnTraceReady"/> with the active-window
    /// events, then drops them so the next window starts clean.
    /// </summary>
    public void Step()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ProfilerSession));
        int prev = _stepIndex;
        int next = prev + 1;
        _stepIndex = next;

        if (_schedule is null) return;
        if (_schedule.IsTraceReadyEdge(prev, next))
        {
            FlushTraceReady();
        }
    }

    /// <summary>
    /// Records a Complete event with explicit timing. Engine-internal callers
    /// use this; user code should prefer <see cref="Profiler.Range(string)"/>
    /// which auto-times via a using-scope.
    /// </summary>
    internal void RecordCompleteInternal(string name, string category, long startTicks, long endTicks, IReadOnlyDictionary<string, string>? args)
    {
        if (_disposed) return;
        var phase = CurrentPhase;
        if (phase != ProfilerSchedulePhase.Active && phase != ProfilerSchedulePhase.Warmup) return;

        long startMicros = TicksToMicros(startTicks - _sessionStartTicks);
        long durationMicros = TicksToMicros(endTicks - startTicks);
        if (durationMicros < 0) durationMicros = 0;

        EnqueueEvent(TraceEvent.Complete(
            name, category, startMicros, durationMicros,
            _processId, System.Environment.CurrentManagedThreadId,
            args));
    }

    /// <summary>
    /// Records an instantaneous event (no duration). Useful for one-shot
    /// markers like "autotune cache miss" or "recompile triggered".
    /// </summary>
    public void RecordInstant(string name, string category, IReadOnlyDictionary<string, string>? args = null)
    {
        if (_disposed) return;
        var phase = CurrentPhase;
        if (phase != ProfilerSchedulePhase.Active && phase != ProfilerSchedulePhase.Warmup) return;

        long ts = TicksToMicros(Stopwatch.GetTimestamp() - _sessionStartTicks);
        EnqueueEvent(TraceEvent.Instant(
            name, category, ts,
            _processId, System.Environment.CurrentManagedThreadId,
            args));
    }

    /// <summary>
    /// Writes the current event log to <paramref name="path"/> as Chrome Trace
    /// Format JSON (gzipped if the path ends in <c>.gz</c>). The session keeps
    /// its events; call <see cref="Reset"/> if you want to clear after export.
    /// </summary>
    public void ExportChromeTrace(string path)
    {
        ChromeTraceWriter.Write(path, FilterRetainedEvents());
    }

    /// <summary>Writes the current event log to <paramref name="stream"/>.</summary>
    public void ExportChromeTrace(Stream stream)
    {
        ChromeTraceWriter.Write(stream, FilterRetainedEvents());
    }

    /// <summary>Drops every retained event. Useful between explicit phases
    /// when running in always-active mode.</summary>
    public void Reset()
    {
        while (_events.TryDequeue(out _)) { }
        System.Threading.Volatile.Write(ref _eventCount, 0);
    }

    /// <summary>Flushes the trace to the registered handler one last time
    /// then unhooks the session from <see cref="Profiler.Current"/>.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try
        {
            // Always fire OnTraceReady on dispose so the user gets the final
            // window. PyTorch does the same — schedule-driven flushes plus a
            // final flush on exit.
            FlushTraceReady();
        }
        finally
        {
            Profiler.ClearCurrent(this);
        }
    }

    private void FlushTraceReady()
    {
        var handler = _options.OnTraceReady;
        handler?.Invoke(this);
        // After the handler runs (it typically writes the trace to disk),
        // drop the events so the next window starts clean. This matches
        // PyTorch's behaviour: `on_trace_ready` consumes the active window.
        Reset();
    }

    private IReadOnlyList<TraceEvent> FilterRetainedEvents()
    {
        // The ConcurrentQueue preserves enqueue order so the snapshot is
        // already chronological. Just materialize.
        return _events.ToArray();
    }

    private void EnqueueEvent(TraceEvent ev)
    {
        // Bounded queue: if we're at capacity, drop the oldest event so a
        // long-running profile can't OOM. The drop is silent — the alternative
        // would be to throw, which is hostile during a profiling run that the
        // user might not be able to easily restart.
        if (System.Threading.Interlocked.Increment(ref _eventCount) > _maxEvents)
        {
            if (_events.TryDequeue(out _))
                System.Threading.Interlocked.Decrement(ref _eventCount);
            else
                System.Threading.Interlocked.Decrement(ref _eventCount);
        }
        _events.Enqueue(ev);
    }

    private long TicksToMicros(long ticks)
    {
        // Stopwatch.Frequency is ticks-per-second; convert to microseconds
        // with a single multiply + divide. Avoid floating-point so the trace
        // ts/dur values are byte-stable across runs.
        return (ticks * 1_000_000L) / _ticksPerSecond;
    }
}
