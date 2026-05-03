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
    private ConcurrentQueue<TraceEvent> _events = new();
    // Metadata events (process_name, thread_name) emitted at construction time. We replay
    // them onto every fresh buffer after a flush/reset so subsequent exports remain
    // self-describing — Reset() previously dropped these silently.
    private readonly List<TraceEvent> _metadataPrologue = new();
    // Serializes buffer swaps against concurrent enqueues so the event counter and the
    // queue identity stay in sync — without it, a producer thread could increment the
    // counter, then the swapper resets it, then the producer dequeues from the wrong
    // queue when it tries to evict at capacity, breaking retention accounting.
    private readonly object _bufferGate = new();
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
        var processName = TraceEvent.Metadata("process_name", _processId, threadId: 0,
            args: new Dictionary<string, string> { ["name"] = "AiDotNet.Tensors" });
        _metadataPrologue.Add(processName);
        EnqueueEvent(processName);
    }

    /// <summary>Configuration this session was started with.</summary>
    public ProfilerOptions Options => _options;

    /// <summary>Snapshot of events recorded so far. Thread-safe — the returned
    /// array is a copy at call time. If accessed from inside an
    /// <see cref="ProfilerOptions.OnTraceReady"/> handler, returns just the events from the
    /// closed window (the buffer that was swapped out before the handler fired).</summary>
    public IReadOnlyList<TraceEvent> Events
    {
        get
        {
            var snap = _flushSnapshot;
            return snap != null ? snap.ToArray() : _events.ToArray();
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
        int prev;
        int next;
        // Update _stepIndex under _bufferGate so RecordInstant's phase
        // check (also under the lock) cannot observe a torn or stale
        // value during the transition. Without this serialisation the
        // accept-bool returned by RecordInstant could disagree with
        // what actually landed in the queue.
        lock (_bufferGate)
        {
            prev = _stepIndex;
            next = prev + 1;
            _stepIndex = next;
        }

        if (_schedule is null) return;
        // Drop the warmup-window samples before the measured window opens. Without
        // this reset, the snapshot flushed at the next trace-ready edge would be
        // contaminated with warmup events even though the schedule documents
        // "warmup = recorded, discarded".
        var prevPhase = _schedule.Classify(prev);
        var nextPhase = _schedule.Classify(next);
        if (prevPhase == ProfilerSchedulePhase.Warmup && nextPhase == ProfilerSchedulePhase.Active)
        {
            Reset();
        }
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
    /// <summary>True iff CPU activity capture is enabled by the session's options.
    /// Honours <see cref="ProfilerOptions.Activities"/>: a session configured with
    /// <c>ProfilerActivities.None</c> or GPU/memory-only flags must not record CPU
    /// ranges or instants.</summary>
    private bool IsCpuCaptureEnabled() => (_options.Activities & ProfilerActivities.Cpu) != 0;

    internal void RecordCompleteInternal(string name, string category, long startTicks, long endTicks, IReadOnlyDictionary<string, string>? args)
    {
        if (_disposed) return;
        if (!IsCpuCaptureEnabled()) return;
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
    /// Returns <c>true</c> when the event was accepted and enqueued;
    /// <c>false</c> when it was dropped because the session is disposed,
    /// CPU capture is disabled (<see cref="ProfilerActivities.None"/>),
    /// or the schedule phase is <see cref="ProfilerSchedulePhase.Wait"/>
    /// or <see cref="ProfilerSchedulePhase.Stopped"/>. Callers that
    /// dedupe against an outer fallback marker need this distinction
    /// so a no-op record doesn't suppress the outer event.
    ///
    /// <para><b>Atomicity:</b> the schedule-phase check and the queue
    /// enqueue happen inside the same <c>_bufferGate</c> lock as
    /// <see cref="Step"/>'s <c>_stepIndex</c> update. A concurrent
    /// <see cref="Step"/> cannot flip the phase between this method's
    /// check and its enqueue, so the returned bool is consistent with
    /// what actually landed in the queue.</para>
    /// </summary>
    public bool RecordInstant(string name, string category, IReadOnlyDictionary<string, string>? args = null)
    {
        if (_disposed) return false;
        if (!IsCpuCaptureEnabled()) return false;

        // Build the event timestamp/payload outside the lock to minimise
        // lock-hold time. Phase classification and enqueue happen under
        // _bufferGate so a concurrent Step() can't desynchronise them.
        long ts = TicksToMicros(Stopwatch.GetTimestamp() - _sessionStartTicks);
        int threadId = System.Environment.CurrentManagedThreadId;
        TraceEvent ev = TraceEvent.Instant(name, category, ts, _processId, threadId, args);

        lock (_bufferGate)
        {
            // Re-read disposal under the lock — Dispose may have raced.
            if (_disposed) return false;
            var phase = CurrentPhase;
            if (phase != ProfilerSchedulePhase.Active && phase != ProfilerSchedulePhase.Warmup) return false;

            // Inline the enqueue body so we don't re-enter the lock via
            // EnqueueEvent (re-entrant locks work but inlining makes the
            // accept/drop transaction unambiguous in one critical section).
            if (++_eventCount > _maxEvents)
            {
                if (_events.TryDequeue(out _)) _eventCount--;
                else _eventCount--;
            }
            _events.Enqueue(ev);
            return true;
        }
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
    /// when running in always-active mode. Atomic-swaps the live buffer for a
    /// fresh one (preserving the metadata prologue so future exports remain
    /// self-describing) so concurrent producers never have to coordinate with
    /// the reset.</summary>
    public void Reset()
    {
        SwapBuffer();
    }

    /// <summary>Atomically replaces <see cref="_events"/> with a fresh queue seeded
    /// with the metadata prologue, returning the old queue to the caller. Producers
    /// that already obtained a reference to the old queue will keep enqueueing into
    /// it harmlessly; their events are still observable on the snapshot returned to
    /// the caller. <see cref="_eventCount"/> is reset to the prologue count so the
    /// counter stays in sync with the new live queue.</summary>
    private ConcurrentQueue<TraceEvent> SwapBuffer()
    {
        // Take the gate so a concurrent enqueue can't observe a partially-rotated state
        // (e.g. counter reset but queue not yet swapped, or vice versa).
        lock (_bufferGate)
        {
            var fresh = new ConcurrentQueue<TraceEvent>();
            foreach (var ev in _metadataPrologue) fresh.Enqueue(ev);
            var old = _events;
            _events = fresh;
            _eventCount = _metadataPrologue.Count;
            return old;
        }
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

    /// <summary>Snapshot exposed to the OnTraceReady handler during a flush so it can
    /// observe exactly the events that belonged to the active window — even if other
    /// threads are still appending into the new live buffer.</summary>
    private ConcurrentQueue<TraceEvent>? _flushSnapshot;

    private void FlushTraceReady()
    {
        var handler = _options.OnTraceReady;
        // Atomically swap to a fresh buffer FIRST so producers immediately move on to
        // the next window. The drained snapshot is what we hand to the handler.
        var snapshot = SwapBuffer();
        if (handler != null)
        {
            _flushSnapshot = snapshot;
            try { handler.Invoke(this); }
            finally { _flushSnapshot = null; }
        }
    }

    private IReadOnlyList<TraceEvent> FilterRetainedEvents()
    {
        // During a flush the handler should see ONLY the events from the just-closed
        // window; outside a flush we materialise the live queue.
        var snap = _flushSnapshot;
        return snap != null ? snap.ToArray() : _events.ToArray();
    }

    private void EnqueueEvent(TraceEvent ev)
    {
        // Bounded queue: if we're at capacity, drop the oldest event so a
        // long-running profile can't OOM. The drop is silent — the alternative
        // would be to throw, which is hostile during a profiling run that the
        // user might not be able to easily restart.
        //
        // Serialise with SwapBuffer via _bufferGate so the increment, the eviction-from-
        // queue, and the queue-identity are always consistent: without the lock, a
        // SwapBuffer running between Increment and TryDequeue would evict from the WRONG
        // queue and corrupt the retention accounting.
        lock (_bufferGate)
        {
            if (++_eventCount > _maxEvents)
            {
                if (_events.TryDequeue(out _)) _eventCount--;
                else _eventCount--;
            }
            _events.Enqueue(ev);
        }
    }

    private long TicksToMicros(long ticks)
    {
        // Stopwatch.Frequency is ticks-per-second; convert to microseconds without
        // floating-point so the trace ts/dur values are byte-stable across runs.
        // Direct (ticks * 1e6) overflows long after ~2.5h on 1 GHz frequencies, so
        // split the conversion into whole-seconds plus sub-second remainder. Both
        // intermediate products stay in range (remainder < _ticksPerSecond).
        long wholeSeconds = ticks / _ticksPerSecond;
        long remainder = ticks % _ticksPerSecond;
        return (wholeSeconds * 1_000_000L) + ((remainder * 1_000_000L) / _ticksPerSecond);
    }
}
