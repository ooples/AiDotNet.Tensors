// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.Profiling.Trace;

/// <summary>
/// One entry in the profiler's chronological event log. Maps directly to the
/// <a href="https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview">
/// Trace Event Format</a> used by Chrome's <c>chrome://tracing</c> and
/// <a href="https://ui.perfetto.dev/">Perfetto UI</a>.
///
/// <para>
/// We use the <b>Complete Event</b> phase (<c>"ph": "X"</c>) for spans with
/// known start + duration, and <b>Instant Event</b> (<c>"ph": "i"</c>) for
/// point markers. This keeps the event log roughly half the size of Begin/End
/// (B/E) pairs and matches what PyTorch emits.
/// </para>
/// </summary>
public readonly struct TraceEvent
{
    /// <summary>Event display name.</summary>
    public string Name { get; }

    /// <summary>Event category — <c>"cpu_op"</c>, <c>"user_annotation"</c>,
    /// <c>"compile_pass"</c>, <c>"gpu_kernel"</c>, etc. Filtered in the trace UI.</summary>
    public string Category { get; }

    /// <summary>Phase. <c>'X'</c> = complete (start + dur), <c>'i'</c> = instant,
    /// <c>'M'</c> = metadata (process/thread name).</summary>
    public char Phase { get; }

    /// <summary>Start timestamp in microseconds since the session began.</summary>
    public long TimestampMicros { get; }

    /// <summary>Duration in microseconds. Only meaningful for Complete events.</summary>
    public long DurationMicros { get; }

    /// <summary>OS process id.</summary>
    public int ProcessId { get; }

    /// <summary>Thread id (managed thread id is fine for our purposes).</summary>
    public int ThreadId { get; }

    /// <summary>Optional structured arguments — small key/value pairs serialised
    /// into the event's <c>"args"</c> object. <c>null</c> writes no <c>args</c>.</summary>
    public System.Collections.Generic.IReadOnlyDictionary<string, string>? Args { get; }

    /// <summary>Constructs a Complete event (most common case).</summary>
    public static TraceEvent Complete(
        string name, string category, long startMicros, long durationMicros,
        int processId, int threadId,
        System.Collections.Generic.IReadOnlyDictionary<string, string>? args = null)
        => new(name, category, 'X', startMicros, durationMicros, processId, threadId, args);

    /// <summary>Constructs an Instant event (point marker, no duration).</summary>
    public static TraceEvent Instant(
        string name, string category, long timestampMicros,
        int processId, int threadId,
        System.Collections.Generic.IReadOnlyDictionary<string, string>? args = null)
        => new(name, category, 'i', timestampMicros, 0, processId, threadId, args);

    /// <summary>Constructs a Metadata event (process/thread name registration).</summary>
    public static TraceEvent Metadata(
        string name, int processId, int threadId,
        System.Collections.Generic.IReadOnlyDictionary<string, string> args)
        => new(name, "__metadata", 'M', 0, 0, processId, threadId, args);

    private TraceEvent(string name, string category, char phase,
        long timestampMicros, long durationMicros,
        int processId, int threadId,
        System.Collections.Generic.IReadOnlyDictionary<string, string>? args)
    {
        Name = name;
        Category = category;
        Phase = phase;
        TimestampMicros = timestampMicros;
        DurationMicros = durationMicros;
        ProcessId = processId;
        ThreadId = threadId;
        Args = args;
    }
}
