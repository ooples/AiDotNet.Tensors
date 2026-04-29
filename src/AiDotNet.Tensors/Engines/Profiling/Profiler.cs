// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace AiDotNet.Tensors.Engines.Profiling;

/// <summary>
/// Top-level facade for the AiDotNet.Tensors profiler — modelled on
/// <c>torch.profiler</c>. Designed so that:
///
/// <list type="bullet">
///   <item><b>Zero overhead when disabled</b>: <see cref="Range(string)"/> returns
///     a singleton no-op disposable when no session is active. The check is a
///     single volatile read of <see cref="Current"/>.</item>
///   <item><b>Single ambient session</b>: there is at most one
///     <see cref="ProfilerSession"/> per process. PyTorch behaves the same;
///     having two profilers fight over the same call stack is a footgun nobody
///     wants. Tests can swap sessions explicitly via
///     <see cref="Profile(ProfilerOptions)"/>.</item>
///   <item><b>Thread-safe event capture</b>: events are accumulated in a
///     <c>ConcurrentQueue</c> on the session, so worker threads spawned by
///     <see cref="System.Threading.Tasks.Parallel"/> emit into the same
///     timeline without locking.</item>
/// </list>
///
/// <para><b>Typical usage:</b></para>
/// <code>
/// using var prof = Profiler.Profile(new ProfilerOptions {
///     Activities = ProfilerActivities.Cpu,
///     Schedule   = new ProfilerSchedule(wait: 1, warmup: 1, active: 3, repeat: 2),
///     OnTraceReady = Profiler.ChromeTraceHandler("./traces"),
/// });
/// for (int step = 0; step &lt; 10; step++) {
///     using (Profiler.Range("step")) {
///         using (Profiler.Range("forward"))  TrainForward();
///         using (Profiler.Range("backward")) TrainBackward();
///     }
///     prof.Step();
/// }
/// </code>
/// </summary>
public static class Profiler
{
    // Single ambient session, exposed via a volatile field so reads from the
    // hot path (`Range`) don't need a lock. Replaces are guarded by a CAS
    // inside Profile() so two threads can't both think they got the slot.
    private static ProfilerSession? _current;

    /// <summary>The currently-active session, or <c>null</c> when none.
    /// <see cref="Range(string)"/> short-circuits to a no-op when this is
    /// <c>null</c>.</summary>
    public static ProfilerSession? Current => System.Threading.Volatile.Read(ref _current);

    /// <summary>
    /// Starts a profiling session with <paramref name="options"/>. Disposing
    /// the returned session ends profiling and fires the final
    /// <c>OnTraceReady</c> callback.
    ///
    /// <para>Throws <see cref="InvalidOperationException"/> when another session
    /// is already active — nested profilers are not supported (matches
    /// PyTorch's behaviour and avoids ambiguous timeline merging).</para>
    /// </summary>
    public static ProfilerSession Profile(ProfilerOptions? options = null)
    {
        var opts = options ?? new ProfilerOptions();
        var session = new ProfilerSession(opts);

        var prior = System.Threading.Interlocked.CompareExchange(ref _current, session, null);
        if (prior is not null)
        {
            throw new InvalidOperationException(
                "A profiler session is already active. Dispose the existing session before starting a new one.");
        }
        return session;
    }

    /// <summary>
    /// Internal hook for <see cref="ProfilerSession.Dispose"/> — atomically
    /// clears the ambient slot iff the session being disposed is the one
    /// currently registered. The CAS protects against two-session races where
    /// a stale dispose would clobber a newly-started session.
    /// </summary>
    internal static void ClearCurrent(ProfilerSession expected)
    {
        System.Threading.Interlocked.CompareExchange(ref _current, null, expected);
    }

    /// <summary>
    /// Opens a user-annotated range. The returned scope, on dispose, emits a
    /// Complete event spanning the range's lifetime. When no profiler is
    /// active, the singleton <see cref="NoOpScope"/> is returned — zero-alloc,
    /// zero-overhead.
    /// </summary>
    public static IDisposable Range(string name)
        => Range(name, category: "user_annotation");

    /// <summary>
    /// Opens a categorized range. <paramref name="category"/> appears in
    /// chrome-trace UIs as the event's tag — typical values are
    /// <c>"cpu_op"</c>, <c>"compile_pass"</c>, <c>"autograd"</c>.
    /// </summary>
    public static IDisposable Range(string name, string category)
    {
        var session = Current;
        if (session is null) return NoOpScope.Instance;
        return new RangeScope(session, name, category, args: null);
    }

    /// <summary>
    /// PyTorch-compatible alias for <see cref="Range(string)"/> — semantically
    /// identical, kept for users porting from <c>torch.profiler.record_function</c>.
    /// </summary>
    public static IDisposable RecordFunction(string name) => Range(name);

    /// <summary>
    /// Opens a range with structured arguments — emitted as the event's
    /// <c>args</c> object in the chrome-trace JSON. Useful for shape /
    /// dtype / dispatch-tier annotations.
    /// </summary>
    public static IDisposable Range(string name, string category, IReadOnlyDictionary<string, string> args)
    {
        var session = Current;
        if (session is null) return NoOpScope.Instance;
        return new RangeScope(session, name, category, args);
    }

    /// <summary>Records an instantaneous event on the active session.
    /// No-op when no session is active.</summary>
    public static void RecordInstant(string name, string category = "instant", IReadOnlyDictionary<string, string>? args = null)
    {
        Current?.RecordInstant(name, category, args);
    }

    /// <summary>
    /// Convenience handler factory: returns a delegate that writes each
    /// active-window flush to a timestamped file under
    /// <paramref name="directory"/>. Equivalent to PyTorch's
    /// <c>tensorboard_trace_handler</c>.
    /// </summary>
    public static Action<ProfilerSession> ChromeTraceHandler(string directory)
    {
        if (string.IsNullOrEmpty(directory))
            throw new ArgumentException("Directory is required.", nameof(directory));
        Directory.CreateDirectory(directory);

        return session =>
        {
            // Millisecond-stamped filenames collide when two flushes land in the same
            // millisecond (back-to-back schedule windows on a fast machine), and File.Create
            // overwrites silently. Append a Guid suffix so each trace lands in its own file.
            string stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff", System.Globalization.CultureInfo.InvariantCulture);
            string path = Path.Combine(directory, $"trace-{stamp}-{Guid.NewGuid():N}.json");
            session.ExportChromeTrace(path);
        };
    }

    private sealed class RangeScope : IDisposable
    {
        private readonly ProfilerSession _session;
        private readonly string _name;
        private readonly string _category;
        private readonly IReadOnlyDictionary<string, string>? _args;
        private readonly long _startTicks;
        private bool _disposed;

        public RangeScope(ProfilerSession session, string name, string category, IReadOnlyDictionary<string, string>? args)
        {
            _session = session;
            _name = name;
            _category = category;
            _args = args;
            _startTicks = Stopwatch.GetTimestamp();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            long endTicks = Stopwatch.GetTimestamp();
            _session.RecordCompleteInternal(_name, _category, _startTicks, endTicks, _args);
        }
    }

    private sealed class NoOpScope : IDisposable
    {
        public static readonly NoOpScope Instance = new();
        public void Dispose() { }
    }
}
