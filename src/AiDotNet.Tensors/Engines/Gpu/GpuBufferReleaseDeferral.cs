using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Thread-local gate that defers GPU buffer releases while a deferred-execution scope is
/// recording/executing (#642). A <see cref="DeferredScope"/> records an op graph against
/// buffers allocated eagerly during recording, but compute is replayed later at
/// <c>Execute</c>. Without this gate the engine's normal tensor lifetime frees those buffers
/// the moment an intermediate tensor is disposed during recording, so by replay time the
/// graph nodes reference freed device pointers (<c>Handle == 0</c> → <c>cuMemcpyHtoDAsync
/// "Invalid value"</c>). While the gate is active, a buffer's release is queued (and the
/// queued closure holds a strong reference, so GC can't reclaim it either) and runs only after
/// the graph has executed.
/// </summary>
internal static class GpuBufferReleaseDeferral
{
    // Per-flow deferral state. Holds the nesting depth (DeferredScopes can nest — a scope's ctor
    // saves the parent — and only the OUTERMOST end flushes, so an inner EndAndRelease can't free the
    // outer scope's still-in-flight buffers) and the queued releases.
    private sealed class State
    {
        public int Depth;
        public readonly List<Action> Releases = new();
    }

    // AsyncLocal, NOT [ThreadStatic]: DeferredScope.ExecuteAsync awaits, and its continuation (and
    // `finally` cleanup) can resume on a DIFFERENT thread-pool thread than the one that called Begin()
    // in the ctor. With [ThreadStatic] the cleanup ran against the wrong thread's (empty) state → the
    // originating thread's gate stayed active forever and its queued buffer releases never ran (a
    // permanent resource leak). AsyncLocal flows the SAME State object across await boundaries and
    // threads, so Begin/TryDefer/EndAndRelease all see one shared, correctly-balanced state. The
    // State is locked because a parallel recording region could touch it from several flow-inheriting
    // threads.
    private static readonly System.Threading.AsyncLocal<State?> _state = new();

    /// <summary>True while releases on the current async flow are being deferred (any nesting depth).</summary>
    public static bool IsActive => _state.Value is { Depth: > 0 };

    /// <summary>Begin deferring releases for the current async flow; increments the nesting depth.</summary>
    public static void Begin()
    {
        var s = _state.Value;
        if (s is null)
        {
            s = new State();
            _state.Value = s;
        }
        lock (s) { s.Depth++; }
    }

    /// <summary>
    /// If deferral is active, queues <paramref name="release"/> to run after the OUTERMOST scope
    /// completes and returns true (caller must NOT release now). Returns false when no scope is active.
    /// </summary>
    public static bool TryDefer(Action release)
    {
        var s = _state.Value;
        if (s is null) return false;
        lock (s)
        {
            if (s.Depth == 0) return false;
            s.Releases.Add(release);
        }
        return true;
    }

    /// <summary>
    /// Ends one deferral nesting level. Only when the outermost level ends (depth → 0) are the queued
    /// releases drained and run. Idempotent when already balanced. Call AFTER the deferred graph has
    /// executed.
    /// </summary>
    public static void EndAndRelease()
    {
        var s = _state.Value;
        if (s is null) return;
        List<Action> toRun;
        lock (s)
        {
            if (s.Depth == 0) return;       // already balanced — no active deferral to end
            if (--s.Depth > 0) return;      // an inner scope ended; keep deferring for the outer scope
            toRun = new List<Action>(s.Releases);
            s.Releases.Clear();
        }
        foreach (var r in toRun)
        {
            try { r(); } catch { /* a single buffer's free must not abort the rest */ }
        }
    }
}
