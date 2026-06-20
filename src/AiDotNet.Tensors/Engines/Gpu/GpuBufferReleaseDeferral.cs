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
    [ThreadStatic] private static List<Action>? _deferred;
    // Nesting depth: DeferredScopes can nest (a scope's ctor saves the parent). A single shared
    // list would let an INNER scope's EndAndRelease() flush the OUTER scope's still-in-flight
    // releases (use-after-free). We only flush when the OUTERMOST scope ends (depth returns to 0);
    // inner-scope releases are held (safe — delayed free) until then. Each Begin() must be paired
    // with exactly one EndAndRelease() (DeferredScope guards against its Execute()+Dispose() double
    // call so the count stays balanced).
    [ThreadStatic] private static int _depth;

    /// <summary>True while releases on the current thread are being deferred (any nesting depth).</summary>
    public static bool IsActive => _depth > 0;

    /// <summary>Begin deferring releases on the calling thread; increments the nesting depth.</summary>
    public static void Begin()
    {
        _depth++;
        _deferred ??= new List<Action>();
    }

    /// <summary>
    /// If deferral is active, queues <paramref name="release"/> to run after the OUTERMOST scope
    /// completes and returns true (caller must NOT release now). Returns false when no scope is active.
    /// </summary>
    public static bool TryDefer(Action release)
    {
        if (_depth == 0) return false;
        (_deferred ??= new List<Action>()).Add(release);
        return true;
    }

    /// <summary>
    /// Ends one deferral nesting level. Only when the outermost level ends (depth → 0) are the queued
    /// releases drained and run. Idempotent when already balanced (depth 0). Call AFTER the deferred
    /// graph has executed.
    /// </summary>
    public static void EndAndRelease()
    {
        if (_depth == 0) return;        // already balanced — no active deferral to end
        if (--_depth > 0) return;       // an inner scope ended; keep deferring for the outer scope
        var d = _deferred;
        _deferred = null;
        if (d is null) return;
        foreach (var r in d)
        {
            try { r(); } catch { /* a single buffer's free must not abort the rest */ }
        }
    }
}
