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

    /// <summary>True while releases on the current thread are being deferred.</summary>
    public static bool IsActive => _deferred != null;

    /// <summary>Begin deferring releases on the calling thread (idempotent within a scope).</summary>
    public static void Begin() => _deferred ??= new List<Action>();

    /// <summary>
    /// If deferral is active, queues <paramref name="release"/> to run after the scope completes
    /// and returns true (caller must NOT release now). Returns false when no scope is active.
    /// </summary>
    public static bool TryDefer(Action release)
    {
        var d = _deferred;
        if (d is null) return false;
        d.Add(release);
        return true;
    }

    /// <summary>
    /// Ends deferral on the calling thread and runs every queued release. Call AFTER the
    /// deferred graph has executed. Clears the gate first so the queued releases run normally
    /// (no re-deferral) and a later scope starts clean.
    /// </summary>
    public static void EndAndRelease()
    {
        var d = _deferred;
        _deferred = null;
        if (d is null) return;
        foreach (var r in d)
        {
            try { r(); } catch { /* a single buffer's free must not abort the rest */ }
        }
    }
}
