using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Static registry for deferred GPU-to-CPU array materialization.
/// When a GPU operation defers its download, it registers a callback here keyed by the result array.
/// When CPU code needs the actual data (via GetDataArray, AsSpan, indexer), VectorBase calls
/// TryMaterialize to populate the array on-demand. This enables zero-copy GPU pipelines
/// where intermediate results stay GPU-resident until explicitly needed by CPU code.
/// </summary>
internal static class DeferredArrayMaterializer
{
    private readonly struct Pending
    {
        public readonly Action<object> Callback;
        public readonly int ThreadId;
        public Pending(Action<object> callback, int threadId) { Callback = callback; ThreadId = threadId; }
    }

    private static readonly ConcurrentDictionary<object, Pending> _pendingMaterializations = new();

    /// <summary>
    /// Lock-free fast-path indicator. Incremented by <see cref="Register"/>, decremented
    /// by <see cref="TryMaterialize"/> and <see cref="Remove"/>. When this is 0,
    /// the CPU-only fast path in <see cref="TryMaterialize"/> returns immediately
    /// with a single volatile-int read, avoiding ConcurrentDictionary bucket-lock
    /// contention that was observed (2026-04-22) to serialize parallel tensor
    /// workloads via <c>ConcurrentDictionary&lt;T,U&gt;.IsEmpty</c>.
    /// </summary>
    private static int _pendingCount;

    // Diagnostics: total deferred GPU→CPU downloads actually performed (each fired callback = one DtoH copy of a
    // resident tensor to host). A test resets this around a training step and asserts it stays ~0 to prove the
    // forward/backward kept every activation/gradient GPU-resident (no per-op host round-trip). Always-on counter
    // on the (already-expensive) transfer path — negligible overhead.
    private static long _materializeCount;

    /// <summary>Total deferred GPU→CPU materializations (DtoH downloads) performed since the last reset.</summary>
    public static long MaterializeCount => Volatile.Read(ref _materializeCount);

    /// <summary>Resets <see cref="MaterializeCount"/> to zero (test instrumentation).</summary>
    public static void ResetMaterializeCount() => Interlocked.Exchange(ref _materializeCount, 0);

    /// <summary>
    /// Registers a deferred materialization callback for the given array.
    /// When TryMaterialize is called with this array, the callback runs to populate it.
    /// </summary>
    /// <remarks>
    /// Ordering: <see cref="Interlocked.Increment(ref int)"/> BEFORE
    /// <c>TryAdd</c>. If the increment happened after a successful TryAdd,
    /// there would be a window where a concurrent <see cref="TryMaterialize"/>
    /// reads <c>_pendingCount == 0</c> and skips the dictionary check even
    /// though the entry is now registered — causing the callback to be
    /// silently missed. Incrementing first makes the counter a conservative
    /// over-estimate during the window: readers see ""might be pending"",
    /// do a harmless dictionary lookup, and find nothing, returning the
    /// correct ""not pending"" result. Rolled back with
    /// <see cref="Interlocked.Decrement"/> if TryAdd fails (duplicate key).
    /// </remarks>
    internal static void Register(object array, Action<object> materializeCallback)
    {
        Interlocked.Increment(ref _pendingCount);
        // Tag with the registering thread. The bulk drain (MaterializeAll) only fires
        // THIS thread's entries — downloading another thread's buffer in a bulk drain
        // would read a GPU buffer that thread's kernel is still writing (shared queue) →
        // CL_INVALID_MEM_OBJECT or a GPU driver fault. See MaterializeAll.
        if (!_pendingMaterializations.TryAdd(array, new Pending(materializeCallback, Environment.CurrentManagedThreadId)))
            Interlocked.Decrement(ref _pendingCount);
    }

    /// <summary>
    /// If the given array has a pending deferred download, materializes it now.
    /// Returns true if materialization occurred, false if no pending download.
    /// </summary>
    internal static bool TryMaterialize(object array)
    {
        // Fast path for CPU-only workloads: a single volatile-int read with no
        // ConcurrentDictionary access at all. Observed 2026-04-22 that using
        // `_pendingMaterializations.IsEmpty` here caused Monitor.Enter_Slowpath
        // contention inside the ConcurrentDictionary under high-fanout parallel
        // tensor forward passes (44s of unmanaged wait time per 30s of parallel
        // work across 4 workers — one of the root causes of the HRE report-card
        // hang).
        if (Volatile.Read(ref _pendingCount) == 0)
            return false;

        if (_pendingMaterializations.TryRemove(array, out var pending))
        {
            Interlocked.Decrement(ref _pendingCount);
            Interlocked.Increment(ref _materializeCount); // a real DtoH download is about to run
            pending.Callback(array);
            return true;
        }
        return false;
    }

    /// <summary>
    /// Checks if the given array has a pending deferred download.
    /// </summary>
    internal static bool IsPending(object array)
    {
        if (Volatile.Read(ref _pendingCount) == 0) return false;
        return _pendingMaterializations.ContainsKey(array);
    }

    /// <summary>
    /// Removes a pending materialization without executing it (e.g., when the GPU buffer is reused).
    /// </summary>
    internal static void Remove(object array)
    {
        if (_pendingMaterializations.TryRemove(array, out _))
            Interlocked.Decrement(ref _pendingCount);
    }

    /// <summary>
    /// Drains all pending materializers by invoking each registered callback.
    /// Used at scope-end (e.g. <see cref="DirectGpuTensorEngine.MaterializeAllDeferred"/>)
    /// so every GPU-resident tensor with a pending download is flushed to CPU.
    /// </summary>
    /// <param name="swallowErrors">
    /// When <c>true</c>, per-entry <see cref="InvalidOperationException"/>s are
    /// swallowed; the entry is still removed from the pending registry before
    /// the callback runs (see below), so subsequent access to the array falls
    /// through the normal data path rather than re-running a broken callback.
    /// Matches the old <c>MaterializeAllDeferred</c> semantics where a torn-down
    /// GPU context during dispose must not bring down the whole teardown path.
    /// When <c>false</c>, exceptions propagate to the caller.
    /// </param>
    /// <remarks>
    /// Each entry is <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey, TValue}.TryRemove(TKey, out TValue)"/>-removed
    /// from the registry *before* its callback is invoked — so whether the
    /// callback succeeds, fails, or is skipped, the array is no longer pending
    /// after this call returns.
    /// </remarks>
    internal static void MaterializeAll(bool swallowErrors = true)
    {
        if (_pendingMaterializations.IsEmpty)
            return;

        // THREAD-SCOPED drain. Only fire the callbacks registered on the CURRENT thread.
        // A bulk drain at scope/engine teardown previously fired EVERY thread's pending
        // download — so one parallel test exiting its GPU scope would DownloadBuffer
        // another concurrent test's IN-FLIGHT buffer (the registry + GPU queue are shared
        // process-wide). That cross-thread read of a buffer a kernel is still writing
        // surfaced as "Failed to read OpenCL buffer: -38" and, when the GPU driver faulted,
        // a hard process kill (no managed/native exception to catch). Each thread flushes
        // only its OWN deferred tensors here; on-demand access (TryMaterialize for a specific
        // array) still works from any thread, and other threads flush at their own scope exit.
        int callerThreadId = Environment.CurrentManagedThreadId;
        var keys = _pendingMaterializations.Keys.ToArray();
        List<Exception>? failures = null;
        foreach (var key in keys)
        {
            // Only claim entries owned by this thread (TryGetValue first to check the owner
            // without removing other threads' entries).
            if (!_pendingMaterializations.TryGetValue(key, out var pending))
                continue;
            if (pending.ThreadId != callerThreadId)
                continue;
            if (!_pendingMaterializations.TryRemove(key, out pending))
                continue;
            Interlocked.Decrement(ref _pendingCount);

            try { pending.Callback(key); }
            catch (InvalidOperationException) when (swallowErrors)
            {
                // GPU buffer may be torn down; the entry was already removed
                // above, so any subsequent call falls through the normal data
                // path rather than re-running a broken callback.
            }
            catch (Exception ex)
            {
                // Drain semantics: a single failing callback must not pin the
                // remaining entries. Collect and surface after the loop so the
                // registry ends in a clean state regardless of how many raised.
                (failures ??= new List<Exception>()).Add(ex);
            }
        }

        if (failures is not null)
        {
            throw failures.Count == 1
                ? failures[0]
                : new AggregateException(
                    "One or more deferred GPU-to-CPU materialization callbacks failed.",
                    failures);
        }
    }
}
