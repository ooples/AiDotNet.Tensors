using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
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
    private sealed class Pending
    {
        public readonly Action<object> Callback;
        public readonly int ThreadId;
        // 0 = pending, 1 = taken (materialized, removed, drained, or a losing duplicate). Exactly one party flips
        // it, and that party owns the _pendingCount decrement.
        private int _state;
        public Pending(Action<object> callback, int threadId) { Callback = callback; ThreadId = threadId; }

        public bool TryClaim()
        {
            if (Interlocked.Exchange(ref _state, 1) != 0) return false;
            Interlocked.Decrement(ref _pendingCount);
            GC.SuppressFinalize(this);
            return true;
        }

        // The key (a result nobody references any more) was collected, so the table dropped this entry without
        // anyone taking it. Keep _pendingCount exact: an inflated count disables the zero fast path in
        // TryMaterialize and sends EVERY host read in the process through the table.
        ~Pending() => TryClaim();
    }

    // WEAKLY keyed. This used to be a ConcurrentDictionary, which strongly rooted every key (a result's vector or
    // array) and, through the callback's closure, the result's GPU buffer - so nothing deferred could ever be
    // collected, and every free had to download first "in case someone reads it". With weak keys, a result nobody
    // references any more takes its pending download and its buffer with it (the buffer's own finalizer frees it,
    // stream-ordered), while a result somebody still holds is materialized on its first host read exactly as before.
    // Keys are compared by reference, as they were (neither arrays nor vectors override Equals).
    // Lock-free: ConditionalWeakTable is thread-safe, and ownership of an entry is decided by Pending.TryClaim. (A
    // global lock here serialized every host read of every tensor across all threads - parallel CPU loops spent
    // a third of a training step spinning in Monitor.Enter_Slowpath on it.)
    private static readonly ConditionalWeakTable<object, Pending> _pendingMaterializations = new();

    // Per-thread weak list of this thread's registered keys, for the thread-scoped bulk drain (MaterializeAll).
    [ThreadStatic] private static List<WeakReference<object>>? t_registeredKeys;
    [ThreadStatic] private static int t_registrationsSincePrune;

    /// <summary>
    /// Lock-free fast-path indicator. Incremented by <see cref="Register"/>, decremented
    /// by <see cref="TryMaterialize"/> and <see cref="Remove"/>. When this is 0,
    /// the CPU-only fast path in <see cref="TryMaterialize"/> returns immediately
    /// with a single volatile-int read, avoiding ConcurrentDictionary bucket-lock
    /// contention that was observed (2026-04-22) to serialize parallel tensor
    /// workloads via <c>ConcurrentDictionary&lt;T,U&gt;.IsEmpty</c>.
    /// </summary>
    private static int _pendingCount;

    internal static bool HasPendingMaterializations => Volatile.Read(ref _pendingCount) != 0;

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
        // First registration wins, as with the previous TryAdd: GetValue stores `mine` only if the key is absent.
        var mine = new Pending(materializeCallback, Environment.CurrentManagedThreadId);
        if (!ReferenceEquals(_pendingMaterializations.GetValue(array, _ => mine), mine))
        {
            mine.TryClaim(); // the duplicate never became pending: undo the increment above
            return;
        }
        var keys = t_registeredKeys ??= new List<WeakReference<object>>();
        keys.Add(new WeakReference<object>(array));
        if (++t_registrationsSincePrune >= 4096)
        {
            t_registrationsSincePrune = 0;
            keys.RemoveAll(static w => !w.TryGetTarget(out var k) || !IsPending(k));
        }
    }

    // Claims the entry (and its _pendingCount decrement) for the caller; false when absent or already claimed.
    private static bool TryTake(object array, out Pending? pending)
    {
        if (_pendingMaterializations.TryGetValue(array, out pending) && pending.TryClaim())
        {
            _pendingMaterializations.Remove(array);
            return true;
        }
        pending = null;
        return false;
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

        if (TryTake(array, out var pending) && pending is not null)
        {
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
        return _pendingMaterializations.TryGetValue(array, out _);
    }

    /// <summary>
    /// Removes a pending materialization without executing it (e.g., when the GPU buffer is reused).
    /// </summary>
    internal static void Remove(object array)
    {
        TryTake(array, out _);
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
        // THREAD-SCOPED drain. Only fire the callbacks registered on the CURRENT thread.
        // A bulk drain at scope/engine teardown previously fired EVERY thread's pending
        // download - so one parallel test exiting its GPU scope would DownloadBuffer
        // another concurrent test's IN-FLIGHT buffer (the registry + GPU queue are shared
        // process-wide). That cross-thread read of a buffer a kernel is still writing
        // surfaced as "Failed to read OpenCL buffer: -38" and, when the GPU driver faulted,
        // a hard process kill. Each thread flushes only its OWN deferred tensors (the
        // per-thread weak key list); keys whose tensors were collected are simply gone -
        // nobody can read them.
        var keys = t_registeredKeys;
        if (keys is null || keys.Count == 0)
            return;
        var snapshot = keys.ToArray();
        keys.Clear();
        List<Exception>? failures = null;
        foreach (var weak in snapshot)
        {
            if (!weak.TryGetTarget(out var key)) continue;
            if (!TryTake(key, out var pending) || pending is null) continue;

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
