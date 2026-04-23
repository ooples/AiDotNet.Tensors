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
    private static readonly ConcurrentDictionary<object, Action<object>> _pendingMaterializations = new();

    /// <summary>
    /// Lock-free fast-path indicator. Incremented by <see cref="Register"/>, decremented
    /// by <see cref="TryMaterialize"/> and <see cref="Remove"/>. When this is 0,
    /// the CPU-only fast path in <see cref="TryMaterialize"/> returns immediately
    /// with a single volatile-int read, avoiding ConcurrentDictionary bucket-lock
    /// contention that was observed (2026-04-22) to serialize parallel tensor
    /// workloads via <c>ConcurrentDictionary&lt;T,U&gt;.IsEmpty</c>.
    /// </summary>
    private static int _pendingCount;

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
        if (!_pendingMaterializations.TryAdd(array, materializeCallback))
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

        if (_pendingMaterializations.TryRemove(array, out var callback))
        {
            Interlocked.Decrement(ref _pendingCount);
            callback(array);
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

        // Snapshot keys so we can safely mutate the dictionary from each callback
        // (Register's semantics are "remove on fire" via TryRemove).
        var keys = _pendingMaterializations.Keys.ToArray();
        List<Exception>? failures = null;
        foreach (var key in keys)
        {
            if (!_pendingMaterializations.TryRemove(key, out var callback))
                continue;
            Interlocked.Decrement(ref _pendingCount);

            try { callback(key); }
            catch (InvalidOperationException) when (swallowErrors)
            {
                // GPU buffer may be torn down; the entry was already removed
                // above, so any subsequent call falls through the normal data
                // path rather than re-running a broken callback.
            }
            catch (Exception ex)
            {
                // Drain-all semantics: a single failing callback must not pin
                // the remaining entries. Collect the exceptions and surface
                // them after the loop finishes so the registry ends in a
                // clean "no pending" state regardless of how many entries
                // raised.
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
