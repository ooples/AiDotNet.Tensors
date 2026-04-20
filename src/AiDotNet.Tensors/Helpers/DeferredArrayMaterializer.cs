using System.Collections.Concurrent;
using System.Linq;

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
    /// Registers a deferred materialization callback for the given array.
    /// When TryMaterialize is called with this array, the callback runs to populate it.
    /// </summary>
    internal static void Register(object array, Action<object> materializeCallback)
    {
        _pendingMaterializations.TryAdd(array, materializeCallback);
    }

    /// <summary>
    /// If the given array has a pending deferred download, materializes it now.
    /// Returns true if materialization occurred, false if no pending download.
    /// </summary>
    internal static bool TryMaterialize(object array)
    {
        // Fast path: if nothing is pending (common case for CPU-only code), skip the dictionary lookup entirely.
        // IsEmpty checks Count == 0 with minimal overhead — avoids full TryRemove lookup.
        if (_pendingMaterializations.IsEmpty)
            return false;

        if (_pendingMaterializations.TryRemove(array, out var callback))
        {
            callback(array);
            return true;
        }
        return false;
    }

    /// <summary>
    /// Checks if the given array has a pending deferred download.
    /// </summary>
    internal static bool IsPending(object array) => _pendingMaterializations.ContainsKey(array);

    /// <summary>
    /// Removes a pending materialization without executing it (e.g., when the GPU buffer is reused).
    /// </summary>
    internal static void Remove(object array) => _pendingMaterializations.TryRemove(array, out _);

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
        foreach (var key in keys)
        {
            if (!_pendingMaterializations.TryRemove(key, out var callback))
                continue;

            if (swallowErrors)
            {
                try { callback(key); }
                catch (InvalidOperationException)
                {
                    // GPU buffer may be torn down; the entry was already removed
                    // above, so any subsequent call falls through the normal data
                    // path rather than re-running a broken callback.
                }
            }
            else
            {
                callback(key);
            }
        }
    }
}
