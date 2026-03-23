using System.Collections.Concurrent;

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
}
