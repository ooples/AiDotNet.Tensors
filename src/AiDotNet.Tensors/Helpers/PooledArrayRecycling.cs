using System;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Implemented by caches that key device copies by a host array's identity.
/// </summary>
internal interface IRecycledArrayListener
{
    /// <summary>
    /// <paramref name="array"/> is going back to a tensor pool: its owner is gone and the next rent hands the
    /// same array to an unrelated tensor. Anything keyed by it must be dropped.
    /// </summary>
    void OnArrayRecycled(object array);
}

/// <summary>
/// Tells array-keyed device caches when a pooled tensor array is recycled.
/// </summary>
/// <remarks>
/// The GPU activation cache is keyed by the host backing array and validated by the storage's GPU-cache
/// version. A tensor that rents a recycled array starts with a fresh storage whose version can equal the
/// one the previous owner's entry was recorded at, so the upload path would serve the previous owner's
/// device data. The tape-gradient parity suite caught it: a different op's GPU gradient came out wrong
/// every few runs while a fresh CPU engine and the fixture's CPU engine agreed. Listeners are held weakly,
/// so an engine nobody references is not kept alive by this registry.
/// </remarks>
internal static class PooledArrayRecycling
{
    private static readonly object s_lock = new();
    private static WeakReference<IRecycledArrayListener>[] s_listeners = Array.Empty<WeakReference<IRecycledArrayListener>>();

    internal static void Register(IRecycledArrayListener listener)
    {
        lock (s_lock)
        {
            var live = new List<WeakReference<IRecycledArrayListener>>(s_listeners.Length + 1);
            foreach (var w in s_listeners)
                if (w.TryGetTarget(out var l) && !ReferenceEquals(l, listener)) live.Add(w);
            live.Add(new WeakReference<IRecycledArrayListener>(listener));
            Volatile.Write(ref s_listeners, live.ToArray());
        }
    }

    internal static void Unregister(IRecycledArrayListener listener)
    {
        lock (s_lock)
        {
            var live = new List<WeakReference<IRecycledArrayListener>>(s_listeners.Length);
            foreach (var w in s_listeners)
                if (w.TryGetTarget(out var l) && !ReferenceEquals(l, listener)) live.Add(w);
            Volatile.Write(ref s_listeners, live.ToArray());
        }
    }

    /// <summary>Reports <paramref name="array"/> as recycled. One volatile read when no device cache exists.</summary>
    internal static void Notify(object array)
    {
        var listeners = Volatile.Read(ref s_listeners);
        if (listeners.Length == 0) return;
        foreach (var w in listeners)
            if (w.TryGetTarget(out var l)) l.OnArrayRecycled(array);
    }
}
