using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Adaptive retention of intermediates that code reads after backward. A storage freed at its last backward use
/// (<see cref="BackwardStorageReleasePlan{T}"/>, on a tape that opted in with
/// <see cref="GradientTapeOptions.ReleaseDeviceActivationsAtLastUse"/>) cannot be read afterwards; PyTorch keeps such a
/// tensor alive only while Python holds a reference, which .NET cannot see without a collection. Instead, the first
/// backward run of each graph shape is a probe: what it would free is quarantined (kept, as before this release existed)
/// and a device or host read of a quarantined storage marks its producer (forward op, shape, ordinal) as retained.
/// Every later run of that graph frees everything else at its last use and keeps the marked ones, with no Retain call in
/// user code and no first-step failure.
/// <para>A graph's quarantine closes when that SAME graph runs backward again (a training loop's next step), not when
/// any backward runs: keeping one model's output, running another model's backward, then comparing the two is ordinary
/// code, and the first output must still be readable. The quarantine holds its storages weakly, so one nothing holds
/// any more falls back to the collector's reclaim instead of staying pinned. A read the probe could not see (a graph's
/// output read after that graph's own next backward run, or through a path that bypasses the engine's buffer lookup)
/// throws the precise <see cref="LastUseReleaseNote"/> error, which names the producer and the Retain call to add.</para>
/// </summary>
/// <remarks>
/// Producer keys and graph signatures are compared by value (<see cref="ProducerKey"/>, <see cref="GraphSignature"/>):
/// a hash alone let two different graphs share a signature, so one graph's probe was spent on another.
/// </remarks>
internal static class AutoRetainProbe
{
    // One per quarantined storage, shared by every quarantine of that storage that is open: the same storage can be
    // freed under two producer keys in one window (a view's producer and its base's), and a read of it must keep it for
    // all of them. A per-entry probe rebound the storage to the newest key, so the older entry still released storage
    // that had been read.
    private sealed class Probe
    {
        internal readonly List<object> Keys = new();   // under s_lock
        internal int OpenQuarantines;                  // under s_lock
        internal volatile bool Hit;
    }

    // Weak: a quarantine must not keep a storage its owner dropped (that one goes back to the collector's reclaim).
    private readonly struct Quarantined
    {
        internal readonly Probe Probe;
        internal readonly WeakReference<object> Vector;
        internal readonly WeakReference<object>? Array;
        internal readonly Action Release;

        internal Quarantined(Probe probe, object vector, object? array, Action release)
        {
            Probe = probe;
            Vector = new WeakReference<object>(vector);
            Array = array is null ? null : new WeakReference<object>(array);
            Release = release;
        }

        internal bool IsAlive => Vector.TryGetTarget(out _);
    }

    // Keyed by value (ProducerKey / GraphSignature implement equality).
    private static readonly ConcurrentDictionary<object, byte> s_retained = new();
    private static readonly HashSet<object> s_probedGraphs = new();
    // Probed graphs in the order they first ran, for the bound below.
    private static readonly Queue<object> s_probeOrder = new();

    /// <summary>
    /// The most graph signatures remembered as probed. A run with variable sequence lengths or changing structure sees
    /// a new signature per shape, each holding its producer keys; past the bound the oldest is forgotten, and that graph
    /// simply probes again the next time it runs (a probe run frees nothing, so forgetting costs memory, never safety).
    /// </summary>
    internal const int MaxProbedGraphs = 1024;
    private static readonly ConditionalWeakTable<object, Probe> s_quarantined = new();
    private static readonly Dictionary<object, List<Quarantined>> s_windows = new();   // per graph signature
    private static readonly object s_lock = new();
    private static volatile bool s_active;

    /// <summary>Whether a probe window is open: only then do buffer lookups and host reads consult the table.</summary>
    internal static bool Active => s_active;

    /// <summary>Whether a probe saw this producer's output read after backward.</summary>
    internal static bool IsRetained(object key) => !s_retained.IsEmpty && s_retained.ContainsKey(key);

    /// <summary>Retains a producer whose released output something tried to read.</summary>
    internal static void Retain(object key) => s_retained.TryAdd(key, 0);

    /// <summary>Producers retained so far (tests and diagnostics).</summary>
    internal static int RetainedCount => s_retained.Count;

    /// <summary>Graph signatures remembered as probed (tests and diagnostics).</summary>
    internal static int ProbedGraphCount
    {
        get { lock (s_lock) return s_probedGraphs.Count; }
    }

    /// <summary>Graphs whose probe quarantine is still open (tests and diagnostics).</summary>
    internal static int OpenWindowCount
    {
        get { lock (s_lock) return s_windows.Count; }
    }

    /// <summary>
    /// A backward run of the graph with this signature is starting. Closes that graph's previous probe window (frees
    /// what nothing read), drops quarantined storages of other graphs that nothing holds any more, and returns true the
    /// first time this graph runs backward: that run probes instead of freeing.
    /// </summary>
    internal static bool BeginRun(object graphSignature)
    {
        if (graphSignature is null) throw new ArgumentNullException(nameof(graphSignature));
        List<Quarantined>? closing = null;
        bool probe;
        lock (s_lock)
        {
            if (s_windows.Count != 0)
            {
                if (s_windows.TryGetValue(graphSignature, out closing)) s_windows.Remove(graphSignature);
                List<object>? empty = null;
                foreach (var pair in s_windows)
                {
                    var dead = pair.Value.FindAll(q => !q.IsAlive);
                    if (dead.Count != 0)
                    {
                        Forget(dead);
                        pair.Value.RemoveAll(q => !q.IsAlive);
                    }
                    if (pair.Value.Count == 0) (empty ??= new List<object>()).Add(pair.Key);
                }
                if (empty is not null) foreach (var key in empty) s_windows.Remove(key);
                if (closing is not null) Forget(closing);
                s_active = s_windows.Count != 0;
            }
            probe = s_probedGraphs.Add(graphSignature);
            if (probe)
            {
                s_probeOrder.Enqueue(graphSignature);
                while (s_probeOrder.Count > MaxProbedGraphs) s_probedGraphs.Remove(s_probeOrder.Dequeue());
            }
        }
        if (closing is not null)
        {
            Exception? firstFailure = null;
            foreach (var q in closing)
            {
                if (q.Probe.Hit || !q.IsAlive) continue;
                try
                {
                    q.Release();
                }
                catch (Exception ex)
                {
                    firstFailure ??= ex;
                }
            }
            if (firstFailure is not null)
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(firstFailure).Throw();
        }
        return probe;
    }

    /// <summary>
    /// Keeps a storage the probe run of <paramref name="graphSignature"/> would free until that graph's next backward
    /// run; <paramref name="release"/> frees it then unless something read it.
    /// </summary>
    internal static void Quarantine(object graphSignature, object vector, object? array, object key, Action release)
    {
        if (graphSignature is null) throw new ArgumentNullException(nameof(graphSignature));
        if (key is null) throw new ArgumentNullException(nameof(key));
        lock (s_lock)
        {
            // A storage already quarantined shares its probe: a read keeps it for every key it is quarantined under.
            if (!s_quarantined.TryGetValue(vector, out var probe)
                && !(array is not null && s_quarantined.TryGetValue(array, out probe)))
                probe = new Probe();
            s_quarantined.Remove(vector);
            s_quarantined.Add(vector, probe);
            if (array is not null)
            {
                s_quarantined.Remove(array);
                s_quarantined.Add(array, probe);
            }
            probe.Keys.Add(key);
            probe.OpenQuarantines++;
            if (probe.Hit) s_retained.TryAdd(key, 0);   // already read after this backward: keep it under this key too
            if (!s_windows.TryGetValue(graphSignature, out var window))
                s_windows.Add(graphSignature, window = new List<Quarantined>());
            window.Add(new Quarantined(probe, vector, array, release));
            s_active = true;
        }
    }

    /// <summary>A storage is being read (device lookup or host download): if quarantined, retain its producers.</summary>
    internal static void NoteAccess(object key)
    {
        if (!s_quarantined.TryGetValue(key, out var probe) || probe.Hit) return;
        lock (s_lock)
        {
            if (probe.Hit) return;
            probe.Hit = true;
            foreach (var producer in probe.Keys) s_retained.TryAdd(producer, 0);
        }
    }

    // Under s_lock. A storage's mapping goes only when no open quarantine shares its probe any more.
    private static void Forget(List<Quarantined> window)
    {
        foreach (var q in window)
        {
            if (--q.Probe.OpenQuarantines > 0) continue;
            if (q.Vector.TryGetTarget(out var vector)
                && s_quarantined.TryGetValue(vector, out var mapped) && ReferenceEquals(mapped, q.Probe))
                s_quarantined.Remove(vector);
            if (q.Array is not null && q.Array.TryGetTarget(out var array)
                && s_quarantined.TryGetValue(array, out var mappedArray) && ReferenceEquals(mappedArray, q.Probe))
                s_quarantined.Remove(array);
        }
    }

    /// <summary>Test hook: forgets every probe, retention and open window.</summary>
    internal static void ResetForTests()
    {
        lock (s_lock)
        {
            foreach (var window in s_windows.Values) Forget(window);
            s_windows.Clear();
            s_probedGraphs.Clear();
            s_probeOrder.Clear();
            s_active = false;
        }
        s_retained.Clear();
    }
}
