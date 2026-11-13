using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Adaptive retention of intermediates that code reads after backward. A storage freed at its last backward use
/// (<see cref="BackwardStorageReleasePlan{T}"/>) cannot be read afterwards; PyTorch keeps such a tensor alive only while
/// Python holds a reference, which .NET cannot see without a collection. Instead, the first backward run of each graph
/// shape is a probe: what it would free is quarantined (kept, as before this release existed) until the next backward
/// run starts, and a device or host read of a quarantined storage in that window marks its producer (forward op,
/// shape, ordinal) as retained. Every later run of that graph frees everything else at its last use and keeps the
/// marked ones, with no Retain call in user code and no first-step failure. A read the probe could not see (one more
/// than a step later, or through a path that bypasses the engine's buffer lookup) throws the precise
/// <see cref="LastUseReleaseNote"/> error, which names the producer and the Retain call to add.
/// </summary>
internal static class AutoRetainProbe
{
    private sealed class Probe
    {
        internal readonly long Key;
        internal volatile bool Hit;
        internal Probe(long key) => Key = key;
    }

    private readonly struct Quarantined
    {
        internal readonly Probe Probe;
        internal readonly object Vector;
        internal readonly object? Array;
        internal readonly Action Release;

        internal Quarantined(Probe probe, object vector, object? array, Action release)
        {
            Probe = probe;
            Vector = vector;
            Array = array;
            Release = release;
        }
    }

    private static readonly ConcurrentDictionary<long, byte> s_retained = new();
    private static readonly HashSet<long> s_probedGraphs = new();
    private static readonly ConditionalWeakTable<object, Probe> s_quarantined = new();
    private static readonly List<Quarantined> s_window = new();
    private static readonly object s_lock = new();
    private static volatile bool s_active;

    /// <summary>Whether a probe window is open: only then do buffer lookups and host reads consult the table.</summary>
    internal static bool Active => s_active;

    /// <summary>Whether a probe saw this producer's output read after backward.</summary>
    internal static bool IsRetained(long key) => !s_retained.IsEmpty && s_retained.ContainsKey(key);

    /// <summary>Retains a producer whose released output something tried to read.</summary>
    internal static void Retain(long key) => s_retained.TryAdd(key, 0);

    /// <summary>Producers retained so far (tests and diagnostics).</summary>
    internal static int RetainedCount => s_retained.Count;

    /// <summary>True the first time a graph with this signature runs backward: that run probes instead of freeing.</summary>
    internal static bool TryBeginProbe(long graphSignature)
    {
        lock (s_lock) return s_probedGraphs.Add(graphSignature);
    }

    /// <summary>Keeps a storage the probe run would free until the window ends; <paramref name="release"/> frees it then.</summary>
    internal static void Quarantine(object vector, object? array, long key, Action release)
    {
        var probe = new Probe(key);
        lock (s_lock)
        {
            s_quarantined.Remove(vector);
            s_quarantined.Add(vector, probe);
            if (array is not null)
            {
                s_quarantined.Remove(array);
                s_quarantined.Add(array, probe);
            }
            s_window.Add(new Quarantined(probe, vector, array, release));
            s_active = true;
        }
    }

    /// <summary>A storage is being read (device lookup or host download): if quarantined, retain its producer.</summary>
    internal static void NoteAccess(object key)
    {
        if (s_quarantined.TryGetValue(key, out var probe) && !probe.Hit)
        {
            probe.Hit = true;
            s_retained.TryAdd(probe.Key, 0);
        }
    }

    /// <summary>Closes the window (the next backward run is starting): frees every quarantined storage nothing read.</summary>
    internal static void EndWindow()
    {
        if (!s_active) return;
        Quarantined[] window;
        lock (s_lock)
        {
            window = s_window.ToArray();
            s_window.Clear();
            foreach (var q in window)
            {
                s_quarantined.Remove(q.Vector);
                if (q.Array is not null) s_quarantined.Remove(q.Array);
            }
            s_active = false;
        }
        foreach (var q in window)
            if (!q.Probe.Hit) q.Release();
    }

    /// <summary>Test hook: forgets every probe, retention and open window.</summary>
    internal static void ResetForTests()
    {
        lock (s_lock)
        {
            foreach (var q in s_window)
            {
                s_quarantined.Remove(q.Vector);
                if (q.Array is not null) s_quarantined.Remove(q.Array);
            }
            s_window.Clear();
            s_probedGraphs.Clear();
            s_active = false;
        }
        s_retained.Clear();
    }
}
