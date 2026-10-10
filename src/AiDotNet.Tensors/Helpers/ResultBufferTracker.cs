using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// The owner of one tracked result array. Every vector or matrix over the array (including zero-copy views of other
/// types, which reach it through <see cref="ResultBufferTracker.Find"/>) holds a reference to it, so it becomes
/// unreachable exactly when the last wrapper does. Tensors reach it through their storage's vector.
/// </summary>
internal sealed class ResultBufferOwner
{
    internal ResultBufferOwner(ResultBufferState state) => State = state;

    /// <summary>Shared with the tracker entry, so the escape mark outlives this owner.</summary>
    internal readonly ResultBufferState State;

    /// <summary>
    /// Records that raw access other than a lease handed out this array (a span, Memory or the array itself), so it
    /// is never recycled: nothing guarantees that access ends before every wrapper is collected.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    internal void MarkEscaped()
    {
        if (!State.Escaped && ResultBufferTracker.TraceEscapes) ResultBufferTracker.RecordEscape();
        State.Escaped = true;
    }
}

/// <summary>Per-array state the owner and the tracker both reference.</summary>
internal sealed class ResultBufferState
{
    /// <summary>Raw access other than a lease reached the array; it must never be recycled.</summary>
    internal volatile bool Escaped;
}

/// <summary>
/// Reuses the arrays of large results that nobody disposed, once every wrapper over them is gone.
/// </summary>
/// <remarks>
/// <para><b>Why.</b> A fresh large result is not just an allocation: the large-object heap hands back memory the OS has
/// to fault in and zero again (about 210 page faults per 2 MB result on Windows; the first write costs about twice a
/// rewrite), and the churn drives gen-2 collections. Reusing an array the previous result no longer needs costs only
/// the arithmetic. Measured on a Threadripper 3990X: a 16K-element double result 43 us fresh vs 4.4 us reused, 1M
/// elements 1311 vs 689 us. Below the large-object threshold the gen-0 allocator is already within 0.15 us of reuse,
/// so small results are left to it (and to <see cref="TensorArena"/> scopes).</para>
/// <para><b>How.</b> The tracker holds each array strongly (so it never becomes large-object garbage) and its
/// <see cref="ResultBufferOwner"/> through ONE weak handle, grouped by element type and length. A large rent that
/// misses the thread cache takes a same-size array whose owner a collection has cleared. When none is free and tracked
/// bytes have grown past the collection mark, it induces a collection (gen 0, escalating while promoted owners keep it
/// from freeing anything) and looks again. Free arrays stay with the tracker until a rent of their size takes them,
/// up to <see cref="FreeCapBytes"/>.</para>
/// <para><b>Contract.</b> Raw access that can outlive a wrapper (a span or Memory taken from a result and used after
/// the result's last reference) must keep the wrapper alive. Wrapping a tracked array whose owner has already died is
/// such a bug; the array then stops being tracked rather than ever being reused (<c>AIDOTNET_RECYCLE_STRESS=1</c>
/// throws instead, so tests surface it).</para>
/// </remarks>
internal static class ResultBufferTracker
{
    private sealed class Entry
    {
        internal GCHandle Owner;
        internal ResultBufferState State = null!;
        internal Array Array = null!;
        internal long Bytes;
        internal List<Entry> Group = null!;
    }

    private sealed class ArrayIdentity : IEqualityComparer<Array>
    {
        internal static readonly ArrayIdentity Instance = new();
        public bool Equals(Array? x, Array? y) => ReferenceEquals(x, y);
        public int GetHashCode(Array obj) => RuntimeHelpers.GetHashCode(obj);
    }

    private static readonly object s_lock = new();
    private static readonly Dictionary<Array, Entry> s_byArray = new(ArrayIdentity.Instance);
    private static readonly Dictionary<(Type, int), List<Entry>> s_bySize = new();
    private static int s_count;
    private static long s_trackedBytes;

    // Collection pacing: fresh large bytes allocated (misses with nothing free) since the last gen-0 collection, natural
    // or induced. Once they pass CollectBudgetBytes, the next miss collects and looks again, so fresh allocation between
    // collections stays bounded. (Pacing on TOTAL tracked bytes let the mark run away: tracked arrays reached the cap
    // with no collection, and 46% of results were fresh.) The generation escalates while a collection frees nothing of
    // the size asked for (owners promoted past gen 0 need a deeper one) and resets once one does.
    private static long s_freshSinceCollect;
    private static int s_lastGen0 = -1;
    private static int s_collectGeneration;
    // Backoff: a collection that freed nothing usable doubles the budget for the next one (up to 32x), so a workload
    // whose large results all escape, or that holds them, does not pay a collection every few calls. Reset on success.
    private static int s_budgetShift;

    /// <summary>Whether result recycling is active (off on .NET Framework, where large results are not pooled).</summary>
    internal static readonly bool Enabled =
#if NET5_0_OR_GREATER
        !EnvTrue("AIDOTNET_DISABLE_RESULT_RECYCLING");
#else
        false;
#endif

    /// <summary>Smallest array, in bytes, that is tracked. Default: the large-object-heap threshold.</summary>
    internal static long MinBytes { get; set; } = EnvLong("AIDOTNET_RESULT_RECYCLE_MIN_BYTES", 85_000);

    /// <summary>Tracked bytes at which a miss with nothing free induces a collection.</summary>
    internal static long CollectBudgetBytes { get; set; } = EnvLong("AIDOTNET_RESULT_RECYCLE_COLLECT_BYTES", 32L << 20);

    /// <summary>Tracked bytes (live and free) past which new results are left untracked (plain GC arrays).</summary>
    internal static long HardCapBytes { get; set; } = EnvLong("AIDOTNET_RESULT_RECYCLE_CAP_BYTES", 1L << 30);

    /// <summary>Free (owner collected, not yet reused) bytes kept for reuse; beyond it the oldest free arrays go to the GC.</summary>
    internal static long FreeCapBytes { get; set; } = EnvLong("AIDOTNET_RESULT_RECYCLE_FREE_BYTES", 64L << 20);

    /// <summary>
    /// Test mode: every large miss collects first, freed arrays are poisoned before reuse, and wrapping a tracked
    /// array whose owner is gone throws. Makes a lifetime bug fail loudly instead of corrupting data occasionally.
    /// </summary>
    internal static bool Stress { get; set; } = EnvTrue("AIDOTNET_RECYCLE_STRESS");

    /// <summary>
    /// Diagnostic: records which member first escaped each tracked array (AIDOTNET_RECYCLE_TRACE_ESCAPES=1), to rank the
    /// call sites still keeping results from being recycled. Costs a stack walk per escape; off by default.
    /// </summary>
    internal static bool TraceEscapes { get; set; } = EnvTrue("AIDOTNET_RECYCLE_TRACE_ESCAPES");

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> s_escapeSites = new();

    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static void RecordEscape()
    {
        // Frame 0 is this method, 1 MarkEscaped, 2 the raw accessor, 3+ the caller; skip accessors of the storage types.
        var trace = new System.Diagnostics.StackTrace(2, false);
        for (int i = 0; i < trace.FrameCount; i++)
        {
            var method = trace.GetFrame(i)?.GetMethod();
            var type = method?.DeclaringType;
            if (method is null || type is null) continue;
            string ns = type.Namespace ?? "";
            if (ns == "AiDotNet.Tensors.LinearAlgebra" && (type.Name.StartsWith("VectorBase") || type.Name.StartsWith("MatrixBase")
                || type.Name.StartsWith("TensorBase") || type.Name.StartsWith("TensorStorage"))) continue;
            s_escapeSites.AddOrUpdate($"{type.Name}.{method.Name}", 1, static (_, n) => n + 1);
            return;
        }
    }

    /// <summary>The most frequent first-escape sites recorded since the last reset, as (site, count).</summary>
    internal static System.Collections.Generic.List<System.Collections.Generic.KeyValuePair<string, int>> TopEscapeSites(int count)
    {
        var list = new System.Collections.Generic.List<System.Collections.Generic.KeyValuePair<string, int>>(s_escapeSites);
        list.Sort((a, b) => b.Value.CompareTo(a.Value));
        if (list.Count > count) list.RemoveRange(count, list.Count - count);
        return list;
    }

    internal static int TrackedCount => Volatile.Read(ref s_count);
    internal static long TrackedBytes => Interlocked.Read(ref s_trackedBytes);
    internal static long InducedCollections;
    internal static long ReusedArrays;
    /// <summary>Arrays whose results were collected but could not be reused because raw access escaped them.</summary>
    internal static long EscapedArrays;

    /// <summary>Whether an array of <paramref name="length"/> elements of <typeparamref name="T"/> is tracked.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool Qualifies<T>(int length)
        => Enabled && TensorPool.Enabled && (long)length * Unsafe.SizeOf<T>() >= MinBytes;

    /// <summary>
    /// Starts tracking <paramref name="array"/>, a pooled array a new result is about to own, and returns the owner the
    /// result must hold. Null when the array is not tracked (too small, disabled, or past the cap).
    /// </summary>
    internal static ResultBufferOwner? Track<T>(T[] array)
    {
        if (!Qualifies<T>(array.Length)) return null;
#if NET5_0_OR_GREATER
        // Arrays of references are never reused: a reused one would keep the previous result's objects reachable.
        if (RuntimeHelpers.IsReferenceOrContainsReferences<T>()) return null;
#endif
        long bytes = (long)array.Length * Unsafe.SizeOf<T>();
        if (Interlocked.Read(ref s_trackedBytes) + bytes > HardCapBytes) return null;

        var state = new ResultBufferState();
        var owner = new ResultBufferOwner(state);
        lock (s_lock)
        {
            // Re-issued without an untrack (a path that returned it to the cache directly): the old owner's death must
            // not hand out an array the new result now uses.
            if (s_byArray.TryGetValue(array, out var previous)) RemoveLocked(previous);

            var key = (typeof(T), array.Length);
            if (!s_bySize.TryGetValue(key, out var group)) s_bySize[key] = group = new List<Entry>();
            var entry = new Entry { Owner = GCHandle.Alloc(owner, GCHandleType.Weak), State = state, Array = array, Bytes = bytes, Group = group };
            group.Add(entry);
            s_byArray[array] = entry;
            s_trackedBytes += bytes;
            s_count++;
        }
        return owner;
    }

    /// <summary>
    /// The owner of the array behind <paramref name="memory"/> when it is tracked, so a zero-copy wrapper of any type
    /// (and any offset) keeps it alive. Null when untracked.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ResultBufferOwner? Find<T>(ReadOnlyMemory<T> memory)
    {
        if (Volatile.Read(ref s_count) == 0) return null;
        return MemoryMarshal.TryGetArray(memory, out var segment) && segment.Array is { } array ? Find(array) : null;
    }

    /// <summary>The owner of a tracked <paramref name="array"/>; null when untracked.</summary>
    internal static ResultBufferOwner? Find(Array array)
    {
        if (Volatile.Read(ref s_count) == 0) return null;
        lock (s_lock)
        {
            if (!s_byArray.TryGetValue(array, out var entry)) return null;
            if (entry.Owner.Target is ResultBufferOwner owner) return owner;

            // Every wrapper is gone, yet something still had the array to wrap: raw access outlived its result.
            if (Stress)
                throw new InvalidOperationException(
                    "A tracked result array was wrapped after every result over it was collected: raw access (a span, " +
                    "Memory<T> or array taken from a result) outlived the result. Keep the result alive while using it.");
            // Never reuse it: from now on it is an ordinary array the GC owns.
            RemoveLocked(entry);
            return null;
        }
    }

    /// <summary>Stops tracking <paramref name="array"/> (it is being returned to the cache explicitly).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Untrack(Array array)
    {
        if (Volatile.Read(ref s_count) == 0) return;
        lock (s_lock)
        {
            if (s_byArray.TryGetValue(array, out var entry)) RemoveLocked(entry);
        }
    }

    /// <summary>
    /// For a large rent that missed the thread cache: an array of exactly <paramref name="length"/> elements whose
    /// result is gone, collecting first when nothing is free and tracked bytes have passed the collection mark. Null
    /// when none is available. The caller tracks it again when a new result takes it.
    /// </summary>
    internal static T[]? TakeFree<T>(int length)
    {
        if (Volatile.Read(ref s_count) == 0) return null;
        if (Stress)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            return TakeFreeOnce<T>(length);
        }

        var array = TakeFreeOnce<T>(length);
        if (array is not null) return array;

        // A collection since the last look (natural or induced) restarts the fresh-bytes count.
        int gen0 = GC.CollectionCount(0);
        if (gen0 != Volatile.Read(ref s_lastGen0))
        {
            Volatile.Write(ref s_lastGen0, gen0);
            Interlocked.Exchange(ref s_freshSinceCollect, 0);
        }
        long bytes = (long)length * Unsafe.SizeOf<T>();
        if (Interlocked.Add(ref s_freshSinceCollect, bytes) < (CollectBudgetBytes << Volatile.Read(ref s_budgetShift))) return null;

        int generation = Volatile.Read(ref s_collectGeneration);
        GC.Collect(generation, GCCollectionMode.Forced, blocking: true);
        Interlocked.Increment(ref InducedCollections);
        Volatile.Write(ref s_lastGen0, GC.CollectionCount(0));
        Interlocked.Exchange(ref s_freshSinceCollect, 0);
        array = TakeFreeOnce<T>(length);
        // Freed nothing of this size: the dead owners may be older than gen 0, so the next collection includes gen 1, and
        // waits twice as long. Never gen 2: a full collection costs too much in a large heap (an inference loop whose
        // results all escaped paid 0.35 ms per call that way) and the GC runs gen 2 on its own schedule anyway.
        if (array is not null)
        {
            Volatile.Write(ref s_collectGeneration, 0);
            Volatile.Write(ref s_budgetShift, 0);
        }
        else
        {
            Volatile.Write(ref s_collectGeneration, 1);
            Volatile.Write(ref s_budgetShift, Math.Min(5, Volatile.Read(ref s_budgetShift) + 1));
        }
        TrimFree();
        return array;
    }

    private static T[]? TakeFreeOnce<T>(int length)
    {
        Entry? found = null;
        lock (s_lock)
        {
            if (!s_bySize.TryGetValue((typeof(T), length), out var group)) return null;
            for (int i = group.Count - 1; i >= 0; i--)
            {
                var candidate = group[i];
                if (candidate.Owner.Target is not null) continue;
                RemoveLocked(candidate);
                // Raw access that is not a lease reached it: whatever holds that access may still be using it, so it
                // is left to the GC, never reused.
                if (candidate.State.Escaped) { Interlocked.Increment(ref EscapedArrays); continue; }
                found = candidate;
                break;
            }
        }
        if (found is null) return null;

        var array = (T[])found.Array;
        // The previous result's device copies and pending downloads are keyed by this array; drop them before reuse.
        PooledArrayRecycling.Notify(array);
        HostSync.ClearReleased(array);
        if (Stress) Poison(array);
        Interlocked.Increment(ref ReusedArrays);
        return array;
    }

    /// <summary>Lets the GC have free arrays beyond <see cref="FreeCapBytes"/> (largest first).</summary>
    private static void TrimFree()
    {
        lock (s_lock)
        {
            long free = 0;
            List<Entry>? candidates = null;
            foreach (var entry in s_byArray.Values)
            {
                if (entry.Owner.Target is not null) continue;
                free += entry.Bytes;
                (candidates ??= new List<Entry>()).Add(entry);
            }
            if (free <= FreeCapBytes || candidates is null) return;
            candidates.Sort((x, y) => y.Bytes.CompareTo(x.Bytes));
            foreach (var entry in candidates)
            {
                if (free <= FreeCapBytes) break;
                free -= entry.Bytes;
                RemoveLocked(entry);
            }
        }
    }

    private static void RemoveLocked(Entry entry)
    {
        entry.Owner.Free();
        entry.Group.Remove(entry);
        s_byArray.Remove(entry.Array);
        s_trackedBytes -= entry.Bytes;
        s_count--;
    }

    private static void Poison(Array array)
    {
        switch (array)
        {
            case float[] f: f.AsSpan().Fill(float.NaN); break;
            case double[] d: d.AsSpan().Fill(double.NaN); break;
            // No poison value for integer types; zero at least erases the previous result's data.
            default: Array.Clear(array, 0, array.Length); break;
        }
    }

    private static bool EnvTrue(string name)
    {
        var value = Environment.GetEnvironmentVariable(name);
        return value is not null && (value == "1" || value.Equals("true", StringComparison.OrdinalIgnoreCase));
    }

    private static long EnvLong(string name, long fallback)
        => long.TryParse(Environment.GetEnvironmentVariable(name), out var value) && value >= 0 ? value : fallback;
}
