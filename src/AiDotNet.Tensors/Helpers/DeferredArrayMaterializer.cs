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

    // Pending downloads held WEAKLY: a step result kept past its tape (the loss, the gradients, a retained tensor).
    // Its device buffer then lives exactly as long as the tensor, as in PyTorch: the closure (and through it the
    // buffer) is reachable only while the key is, so an unread result is freed by the buffer's finalizer when the
    // tensor is collected, and a read downloads it once. The strong registry would have pinned every kept result
    // (and its device memory) until a host read that may never come.
    private sealed class WeakPending
    {
        public readonly Action<object> Callback;
        public WeakPending(Action<object> callback) { Callback = callback; }
    }
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, WeakPending> _weakPending = new();
    private static int _anyWeakPending;

    /// <summary>
    /// Moves <paramref name="array"/>'s pending download (if any) from the strong registry to the weak one, so the
    /// device data it downloads from lives as long as the key does. Returns true when an entry was moved.
    /// </summary>
    internal static bool HoldWeakly(object array)
    {
        if (!_pendingMaterializations.TryRemove(array, out var pending)) return false;
        Interlocked.Decrement(ref _pendingCount);
        Volatile.Write(ref _anyWeakPending, 1);
#if NETFRAMEWORK || NETSTANDARD2_0
        _weakPending.Remove(array);
        _weakPending.Add(array, new WeakPending(pending.Callback));
#else
        _weakPending.AddOrUpdate(array, new WeakPending(pending.Callback));
#endif
        return true;
    }

    private static bool TryTakeWeak(object array, out WeakPending pending)
    {
        pending = null!;
        if (Volatile.Read(ref _anyWeakPending) == 0) return false;
        if (!_weakPending.TryGetValue(array, out var found)) return false;
        _weakPending.Remove(array);
        pending = found;
        return true;
    }

    // Released keys: device data that was freed WITHOUT a host copy (PyTorch-style release of a dead step
    // intermediate). Weakly keyed, so an entry disappears with its array/vector and the table cannot grow with
    // training. A later host read of a released key throws the stored message instead of returning undefined
    // contents. _anyReleased only ever goes 0 -> 1: it keeps the pre-existing zero-cost fast path for processes
    // that never release anything (CPU-only workloads).
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, string> _released = new();
    private static int _anyReleased;

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

    private static long _releaseCount;

    /// <summary>
    /// Pending downloads dropped by a PyTorch-style release since the last reset. Like <see cref="MaterializeCount"/>
    /// it proves a result was GPU-resident: its only copy was on the device.
    /// </summary>
    public static long ReleaseCount => Volatile.Read(ref _releaseCount);

    /// <summary>Resets <see cref="MaterializeCount"/> and <see cref="ReleaseCount"/> to zero (test instrumentation).</summary>
    public static void ResetMaterializeCount()
    {
        Interlocked.Exchange(ref _materializeCount, 0);
        Interlocked.Exchange(ref _releaseCount, 0);
    }

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
        // A key that is registered again holds a new result, so a previous release or weak hold no longer applies.
        if (Volatile.Read(ref _anyReleased) != 0) _released.Remove(array);
        if (Volatile.Read(ref _anyWeakPending) != 0) _weakPending.Remove(array);
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
        if (Volatile.Read(ref _pendingCount) == 0 && Volatile.Read(ref _anyReleased) == 0
            && Volatile.Read(ref _anyWeakPending) == 0)
            return false;

        if (Volatile.Read(ref _pendingCount) != 0 && _pendingMaterializations.TryRemove(array, out var pending))
        {
            Interlocked.Decrement(ref _pendingCount);
            Interlocked.Increment(ref _materializeCount); // a real DtoH download is about to run
            try
            {
                pending.Callback(array);
            }
            catch
            {
                // The download did not happen, so the device copy is still the only valid one: put the
                // registration back before propagating. Dropping it left the host array permanently stale and
                // no longer marked pending. Seen when a host read runs during CUDA-graph capture (the download's
                // stream sync is illegal there, CUDA 900): the capture is abandoned and training continues eagerly,
                // but every later read of that array silently returned its pre-capture contents.
                if (_pendingMaterializations.TryAdd(array, pending))
                    Interlocked.Increment(ref _pendingCount);
                throw;
            }
            return true;
        }
        if (TryTakeWeak(array, out var weak))
        {
            Interlocked.Increment(ref _materializeCount);
            try
            {
                weak.Callback(array);
            }
            catch
            {
                // Not downloaded: the device copy is still the only valid one; keep it held.
#if NETFRAMEWORK || NETSTANDARD2_0
                _weakPending.Remove(array);
                _weakPending.Add(array, weak);
#else
                _weakPending.AddOrUpdate(array, weak);
#endif
                throw;
            }
            return true;
        }
        if (Volatile.Read(ref _anyReleased) != 0 && _released.TryGetValue(array, out var releasedMessage))
            throw new InvalidOperationException(releasedMessage);
        return false;
    }

    /// <summary>
    /// Drops the pending download of <paramref name="array"/> WITHOUT running it, because the device data is being
    /// freed as dead (PyTorch-style release: no host copy of a step intermediate). A later host read of the key
    /// throws <paramref name="message"/> rather than returning undefined contents. Returns true when a pending
    /// download was dropped.
    /// </summary>
    internal static bool Release(object array, string message)
    {
        bool dropped = false;
        if (_pendingMaterializations.TryRemove(array, out _))
        {
            Interlocked.Decrement(ref _pendingCount);
            Interlocked.Increment(ref _releaseCount);
            dropped = true;
        }
        else if (TryTakeWeak(array, out _))
        {
            Interlocked.Increment(ref _releaseCount);
            dropped = true;
        }
        Volatile.Write(ref _anyReleased, 1);
#if NETFRAMEWORK || NETSTANDARD2_0
        _released.Remove(array);
        _released.Add(array, message);
#else
        _released.AddOrUpdate(array, message);
#endif
        return dropped;
    }

    /// <summary>True when <paramref name="array"/>'s device data was released without a host copy.</summary>
    internal static bool IsReleased(object array)
        => Volatile.Read(ref _anyReleased) != 0 && _released.TryGetValue(array, out _);

    // Keys a caller asked to keep past their tape (GradientTape.Retain, and what ComputeGradients returns): the
    // step-end and last-use releases skip them. Weakly keyed, like the release marks.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, object> _retained = new();
    private static int _anyRetained;

    /// <summary>Exempts <paramref name="array"/> from PyTorch-style release of dead step intermediates.</summary>
    internal static void MarkRetained(object array)
    {
        Volatile.Write(ref _anyRetained, 1);
#if NETFRAMEWORK || NETSTANDARD2_0
        _retained.Remove(array);
        _retained.Add(array, array);
#else
        _retained.AddOrUpdate(array, array);
#endif
    }

    /// <summary>True when <paramref name="array"/> was exempted from release (see <see cref="MarkRetained"/>).</summary>
    internal static bool IsRetained(object array)
        => Volatile.Read(ref _anyRetained) != 0 && _retained.TryGetValue(array, out _);

    /// <summary>Throws the release message when <paramref name="array"/>'s device data was released.</summary>
    internal static void ThrowIfReleased(object array)
    {
        if (Volatile.Read(ref _anyReleased) != 0 && _released.TryGetValue(array, out var releasedMessage))
            throw new InvalidOperationException(releasedMessage);
    }

    /// <summary>Clears a release mark: the key's contents were fully rewritten on the host.</summary>
    internal static void ClearReleased(object array)
    {
        if (Volatile.Read(ref _anyReleased) != 0) _released.Remove(array);
    }

    /// <summary>
    /// Checks if the given array has a pending deferred download.
    /// </summary>
    internal static bool IsPending(object array)
    {
        if (Volatile.Read(ref _pendingCount) != 0 && _pendingMaterializations.ContainsKey(array)) return true;
        return Volatile.Read(ref _anyWeakPending) != 0 && _weakPending.TryGetValue(array, out _);
    }

    /// <summary>
    /// Removes a pending materialization without executing it (e.g., when the GPU buffer is reused).
    /// </summary>
    internal static void Remove(object array)
    {
        if (_pendingMaterializations.TryRemove(array, out _))
            Interlocked.Decrement(ref _pendingCount);
        if (Volatile.Read(ref _anyWeakPending) != 0) _weakPending.Remove(array);
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
