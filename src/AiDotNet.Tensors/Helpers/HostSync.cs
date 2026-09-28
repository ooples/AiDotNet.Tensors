using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace AiDotNet.Tensors.Helpers;

/// <summary>Host storage that can carry a <see cref="HostSync"/> (a vector or matrix over a host array).</summary>
internal interface IHostSyncOwner
{
    /// <summary>This storage's sync state, created (and shared with every alias of its host array) on demand.</summary>
    HostSync GetOrCreateHostSync();

    /// <summary>This storage's sync state if it (or an alias of its host array) has one; never creates.</summary>
    HostSync? FindHostSync();
}

/// <summary>
/// Whether a host array holds current data, and how to make it current when the device holds the only valid copy
/// (PyTorch's rule: a tensor's data lives where it was produced; a host read downloads it once).
/// </summary>
/// <remarks>
/// <para>ONE state per host array, shared by every vector, matrix, tensor or segment view over that array, so an
/// alias can never read stale host data while another alias's device result is pending. The state is found
/// through a weak table keyed by the array (it lives exactly as long as the array) and cached on each owner, so a
/// repeated host read costs a field read, not a global lookup. A storage whose host array is not allocated yet
/// (GPU-resident lazy vector) keys its state by the storage object until the array appears.</para>
/// <para>Replaces the process-wide DeferredArrayMaterializer registry: no global dictionary or lock on the hot
/// path, one lock per storage, held across the download so a concurrent reader waits for the data instead of
/// reading a half-filled array. A newer registration replaces a pending older one (the newest device result is
/// the truth), and the pending download lives as long as the host array (never pins device memory past it).</para>
/// </remarks>
internal sealed class HostSync
{
    private readonly object _lock = new();
    private Action<object>? _pending;   // downloads the device copy into the host array
    private object? _pendingKey;        // the object the download was registered with (its callback's argument)
    private string? _released;          // device data freed without a host copy: host reads throw this
    private bool _retained;             // exempt from PyTorch-style release of dead step intermediates
    private bool _downloading;          // a download is filling the host array: other readers wait on _lock

    // array (or not-yet-backed storage) -> its state. Weak on the key: an entry disappears with its array.
    private static readonly ConditionalWeakTable<object, HostSync> s_byKey = new();
    // 0 until the first state exists: CPU-only processes never touch the table.
    private static int s_any;

    private static long s_materializeCount;
    private static long s_releaseCount;

    /// <summary>Device-to-host downloads performed since the last reset (each one is a real DtoH copy).</summary>
    public static long MaterializeCount => Volatile.Read(ref s_materializeCount);

    /// <summary>Pending downloads dropped by a PyTorch-style release since the last reset.</summary>
    public static long ReleaseCount => Volatile.Read(ref s_releaseCount);

    /// <summary>Resets <see cref="MaterializeCount"/> and <see cref="ReleaseCount"/> (test instrumentation).</summary>
    public static void ResetMaterializeCount()
    {
        Interlocked.Exchange(ref s_materializeCount, 0);
        Interlocked.Exchange(ref s_releaseCount, 0);
    }

    internal static bool AnyExists => Volatile.Read(ref s_any) != 0;

    /// <summary>
    /// Cheap pre-check before an <see cref="IsPending"/> lookup: false guarantees nothing is pending anywhere (no state
    /// exists yet); true only means a per-storage check is needed.
    /// </summary>
    internal static bool HasPendingMaterializations => AnyExists;

    /// <summary>The state shared by every storage over <paramref name="array"/>, created on demand.</summary>
    internal static HostSync ForArray(object array)
    {
        Volatile.Write(ref s_any, 1);
        return s_byKey.GetValue(array, static _ => new HostSync());
    }

    /// <summary>The state of <paramref name="array"/> if one exists.</summary>
    internal static HostSync? FindForArray(object array)
        => AnyExists && s_byKey.TryGetValue(array, out var sync) ? sync : null;

    /// <summary>
    /// Makes <paramref name="array"/> share <paramref name="sync"/> (a lazily backed storage just got its host
    /// array). An array that already has a state keeps it.
    /// </summary>
    internal static HostSync Adopt(object array, HostSync sync)
    {
        Volatile.Write(ref s_any, 1);
        return s_byKey.GetValue(array, _ => sync);
    }

    private static HostSync For(object key)
        => key is IHostSyncOwner owner ? owner.GetOrCreateHostSync() : ForArray(key);

    private static HostSync? Find(object key)
    {
        if (!AnyExists) return null;
        return key is IHostSyncOwner owner ? owner.FindHostSync()
            : s_byKey.TryGetValue(key, out var sync) ? sync : null;
    }

    /// <summary>
    /// Records that the device holds <paramref name="key"/>'s current data: the next host read runs
    /// <paramref name="download"/> (with <paramref name="key"/> as its argument). Replaces a pending older download
    /// and clears a release mark: the key holds a new result.
    /// </summary>
    internal static void Register(object key, Action<object> download)
    {
        var sync = For(key);
        lock (sync._lock)
        {
            sync._pending = download;
            sync._pendingKey = key;
            sync._released = null;
        }
    }

    /// <summary>
    /// Runs <paramref name="key"/>'s pending download, if any; throws the release message when its device data
    /// was freed without a host copy. Returns true when a download ran.
    /// </summary>
    internal static bool TryMaterialize(object key)
    {
        var sync = Find(key);
        return sync is not null && sync.MakeHostCurrent();
    }

    /// <summary>Downloads the device copy into the host array if it is pending (see <see cref="TryMaterialize"/>).</summary>
    internal bool MakeHostCurrent()
    {
        if (Volatile.Read(ref _pending) is null && !Volatile.Read(ref _downloading)
            && Volatile.Read(ref _released) is null) return false;
        lock (_lock)
        {
            var download = _pending;
            if (download is null)
            {
                // Another thread's download finished while this one waited on the lock, or this is the download's
                // own (re-entrant) read of the array it is filling: either way the host array is the current copy.
                if (!_downloading && _released is { } message) throw new InvalidOperationException(message);
                return false;
            }
            var key = _pendingKey!;
            _pending = null;
            _pendingKey = null;
            _downloading = true;
            Interlocked.Increment(ref s_materializeCount);
            try
            {
                download(key);
            }
            catch
            {
                // Not downloaded (e.g. a host read during CUDA-graph capture, where the stream sync is illegal):
                // the device copy is still the only valid one, so it stays pending.
                _pending = download;
                _pendingKey = key;
                throw;
            }
            finally
            {
                _downloading = false;
            }
            return true;
        }
    }

    /// <summary>True when <paramref name="key"/>'s host data is stale (a device download is pending or running).</summary>
    internal static bool IsPending(object key)
    {
        var sync = Find(key);
        // In progress counts as pending: a caller about to free the device buffer then materializes, which waits on
        // the storage lock until the running download has finished reading that buffer.
        return sync is not null && (Volatile.Read(ref sync._pending) is not null || Volatile.Read(ref sync._downloading));
    }

    /// <summary>Drops <paramref name="key"/>'s pending download without running it (its host data was replaced).</summary>
    internal static void Remove(object key)
    {
        var sync = Find(key);
        if (sync is null) return;
        lock (sync._lock)
        {
            sync._pending = null;
            sync._pendingKey = null;
        }
    }

    /// <summary>
    /// Drops <paramref name="key"/>'s pending download because its device data is being freed as dead (no host
    /// copy of a step intermediate); a later host read throws <paramref name="message"/>. Returns true when a
    /// pending download was dropped.
    /// </summary>
    internal static bool Release(object key, string message)
    {
        var sync = For(key);
        lock (sync._lock)
        {
            bool dropped = sync._pending is not null;
            if (dropped) Interlocked.Increment(ref s_releaseCount);
            sync._pending = null;
            sync._pendingKey = null;
            sync._released = message;
            return dropped;
        }
    }

    /// <summary>True when <paramref name="key"/>'s device data was released without a host copy.</summary>
    internal static bool IsReleased(object key) => Find(key) is { } sync && Volatile.Read(ref sync._released) is not null;

    /// <summary>Throws the release message when <paramref name="key"/>'s device data was released.</summary>
    internal static void ThrowIfReleased(object key)
    {
        if (Find(key) is { } sync && Volatile.Read(ref sync._released) is { } message)
            throw new InvalidOperationException(message);
    }

    /// <summary>Clears a release mark: the key's host contents were fully rewritten.</summary>
    internal static void ClearReleased(object key)
    {
        if (Find(key) is { } sync) Volatile.Write(ref sync._released, null);
    }

    /// <summary>Exempts <paramref name="key"/> from release of dead step intermediates (kept past its tape).</summary>
    internal static void MarkRetained(object key)
    {
        var sync = For(key);
        Volatile.Write(ref sync._retained, true);
    }

    /// <summary>True when <paramref name="key"/> was exempted from release (see <see cref="MarkRetained"/>).</summary>
    internal static bool IsRetained(object key) => Find(key) is { } sync && Volatile.Read(ref sync._retained);
}
