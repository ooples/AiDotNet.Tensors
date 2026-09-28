using System;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The engine's device entries for its storages (buffer, shape, backend of a result or upload), replacing the
/// engine-wide activation-cache dictionary.
/// </summary>
/// <remarks>
/// <para>An entry lives ON its storage (<see cref="HostSync.DeviceEntry"/>): a lookup is a field read on the
/// storage's state, the host array and the vector over it resolve to the same entry, and the entry lives exactly as
/// long as the storage -- an unreferenced result's buffer is freed by its finalizer, with no global table keeping it
/// alive and no LRU deciding what to offload.</para>
/// <para>Enumeration (the step-end release of a tape or compiled step, the transient drops) walks a per-FLOW ledger
/// of weak references to the entries created on that flow, so a thread only ever enumerates and frees its own
/// step's buffers -- another thread's in-flight buffers are never visible to it. The flow is inherited by the
/// Parallel/Task workers a step fans out to.</para>
/// </remarks>
internal sealed class DeviceEntryRegistry
{
    private sealed class Ledger
    {
        public readonly List<(WeakReference<object> Key, long Timestamp)> Items = new();
        public int AddsSincePrune;
    }

    private readonly AsyncLocal<Ledger?> _ledger = new();

    private Ledger CurrentLedger => _ledger.Value ??= new Ledger();

    private ActivationCacheEntry? OwnEntry(HostSync? sync)
        => sync?.DeviceEntry is ActivationCacheEntry entry && ReferenceEquals(entry.Owner, this) ? entry : null;

    /// <summary>This engine's entry for <paramref name="key"/> (a host array, vector or matrix).</summary>
    public bool TryGetValue(object key, out ActivationCacheEntry entry)
    {
        var found = OwnEntry(HostSync.Find(key));
        entry = found!;
        return found is not null;
    }

    public bool ContainsKey(object key) => OwnEntry(HostSync.Find(key)) is not null;

    /// <summary>Binds <paramref name="entry"/> to <paramref name="key"/>'s storage; false when this engine already has one.</summary>
    public bool TryAdd(object key, ActivationCacheEntry entry)
    {
        var sync = HostSync.For(key);
        lock (sync)
        {
            if (OwnEntry(sync) is not null) return false;
            entry.Owner = this;
            sync.DeviceEntry = entry;
        }
        var ledger = CurrentLedger;
        lock (ledger)
        {
            ledger.Items.Add((new WeakReference<object>(key), entry.Timestamp));
            if (++ledger.AddsSincePrune >= 4096) PruneUnsafe(ledger);
        }
        return true;
    }

    /// <summary>
    /// Replaces <paramref name="key"/>'s entry in place (an FP16/FP32 re-encode of the same data). The replacement keeps
    /// the original timestamp, so the flow ledger's item for it stays valid.
    /// </summary>
    public ActivationCacheEntry this[object key]
    {
        set
        {
            var sync = HostSync.For(key);
            lock (sync)
            {
                value.Owner = this;
                sync.DeviceEntry = value;
            }
        }
    }

    /// <summary>Unbinds this engine's entry for <paramref name="key"/> (the caller disposes it).</summary>
    public bool TryRemove(object key, out ActivationCacheEntry entry)
    {
        entry = null!;
        var sync = HostSync.Find(key);
        if (sync is null) return false;
        lock (sync)
        {
            var own = OwnEntry(sync);
            if (own is null) return false;
            sync.DeviceEntry = null;
            entry = own;
            return true;
        }
    }

    /// <summary>The live entries created on the CURRENT flow, oldest first.</summary>
    public KeyValuePair<object, ActivationCacheEntry>[] ToArray()
    {
        var ledger = _ledger.Value;
        if (ledger is null) return Array.Empty<KeyValuePair<object, ActivationCacheEntry>>();
        lock (ledger)
        {
            PruneUnsafe(ledger);
            // Re-check each item as it is read: a storage can be collected (or its entry unbound) after the prune.
            var result = new List<KeyValuePair<object, ActivationCacheEntry>>(ledger.Items.Count);
            foreach (var item in ledger.Items)
            {
                if (!item.Key.TryGetTarget(out var key)) continue;
                if (OwnEntry(HostSync.Find(key)) is { } entry && entry.Timestamp == item.Timestamp)
                    result.Add(new KeyValuePair<object, ActivationCacheEntry>(key, entry));
            }
            return result.ToArray();
        }
    }

    /// <summary>True when the current flow has no live entries.</summary>
    public bool IsEmpty => ToArray().Length == 0;

    /// <summary>Number of live entries created on the current flow.</summary>
    public int Count => ToArray().Length;

    /// <summary>Unbinds every live entry of the current flow (the caller disposed them).</summary>
    public void Clear()
    {
        foreach (var pair in ToArray()) TryRemove(pair.Key, out _);
        var ledger = _ledger.Value;
        if (ledger is not null) lock (ledger) ledger.Items.Clear();
    }

    // Drops ledger items whose storage died, was unbound, or was re-bound to a newer entry (a newer item exists).
    private void PruneUnsafe(Ledger ledger)
    {
        ledger.AddsSincePrune = 0;
        ledger.Items.RemoveAll(item =>
            !item.Key.TryGetTarget(out var key)
            || OwnEntry(HostSync.Find(key)) is not { } entry
            || entry.Timestamp != item.Timestamp);
    }
}
