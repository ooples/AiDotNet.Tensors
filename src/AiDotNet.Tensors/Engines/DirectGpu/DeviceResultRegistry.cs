// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Device-resident storage a releasing engine can move to the host before it frees the backend.</summary>
internal interface IDeviceResultStorage
{
    /// <summary>
    /// When this storage's device copy lives on <paramref name="backend"/>: downloads it if the host copy is stale, then
    /// frees the device buffer and unbinds it, so the host array is the only copy. Otherwise does nothing.
    /// </summary>
    void EvacuateToHost(IDirectGpuBackend backend);
}

/// <summary>
/// The device results a GPU engine handed out (results whose tensors own their buffers), held weakly, so releasing the
/// engine can download every one still alive before it frees the backend: a result read after the release returns its
/// values instead of faulting on a freed buffer. A collected result costs nothing here beyond its slot until the next
/// compaction.
/// </summary>
/// <remarks>
/// <para>Registration is on the result path of every op, so it touches only the calling thread's shard: an
/// uncontended lock and one weak GC handle, no shared cache line between threads running ops in parallel. Each shard
/// compacts (frees the handles of collected results) when it doubles past its live count, so it stays proportional to
/// the results alive, amortized O(1) per registration.</para>
/// <para>The weak handles are freed by <see cref="EvacuateAll"/>, by compaction, or by the shard's finalizer when an
/// engine is dropped without being disposed.</para>
/// </remarks>
internal sealed class DeviceResultRegistry
{
    private const int MinCompactAt = 64;

    private sealed class Shard
    {
        internal GCHandle[] Handles = new GCHandle[MinCompactAt];
        internal IDirectGpuBackend?[] Backends = new IDirectGpuBackend?[MinCompactAt];
        internal int Count;
        internal int CompactAt = MinCompactAt;

        ~Shard()
        {
            for (int i = 0; i < Count; i++)
                if (Handles[i].IsAllocated) Handles[i].Free();
        }
    }

    private readonly ThreadLocal<Shard> _shards = new(() => new Shard(), trackAllValues: true);

    [ThreadStatic] private static DeviceResultRegistry? t_lastRegistry;
    [ThreadStatic] private static Shard? t_lastShard;

    /// <summary>Records <paramref name="storage"/>, whose device copy was just bound on <paramref name="backend"/>.</summary>
    public void Add(IDeviceResultStorage storage, IDirectGpuBackend backend)
    {
        var shard = ReferenceEquals(t_lastRegistry, this) ? t_lastShard! : ShardForThisThread(this);
        lock (shard)
        {
            if (shard.Count == shard.CompactAt) Compact(shard);
            if (shard.Count == shard.Handles.Length)
            {
                Array.Resize(ref shard.Handles, shard.Count * 2);
                Array.Resize(ref shard.Backends, shard.Count * 2);
            }
            shard.Handles[shard.Count] = GCHandle.Alloc(storage, GCHandleType.Weak);
            shard.Backends[shard.Count] = backend;
            shard.Count++;
        }
    }

    // Static: it fills the thread's one-entry cache, which is per thread rather than per registry.
    private static Shard ShardForThisThread(DeviceResultRegistry registry)
    {
        var shard = registry._shards.Value!;
        t_lastRegistry = registry;
        t_lastShard = shard;
        return shard;
    }

    private static void Compact(Shard shard)
    {
        int live = 0;
        for (int i = 0; i < shard.Count; i++)
        {
            if (shard.Handles[i].Target is null)
            {
                shard.Handles[i].Free();
                continue;
            }
            shard.Handles[live] = shard.Handles[i];
            shard.Backends[live] = shard.Backends[i];
            live++;
        }
        for (int i = live; i < shard.Count; i++)
        {
            shard.Handles[i] = default;
            shard.Backends[i] = null;
        }
        shard.Count = live;
        shard.CompactAt = Math.Max(MinCompactAt, live * 2);
    }

    /// <summary>The live entries across every thread's shard (a diagnostic for tests).</summary>
    internal int CountLive()
    {
        int live = 0;
        foreach (var shard in _shards.Values)
            lock (shard)
                for (int i = 0; i < shard.Count; i++)
                    if (shard.Handles[i].Target is not null) live++;
        return live;
    }

    /// <summary>The slots in use, live or collected-but-not-yet-compacted (a diagnostic for tests).</summary>
    internal int CountSlots()
    {
        int slots = 0;
        foreach (var shard in _shards.Values)
            lock (shard)
                slots += shard.Count;
        return slots;
    }

    /// <summary>
    /// Moves every live result's device copy to the host (see <see cref="IDeviceResultStorage.EvacuateToHost"/>) and
    /// frees every handle. Called by the releasing engine after its ops have drained and before it frees the backend.
    /// A result whose download fails keeps its error for the read that would have needed it; the rest still move.
    /// </summary>
    public int EvacuateAll()
    {
        int moved = 0;
        foreach (var shard in _shards.Values)
        {
            lock (shard)
            {
                for (int i = 0; i < shard.Count; i++)
                {
                    if (shard.Handles[i].Target is IDeviceResultStorage storage && shard.Backends[i] is { } backend)
                    {
                        try
                        {
                            storage.EvacuateToHost(backend);
                            moved++;
                        }
                        catch (Exception ex) when (ex is InvalidOperationException or ObjectDisposedException
                                                       or NotSupportedException)
                        {
                            Trace.TraceWarning($"[DirectGpu] A device result could not be downloaded before the " +
                                               $"engine was released: {ex.Message}");
                        }
                    }
                    shard.Handles[i].Free();
                    shard.Handles[i] = default;
                    shard.Backends[i] = null;
                }
                shard.Count = 0;
                shard.CompactAt = MinCompactAt;
            }
        }
        return moved;
    }
}
