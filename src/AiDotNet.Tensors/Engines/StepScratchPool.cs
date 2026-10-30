// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// One compiled plan's step scratch: the stable device buffers its actions draw their outputs and transient scratch
/// from inside a compiled step (keyed by action and allocation order, so the pre-pass, the capture and every replay
/// see the same addresses), and the cursor of the action running now.
/// </summary>
/// <remarks>
/// <para>
/// Owned by the plan, entered by the thread running its step body (<see cref="DirectGpuTensorEngine.EnterScratchPool"/>).
/// It used to be one pool per ENGINE with a per-engine cursor, and every plan on the engine shared it: two plans' action
/// 5 got the same buffer, so two captured graphs baked in and wrote the same memory, a size mismatch freed a buffer the
/// other plan's graph still replayed, and a second thread training on the same engine moved the running step's cursor.
/// Concurrent GPU Autoformer training faulted with CUDA 700 once its step started capturing (AiDotNet#1804).
/// </para>
/// <para>
/// The pool frees nothing itself when its plan retires: tensors that outlive the plan (a result, a gradient view) may
/// still be bound to its buffers, so each buffer is released by its finalizer once nothing references it.
/// </para>
/// </remarks>
internal sealed class StepScratchPool
{
    // Every pooled buffer, whichever pool holds it: a tensor wrapping one does not own it (ReleaseDeadDeviceStorage).
    private static readonly ConditionalWeakTable<IGpuBuffer, StepScratchPool> s_pooled = new();

    internal StepScratchPool(DirectGpuTensorEngine engine) => Engine = engine;

    /// <summary>The engine whose compiled steps draw from this pool.</summary>
    internal DirectGpuTensorEngine Engine { get; }

    /// <summary>Output slots by (action, local allocation index).</summary>
    internal Dictionary<long, IGpuBuffer> Slots { get; } = new();

    /// <summary>The (action, local) key of each output slot, so a first-write gradient copy can name its slot.</summary>
    internal Dictionary<IGpuBuffer, long> Keys { get; } = new();

    /// <summary>Transient scratch by (local index within the action, size class), shared by the plan's actions.</summary>
    internal Dictionary<(int Local, long SizeClass), IGpuBuffer> Transient { get; } = new();

    /// <summary>The compiled action running now, or -1 outside one (the pre-pass, the optimizer).</summary>
    internal int Action { get; set; } = -1;

    internal int Local { get; set; }

    internal int TransientLocal { get; set; }

    /// <summary>Positions the cursor at the start of <paramref name="action"/>.</summary>
    internal void Begin(int action)
    {
        Action = action;
        Local = 0;
        TransientLocal = 0;
    }

    internal static bool IsPooled(IGpuBuffer buffer) => s_pooled.TryGetValue(buffer, out _);

    internal void Track(IGpuBuffer buffer)
    {
        s_pooled.Remove(buffer);
        s_pooled.Add(buffer, this);
    }

    internal static void Untrack(IGpuBuffer buffer) => s_pooled.Remove(buffer);

    /// <summary>
    /// Drops output slot <paramref name="key"/> without freeing its buffer: a tensor may still be bound to it, so its
    /// finalizer frees it once nothing references it. The next rent at that key allocates afresh. Returns its bytes.
    /// </summary>
    internal long Forget(long key)
    {
        if (!Slots.TryGetValue(key, out var buffer)) return 0;
        Slots.Remove(key);
        Keys.Remove(buffer);
        Untrack(buffer);
        return buffer.SizeInBytes;
    }

    /// <summary>
    /// The first compiled action whose output slots are leased (<see cref="Lease"/>) rather than kept for the plan's life,
    /// or <see cref="int.MaxValue"/> when none are. The plan sets it to its first backward action on a backend without
    /// graph capture: a backward output is consumed inside the action that writes it, and keeping every one of them
    /// for the plan's life made a GPU step's footprint the sum of its backward outputs, not the most live at once.
    /// </summary>
    internal int FirstLeasedAction { get; set; } = int.MaxValue;

    /// <summary>Slots once seen outliving their action (<see cref="EndLeases"/>): kept for good from then on.</summary>
    private readonly HashSet<long> _pinned = new();
    private readonly List<IGpuBuffer> _free = new();
    private readonly List<(long Key, IGpuBuffer Buffer)> _leases = new();
    // Buffers leased since the last TrimUnleased: a free buffer outside it served no lease in the last run.
    private readonly HashSet<IGpuBuffer> _leasedThisRun = new();

    /// <summary>Whether slot <paramref name="key"/> of the running action is leased.</summary>
    internal bool Leases(long key) => Action >= FirstLeasedAction && !_pinned.Contains(key);

    /// <summary>The device buffers the pool holds: kept slots, the transient arena and free leased buffers.</summary>
    internal int BufferCount => Slots.Count + Transient.Count + _free.Count + _leases.Count;

    /// <summary>The bytes of those buffers.</summary>
    internal long BufferBytes
    {
        get
        {
            long bytes = 0;
            foreach (var b in Slots.Values) bytes += b.SizeInBytes;
            foreach (var b in Transient.Values) bytes += b.SizeInBytes;
            foreach (var b in _free) bytes += b.SizeInBytes;
            foreach (var (_, b) in _leases) bytes += b.SizeInBytes;
            return bytes;
        }
    }

    /// <summary>Whether any slot is leased now.</summary>
    internal bool HasLeases => _leases.Count > 0;

    /// <summary>
    /// Leases slot <paramref name="key"/> a buffer of at least <paramref name="length"/> elements: the smallest free one
    /// within twice the length, else a new one. The same actions lease the same sizes in the same order every step, so
    /// the assignment repeats.
    /// </summary>
    internal IGpuBuffer Lease(long key, IDirectGpuBackend backend, int length)
    {
        int pick = -1;
        for (int i = 0; i < _free.Count; i++)
        {
            var candidate = _free[i];
            if (candidate.Handle == IntPtr.Zero || candidate.Size < length || candidate.Size > 2L * length) continue;
            if (pick < 0 || candidate.Size < _free[pick].Size) pick = i;
        }
        IGpuBuffer buffer;
        if (pick >= 0)
        {
            buffer = _free[pick];
            _free[pick] = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
        }
        else
        {
            buffer = backend.AllocateBuffer(length);
            Track(buffer);
        }
        _leases.Add((key, buffer));
        _leasedThisRun.Add(buffer);
        Keys[buffer] = key;
        return buffer;
    }

    /// <summary>
    /// Ends the running action's leases. A slot whose buffer is in <paramref name="escaped"/> (still referenced after
    /// the action: a parked gradient contribution, an accumulator) keeps it for good; every other buffer is free for
    /// the next lease.
    /// </summary>
    internal void EndLeases(HashSet<IGpuBuffer>? escaped)
    {
        foreach (var (key, buffer) in _leases)
        {
            if (escaped is not null && escaped.Contains(buffer))
            {
                _pinned.Add(key);
                Slots[key] = buffer;
            }
            else
            {
                _free.Add(buffer);
            }
        }
        _leases.Clear();
    }

    /// <summary>
    /// After a whole executed run: drops the free buffers that run leased none of (the first run leases a buffer for
    /// every backward output, and the slots it then redirects into their accumulators never lease again). Like
    /// <see cref="Forget"/>, a dropped buffer is freed by its finalizer once nothing references it. Returns their bytes.
    /// </summary>
    internal long TrimUnleased()
    {
        long dropped = 0;
        for (int i = _free.Count - 1; i >= 0; i--)
        {
            var buffer = _free[i];
            if (_leasedThisRun.Contains(buffer)) continue;
            Keys.Remove(buffer);
            Untrack(buffer);
            dropped += buffer.SizeInBytes;
            _free[i] = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
        }
        _leasedThisRun.Clear();
        return dropped;
    }

    /// <summary>Frees every buffer now. Only for a caller that knows no captured graph or tensor still uses them.</summary>
    internal void Release()
    {
        EndLeases(null);
        foreach (var buffer in Slots.Values) { Untrack(buffer); try { buffer.Dispose(); } catch (ObjectDisposedException) { } }
        foreach (var buffer in Transient.Values) { Untrack(buffer); try { buffer.Dispose(); } catch (ObjectDisposedException) { } }
        foreach (var buffer in _free) { Untrack(buffer); try { buffer.Dispose(); } catch (ObjectDisposedException) { } }
        Slots.Clear();
        Keys.Clear();
        Transient.Clear();
        _free.Clear();
        _leasedThisRun.Clear();
        Begin(-1);
    }
}
