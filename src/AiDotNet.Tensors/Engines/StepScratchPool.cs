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
    /// or <see cref="int.MaxValue"/> when none are. The plan sets it to its first backward action: a backward output is
    /// consumed inside the action that writes it, and keeping every one of them for the plan's life made a GPU step's
    /// footprint the sum of its backward outputs, not the most live at once. A captured step graph bakes in the
    /// assignment of the run it recorded; it is replayed in stream order, so the reuse it recorded stays valid.
    /// </summary>
    internal int FirstLeasedAction { get; set; } = int.MaxValue;

    /// <summary>Slots once seen outliving their action (<see cref="EndLeases"/>): kept for good from then on.</summary>
    private readonly HashSet<long> _pinned = new();
    // Forward output slots leased across actions (SpanLease): from the action that writes them to their last forward
    // reader. Keyed to the slot they were converted from, and held now by buffer.
    private readonly HashSet<long> _spanKeys = new();
    private readonly Dictionary<IGpuBuffer, long> _spanning = new();
    // Bumped by every change to which slots lease: a pin, a slot converted to a span lease.
    private int _assignmentChanges;
    private readonly List<IGpuBuffer> _free = new();
    private readonly List<(long Key, IGpuBuffer Buffer)> _leases = new();
    // Buffers leased since the last TrimUnleased: a free buffer outside it served no lease in the last run.
    private readonly HashSet<IGpuBuffer> _leasedThisRun = new();

    /// <summary>
    /// How many times the lease assignment has changed: a slot pinned, a forward slot turned into a span lease. It only
    /// grows. A captured step graph bakes in the lease assignment of the run it recorded, so a change first seen after
    /// that run means the graph may have handed a buffer to a later action while it was still live, or keeps one the
    /// next run leases elsewhere. The plan compares this count against the one at capture and re-captures.
    /// </summary>
    internal int PinnedCount => _assignmentChanges;

    /// <summary>Whether slot <paramref name="key"/> of the running action is leased.</summary>
    internal bool Leases(long key) => (Action >= FirstLeasedAction || _spanKeys.Contains(key)) && !_pinned.Contains(key);

    /// <summary>The device buffers the pool holds: kept slots, the transient arena and free leased buffers.</summary>
    internal int BufferCount => Slots.Count + Transient.Count + _free.Count + _leases.Count + _spanning.Count;

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
            foreach (var b in _spanning.Keys) bytes += b.SizeInBytes;
            return bytes;
        }
    }

    /// <summary>
    /// Set by the plan around a forward action that writes an output it will span-lease: a slot that action allocates for
    /// the first time takes a free buffer (<see cref="TryTakeFree"/>) instead of a new one.
    /// </summary>
    internal bool ReuseFreeForNewSlots { get; set; }

    /// <summary>
    /// Takes the smallest free buffer of at least <paramref name="length"/> elements, within twice the length, for a slot
    /// allocated for the first time; false when none fits. Nothing holds a free buffer, so the slot owns it from now on.
    /// </summary>
    internal bool TryTakeFree(int length, out IGpuBuffer buffer)
    {
        int pick = -1;
        for (int i = 0; i < _free.Count; i++)
        {
            var candidate = _free[i];
            if (candidate.Handle == IntPtr.Zero || candidate.Size < length || candidate.Size > 2L * length) continue;
            if (pick < 0 || candidate.Size < _free[pick].Size) pick = i;
        }
        if (pick < 0)
        {
            buffer = null!;
            return false;
        }
        buffer = _free[pick];
        _free[pick] = _free[_free.Count - 1];
        _free.RemoveAt(_free.Count - 1);
        return true;
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
        if (_spanKeys.Contains(key)) _spanning[buffer] = key;   // a forward output: held until its last reader
        else _leases.Add((key, buffer));
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
                // A lent forward slot that escaped its lease becomes this slot's for good: its forward slot takes a new
                // buffer on the next run (a buffer cannot be both, or a later lease could take it while it is pinned).
                if (_lent.TryGetValue(buffer, out long forwardKey))
                {
                    _lent.Remove(buffer);
                    if (Slots.TryGetValue(forwardKey, out var bound) && ReferenceEquals(bound, buffer)) Slots.Remove(forwardKey);
                }
                if (_pinned.Add(key)) _assignmentChanges++;
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
    /// Whether <paramref name="buffer"/> is a forward output's span lease (<see cref="SpanLease"/>) held now.
    /// </summary>
    internal bool IsSpanLeased(IGpuBuffer buffer) => _spanning.ContainsKey(buffer);

    /// <summary>
    /// Leases kept slot <paramref name="key"/> across actions from now on: a forward output no backward reads, needed
    /// only until its last forward reader, as PyTorch frees such an intermediate once it is consumed. Its buffer is held
    /// now (the slot's first run kept it) and returns to the free list at <see cref="EndSpanLease"/>; later runs lease
    /// the slot a free buffer instead. Returns false when <paramref name="key"/> is not a kept slot.
    /// </summary>
    internal bool SpanLease(long key)
    {
        if (_pinned.Contains(key) || !Slots.TryGetValue(key, out var buffer)) return false;
        Slots.Remove(key);
        _spanKeys.Add(key);
        _spanning[buffer] = key;
        _leasedThisRun.Add(buffer);
        _assignmentChanges++;
        return true;
    }

    /// <summary>Ends a forward output's span lease: its last reader ran, so the buffer is free for the next lease.</summary>
    internal void EndSpanLease(IGpuBuffer buffer)
    {
        if (_spanning.Remove(buffer)) _free.Add(buffer);
    }

    /// <summary>
    /// A span-leased forward output turned out to be read past its forward (a tensor still bound to its buffer is read
    /// later): its slot keeps the buffer for good from now on, as a backward slot that escapes its action does.
    /// </summary>
    internal void PinSpanLease(IGpuBuffer buffer)
    {
        if (!_spanning.TryGetValue(buffer, out long key)) return;
        _spanning.Remove(buffer);
        _spanKeys.Remove(key);
        _pinned.Add(key);
        Slots[key] = buffer;
        _assignmentChanges++;
    }

    // Forward slots lent to the free list for the rest of the backward (Lend), by buffer, with their forward key; withdrawn
    // when it ends.
    private readonly Dictionary<IGpuBuffer, long> _lent = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);

    /// <summary>
    /// Lends forward slots no later backward action reads to the free list, so the backward's leases reuse them. Each
    /// stays its slot's: <see cref="WithdrawLent"/> takes it back before the next forward writes it. A slot no longer
    /// bound to the buffer the schedule recorded is skipped.
    /// </summary>
    internal void Lend(List<(long Key, IGpuBuffer Buffer)> slots)
    {
        foreach (var (key, buffer) in slots)
        {
            if (!Slots.TryGetValue(key, out var bound) || !ReferenceEquals(bound, buffer) || buffer.Handle == IntPtr.Zero)
                continue;
            if (_lent.ContainsKey(buffer)) continue;
            _free.Add(buffer);
            _lent[buffer] = key;
        }
    }

    /// <summary>Takes every lent forward slot back off the free list (one a lease still holds ended with its action).</summary>
    internal void WithdrawLent()
    {
        if (_lent.Count == 0) return;
        _free.RemoveAll(_lent.ContainsKey);
        // A lease names its slot in Keys; the buffer is its forward slot's again.
        foreach (var entry in _lent) Keys[entry.Key] = entry.Value;
        _lent.Clear();
    }

    /// <summary>Ends every span lease still held (the forward is over: nothing reads its forward-only outputs again).</summary>
    internal void EndSpanLeases()
    {
        if (_spanning.Count == 0) return;
        foreach (var buffer in _spanning.Keys) _free.Add(buffer);
        _spanning.Clear();
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
        EndSpanLeases();
        foreach (var buffer in Slots.Values) DisposeReleased(buffer);
        foreach (var buffer in Transient.Values) DisposeReleased(buffer);
        foreach (var buffer in _free) DisposeReleased(buffer);
        Slots.Clear();
        Keys.Clear();
        Transient.Clear();
        _free.Clear();
        _leasedThisRun.Clear();
        Begin(-1);
    }

    private static void DisposeReleased(IGpuBuffer buffer)
    {
        Untrack(buffer);
        try { buffer.Dispose(); }
        catch (ObjectDisposedException ex)
        {
            // Released already, with its backend (an engine disposed before the plan): nothing is left to free.
            System.Diagnostics.Trace.TraceInformation($"Step scratch buffer already released: {ex.ObjectName}");
        }
    }
}
