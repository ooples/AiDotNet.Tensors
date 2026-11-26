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
internal sealed partial class StepScratchPool
{
    // Every pooled buffer, whichever pool holds it: a tensor wrapping one does not own it (ReleaseDeadDeviceStorage).
    private static readonly ConditionalWeakTable<IGpuBuffer, object> s_pooled = new();
    private static readonly object s_pooledMarker = new();

    internal StepScratchPool(DirectGpuTensorEngine engine) => Engine = engine;

    /// <summary>The engine whose compiled steps draw from this pool.</summary>
    internal DirectGpuTensorEngine Engine { get; }

    /// <summary>Output slots by (action, local allocation index).</summary>
    internal Dictionary<long, IGpuBuffer> Slots { get; } = new();

    /// <summary>The (action, local) key of each output slot, so a first-write gradient copy can name its slot.</summary>
    internal Dictionary<IGpuBuffer, long> Keys { get; } = new();

    /// <summary>
    /// Transient scratch shared by the plan's actions: by (local index within the action, size class), or, where views
    /// are free (<see cref="GrowableTransientClass"/>), one slot per local index grown to the largest request.
    /// </summary>
    internal Dictionary<(int Local, long SizeClass), IGpuBuffer> Transient { get; } = new();

    /// <summary>The size-class key of a local index's single growable transient slot.</summary>
    internal const long GrowableTransientClass = -1;

    /// <summary>
    /// Whether a captured graph of this pool's plan may name its buffers: set before each capture begins, cleared once
    /// the plan's graphs are destroyed. While it is set, a transient slot that grows keeps its old buffer (retired, not
    /// freed); clearing it frees the retired ones.
    /// </summary>
    internal bool GraphsHoldBuffers
    {
        get => _graphsHoldBuffers;
        set
        {
            _graphsHoldBuffers = value;
            if (value) return;
            foreach (var buffer in _retiredTransient) DisposeReleased(buffer);
            _retiredTransient.Clear();
        }
    }

    private bool _graphsHoldBuffers;
    private readonly List<IGpuBuffer> _retiredTransient = new();

    /// <summary>Transient buffers replaced by a larger one while a graph could name them (tests, diagnostics).</summary>
    internal int RetiredTransientCount => _retiredTransient.Count;

    /// <summary>
    /// A growable transient slot moved to a larger buffer: the old one is freed now (its release is stream-ordered, so the
    /// kernels already queued on it finish first) unless a captured graph may name it.
    /// </summary>
    internal void RetireTransient(IGpuBuffer old)
    {
        if (_graphsHoldBuffers) _retiredTransient.Add(old);
        else DisposeReleased(old);
    }

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
        // Ownership lives in Slots/free lists. The marker only prevents tensor/cache owners from disposing scratch.
        // Moving backing between a plan and the broker must not allocate a new weak-table entry or retain its old pool.
        s_pooled.GetValue(buffer, static _ => s_pooledMarker);
    }

    // A co-owned buffer (SharedStepScratch) stays step scratch while any plan may use it.
    internal static void Untrack(IGpuBuffer buffer)
    {
        if (!SharedStepScratch.IsShared(buffer)) s_pooled.Remove(buffer);
    }

    /// <summary>
    /// The co-owned step scratch this pool may take a new buffer from (set by its plan for the step body when the plan
    /// can co-own), or null.
    /// </summary>
    internal SharedStepScratch? Shared
    {
        get => _shared;
        set
        {
            if (NeverCoOwn) value = null;
            _shared = value;
            if (value is not null) _sharedRegistry = value;
        }
    }

    private SharedStepScratch? _shared;
    // The registry this pool ever took from or offered to: told whenever the pool stops using a buffer.
    private SharedStepScratch? _sharedRegistry;

    /// <summary>
    /// This pool stops using <paramref name="buffer"/>. True when another plan still co-owns it: the caller must neither
    /// free nor untrack it. Otherwise it is untracked, left to the caller as before.
    /// </summary>
    internal bool ReleaseOwnership(IGpuBuffer buffer)
    {
        if (_sharedRegistry?.Drop(this, buffer) == true) return true;
        s_pooled.Remove(buffer);
        return false;
    }

    private readonly List<IGpuBuffer> _takenThisRun = new();
    private readonly List<IGpuBuffer> _revoked = new();
    private readonly List<long> _keysScratch = new();
    private readonly List<(int Local, long SizeClass)> _transientScratch = new();

    /// <summary>The co-owned buffers this pool took since the last call (the run's takes), for its plan to validate.</summary>
    internal List<IGpuBuffer> TakeTakenThisRun()
    {
        var taken = new List<IGpuBuffer>(_takenThisRun);
        _takenThisRun.Clear();
        return taken;
    }

    /// <summary>
    /// Another co-owner turned out to keep a value in <paramref name="buffer"/> past its step: this pool stops using it at
    /// once, without freeing it (it is that owner's now), and its plan must drop a captured graph that names it before
    /// stepping again (<see cref="TakeRevoked"/>). Called between steps (they run one at a time per engine).
    /// </summary>
    internal void Revoke(IGpuBuffer buffer)
    {
        _keysScratch.Clear();
        foreach (var slot in Slots) if (ReferenceEquals(slot.Value, buffer)) _keysScratch.Add(slot.Key);
        foreach (long key in _keysScratch) { Slots.Remove(key); _pinned.Remove(key); _spanKeys.Remove(key); }
        _transientScratch.Clear();
        foreach (var entry in Transient) if (ReferenceEquals(entry.Value, buffer)) _transientScratch.Add(entry.Key);
        foreach (var key in _transientScratch) Transient.Remove(key);
        Keys.Remove(buffer);
        _free.RemoveAll(b => ReferenceEquals(b, buffer));
        _leases.RemoveAll(lease => ReferenceEquals(lease.Buffer, buffer));
        _spanning.Remove(buffer);
        _lent.Remove(buffer);
        _leasedThisRun.Remove(buffer);
        _takenThisRun.RemoveAll(b => ReferenceEquals(b, buffer));
        _revoked.Add(buffer);
    }

    /// <summary>Set once this pool's plan ran a body nested inside another plan's: it never co-owns again.</summary>
    internal bool NeverCoOwn { get; private set; }

    /// <summary>
    /// This pool's plan is running nested inside another plan's step, which co-owned memory cannot survive (the outer step
    /// may hold live values in it). It leaves co-ownership for good: every buffer another plan still co-owns is dropped
    /// here without being freed (this run allocates afresh, and the plan recaptures, see <see cref="TakeRevoked"/>); one
    /// no other plan owns stays this pool's alone.
    /// </summary>
    internal void LeaveSharing()
    {
        NeverCoOwn = true;
        _shared = null;
        if (_sharedRegistry is not { } registry) return;
        var held = new List<IGpuBuffer>();
        foreach (var buffer in AllBuffers()) if (SharedStepScratch.IsShared(buffer)) held.Add(buffer);
        foreach (var buffer in held)
            if (registry.Drop(this, buffer)) Revoke(buffer);
    }

    /// <summary>Whether a co-owned buffer was revoked from this pool since the last call (its captured graph names it).</summary>
    internal bool TakeRevoked()
    {
        bool any = _revoked.Count != 0;
        _revoked.Clear();
        return any;
    }

    private bool Holds(IGpuBuffer buffer)
    {
        foreach (var held in AllBuffers()) if (ReferenceEquals(held, buffer)) return true;
        return false;
    }

    /// <summary>
    /// A new buffer of at least <paramref name="length"/> elements for this pool: one the engine's other plans co-own
    /// when <paramref name="mayCoOwn"/> and <see cref="Shared"/> offers one (zeroed unless the caller overwrites it all,
    /// as a fresh allocation is), else a fresh allocation. Tracked as step scratch either way.
    /// </summary>
    /// <remarks>A taken buffer is recorded (<see cref="TakeTakenThisRun"/>): another plan writes it on its own steps, so
    /// this plan's run must prove it holds nothing read after the step, or revoke the other owners'
    /// (SharedStepScratch.Revoke) before any of them steps again.</remarks>
    internal IGpuBuffer AllocateNew(IDirectGpuBackend backend, int length, bool fullyWritten)
    {
        // Never one this pool still holds elsewhere (a forgotten slot's buffer still on the free list): two of its live
        // values would share memory.
        if (Shared?.TryTake(this, backend, length, Holds) is { } shared)
        {
            _takenThisRun.Add(shared);
            if (!fullyWritten)
            {
                if (backend is IDeviceZeroFill zero) zero.ZeroBuffer(shared, length);
                else backend.Fill(shared, 0f, length);
            }
            Track(shared);
            return shared;
        }
        var fresh = backend.AllocateBuffer(length);
        Track(fresh);
        return fresh;
    }

    /// <summary>Every buffer the pool holds now (kept slots, the transient arena, free and leased buffers).</summary>
    internal IEnumerable<IGpuBuffer> AllBuffers()
    {
        foreach (var b in Slots.Values) yield return b;
        foreach (var b in Transient.Values) yield return b;
        foreach (var b in _retiredTransient) yield return b;
        foreach (var b in _free) yield return b;
        foreach (var (_, b) in _leases) yield return b;
        foreach (var b in _spanning.Keys) yield return b;
    }

    /// <summary>
    /// Drops output slot <paramref name="key"/> without freeing its buffer: a tensor may still be bound to it, so its
    /// finalizer frees it once nothing references it. The next rent at that key allocates afresh. Returns its bytes.
    /// </summary>
    internal long Forget(long key)
    {
        if (!Slots.TryGetValue(key, out var buffer)) return 0;
        Slots.Remove(key);
        Keys.Remove(buffer);
        ReleaseOwnership(buffer);
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

    // Set only for a permanently uncaptured plan's executing body. Other plans never see this route.
    internal CompiledTransientWorkspaceBroker? WorkspaceBroker { get; set; }
    internal bool HasOutstandingWorkspaceLeases => _leases.Count != 0 || _spanning.Count != 0;
    private readonly List<long> _workspaceTransferKeys = new();
    private readonly HashSet<IGpuBuffer> _workspaceTransferred = new();

    internal void CollectTransferableForwardSlots(int forwardActions, Func<IGpuBuffer, bool> eligible,
        HashSet<IGpuBuffer> result)
    {
        foreach (var slot in Slots)
            if ((int)(slot.Key >> 32) < forwardActions && !_pinned.Contains(slot.Key) && eligible(slot.Value)
                && !SharedStepScratch.IsShared(slot.Value))   // co-owned: its other owners keep using it
                result.Add(slot.Value);
        // One backing cannot be lent while a pinned/backward/other arena slot still names it.
        foreach (var slot in Slots)
            if ((int)(slot.Key >> 32) >= forwardActions || _pinned.Contains(slot.Key)) result.Remove(slot.Value);
        foreach (var buffer in Transient.Values) result.Remove(buffer);
    }

    /// <summary>
    /// The buffers idle between two runs of a captured step: forward output slots (as
    /// <see cref="CollectTransferableForwardSlots"/>), the transient scratch and the free leased buffers. Every one is
    /// written by the step before the step reads it, so between runs nothing reads it. A buffer held by a lease, a span
    /// lease or a slot outside that set is left out.
    /// </summary>
    internal void CollectIdleBetweenRuns(int forwardActions, Func<IGpuBuffer, bool> eligible, HashSet<IGpuBuffer> result)
    {
        CollectTransferableForwardSlots(forwardActions, eligible, result);
        foreach (var buffer in Transient.Values) if (eligible(buffer)) result.Add(buffer);
        foreach (var buffer in _free) if (eligible(buffer)) result.Add(buffer);
        foreach (var slot in Slots)
            if ((int)(slot.Key >> 32) >= forwardActions || _pinned.Contains(slot.Key)) result.Remove(slot.Value);
        foreach (var (_, buffer) in _leases) result.Remove(buffer);
        foreach (var buffer in _spanning.Keys) result.Remove(buffer);
        foreach (var buffer in _lent.Keys) result.Remove(buffer);
    }

    internal long TransferForwardSlots(HashSet<IGpuBuffer> buffers, IDirectGpuBackend backend,
        CompiledTransientWorkspaceBroker broker, out int transferred)
    {
        _workspaceTransferKeys.Clear();
        _workspaceTransferred.Clear();
        foreach (var slot in Slots) if (buffers.Contains(slot.Value)) _workspaceTransferKeys.Add(slot.Key);
        long bytes = 0;
        foreach (long key in _workspaceTransferKeys)
        {
            var buffer = Slots[key];
            Slots.Remove(key);
            Keys.Remove(buffer);
            _leasedThisRun.Remove(buffer);
            if (!_workspaceTransferred.Add(buffer)) continue;
            bytes += buffer.SizeInBytes;
            broker.Return(backend, buffer);
        }
        transferred = _workspaceTransferred.Count;
        return bytes;
    }


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
    internal bool Leases(long key)
        => (Action >= FirstLeasedAction || _spanKeys.Contains(key) || HoldLeasesIn is not null) && !_pinned.Contains(key);

    /// <summary>
    /// While set, every slot the running action leases is held across actions, as a span lease, and added to this list,
    /// until the plan ends it (<see cref="EndSpanLease"/>) or keeps it (<see cref="PinSpanLease"/>). The plan sets it
    /// around an action of an activation-checkpoint segment: the forward actions whose outputs only the segment reads,
    /// and the actions that recompute them for the segment's backward. Their buffers live for a stretch of actions the
    /// plan knows and the pool does not.
    /// </summary>
    internal List<IGpuBuffer>? HoldLeasesIn { get; set; }

    /// <summary>The device buffers the pool holds: kept slots, the transient arena and free leased buffers.</summary>
    internal int BufferCount => Slots.Count + Transient.Count + _retiredTransient.Count + _free.Count + _leases.Count + _spanning.Count;

    /// <summary>The bytes of those buffers.</summary>
    internal long BufferBytes
    {
        get
        {
            long bytes = 0;
            foreach (var b in Slots.Values) bytes += b.SizeInBytes;
            foreach (var b in Transient.Values) bytes += b.SizeInBytes;
            foreach (var b in _retiredTransient) bytes += b.SizeInBytes;
            foreach (var b in _free) bytes += b.SizeInBytes;
            foreach (var (_, b) in _leases) bytes += b.SizeInBytes;
            foreach (var b in _spanning.Keys) bytes += b.SizeInBytes;
            return bytes;
        }
    }

    /// <summary>
    /// Metadata-only retention snapshot. Each live native handle is assigned to its first owning category, in the
    /// order listed below; borrowed/lent references therefore cannot inflate the total. Does not download or free data.
    /// </summary>
    internal IReadOnlyDictionary<string, (int Count, long Bytes)> SnapshotStorage(int forwardActionCount)
    {
        var result = new Dictionary<string, (int Count, long Bytes)>
        {
            ["slots_forward"] = (0, 0),
            ["slots_backward"] = (0, 0),
            ["transient"] = (0, 0),
            ["free"] = (0, 0),
            ["leases"] = (0, 0),
            ["spanning"] = (0, 0),
        };
        var handles = new HashSet<IntPtr>();
        var handleless = new HashSet<IGpuBuffer>();
        void Add(string category, IGpuBuffer buffer)
        {
            buffer = GpuBufferViewRoot.Of(buffer);
            if (!DirectGpuTensorEngine.IsLiveDeviceBuffer(buffer)) return;
            if (buffer.Handle == IntPtr.Zero ? !handleless.Add(buffer) : !handles.Add(buffer.Handle)) return;
            var previous = result[category];
            result[category] = (previous.Count + 1, previous.Bytes + buffer.SizeInBytes);
        }
        foreach (var slot in Slots)
            if ((int)(slot.Key >> 32) < forwardActionCount) Add("slots_forward", slot.Value);
        foreach (var slot in Slots)
            if ((int)(slot.Key >> 32) >= forwardActionCount) Add("slots_backward", slot.Value);
        foreach (var buffer in Transient.Values) Add("transient", buffer);
        foreach (var buffer in _free) Add("free", buffer);
        foreach (var lease in _leases) Add("leases", lease.Buffer);
        foreach (var buffer in _spanning.Keys) Add("spanning", buffer);
        return result;
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
        BufferAllocationDiagnostics.RecordStepScratchReuse(length);
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
            // The caller zeroes a lease it does not overwrite, so a co-owned buffer's contents never show.
            buffer = AllocateNew(backend, length, fullyWritten: true);
        }
        if (HoldLeasesIn is { } held)
        {
            _spanning[buffer] = key;   // a checkpoint segment's buffer: held until the plan ends or keeps it
            held.Add(buffer);
        }
        else if (_spanKeys.Contains(key)) _spanning[buffer] = key;   // a forward output: held until its last reader
        else _leases.Add((key, buffer));
        _leasedThisRun.Add(buffer);
        Keys[buffer] = key;
        if (pick >= 0) BufferAllocationDiagnostics.RecordStepScratchReuse(length);
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

    /// <summary>Takes every lent forward slot back off the free list, and back from a lease still holding one.</summary>
    internal void WithdrawLent()
    {
        if (_lent.Count == 0) return;
        _free.RemoveAll(_lent.ContainsKey);
        // A backward action that threw has not ended its leases. Ending one that holds a lent buffer after this would
        // put the buffer on the free list while its forward slot still binds it, so a later lease could take it before
        // the forward output's last reader (and the next Lend would add a second copy): it is its forward slot's again.
        _leases.RemoveAll(lease => _lent.ContainsKey(lease.Buffer));
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
    /// Return idle, unescaped backward leases unused by a completed run. Redirected gradient writes
    /// can permanently reduce the set of work buffers following the first backward pass.
    /// </summary>
    internal void TrimUnleased(IDirectGpuBackend backend)
    {
        ReturnFreeLeases(backend, unusedOnly: true);
        _leasedThisRun.Clear();
    }

    /// <summary>Frees every buffer now. Only for a caller that knows no captured graph or tensor still uses them.</summary>
    internal void Release()
    {
        EndLeases(null);
        EndSpanLeases();
        foreach (var buffer in Slots.Values) DisposeReleased(buffer);
        foreach (var buffer in Transient.Values) DisposeReleased(buffer);
        foreach (var buffer in _retiredTransient) DisposeReleased(buffer);
        foreach (var buffer in _free) DisposeReleased(buffer);
        Slots.Clear();
        Keys.Clear();
        Transient.Clear();
        _retiredTransient.Clear();
        _free.Clear();
        _leasedThisRun.Clear();
        Begin(-1);
    }

    private void DisposeReleased(IGpuBuffer buffer)
    {
        if (ReleaseOwnership(buffer)) return;   // another plan still co-owns it
        try { buffer.Dispose(); }
        catch (ObjectDisposedException ex)
        {
            // Released already, with its backend (an engine disposed before the plan): nothing is left to free.
            System.Diagnostics.Trace.TraceInformation($"Step scratch buffer already released: {ex.ObjectName}");
        }
    }
}
