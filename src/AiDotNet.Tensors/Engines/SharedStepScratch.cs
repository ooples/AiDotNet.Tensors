// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Step scratch that compiled plans on one engine co-own: a buffer one plan's step uses only within the step (written
/// before it is read on every run, and read by nothing between runs) is offered to the engine's other plans, and a plan
/// allocating a new slot takes one instead of allocating. Two plans whose steps never overlap -- a GAN's discriminator
/// and generator, each captured as its own graph -- then hold the larger working set, not the sum, as one caching
/// allocator would give PyTorch.
/// </summary>
/// <remarks>
/// <para>Sound because every co-owner treats the memory as transient: a step re-zeroes a slot it does not fully write
/// before using it (StepScratchPool's alloc-zero invariant, captured as a memset node), so nothing one plan left there
/// is read by another, and the plan publishing a buffer has proved that nothing outside its step reads it (the
/// parameters, gradients, inputs, loss, retained outputs, in-place state and optimizer buffers are excluded: the
/// protection the cross-plan workspace broker applies). Steps run one at a time per engine (CompiledStepLock), and a
/// buffer is shared only between plans stepping on the thread that published it, so their work is ordered on that
/// thread's stream.</para>
/// <para>A shared buffer stays allocated while the engine lives: one owner's graph may replay into it after another
/// plan retires, so no owner frees it, and it stays marked as step scratch so no tensor bound to it can free it.</para>
/// </remarks>
internal sealed class SharedStepScratch
{
    /// <summary>Process switch (AIDOTNET_GPU_SHARE_STEP_SCRATCH=0 turns it off); tests compare both.</summary>
    internal static bool Enabled { get; set; } = Environment.GetEnvironmentVariable("AIDOTNET_GPU_SHARE_STEP_SCRATCH") != "0";

    private static readonly ConditionalWeakTable<IGpuBuffer, object> s_shared = new();
    private static readonly object s_marker = new();

    private sealed class Entry
    {
        internal Entry(IGpuBuffer buffer, IDirectGpuBackend backend, int thread)
        {
            Buffer = buffer;
            Backend = backend;
            Thread = thread;
        }

        internal IGpuBuffer Buffer { get; }
        internal IDirectGpuBackend Backend { get; }
        internal int Thread { get; }
        internal HashSet<StepScratchPool> Owners { get; } = new(Autodiff.ReferenceEqualityComparer<StepScratchPool>.Instance);
    }

    private readonly object _lock = new();
    private readonly Dictionary<IGpuBuffer, Entry> _entries = new(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);

    /// <summary>Whether <paramref name="buffer"/> is co-owned step scratch (never disposed, untracked or lent by an owner).</summary>
    internal static bool IsShared(IGpuBuffer buffer) => s_shared.TryGetValue(buffer, out _);

    /// <summary>Buffers offered, and how many times a plan took one instead of allocating (test hooks).</summary>
    internal int Count { get { lock (_lock) return _entries.Count; } }

    internal int Takes { get; private set; }

    internal long TakenBytes { get; private set; }

    /// <summary>
    /// Offers <paramref name="owner"/>'s buffers to the engine's other plans stepping on this thread. Only buffers the
    /// caller has proved transient to its step.
    /// </summary>
    internal void Publish(StepScratchPool owner, IDirectGpuBackend backend, IEnumerable<IGpuBuffer> buffers)
    {
        int thread = Environment.CurrentManagedThreadId;
        lock (_lock)
        {
            foreach (var buffer in buffers)
            {
                if (buffer.Handle == IntPtr.Zero) continue;
                if (!_entries.TryGetValue(buffer, out var entry))
                {
                    entry = new Entry(buffer, backend, thread);
                    _entries[buffer] = entry;
                    s_shared.GetValue(buffer, static _ => s_marker);
                }
                entry.Owners.Add(owner);
            }
        }
    }

    /// <summary>
    /// A shared buffer of at least <paramref name="length"/> elements (the smallest within twice it) that
    /// <paramref name="taker"/> does not already hold, now co-owned by it; or null. Its contents are another plan's.
    /// </summary>
    internal IGpuBuffer? TryTake(StepScratchPool taker, IDirectGpuBackend backend, int length,
        Func<IGpuBuffer, bool>? held = null)
    {
        int thread = Environment.CurrentManagedThreadId;
        lock (_lock)
        {
            Entry? pick = null;
            foreach (var entry in _entries.Values)
            {
                var buffer = entry.Buffer;
                if (!ReferenceEquals(entry.Backend, backend) || entry.Thread != thread || entry.Owners.Contains(taker)
                    || buffer.Handle == IntPtr.Zero || buffer.Size < length || buffer.Size > 2L * length
                    || held?.Invoke(buffer) == true)
                    continue;
                if (pick is null || buffer.Size < pick.Buffer.Size) pick = entry;
            }
            if (pick is null) return null;
            pick.Owners.Add(taker);
            Takes++;
            TakenBytes += pick.Buffer.SizeInBytes;
            return pick.Buffer;
        }
    }

    /// <summary>
    /// <paramref name="keeper"/>'s step keeps a value in <paramref name="buffer"/> past the step (it took the buffer for a
    /// slot that turned out to hold its loss, a retained output or other persistent state): every other owner stops using
    /// it (StepScratchPool.Revoke, which makes its plan recapture), and it is no longer shared -- the keeper's alone.
    /// </summary>
    internal void Revoke(IGpuBuffer buffer, StepScratchPool keeper)
    {
        lock (_lock)
        {
            if (!_entries.TryGetValue(buffer, out var entry)) return;
            foreach (var owner in entry.Owners)
                if (!ReferenceEquals(owner, keeper)) owner.Revoke(buffer);
            _entries.Remove(buffer);
            s_shared.Remove(buffer);
            Revocations++;
        }
    }

    /// <summary>How many buffers a taker revoked (test hook).</summary>
    internal int Revocations { get; private set; }

    /// <summary>
    /// <paramref name="owner"/> no longer uses <paramref name="buffer"/>. True when another plan still co-owns it (the
    /// caller must neither free nor untrack it); false when it was not shared or this was its last owner, which leaves it
    /// to the caller as before.
    /// </summary>
    internal bool Drop(StepScratchPool owner, IGpuBuffer buffer)
    {
        lock (_lock)
        {
            if (!_entries.TryGetValue(buffer, out var entry)) return false;
            entry.Owners.Remove(owner);
            if (entry.Owners.Count > 0) return true;
            _entries.Remove(buffer);
            s_shared.Remove(buffer);
            return false;
        }
    }
}
