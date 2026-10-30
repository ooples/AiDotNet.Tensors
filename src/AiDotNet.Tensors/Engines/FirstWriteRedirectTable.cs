// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// A compiled plan's first-write redirects: pooled scratch slots (compiled action, local allocation index) whose output
/// is a gradient accumulator's first contribution of the step, each mapped to that accumulator's device buffer. While the
/// plan's captured step body runs, the engine hands such a slot the accumulator itself, so the op writes the gradient in
/// place and the first-write copy issues nothing.
/// </summary>
/// <remarks>
/// <para>A slot is redirected only from evidence of a full uncaptured run of the body in which its output was copied into
/// the accumulator as that accumulator's first write AND was consumed by no other accumulation. A contribution with a
/// second consumer (another accumulator, or the same one again) must stay a separate buffer: written in place, a later
/// contribution to the accumulator would change what the second consumer reads. A slot once seen with a second consumer
/// is blocked for good, including when the second use shows up in a run where it was already redirected (its
/// contribution is then the accumulator's own buffer).</para>
/// <para>The backward runs in a fixed order, so the slot that first-writes an accumulator is the same every step. The
/// table belongs to the plan and is lent to the engine only while the plan's body runs.</para>
/// </remarks>
internal sealed class FirstWriteRedirectTable
{
    private readonly Dictionary<long, IGpuBuffer> _redirects = new();
    private readonly HashSet<long> _blocked = new();
    private readonly Dictionary<long, (IGpuBuffer Contribution, IGpuBuffer Accumulator)> _offers = new();
    // Uses by (buffer, backward action): pooled gradients share one device buffer across tenants whose lifetimes do not
    // overlap, so one buffer is the accumulator of several redirected slots in different actions. Counted per buffer alone,
    // every such slot looked consumed twice and was blocked. A contribution is consumed in the action that produced it.
    private readonly Dictionary<(IGpuBuffer Buffer, int Action), int> _uses = new();

    /// <summary>Number of redirected slots (diagnostics and tests).</summary>
    internal int Count => _redirects.Count;

    /// <summary>Starts a run of the body: forgets the previous run's offers and use counts.</summary>
    internal void BeginRun()
    {
        _offers.Clear();
        _uses.Clear();
    }

    /// <summary>Counts one accumulation in compiled action <paramref name="action"/> consuming <paramref name="contribution"/>'s
    /// device buffer.</summary>
    internal void NoteContribution(IGpuBuffer? contribution, int action)
    {
        if (contribution is null) return;
        var key = (contribution, action);
        _uses[key] = _uses.TryGetValue(key, out int n) ? n + 1 : 1;
    }

    private static int ActionOf(long slot) => (int)(slot >> 32);

    /// <summary>A slot's output was just copied into an accumulator as its first write: a candidate for this run.</summary>
    internal void Offer(long slot, IGpuBuffer contribution, IGpuBuffer accumulator)
    {
        if (!_blocked.Contains(slot)) _offers[slot] = (contribution, accumulator);
    }

    /// <summary>The redirected slots.</summary>
    internal IEnumerable<long> RedirectedSlots => _redirects.Keys;

    internal bool TryGetTarget(long slot, out IGpuBuffer target) => _redirects.TryGetValue(slot, out target!);

    internal void Drop(long slot) => _redirects.Remove(slot);

    /// <summary>
    /// Ends a run of the body. After an executed (uncaptured) run, commits the offers whose contribution had no other
    /// consumer, and blocks every slot whose buffer had a second one. A captured run only recorded, so it changes nothing.
    /// </summary>
    internal void EndRun(bool executed)
    {
        if (!executed) return;
        foreach (var kv in _offers)
        {
            if (_uses.TryGetValue((kv.Value.Contribution, ActionOf(kv.Key)), out int n) && n == 1) _redirects[kv.Key] = kv.Value.Accumulator;
            else { _blocked.Add(kv.Key); _redirects.Remove(kv.Key); }
        }
        List<long>? reused = null;
        foreach (var kv in _redirects)
            if (_uses.TryGetValue((kv.Value, ActionOf(kv.Key)), out int n) && n > 1) (reused ??= new()).Add(kv.Key);
        if (reused is not null)
            foreach (long slot in reused) { _blocked.Add(slot); _redirects.Remove(slot); }
    }

    /// <summary>Forgets every redirect (the eager step, or the pooled slots were freed). Blocked slots stay blocked.</summary>
    internal void Clear()
    {
        _redirects.Clear();
        _offers.Clear();
        _uses.Clear();
    }
}
