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
    /// finalizer frees it once nothing references it. The next rent at that key allocates afresh.
    /// </summary>
    internal void Forget(long key)
    {
        if (!Slots.TryGetValue(key, out var buffer)) return;
        Slots.Remove(key);
        Keys.Remove(buffer);
        Untrack(buffer);
    }

    /// <summary>Frees every buffer now. Only for a caller that knows no captured graph or tensor still uses them.</summary>
    internal void Release()
    {
        foreach (var buffer in Slots.Values) { Untrack(buffer); try { buffer.Dispose(); } catch (ObjectDisposedException) { } }
        foreach (var buffer in Transient.Values) { Untrack(buffer); try { buffer.Dispose(); } catch (ObjectDisposedException) { } }
        Slots.Clear();
        Keys.Clear();
        Transient.Clear();
        Begin(-1);
    }
}
