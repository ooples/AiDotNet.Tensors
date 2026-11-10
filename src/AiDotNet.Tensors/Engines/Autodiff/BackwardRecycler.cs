using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Hands the buffers a backward pass rents back to the step's <see cref="TensorArena"/> as soon as no gradient
/// references them, so the next rent of the same size reuses them inside the same step.
/// </summary>
/// <remarks>
/// <para><b>Why.</b> An arena rewinds only at the step boundary, so a training step's backing memory was the sum of
/// every gradient and backward temporary it rented. Measured on AiDotNet's FasterRCNN (320x320, CPU): 785 MB of arena
/// rentals per step, 530 MB of them in backward, all retained until the step ended. PyTorch frees each gradient once
/// backward has consumed it, so its peak is the most that is live at one time.</para>
/// <para><b>What is dead.</b> In the reverse walk a node's output gradient is final when that node's backward runs and
/// no later node reads it, so the walk drops it from the gradient map (as the streaming backward does). A buffer
/// rented during backward is then dead once no gradient left in the map shares its backing array: a backward kernel's
/// temporaries, and gradients the walk has consumed. Forward activations were rented before backward and are never
/// considered. The loss, the requested sources and retained tensors keep their gradients.</para>
/// <para><b>When.</b> Only for a CPU backward that returns requested sources (the full map would hand every
/// intermediate gradient to the caller), without higher-order recording, indexed gradients or a nested recycler, and
/// only inside an arena. A gradient that is lazy or not array-backed stops recycling for the rest of the pass, because
/// what it reads cannot be seen. <c>AIDOTNET_BACKWARD_RECYCLE=0</c> turns it off;
/// <c>AIDOTNET_BACKWARD_RECYCLE_STRESS=1</c> poisons each dead buffer with NaN instead of reusing it, so code that
/// still reads one fails loudly in tests, and throws at the end of the pass if one was written (a second owner, such as
/// a cache that pooled the buffer).</para>
/// </remarks>
internal sealed class BackwardRecycler<T>
{
    private static readonly bool s_enabled =
        Environment.GetEnvironmentVariable("AIDOTNET_BACKWARD_RECYCLE") is not ("0" or "false" or "False");

    /// <summary>Test override for whether recycling runs; null follows <c>AIDOTNET_BACKWARD_RECYCLE</c>.</summary>
    internal static bool? EnabledOverride;

    /// <summary>Test override for the poisoning mode; null follows <c>AIDOTNET_BACKWARD_RECYCLE_STRESS</c>.</summary>
    internal static bool? StressOverride;

    private static readonly bool s_stress =
        Environment.GetEnvironmentVariable("AIDOTNET_BACKWARD_RECYCLE_STRESS") is "1" or "true" or "True";

    [ThreadStatic] private static bool t_active;

    private readonly TensorArena _arena;
    private readonly bool _stress;
    private readonly List<TensorArena.IssuedBuffer> _issued = new();
    private readonly HashSet<Array> _live = new(ReferenceEqualityComparer<Array>.Instance);
    private readonly HashSet<Tensor<T>> _keep = new(ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly HashSet<Tensor<T>> _seen = new(ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly HashSet<Tensor<T>> _repeated = new(ReferenceEqualityComparer<Tensor<T>>.Instance);
    private readonly List<Array>? _poisoned;
    private bool _stopped;

    /// <summary>Buffers returned to the arena (or poisoned) by this pass. Diagnostic.</summary>
    internal int ReturnedCount { get; private set; }

    /// <summary>Buffers the last finished pass on this thread returned (or poisoned), so tests can see it engaged.</summary>
    [ThreadStatic] internal static int LastPassReturned;

    private BackwardRecycler(TensorArena arena, bool stress)
    {
        _arena = arena;
        _stress = stress;
        if (stress) _poisoned = new List<Array>();
    }

    /// <summary>
    /// Starts recycling for one backward pass, or returns null when this pass cannot recycle. Call
    /// <see cref="NoteOutput"/> for every step's output before the walk, <see cref="AfterStep"/> after each step that
    /// ran, and <see cref="End"/> in a finally.
    /// </summary>
    internal static BackwardRecycler<T>? Begin(IEngine engine, Tensor<T> loss,
        IReadOnlyList<Tensor<T>>? sources, IReadOnlyCollection<Tensor<T>>? retained)
    {
        if (!(EnabledOverride ?? s_enabled) || t_active || sources is null || engine.SupportsGpu
            || DifferentiableOps._isBackwardCreateGraph || DifferentiableOps._indexedGrads is not null
            || TensorArena.Current is not { } arena)
            return null;
        var recycler = new BackwardRecycler<T>(arena, StressOverride ?? s_stress);
        // t_active is per element type; the arena's log is per arena, so a pass of another type may already hold it.
        if (!arena.OpenIssueLog(recycler._issued)) return null;
        recycler._keep.Add(loss);
        foreach (var source in sources.Where(s => s is not null))
            recycler._keep.Add(source);
        if (retained is not null)
            foreach (var tensor in retained) recycler._keep.Add(tensor);
        t_active = true;
        return recycler;
    }

    /// <summary>
    /// Records a step's output. An output produced by more than one step (an in-place op recorded twice) keeps its
    /// gradient for the whole pass, since a later step may still read it.
    /// </summary>
    internal void NoteOutput(Tensor<T> output)
    {
        if (!_seen.Add(output)) _repeated.Add(output);
    }

    /// <summary>Drops the consumed output's gradient and returns every backward buffer no gradient still references.</summary>
    internal void AfterStep(Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> consumed)
    {
        if (_stopped) return;
        if (!_keep.Contains(consumed) && !_repeated.Contains(consumed) && grads.Remove(consumed))
            consumed.Grad = null;
        if (_issued.Count == 0) return;

        _live.Clear();
        foreach (var gradient in grads.Values)
        {
            if (gradient.LazySource is not null
                || !gradient._storage.TryGetBackingArrayIdentity(out var array, out _) || array is null)
            {
                Stop();
                return;
            }
            _live.Add(array);
        }

        for (int i = _issued.Count - 1; i >= 0; i--)
        {
            var buffer = _issued[i];
            if (_live.Contains(buffer.Array)) continue;
            if (_poisoned is not null)
            {
                ResultBufferTracker.Poison(buffer.Array);
                _poisoned.Add(buffer.Array);
            }
            else _arena.ReturnIssued(buffer);
            ReturnedCount++;
            int last = _issued.Count - 1;
            _issued[i] = _issued[last];
            _issued.RemoveAt(last);
        }
    }

    // Stops returning buffers for the rest of the pass; those already returned were proven dead.
    private void Stop()
    {
        _stopped = true;
        _arena.CloseIssueLog(_issued);
        _issued.Clear();
    }

    // A poisoned buffer that no longer holds NaN was written after it was proven dead: something else owns it too (a
    // cache that pooled it), and reusing it would have put one buffer behind two tensors.
    private static void ThrowIfPoisonOverwritten(List<Array> poisoned)
    {
        foreach (var array in poisoned)
        {
            bool intact = array switch
            {
                float[] f => Array.TrueForAll(f, float.IsNaN),
                double[] d => Array.TrueForAll(d, double.IsNaN),
                _ => true,
            };
            if (!intact)
                throw new InvalidOperationException(
                    $"BackwardRecycler stress: a {array.Length}-element {(array.GetType().GetElementType() ?? array.GetType()).Name} buffer was "
                    + "written after backward proved it dead; another owner still holds it.");
        }
    }

    // The per-thread pass state lives in [ThreadStatic] fields: one pass per thread at a time, by design.
    private static void FinishPass(int returned)
    {
        t_active = false;
        LastPassReturned = returned;
    }

    /// <summary>Ends the pass. Buffers still referenced by gradients stay issued until the step boundary.</summary>
    internal void End()
    {
        _arena.CloseIssueLog(_issued);
        _issued.Clear();
        FinishPass(ReturnedCount);
        if (_poisoned is not null) ThrowIfPoisonOverwritten(_poisoned);
    }
}
