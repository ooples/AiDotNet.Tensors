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

    // Forward activations (GradientTapeOptions.ReleaseActivationsDuringBackward): the forward's rentals, the last step
    // that reads each backing array, and the arrays the caller keeps.
    private readonly TensorArena.ForwardActivationLog? _forward;
    private readonly Dictionary<Array, int>? _lastUse;
    private readonly HashSet<Array>? _kept;
    private readonly HashSet<object>? _visited;
    private List<TensorArena.IssuedBuffer>?[]? _forwardReleases;
    private bool _forwardBlind;
    private int _stepCount;

    /// <summary>Buffers returned to the arena (or poisoned) by this pass. Diagnostic.</summary>
    internal int ReturnedCount { get; private set; }

    /// <summary>Forward activations this pass returned to the arena (or poisoned). Diagnostic.</summary>
    internal int ForwardReturnedCount { get; private set; }

    /// <summary>Forward activations the last finished pass on this thread returned (or poisoned).</summary>
    [ThreadStatic] internal static int LastPassForwardReturned;

    /// <summary>
    /// True when this pass also hands back forward activations: call <see cref="NoteStep"/> for every step, in walk
    /// order, before the walk.
    /// </summary>
    internal bool TracksForward => _forward is not null;

    /// <summary>Buffers the last finished pass on this thread returned (or poisoned), so tests can see it engaged.</summary>
    [ThreadStatic] internal static int LastPassReturned;

    private BackwardRecycler(TensorArena arena, bool stress, TensorArena.ForwardActivationLog? forward)
    {
        _arena = arena;
        _stress = stress;
        if (stress) _poisoned = new List<Array>();
        if (forward is { Buffers.Count: > 0 })
        {
            _forward = forward;
            _lastUse = new Dictionary<Array, int>(ReferenceEqualityComparer<Array>.Instance);
            _kept = new HashSet<Array>(ReferenceEqualityComparer<Array>.Instance);
            _visited = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        }
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
        var recycler = new BackwardRecycler<T>(arena, StressOverride ?? s_stress, arena.TakeForwardActivationLog());
        recycler._keep.Add(loss);
        foreach (var source in sources.Where(s => s is not null))
            recycler._keep.Add(source);
        if (retained is not null)
            foreach (var tensor in retained) recycler._keep.Add(tensor);
        if (recycler._kept is { } kept)
            foreach (var tensor in recycler._keep)
                if (tensor.GetBackingArrayForCacheLookupUnsafe() is { } array) kept.Add(array);
        t_active = true;
        arena.OpenIssueLog(recycler._issued);
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

    /// <summary>
    /// Records what step <paramref name="step"/> of the walk reads: its output, inputs, saved state and whatever its
    /// backward delegate captured. Only when <see cref="TracksForward"/>.
    /// </summary>
    internal void NoteStep(int step, Tensor<T>? output, Tensor<T>? input0, Tensor<T>? input1, Tensor<T>? input2,
        Tensor<T>[]? inputs, object[]? savedState, Delegate? backward)
    {
        if (_lastUse is null || _forwardBlind) return;
        if (step >= _stepCount) _stepCount = step + 1;
        _visited!.Clear();
        NoteUse(output, step, 0);
        NoteUse(input0, step, 0);
        NoteUse(input1, step, 0);
        NoteUse(input2, step, 0);
        NoteUse(inputs, step, 0);
        NoteUse(savedState, step, 0);
        NoteUse(backward?.Target, step, 0);
    }

    // Marks every array a step can reach as read at that step. Saved state and closures are walked a few levels deep,
    // through tensors of any element type, arrays, saved-state containers and the fields of other objects, so a
    // backward that reads a forward buffer through a captured layer or a raw mask array still counts as a use. A
    // tensor whose value is still deferred could read buffers this cannot name: the pass then keeps every activation.
    private void NoteUse(object? value, int step, int depth)
    {
        if (value is null || _forwardBlind || value is string || value is Type || value is IEngine
            || value is TensorArena || value is System.Reflection.MemberInfo || !_visited!.Add(value))
            return;
        if (value is IBackingArraySource tensor)
        {
            if (!tensor.TryGetBackingArray(out var backing)) { _forwardBlind = true; return; }
            if (backing is not null) _lastUse![backing] = step;
            return;
        }
        if (value is Array array)
        {
            var element = array.GetType().GetElementType();
            if (element is not null && element.IsValueType) { _lastUse![array] = step; return; }
            if (depth < MaxUseDepth)
                foreach (var item in array) NoteUse(item, step, depth + 1);
            return;
        }
        if (depth >= MaxUseDepth) return;
        if (value is ISavedStateTensorContainer container)
        {
            foreach (var item in container.SavedStateValues) NoteUse(item, step, depth + 1);
            return;
        }
        if (value is Delegate del)
        {
            NoteUse(del.Target, step, depth + 1);
            return;
        }
        var type = value.GetType();
        if (type.IsPrimitive || type.IsEnum) return;
        foreach (var field in ReferenceFields(type))
            NoteUse(field.GetValue(value), step, depth + 1);
    }

    private const int MaxUseDepth = 3;

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, System.Reflection.FieldInfo[]> s_referenceFields = new();

    // Instance fields that can hold a reference, through the base types.
    private static System.Reflection.FieldInfo[] ReferenceFields(Type type) => s_referenceFields.GetOrAdd(type, static t =>
    {
        var fields = new List<System.Reflection.FieldInfo>();
        for (var current = t; current is not null && current != typeof(object); current = current.BaseType)
            foreach (var field in current.GetFields(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public
                | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.DeclaredOnly))
                if (!field.FieldType.IsPrimitive && !field.FieldType.IsEnum && !field.FieldType.IsPointer)
                    fields.Add(field);
        return fields.ToArray();
    });

    // Groups the forward's buffers by the walk step after which nothing reads them. A buffer no step reads is left
    // alone: it may be a forward output the caller holds, or scratch some other owner manages.
    private void PlanForwardReleases()
    {
        _forwardReleases = new List<TensorArena.IssuedBuffer>?[_stepCount];
        if (!_forwardBlind)
        {
            var planned = new HashSet<Array>(ReferenceEqualityComparer<Array>.Instance);
            foreach (var buffer in _forward!.Buffers)
            {
                if (!planned.Add(buffer.Array) || _kept!.Contains(buffer.Array)) continue;
                if (!_lastUse!.TryGetValue(buffer.Array, out int last)) continue;
                (_forwardReleases[last] ??= new List<TensorArena.IssuedBuffer>()).Add(buffer);
            }
        }
        _lastUse!.Clear();
        _visited!.Clear();
    }

    /// <summary>
    /// Drops the consumed output's gradient and returns every backward buffer no gradient still references, then the
    /// forward activations step <paramref name="step"/> was the last to read. Call it for every step of the walk,
    /// including the ones that had no gradient to propagate.
    /// </summary>
    internal void AfterStep(Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> consumed, int step)
    {
        if (_stopped) return;
        if (!_keep.Contains(consumed) && !_repeated.Contains(consumed) && grads.Remove(consumed))
            consumed.Grad = null;
        if (_forward is not null && _forwardReleases is null) PlanForwardReleases();
        var forwardDead = _forwardReleases is { } releases && (uint)step < (uint)releases.Length ? releases[step] : null;
        if (_issued.Count == 0 && forwardDead is null) return;

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
            Release(buffer);
            ReturnedCount++;
            int last = _issued.Count - 1;
            _issued[i] = _issued[last];
            _issued.RemoveAt(last);
        }

        if (forwardDead is null) return;
        foreach (var buffer in forwardDead)
        {
            // A backward that wrote a gradient into an activation's buffer made it a gradient: it stays.
            if (_live.Contains(buffer.Array)) continue;
            Release(buffer);
            ForwardReturnedCount++;
            _forward!.Released = true;
        }
    }

    private void Release(in TensorArena.IssuedBuffer buffer)
    {
        if (_poisoned is not null)
        {
            ResultBufferTracker.Poison(buffer.Array);
            _poisoned.Add(buffer.Array);
        }
        else _arena.ReturnIssued(buffer);
    }

    // Stops returning buffers for the rest of the pass; those already returned were proven dead.
    private void Stop()
    {
        _stopped = true;
        _arena.CloseIssueLog();
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
    private static void FinishPass(int returned, int forwardReturned)
    {
        t_active = false;
        LastPassReturned = returned;
        LastPassForwardReturned = forwardReturned;
    }

    /// <summary>Ends the pass. Buffers still referenced by gradients stay issued until the step boundary.</summary>
    internal void End()
    {
        _arena.CloseIssueLog();
        _issued.Clear();
        FinishPass(ReturnedCount, ForwardReturnedCount);
        if (_poisoned is not null) ThrowIfPoisonOverwritten(_poisoned);
    }
}
