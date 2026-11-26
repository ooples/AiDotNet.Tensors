using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Activation checkpointing inside a compiled training plan: the compiled counterpart of <c>torch.utils.checkpoint</c>
/// under <c>torch.compile</c>, where the partitioner puts the recomputation into the backward graph.
/// </summary>
/// <remarks>
/// <para>
/// The trace marks segments (GraphMode.BeginCheckpointSegment). A step of a segment whose output only that segment reads
/// is <i>internal</i>; its forward buffers are released when the segment's forward ends. The internal steps whose values
/// the segment's backward reads (an input or output the backward function uses, or its saved tensors), with the internal
/// steps they are computed from, are the segment's <i>recompute set</i>: just before the segment's first backward action
/// the plan runs their forward actions again, into buffers held until the segment's last backward action. Everything a
/// reader outside the segment needs (the segment's outputs, an auxiliary loss, parameters, in-place state) is kept, as
/// without checkpointing.
/// </para>
/// <para>
/// Both runs are ordinary plan actions, so the recompute is inside the captured CUDA graph and replays with the step;
/// a replay recomputes exactly what the first forward computed, from the same inputs, with the same kernels.
/// </para>
/// <para>
/// State a forward updates in place (a router's selection bias nudged from its loads, a running statistic) has moved on
/// by the time the backward runs. A recomputed step that reads such state reads the value its first forward read: the
/// plan copies the state on the device when the segment's forward starts, swaps that copy in for the recompute, and puts
/// the current value back after it. The in-place updates themselves are not recomputed, so they still happen once per
/// step. A segment that updates a state and then reads it in a step it recomputes is refused: the copy taken at the
/// segment's start would not be what that step read.
/// </para>
/// <para>
/// A released buffer goes back to the step's scratch pool and is leased again, so every tensor of the segment bound to
/// it is unbound at the release: a stale binding would read the next tenant, or have an op that writes into its
/// output's existing device copy overwrite that tenant (measured: wrong gradients in the first, uncaptured step).
/// The host-centric eager step keeps every forward output for the whole step and releases nothing, so it needs no
/// recompute and runs the segments as the plain plan does.
/// </para>
/// </remarks>
internal sealed partial class CompiledTrainingPlan<T>
{
    private CheckpointSchedule? _checkpoint;

    /// <summary>The number of activation-checkpoint segments the plan recomputes in its backward (0 = none).</summary>
    internal int CheckpointSegmentCount => _checkpoint?.RecomputeActions.Count(a => a.Length > 0) ?? 0;

    /// <summary>Forward actions the plan runs again in each step's backward (the sum over its checkpoint segments).</summary>
    internal int CheckpointRecomputedActionCount => _checkpoint?.RecomputeActions.Sum(a => a.Length) ?? 0;

    /// <summary>Forward actions whose buffers a checkpoint segment releases when its forward ends.</summary>
    internal int CheckpointReleasedActionCount => _checkpoint?.SegmentOfForwardAction.Count(s => s >= 0) ?? 0;

    /// <summary>Times the plan has run a segment's recompute (diagnostics and tests; a captured replay is not counted).</summary>
    internal long CheckpointRecomputeRuns => _checkpoint?.RecomputeRuns ?? 0;

    /// <summary>Forward step indices in each segment's recompute set (tests).</summary>
    internal IReadOnlyList<int[]> CheckpointRecomputeSteps => _checkpoint?.RecomputeSteps ?? Array.Empty<int[]>();

    private sealed class CheckpointSchedule
    {
        internal int SegmentCount;
        internal int[] SegmentOfForwardAction = Array.Empty<int>();       // the segment whose buffers it releases, or -1
        internal int[]?[] ReleaseAfterForwardAction = Array.Empty<int[]?>();   // segments checked after the action
        internal int[] LastForwardAction = Array.Empty<int>();                 // per segment: its last forward action
        internal (object Item, int ReleaseAfter)[][] ForwardReads = Array.Empty<(object, int)[]>(); // per segment
        internal int[]?[] StashBeforeForwardAction = Array.Empty<int[]?>();
        // Per forward action: the engine's stochastic seed position when the action's forward started, which its
        // recompute starts from again.
        internal long[] SeedBeforeForwardAction = Array.Empty<long>();
        internal int[][] RecomputeActions = Array.Empty<int[]>();          // per segment, in forward order
        internal int[][] RecomputeSteps = Array.Empty<int[]>();
        internal int[]?[] RecomputeBeforeBackward = Array.Empty<int[]?>();
        internal int[] LastBackward = Array.Empty<int>();
        internal int[] FirstBackward = Array.Empty<int>();                 // per segment: its first backward action
        internal object[][] Keep = Array.Empty<object[]>();               // per segment: tensors read past its forward
        internal object[][] Internal = Array.Empty<object[]>();           // per segment: what its released buffers can back
        internal Tensor<T>[][] State = Array.Empty<Tensor<T>[]>();        // per segment: in-place state its recompute reads
        internal List<IGpuBuffer>[] HeldForward = Array.Empty<List<IGpuBuffer>>();
        internal List<IGpuBuffer>[] HeldRecompute = Array.Empty<List<IGpuBuffer>>();
        internal (object Item, int LastRead)[][] BackwardReads = Array.Empty<(object, int)[]>(); // per segment
        internal List<(IGpuBuffer Buffer, int ReleaseAt)>[] Pending = Array.Empty<List<(IGpuBuffer, int)>>();
        internal IGpuBuffer?[][] Stash = Array.Empty<IGpuBuffer?[]>();
        internal IGpuBuffer?[][] Current = Array.Empty<IGpuBuffer?[]>();
        internal long RecomputeRuns;
    }

    /// <summary>
    /// Builds the checkpoint schedule from the forward steps' segments, or returns null when the trace marked none.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// A segment cannot be recomputed exactly by this plan; the message names why. The plan refuses rather than keep the
    /// activations it was asked to release, or recompute something other than what the forward computed.
    /// </exception>
    private static CheckpointSchedule? BuildCheckpointSchedule(
        List<CompiledStep<T>> forwardSteps, int[] genericActionOfStep, int forwardActionCount,
        List<int[]?> actionCoverage, int backwardActionCount, Tensor<T>[] parameters, Tensor<T> lossOutput,
        HashSet<Tensor<T>> retainedOutputs, IEngine engine)
    {
        int n = forwardSteps.Count;
        int segments = 0;
        for (int i = 0; i < n; i++) segments = Math.Max(segments, forwardSteps[i].CheckpointSegment + 1);
        if (segments == 0) return null;

        if (typeof(T) != typeof(float) || !engine.Is<DirectGpuTensorEngine>())
            throw new NotSupportedException(
                "Activation checkpointing inside a compiled training plan needs a float plan on a DirectGpu engine: a host "
                + "plan's forward outputs are fixed arrays, so releasing and recomputing them saves no memory.");
        if (actionCoverage.Count != backwardActionCount || actionCoverage.Any(c => c is null))
            throw new NotSupportedException(
                "Activation checkpointing inside a compiled training plan needs every backward action to report the forward "
                + "steps it covers (a fused or analytic backward does not), to know where each segment's backward starts.");

        // Every backward action's covered steps (none is missing: refused above).
        var coverage = new int[actionCoverage.Count][];
        for (int a = 0; a < coverage.Length; a++) coverage[a] = actionCoverage[a] ?? Array.Empty<int>();

        var cmp = Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance;
        var seg = new int[n];
        for (int i = 0; i < n; i++) seg[i] = forwardSteps[i].CheckpointSegment;

        // Producers (an in-place write is not a producer: its output is state), state writers and forward readers.
        var producer = new Dictionary<Tensor<T>, int>(cmp);
        var writers = new Dictionary<Tensor<T>, List<int>>(cmp);
        var readerSegments = new Dictionary<Tensor<T>, HashSet<int>>(cmp);
        var inPlace = new bool[n];
        for (int i = 0; i < n; i++)
        {
            var step = forwardSteps[i];
            inPlace[i] = IsInPlaceState(step);
            if (inPlace[i])
            {
                if (!writers.TryGetValue(step.OutputBuffer, out var w)) writers[step.OutputBuffer] = w = new List<int>();
                w.Add(i);
            }
            else if (!producer.ContainsKey(step.OutputBuffer)) producer[step.OutputBuffer] = i;
            foreach (var input in step.Inputs)
            {
                if (input is null) continue;
                if (!readerSegments.TryGetValue(input, out var set)) readerSegments[input] = set = new HashSet<int>();
                set.Add(seg[i]);
            }
        }
        // A second writer of the same output makes it state as well.
        for (int i = 0; i < n; i++)
            if (!inPlace[i] && producer.TryGetValue(forwardSteps[i].OutputBuffer, out int first) && first != i)
                (writers.TryGetValue(forwardSteps[i].OutputBuffer, out var w) ? w : writers[forwardSteps[i].OutputBuffer] = new List<int>()).Add(i);

        var kept = new HashSet<Tensor<T>>(parameters, cmp) { lossOutput };
        kept.UnionWith(retainedOutputs);
        kept.UnionWith(writers.Keys);

        // A step computed from parameters alone (a weight reshaped, flipped or transposed for a kernel) is not
        // checkpointed: the plan hoists it to the start of the forward and runs its backward at the very end of the step,
        // so recomputing it would hold the segment's recompute for the whole backward; its output is weight-sized.
        var parameterOnly = new bool[n];
        var parameterSet = new HashSet<Tensor<T>>(parameters, cmp);
        for (int i = 0; i < n; i++)
            parameterOnly[i] = !inPlace[i] && forwardSteps[i].Inputs.Length > 0 && forwardSteps[i].Inputs.All(input =>
                input is not null && (parameterSet.Contains(input)
                    || producer.TryGetValue(input, out int p) && p < i && parameterOnly[p]));

        // A step's output is internal to its segment when only that segment's steps read it and nothing else needs it.
        var internalOutput = new bool[n];
        for (int i = 0; i < n; i++)
        {
            if (seg[i] < 0 || inPlace[i] || parameterOnly[i]) continue;
            var output = forwardSteps[i].OutputBuffer;
            if (kept.Contains(output) || producer[output] != i) continue;
            internalOutput[i] = !readerSegments.TryGetValue(output, out var readers) || readers.All(s => s == seg[i]);
        }

        // The last forward action that reads each value (int.MaxValue when a step fused into another action reads it).
        var lastForwardRead = new Dictionary<Tensor<T>, int>(cmp);
        for (int i = 0; i < n; i++)
        {
            int a = genericActionOfStep[i] >= 0 ? genericActionOfStep[i] : int.MaxValue;
            foreach (var input in forwardSteps[i].Inputs)
                if (input is not null)
                    lastForwardRead[input] = lastForwardRead.TryGetValue(input, out int prior) ? Math.Max(prior, a) : a;
        }

        var firstBackward = Enumerable.Repeat(int.MaxValue, segments).ToArray();
        var lastBackward = Enumerable.Repeat(-1, segments).ToArray();
        var backwardRuns = new bool[n];
        for (int a = 0; a < coverage.Length; a++)
            foreach (int s in coverage[a])
            {
                backwardRuns[s] = true;
                if (seg[s] < 0) continue;
                firstBackward[seg[s]] = Math.Min(firstBackward[seg[s]], a);
                lastBackward[seg[s]] = Math.Max(lastBackward[seg[s]], a);
            }

        // Recompute sets: the internal producers of what each segment's backward reads, closed over their own inputs.
        var recompute = new HashSet<int>[segments];
        for (int k = 0; k < segments; k++) recompute[k] = new HashSet<int>();
        void NeedStep(int p, int k)
        {
            if (seg[p] != k || !internalOutput[p] || !recompute[k].Add(p)) return;
            foreach (var input in forwardSteps[p].Inputs) Need(input, k);
        }
        void Need(Tensor<T>? tensor, int k)
        {
            if (tensor is not null && producer.TryGetValue(tensor, out int p)) NeedStep(p, k);
        }
        for (int s = 0; s < n; s++)
        {
            if (seg[s] < 0 || !backwardRuns[s]) continue;
            var step = forwardSteps[s];
            int k = seg[s];
            if (!BackwardIgnoresForwardInputs(step))
                foreach (var input in step.Inputs) Need(input, k);
            if (!BackwardIgnoresForwardOutput(step)) Need(step.OutputBuffer, k);
            // Saved state is written by the step's own forward: an internal step that saves any is recomputed itself.
            if (step.SavedState is { Length: > 0 } saved && saved.Any(o => o is not null)) NeedStep(s, k);
        }

        var schedule = new CheckpointSchedule
        {
            SegmentCount = segments,
            SegmentOfForwardAction = Enumerable.Repeat(-1, forwardActionCount).ToArray(),
            ReleaseAfterForwardAction = new int[]?[forwardActionCount],
            LastForwardAction = Enumerable.Repeat(-1, segments).ToArray(),
            ForwardReads = new (object, int)[segments][],
            StashBeforeForwardAction = new int[]?[forwardActionCount],
            SeedBeforeForwardAction = new long[forwardActionCount],
            RecomputeActions = new int[segments][],
            RecomputeSteps = new int[segments][],
            RecomputeBeforeBackward = new int[]?[backwardActionCount],
            Keep = new object[segments][],
            Internal = new object[segments][],
            State = new Tensor<T>[segments][],
            HeldForward = Enumerable.Range(0, segments).Select(_ => new List<IGpuBuffer>()).ToArray(),
            HeldRecompute = Enumerable.Range(0, segments).Select(_ => new List<IGpuBuffer>()).ToArray(),
            BackwardReads = new (object, int)[segments][],
            Pending = Enumerable.Range(0, segments).Select(_ => new List<(IGpuBuffer, int)>()).ToArray(),
            Stash = new IGpuBuffer?[segments][],
            Current = new IGpuBuffer?[segments][],
            FirstBackward = firstBackward,
            LastBackward = new int[segments],
        };
        static void Append(int[]?[] table, int index, int value)
        {
            var prior = table[index];
            if (prior is null) table[index] = new[] { value };
            else if (Array.IndexOf(prior, value) < 0) table[index] = prior.Concat(new[] { value }).ToArray();
        }

        for (int k = 0; k < segments; k++)
        {
            if (lastBackward[k] < 0) recompute[k].Clear();   // nothing of the segment is differentiated
            var steps = recompute[k].OrderBy(s => s).ToArray();
            var actions = new int[steps.Length];
            for (int j = 0; j < steps.Length; j++)
            {
                actions[j] = genericActionOfStep[steps[j]];
                if (actions[j] < 0)
                    throw new NotSupportedException(
                        $"Checkpoint segment {k}: its backward reads the output of {forwardSteps[steps[j]].OpName}, which the "
                        + "plan runs as part of a fused or specialized action, so it cannot run it again alone. Mark the "
                        + "segment so the op is outside it, or turn the fusion off.");
            }
            schedule.RecomputeSteps[k] = steps;
            // Every value the segment's backward reads (inputs, outputs and saved objects of its steps, whoever produced
            // them) with the last backward action that reads it: a recomputed buffer bound to one is held until then.
            var lastRead = new Dictionary<object, int>(Autodiff.ReferenceEqualityComparer<object>.Instance);
            void Read(object? item, int a)
            {
                if (item is null) return;
                lastRead[item] = lastRead.TryGetValue(item, out int prior) ? Math.Max(prior, a) : a;
            }
            for (int a = 0; a < coverage.Length; a++)
                foreach (int st in coverage[a])
                {
                    if (seg[st] != k) continue;
                    var step = forwardSteps[st];
                    if (!BackwardIgnoresForwardInputs(step))
                        foreach (var input in step.Inputs) Read(input, a);
                    if (!BackwardIgnoresForwardOutput(step)) Read(step.OutputBuffer, a);
                    if (step.SavedState is { } saved) foreach (var item in saved) Read(item, a);
                }
            schedule.BackwardReads[k] = lastRead.Select(kv => (kv.Key, kv.Value)).ToArray();
            // The last backward action that runs the backward of a recomputed step or of a step reading a recomputed value:
            // the end of the recompute's use (a parameter-only step of the segment, hoisted to the start of the forward,
            // has its backward at the very end of the step and reads nothing the recompute wrote).
            var recomputedOutputs = new HashSet<Tensor<T>>(steps.Select(st => forwardSteps[st].OutputBuffer), cmp);
            int useEnd = -1;
            for (int a = 0; a < coverage.Length; a++)
                foreach (int st in coverage[a])
                    if (seg[st] == k && (recompute[k].Contains(st) || forwardSteps[st].Inputs.Any(i => i is not null && recomputedOutputs.Contains(i))))
                        useEnd = Math.Max(useEnd, a);
            schedule.LastBackward[k] = useEnd;
            schedule.RecomputeActions[k] = actions;

            int firstStep = Array.IndexOf(seg, k);
            // In-place state a recomputed step reads: restored to the value it had when the segment started.
            var state = new List<Tensor<T>>();
            foreach (int s in steps)
                foreach (var input in forwardSteps[s].Inputs)
                {
                    if (input is null || !writers.TryGetValue(input, out var stateWriters)) continue;
                    if (stateWriters.Any(w => w >= firstStep && w < s))
                        throw new NotSupportedException(
                            $"Checkpoint segment {k} updates a state in place and then reads it in {forwardSteps[s].OpName}, "
                            + "which its backward recomputes: the recompute could not read the value that step read. Keep the "
                            + "update and its reader in different segments, or the reader outside the segment.");
                    if (!state.Contains(input, cmp)) state.Add(input);
                }
            schedule.State[k] = state.ToArray();
            schedule.Stash[k] = new IGpuBuffer?[state.Count];
            schedule.Current[k] = new IGpuBuffer?[state.Count];

            int firstAction = -1, lastAction = -1;
            for (int i = 0; i < n; i++)
            {
                if (seg[i] != k || genericActionOfStep[i] < 0) continue;
                int a = genericActionOfStep[i];
                if (firstAction < 0) firstAction = a;
                lastAction = Math.Max(lastAction, a);
                if (internalOutput[i]) schedule.SegmentOfForwardAction[a] = k;
            }
            // A segment step that is not its own action (fused into another) may read or write past the segment's last
            // generic action: release only once every action up to the last one covering any of its steps has run.
            for (int i = 0; i < n; i++)
                if (seg[i] == k && genericActionOfStep[i] < 0)
                    lastAction = Math.Max(lastAction, PrecedingAction(genericActionOfStep, i, forwardActionCount));
            int segmentEnd = Math.Min(lastAction, forwardActionCount - 1);
            schedule.LastForwardAction[k] = segmentEnd;
            if (lastAction >= 0) Append(schedule.ReleaseAfterForwardAction, segmentEnd, k);
            // Each internal value is released after its last forward reader, as a forward span lease is (the segment's
            // last action can be much later: an in-place state update the plan runs at the end of the forward). What a
            // step saves for its backward is recomputed, so it goes once its producer ran.
            var forwardReads = new List<(object, int)>();
            for (int i = 0; i < n; i++)
            {
                if (seg[i] != k || !internalOutput[i]) continue;
                int produced = genericActionOfStep[i] >= 0 ? genericActionOfStep[i] : int.MaxValue;
                var output = forwardSteps[i].OutputBuffer;
                int release = lastForwardRead.TryGetValue(output, out int read) ? Math.Max(read, produced) : produced;
                forwardReads.Add((output, release));
                if (forwardSteps[i].SavedState is { } savedItems)
                    foreach (var item in savedItems)
                        if (item is not null && !(item is Tensor<T> t && kept.Contains(t))) forwardReads.Add((item, produced));
            }
            schedule.ForwardReads[k] = forwardReads.ToArray();
            if (lastAction >= 0)
                foreach (var (_, release) in forwardReads)
                    if (release < segmentEnd && release >= 0) Append(schedule.ReleaseAfterForwardAction, release, k);
            if (state.Count > 0 && firstAction >= 0) Append(schedule.StashBeforeForwardAction, firstAction, k);
            if (steps.Length > 0)
            {
                Append(schedule.RecomputeBeforeBackward, firstBackward[k], k);
            }

            // What outlives the segment's forward: what readers outside it read (its outputs, an auxiliary term), what the
            // backward of its steps that are not recomputed reads, and state. Its buffers stay kept, found by identity.
            var keep = new List<object>();
            for (int i = 0; i < n; i++)
            {
                if (seg[i] != k) continue;
                var step = forwardSteps[i];
                if (!internalOutput[i]) keep.Add(step.OutputBuffer);
                if (!recompute[k].Contains(i) && step.SavedState is { } saved)
                    foreach (var item in saved)
                        if (item is not null) keep.Add(item);
            }
            keep.AddRange(state);
            schedule.Keep[k] = keep.ToArray();
            // The tensors the segment's released buffers can back: its internal outputs, and what its steps save.
            var internals = new HashSet<object>(Autodiff.ReferenceEqualityComparer<object>.Instance);
            for (int i = 0; i < n; i++)
            {
                if (seg[i] != k || !internalOutput[i]) continue;
                internals.Add(forwardSteps[i].OutputBuffer);
                if (forwardSteps[i].SavedState is { } saved)
                    foreach (var item in saved)
                        if (item is not null && !(item is Tensor<T> t && kept.Contains(t))) internals.Add(item);
            }
            // A recompute writes its steps' values again, also ones a step that is not recomputed saved (an index tensor):
            // whatever the segment's backward reads can end up bound to a recompute buffer.
            foreach (var (item, _) in schedule.BackwardReads[k])
                if (!(item is Tensor<T> t && kept.Contains(t))) internals.Add(item);
            schedule.Internal[k] = internals.ToArray();
        }
        if (CompiledTrainingDiagnostics.GraphStepTraceEnabled)
            for (int k = 0; k < segments; k++)
            {
                AiDotNetEngine.LogDiagnostic($"[PLAN-CKPT] segment {k}: forward actions ..{schedule.LastForwardAction[k]} "
                    + $"backward actions {firstBackward[k]}..{lastBackward[k]} of {backwardActionCount}, recomputes "
                    + $"{schedule.RecomputeSteps[k].Length} steps used through backward action {schedule.LastBackward[k]}");
            }
        return schedule;
    }

    private static int PrecedingAction(int[] genericActionOfStep, int step, int forwardActionCount)
    {
        for (int i = step; i >= 0; i--)
            if (genericActionOfStep[i] >= 0) return genericActionOfStep[i];
        return Math.Min(0, forwardActionCount - 1);
    }

    /// <summary>
    /// The tensors no other mechanism may release or lend before a segment's recompute reads them: every input of a
    /// recomputed step that the recompute does not produce itself (the segment's input, a value of another segment).
    /// </summary>
    private static HashSet<Tensor<T>> CheckpointRecomputeInputs(CheckpointSchedule schedule, IReadOnlyList<CompiledStep<T>> forwardSteps)
    {
        var cmp = Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance;
        var inputs = new HashSet<Tensor<T>>(cmp);
        foreach (var steps in schedule.RecomputeSteps)
        {
            var produced = new HashSet<Tensor<T>>(steps.Select(s => forwardSteps[s].OutputBuffer), cmp);
            foreach (int s in steps)
                foreach (var input in forwardSteps[s].Inputs)
                    if (input is not null && !produced.Contains(input)) inputs.Add(input);
        }
        return inputs;
    }

    /// <summary>
    /// Holds every forward-step output of a checkpointed segment's released actions out of the forward span leases, and
    /// keeps the lend schedule from lending a recomputed step's input before the recompute that reads it.
    /// </summary>
    private void ApplyCheckpointLifetimes()
    {
        if (_checkpoint is not { } schedule || _forwardSteps is not { } forwardSteps) return;
        var firstBackwardOfSegment = schedule.FirstBackward;
        if (_lastBackwardReads is not { } lastReads) return;
        for (int k = 0; k < schedule.SegmentCount; k++)
        {
            var produced = new HashSet<Tensor<T>>(schedule.RecomputeSteps[k].Select(s => forwardSteps[s].OutputBuffer),
                Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (int s in schedule.RecomputeSteps[k])
                foreach (var input in forwardSteps[s].Inputs)
                {
                    if (input is null || produced.Contains(input)) continue;
                    if (lastReads.TryGetValue(input, out int last))
                    {
                        if (last >= 0) lastReads[input] = Math.Max(last, firstBackwardOfSegment[k]);
                    }
                    else lastReads[input] = firstBackwardOfSegment[k];
                }
        }
    }

    private static IGpuBuffer? TensorBufferOf(object item) => item switch
    {
        TensorBase<float> t => t._gpuBuffer,
        TensorBase<double> t => t._gpuBuffer,
        TensorBase<int> t => t._gpuBuffer,
        TensorBase<long> t => t._gpuBuffer,
        TensorBase<Half> t => t._gpuBuffer,
        TensorBase<byte> t => t._gpuBuffer,
        TensorBase<bool> t => t._gpuBuffer,
        _ => null,
    };

    // ---- the resident (and captured) step body ----

    /// <summary>Before forward action <paramref name="action"/>: copies the state its segments' recomputes will read.</summary>
    private void CheckpointBeforeForwardAction(int action, DirectGpuTensorEngine? de, IDirectGpuBackend backend, bool capturing)
    {
        if (_checkpoint is not { } schedule) return;
        if (de is not null) schedule.SeedBeforeForwardAction[action] = de.StochasticSeedPosition;
        if (schedule.StashBeforeForwardAction[action] is not { } segments) return;
        foreach (int k in segments)
        {
            var state = _checkpoint.State[k];
            for (int j = 0; j < state.Length; j++)
            {
                var source = StateBuffer(state[j], k);
                var stash = _checkpoint.Stash[k][j] ??= AllocateCheckpointBuffer(backend, state[j].Length, capturing);
                CopyDevice(backend, source, stash, state[j].Length);
            }
        }
    }

    /// <summary>Starts holding the leases of forward action <paramref name="action"/> when its segment releases them.</summary>
    private bool CheckpointHoldsForwardAction(int action, StepScratchPool pool)
    {
        if (_checkpoint is not { } schedule) return false;
        int k = schedule.SegmentOfForwardAction[action];
        if (k < 0) return false;
        pool.HoldLeasesIn = schedule.HeldForward[k];
        return true;
    }

    /// <summary>
    /// After forward action <paramref name="action"/>: for each segment whose forward ended there, releases the buffers
    /// its actions leased, except those still bound to a tensor that outlives the forward, which become kept slots.
    /// </summary>
    private void CheckpointAfterForwardAction(int action, StepScratchPool pool)
    {
        pool.HoldLeasesIn = null;
        if (_checkpoint is null || _checkpoint.ReleaseAfterForwardAction[action] is not { } segments) return;
        foreach (int k in segments)
        {
            var held = _checkpoint.HeldForward[k];
            if (held.Count == 0) continue;
            bool segmentEnds = action >= _checkpoint.LastForwardAction[k];
            var keep = new HashSet<IGpuBuffer>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
            foreach (var item in _checkpoint.Keep[k])
                if (TensorBufferOf(item) is { } bound) keep.Add(GpuBufferViewRoot.Of(bound));
            // Before the segment's end, a buffer goes once every value bound to it had its last forward reader; one
            // bound to no tracked value waits for the segment's end: a later forward step can read an action's scratch
            // through an object it captured (measured: releasing a router's unbound scratch at the end of its layer
            // changed the first step's loss).
            Dictionary<IGpuBuffer, int>? releaseAt = null;
            if (!segmentEnds)
            {
                releaseAt = new Dictionary<IGpuBuffer, int>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
                foreach (var (item, after) in _checkpoint.ForwardReads[k])
                {
                    if (TensorBufferOf(item) is not { } bound) continue;
                    var root = GpuBufferViewRoot.Of(bound);
                    releaseAt[root] = releaseAt.TryGetValue(root, out int prior) ? Math.Max(prior, after) : after;
                }
            }
            var released = new HashSet<IGpuBuffer>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
            for (int j = held.Count - 1; j >= 0; j--)
            {
                var buffer = held[j];
                var root = GpuBufferViewRoot.Of(buffer);
                if (keep.Contains(root)) pool.PinSpanLease(buffer);
                else if (segmentEnds || (releaseAt is not null && releaseAt.TryGetValue(root, out int after) && after <= action))
                {
                    pool.EndSpanLease(buffer);
                    released.Add(root);
                }
                else continue;
                held[j] = held[held.Count - 1];
                held.RemoveAt(held.Count - 1);
            }
            UnbindReleased(_checkpoint.Internal[k], released);
        }
    }

    /// <summary>
    /// Before backward action <paramref name="action"/>: recomputes each segment whose backward starts there, with the
    /// state its forward read, into buffers held until that backward ends.
    /// </summary>
    private void CheckpointBeforeBackwardAction(int action, IEngine engine, DirectGpuTensorEngine? de, StepScratchPool? pool,
        IDirectGpuBackend backend, bool capturing, Action<IEngine>[] forward, int recomputeKeyBase)
    {
        if (_checkpoint is null || _checkpoint.RecomputeBeforeBackward[action] is not { } segments) return;
        foreach (int k in segments)
        {
            var state = _checkpoint.State[k];
            for (int j = 0; j < state.Length; j++)
            {
                var live = StateBuffer(state[j], k);
                var current = _checkpoint.Current[k][j] ??= AllocateCheckpointBuffer(backend, state[j].Length, capturing);
                var stash = _checkpoint.Stash[k][j]
                    ?? throw new InvalidOperationException($"Checkpoint segment {k}: its state was not saved in the forward.");
                CopyDevice(backend, live, current, state[j].Length);
                CopyDevice(backend, stash, live, state[j].Length);
            }
            // A recomputed stochastic op (a dropout) draws what its forward drew: the segment's kept outputs were
            // computed from that draw, so a new one would backpropagate through a different function. Each action starts
            // from the seed position its forward started from. The replay is this thread's: the engine's sequence is not
            // rewound, so another thread's ops on the engine meanwhile cannot draw seeds the step already used.
            try
            {
                foreach (int a in _checkpoint.RecomputeActions[k])
                {
                    de?.SetCurrentScratchAction(recomputeKeyBase + a);
                    if (pool is not null) pool.HoldLeasesIn = _checkpoint.HeldRecompute[k];
                    if (de is not null)
                        using (de.ReplayStochasticSeedsFrom(_checkpoint.SeedBeforeForwardAction[a]))
                            forward[a](engine);
                    else
                        forward[a](engine);
                    if (pool is not null) pool.HoldLeasesIn = null;
                    de?.EndScratchAction();
                }
            }
            catch (Exception failure)
            {
                // The live state (running statistics, router bias) holds the segment start's values now: put the step's
                // back before the failure reaches a caller that may carry on (the eager fallback), or report both.
                try { RestoreCurrentState(); }
                catch (Exception restore)
                {
                    throw new AggregateException(
                        $"Checkpoint segment {k}: the recompute failed, and the state it rewound could not be restored.",
                        failure, restore);
                }
                throw;
            }
            finally
            {
                if (pool is not null) pool.HoldLeasesIn = null;
            }
            RestoreCurrentState();
            void RestoreCurrentState()
            {
                for (int j = 0; j < state.Length; j++)
                    CopyDevice(backend,
                        _checkpoint.Current[k][j] ?? throw new InvalidOperationException($"Checkpoint segment {k}: its current state was not saved before the recompute."),
                        StateBuffer(state[j], k), state[j].Length);
            }
            if (!capturing) _checkpoint.RecomputeRuns++;
            ScheduleRecomputeReleases(k, action, pool);
        }
    }

    /// <summary>
    /// After segment <paramref name="k"/>'s recompute: gives each buffer it leased the backward action after which it is
    /// released, the last one of the segment's backward reading a value bound to it (as PyTorch frees a recomputed
    /// tensor once its last backward user ran). Bindings are read after the recompute, so a value a step that is not
    /// recomputed saved (an index tensor its producer's recompute wrote again) counts too. Any other buffer the
    /// recompute leased is held until the last backward action of a step that is recomputed or reads a recomputed value.
    /// </summary>
    private void ScheduleRecomputeReleases(int k, int action, StepScratchPool? pool)
    {
        if (_checkpoint is not { } cp) return;
        var releaseAt = new Dictionary<IGpuBuffer, int>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
        foreach (var (item, at) in cp.BackwardReads[k])
        {
            if (TensorBufferOf(item) is not { } bound) continue;
            var root = GpuBufferViewRoot.Of(bound);
            releaseAt[root] = releaseAt.TryGetValue(root, out int prior) ? Math.Max(prior, at) : at;
        }
        var held = cp.HeldRecompute[k];
        var now = new HashSet<IGpuBuffer>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance);
        foreach (var buffer in held)
        {
            var root = GpuBufferViewRoot.Of(buffer);
            // A buffer bound to no value the plan sees the backward read can still be read through an object a backward
            // function captured (measured: a router's first-run gradients changed when such buffers were released right
            // after the recompute), so it is held until the segment's backward ends.
            int at = releaseAt.TryGetValue(root, out int last) ? last : cp.LastBackward[k];
            if (at >= action) cp.Pending[k].Add((buffer, at));
            else
            {
                pool?.EndSpanLease(buffer);
                now.Add(root);
            }
        }
        held.Clear();
        UnbindReleased(cp.Internal[k], now);
    }

    /// <summary>After backward action <paramref name="action"/>: releases the recomputed buffers whose last reader ran.</summary>
    private void CheckpointAfterBackwardAction(int action, StepScratchPool? pool)
    {
        if (_checkpoint is not { } cp) return;
        for (int k = 0; k < cp.SegmentCount; k++)
        {
            var pending = cp.Pending[k];
            if (pending.Count == 0) continue;
            HashSet<IGpuBuffer>? released = null;
            for (int j = pending.Count - 1; j >= 0; j--)
            {
                if (pending[j].ReleaseAt > action) continue;
                pool?.EndSpanLease(pending[j].Buffer);
                (released ??= new HashSet<IGpuBuffer>(Autodiff.ReferenceEqualityComparer<IGpuBuffer>.Instance))
                    .Add(GpuBufferViewRoot.Of(pending[j].Buffer));
                pending[j] = pending[pending.Count - 1];
                pending.RemoveAt(pending.Count - 1);
            }
            if (released is not null) UnbindReleased(cp.Internal[k], released);
        }
    }

    /// <summary>
    /// Drops the device binding of every tensor in <paramref name="tensors"/> bound to a buffer in
    /// <paramref name="released"/>. A released buffer goes back to the pool and is leased again; a tensor still bound to
    /// it would read the next tenant's data, and a write into its binding (an op that writes its result into the
    /// output's existing device copy) would overwrite that tenant. Unbound, the tensor's next write takes a new buffer.
    /// </summary>
    private static void UnbindReleased(object[] tensors, HashSet<IGpuBuffer> released)
    {
        if (released.Count == 0) return;
        foreach (var item in tensors)
            if (TensorBufferOf(item) is { } bound && released.Contains(GpuBufferViewRoot.Of(bound)))
                switch (item)
                {
                    case TensorBase<float> t: Detach(t); break;
                    case TensorBase<double> t: Detach(t); break;
                    case TensorBase<int> t: Detach(t); break;
                    case TensorBase<long> t: Detach(t); break;
                    case TensorBase<Half> t: Detach(t); break;
                    case TensorBase<byte> t: Detach(t); break;
                    case TensorBase<bool> t: Detach(t); break;
                }

        static void Detach<U>(TensorBase<U> t)
        {
            if (t.GetBackingArrayForCacheLookupUnsafe() is { } key) Helpers.HostSync.Remove(key);
            t._gpuBuffer = null;
            t._gpuBackend = null;
            t._gpuBufferVersion = -1;
        }
    }

    /// <summary>Settles a body that threw: nothing stays held or flagged for the next run.</summary>
    private void CheckpointAbandonRun(StepScratchPool? pool)
    {
        if (_checkpoint is not { } schedule) return;
        if (pool is not null) pool.HoldLeasesIn = null;
        for (int k = 0; k < schedule.Pending.Length; k++)
        {
            foreach (var (buffer, _) in schedule.Pending[k]) schedule.HeldRecompute[k].Add(buffer);
            schedule.Pending[k].Clear();
        }
        foreach (var held in schedule.HeldForward.Concat(schedule.HeldRecompute))
        {
            if (pool is not null)
                foreach (var buffer in held) pool.EndSpanLease(buffer);
            held.Clear();
        }
    }

    private static IGpuBuffer StateBuffer(Tensor<T> state, int segment)
        => state._gpuBuffer is { Handle: var handle } buffer && handle != IntPtr.Zero
            ? buffer
            : throw new InvalidOperationException(
                $"Checkpoint segment {segment}: the in-place state its recompute reads is not on the device, so the plan "
                + "cannot give the recompute the value the forward read.");

    private static IGpuBuffer AllocateCheckpointBuffer(IDirectGpuBackend backend, int length, bool capturing)
        => capturing
            ? throw new InvalidOperationException("A checkpoint state copy must be allocated before the step is captured.")
            : backend.AllocateBuffer(length);

    private static void CopyDevice(IDirectGpuBackend backend, IGpuBuffer source, IGpuBuffer destination, int length)
    {
        if (backend is IFp16ActivationBackend raw)
            raw.CopyBufferDtoD(source, destination, (long)length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>());
        else backend.Copy(source, destination, length);
    }

    private void DisposeCheckpointBuffers()
    {
        if (_checkpoint is not { } schedule) return;
        foreach (var row in schedule.Stash.Concat(schedule.Current))
            for (int j = 0; j < row.Length; j++)
            {
                row[j]?.Dispose();
                row[j] = null;
            }
    }
}
