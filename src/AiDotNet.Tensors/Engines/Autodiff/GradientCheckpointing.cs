using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Provides gradient checkpointing for memory-efficient training.
/// Instead of storing all intermediate activations, only stores activations at
/// checkpoints and recomputes the rest during backward (trades compute for memory).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>For a model with N layers, standard backprop stores O(N) activations.
/// Checkpointing every sqrt(N) layers reduces this to O(sqrt(N)) with ~33% more compute.</para>
/// <para>Reference: Chen et al. "Training Deep Nets with Sublinear Memory Cost" (2016)</para>
/// </remarks>
public static class GradientCheckpointing<T>
{
    /// <summary>
    /// Runs a sequence of functions with gradient checkpointing.
    /// Only stores activations at segment boundaries. During backward,
    /// recomputes activations between checkpoints.
    /// </summary>
    /// <param name="functions">The sequence of differentiable functions (e.g., layer forwards).</param>
    /// <param name="input">The input tensor.</param>
    /// <param name="segmentSize">Number of functions per checkpoint segment.
    /// Smaller = less memory but more recomputation.</param>
    /// <returns>The final output tensor, with backward support.</returns>
    public static Tensor<T> Checkpoint(
        IReadOnlyList<Func<Tensor<T>, Tensor<T>>> functions,
        Tensor<T> input,
        int segmentSize = 2)
        => CheckpointCore(functions, input, parameterSourceFactory: null, segmentParameters: null, segmentSize, tryConsumeParameterGradient: null);

    /// <summary>
    /// Runs a checkpointed sequence while differentiating only the explicitly selected parameters.
    /// </summary>
    /// <remarks>
    /// The source factory is evaluated after the no-grad forward has materialized lazy parameters.
    /// This is the activation-checkpointing equivalent of freezing a backbone: the recompute still
    /// calculates the input VJP needed to cross the frozen block, but it neither allocates nor returns
    /// gradients for parameters outside <paramref name="parameterSourceFactory"/>.
    /// </remarks>
    public static Tensor<T> Checkpoint(
        IReadOnlyList<Func<Tensor<T>, Tensor<T>>> functions,
        Tensor<T> input,
        Func<IReadOnlyList<Tensor<T>>> parameterSourceFactory,
        int segmentSize = 2)
    {
        if (parameterSourceFactory is null)
            throw new ArgumentNullException(nameof(parameterSourceFactory));
        return CheckpointCore(functions, input, parameterSourceFactory, segmentParameters: null, segmentSize, tryConsumeParameterGradient: null);
    }

    /// <summary>
    /// Runs a checkpointed sequence with explicit per-segment parameter lists and an optional in-backward
    /// parameter-gradient consumer (for post-accumulate optimizer stepping that immediately reclaims gradient VRAM).
    /// </summary>
    /// <param name="functions">The sequence of differentiable layer forwards.</param>
    /// <param name="input">The input tensor.</param>
    /// <param name="segmentParameters">Optional per-segment trainable parameter lists (length must equal segment count).</param>
    /// <param name="segmentSize">Number of functions per checkpoint segment.</param>
    /// <param name="tryConsumeParameterGradient">
    /// Optional callback invoked during segment backward for each parameter gradient produced by that segment.
    /// Return <c>true</c> when the parameter was updated in-place and its gradient storage can be reclaimed immediately;
    /// return <c>false</c> to accumulate the gradient into the outer tape's gradient dictionary (e.g., for shared/tied weights).
    /// Requires <paramref name="segmentParameters"/>: without it the recompute differentiates every touched leaf, and the
    /// callback would be offered intermediate tensors as parameters. The callback runs outside the recompute's tensor
    /// scope so the optimizer state it creates persists; it should dispose any temporary it creates (an update tensor
    /// it discards), which otherwise waits for finalization to release its device memory.
    /// </param>
    /// <returns>The final output tensor, with backward support.</returns>
    public static Tensor<T> CheckpointSegments(
        IReadOnlyList<Func<Tensor<T>, Tensor<T>>> functions,
        Tensor<T> input,
        IReadOnlyList<IReadOnlyList<Tensor<T>>>? segmentParameters,
        int segmentSize = 2,
        Func<Tensor<T>, Tensor<T>, bool>? tryConsumeParameterGradient = null)
    {
        if (tryConsumeParameterGradient is not null && segmentParameters is null)
            throw new ArgumentException(
                "A parameter-gradient consumer needs explicit segment parameter lists: without them every touched leaf, "
                + "intermediates included, would be offered to it as a parameter.", nameof(segmentParameters));
        return CheckpointCore(functions, input, parameterSourceFactory: null, segmentParameters, segmentSize, tryConsumeParameterGradient);
    }

    private static Tensor<T> CheckpointCore(
        IReadOnlyList<Func<Tensor<T>, Tensor<T>>> functions,
        Tensor<T> input,
        Func<IReadOnlyList<Tensor<T>>>? parameterSourceFactory,
        IReadOnlyList<IReadOnlyList<Tensor<T>>>? segmentParameters,
        int segmentSize,
        Func<Tensor<T>, Tensor<T>, bool>? tryConsumeParameterGradient)
    {
        if (functions == null || functions.Count == 0) return input;
        if (segmentSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(segmentSize), "Must be positive.");

        // 1 + (n - 1) / size, not (n + size - 1) / size: the latter overflows for a segment size near int.MaxValue.
        int numSegments = 1 + (functions.Count - 1) / segmentSize;
        if (segmentParameters is not null)
        {
            if (segmentParameters.Count != numSegments)
                throw new ArgumentException($"Expected {numSegments} segment parameter lists, got {segmentParameters.Count}.", nameof(segmentParameters));
            // A null list would silently select the unfiltered mode (every touched leaf gets a gradient) for that segment.
            for (int seg = 0; seg < numSegments; seg++)
            {
                if (segmentParameters[seg] is null)
                    throw new ArgumentException($"Segment parameter list {seg} is null; pass an empty list for a segment with no trainable parameters.", nameof(segmentParameters));
            }
        }

        var tape = GradientTape<T>.Current;
        if (tape is null)
        {
            // No tape active — just run forward without checkpointing
            var current = input;
            foreach (var fn in functions)
                current = fn(current);
            return current;
        }

        var engine = AiDotNetEngine.Current;
        var current2 = input;

        for (int seg = 0; seg < numSegments; seg++)
        {
            int startIdx = seg * segmentSize;
            int endIdx = Math.Min(startIdx + segmentSize, functions.Count);
            var segmentInput = current2;

            // Run segment forward WITHOUT recording (save memory) on both CPU (TensorArena) and GPU (TensorScope).
            Tensor<T> segmentOutput;
            using (var segmentGpuScope = TensorScope.Begin())
            using (var segmentArena = TensorArena.Create(poolWhenNested: true))
            using (GradientTape<T>.NoGrad())
            {
                segmentOutput = segmentInput;
                for (int i = startIdx; i < endIdx; i++)
                    segmentOutput = functions[i](segmentOutput);
                // Only the segment boundary escapes. Copy it out (device-resident on GPU, arena-independent on CPU),
                // then release every internal activation immediately on scope disposal.
                segmentOutput = segmentGpuScope.Keep(CloneCheckpointBoundary(segmentOutput, engine));
            }

            // Record a single "checkpoint" op that recomputes during backward. A selective
            // checkpoint records its chosen parameters as logical inputs as well as the segment
            // boundary. Besides making the dependency explicit, this lets requested-source graph
            // pruning retain the checkpoint when a caller asks only for trainable parameters.
            Tensor<T>[]? checkpointInputs = null;
            IReadOnlyList<Tensor<T>>? selected = segmentParameters is not null
                ? segmentParameters[seg]
                : parameterSourceFactory?.Invoke();
            if (selected is not null)
            {
                var unique = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
                checkpointInputs = new Tensor<T>[selected.Count + 1];
                checkpointInputs[0] = segmentInput;
                int write = 1;
                for (int i = 0; i < selected.Count; i++)
                {
                    var parameter = selected[i];
                    if (parameter is not null && !ReferenceEquals(parameter, segmentInput) && unique.Add(parameter))
                        checkpointInputs[write++] = parameter;
                }
                if (write != checkpointInputs.Length)
                    Array.Resize(ref checkpointInputs, write);
            }

            int capturedStart = startIdx;
            int capturedEnd = endIdx;
            var capturedFunctions = functions;

            BackwardFunction<T> backward = (gradOutput, inputs, output, savedState, eng, grads) =>
                {
                    // RECOMPUTE: run the segment forward again WITH recording so
                    // backward can flow through the ops. Use a fresh, NON-persistent
                    // inner tape so the recomputed forward / backward do NOT
                    // pollute the outer tape that's currently mid-backward, and so
                    // each recompute step bypasses AutoTrainingCompiler's plan
                    // cache (the compiler is keyed on tape-pattern signatures and
                    // the compiled-plan replay path would otherwise try to attach
                    // a CompiledBackwardGraph to the per-step throw-away tape,
                    // which produces NullReferenceException in
                    // SymbolicBackwardGraphBuilder.Analyze when the cached pattern
                    // does not match the current tape's reachable entries).
                    var reInput = inputs[0];
                    List<(Tensor<T> Key, Tensor<T> Gradient)> detachedGradients;
                    using (var recomputeGpuScope = TensorScope.Begin())
                    using (var recomputeArena = TensorArena.Create(poolWhenNested: true))
                    using (var recomputeTape = new GradientTape<T>(
                        // A dedicated nested arena owns this segment's replay. SuppressArenaScope
                        // keeps the tape from resetting it before gradients are copied out below.
                        new GradientTapeOptions { Persistent = false, SuppressArenaScope = true }))
                    {
                    // Detach the segment input so the recompute graph is SELF-CONTAINED: its backward
                    // stops at the segment boundary instead of following reInput's producer into an
                    // EARLIER checkpoint segment. Without this, ComputeGradients(sources: null) below
                    // re-enters the previous segment's recompute (nested) and scatters its gradients,
                    // which the outer reverse walk then ALSO computes — double-counting every earlier
                    // segment's gradients (2x with two segments; (N-i)x for segment i of N). This
                    // mirrors torch.utils.checkpoint's detach_variable(inputs). StopGradient returns a
                    // fresh leaf (data copy, no GradFn); the input gradient it produces is remapped
                    // back onto the original reInput tensor when scattering below.
                        var reInputDetached = eng.StopGradient(reInput);
                        var reOutput = reInputDetached;
                        for (int i = capturedStart; i < capturedEnd; i++)
                            reOutput = capturedFunctions[i](reOutput);

                        // Vector-Jacobian product (VJP) seed: define an inner scalar pseudo-loss
                        //     pseudoLoss = sum(reOutput * gradOutput.detach())
                        // whose gradient w.r.t. reInput is, by the chain rule, exactly the VJP
                        //     d(pseudoLoss)/d(reInput) = (d(reOutput)/d(reInput))^T @ gradOutput
                        // needed even when the segment changes shape (AiDotNet#1341) or uses non-elementwise ops.
                        var weighted = eng.TensorMultiply(reOutput, gradOutput);
                        var pseudoLoss = eng.ReduceSum(weighted);

                        // Differentiate the recomputed segment w.r.t. the detached segment input and the
                        // segment's parameters (or every touched leaf when checkpointInputs is null).
                        // Because reInput was detached above, this recompute graph is self-contained and never
                        // double-counts earlier segments; gradOutput is excluded below as an outer-tape seed.
                        Tensor<T>[]? recomputeSources = null;
                        if (checkpointInputs is not null)
                        {
                            recomputeSources = new Tensor<T>[inputs.Length];
                            recomputeSources[0] = reInputDetached;
                            for (int i = 1; i < inputs.Length; i++) recomputeSources[i] = inputs[i];
                        }

                        // Suspend any outer tape's thread-static indexed/sparse gradient tables during the nested
                        // recompute backward: otherwise Embedding/Gather backwards inside this segment observe the
                        // outer streaming tape's _indexedSparseGrads, skip writing into segGrads, and stash a
                        // SparseEmbeddingGradient wrapping a temporary recomputeArena/recomputeGpuScope tensor that
                        // is recycled as soon as this scope disposes.
                        var savedIndexedGrads = DifferentiableOps._indexedGrads;
                        var savedSparseGrads = DifferentiableOps._indexedSparseGrads;
                        // The tables are addressed by each tensor's _gradIndex, so those are suspended too. A parameter
                        // shared with the outer pass keeps the index that pass assigned: the nested tape walk would not
                        // reassign it (so a new nested tensor could take the same slot and the two would accumulate
                        // into one), and its cleanup sets every index it touched to -1 (so the outer pass would lose
                        // the parameter's indexed and sparse gradients). Clear them for the nested pass and restore them.
                        var savedGradIndices = SuspendGradIndices(recomputeTape);
                        DifferentiableOps.ClearIndexedGrads();
                        DifferentiableOps.ClearIndexedSparseGrads();
                        Dictionary<Tensor<T>, Tensor<T>> segGrads;
                        try
                        {
                            segGrads = recomputeTape.ComputeGradients(pseudoLoss, recomputeSources);
                        }
                        finally
                        {
                            DifferentiableOps._indexedGrads = savedIndexedGrads;
                            DifferentiableOps._indexedSparseGrads = savedSparseGrads;
                            foreach (var (tensor, index) in savedGradIndices)
                                tensor._gradIndex = index;
                        }

                        // Segment-recompute arrays and GPU intermediates must not escape their bounded scope.
                        // Preserve only the selected VJPs (or consume single-segment parameter gradients in-place).
                        detachedGradients = new List<(Tensor<T>, Tensor<T>)>(segGrads.Count);
                        bool hasInputGrad = false;
                        foreach (var kvp in segGrads)
                        {
                            if (ReferenceEquals(kvp.Key, gradOutput) || kvp.Value is null) continue;
                            bool isSegmentInput = ReferenceEquals(kvp.Key, reInputDetached);
                            if (isSegmentInput)
                            {
                                hasInputGrad = true;
                            }
                            else if (tryConsumeParameterGradient is not null)
                            {
                                bool consumed;
                                using (TensorScope.Suspend())
                                {
                                    consumed = tryConsumeParameterGradient(kvp.Key, kvp.Value);
                                }
                                if (consumed)
                                {
                                    // The nested backward mirrored the gradient into Grad; it belongs to
                                    // recomputeGpuScope and is released below, and no outer accumulation will
                                    // replace it. Clear the mirror only while it still points at that gradient.
                                    if (ReferenceEquals(kvp.Key.Grad, kvp.Value)) kvp.Key.Grad = null;
                                    continue;
                                }
                            }
                            var key = isSegmentInput ? reInput : kvp.Key;
                            var copied = recomputeGpuScope.Keep(CloneCheckpointBoundary(kvp.Value, eng));
                            detachedGradients.Add((key, copied));
                        }

                        if (!hasInputGrad && ReferenceEquals(reOutput, reInputDetached))
                        {
                            var copied = recomputeGpuScope.Keep(CloneCheckpointBoundary(gradOutput, eng));
                            detachedGradients.Add((reInput, copied));
                        }
                    }

                    foreach (var (key, gradient) in detachedGradients)
                    {
                        DifferentiableOps.AccumulateGrad(grads, key, gradient, eng);
                    }
                };

            if (checkpointInputs is null)
            {
                DifferentiableOps.RecordUnary<T>(
                    $"Checkpoint_seg{seg}", segmentOutput, segmentInput, backward);
            }
            else
            {
                DifferentiableOps.RecordIfActive<T>(
                    $"Checkpoint_seg{seg}", segmentOutput, checkpointInputs, backward);
            }

            current2 = segmentOutput;
        }

        return current2;
    }

    /// <summary>
    /// Sets every gradient index assigned on a tensor the tape recorded to -1 and returns the previous values, so a
    /// nested backward over that tape starts from unindexed tensors and the caller can restore the outer pass's indices.
    /// </summary>
    private static List<(Tensor<T> Tensor, int Index)> SuspendGradIndices(GradientTape<T> tape)
    {
        var saved = new List<(Tensor<T>, int)>();
        var entries = tape.Entries;
        for (int i = 0; i < entries.Count; i++)
        {
            ref var entry = ref entries[i];
            Suspend(entry.Output, saved);
            Suspend(entry.Input0, saved);
            if (entry.InputCount >= 2) Suspend(entry.Input1, saved);
            if (entry.InputCount >= 3) Suspend(entry.Input2, saved);
            if (entry.InputsOverflow is not null)
            {
                foreach (var extra in entry.InputsOverflow)
                    Suspend(extra, saved);
            }
        }
        return saved;

        static void Suspend(Tensor<T>? tensor, List<(Tensor<T>, int)> into)
        {
            if (tensor is null || tensor._gradIndex < 0) return;
            into.Add((tensor, tensor._gradIndex));
            tensor._gradIndex = -1;
        }
    }

    /// <summary>
    /// Copies a checkpoint boundary into ordinary owned storage (device-to-device on DirectGpuTensorEngine
    /// without any host download, or arena-independent CPU array on CpuEngine).
    /// </summary>
    private static Tensor<T> CloneCheckpointBoundary(Tensor<T> source, IEngine engine)
    {
        if (engine is DirectGpuTensorEngine && source.TryGetGpuBuffer() is not null)
        {
            using (GradientTape<T>.NoGrad())
            {
                var gpuContiguous = source.IsContiguous ? source : source.Contiguous();
                return engine.StopGradient(gpuContiguous);
            }
        }
        var contiguous = source.IsContiguous ? source : source.Contiguous();
        return new Tensor<T>(contiguous.AsSpan().ToArray(), source.Shape.ToArray());
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (ReferenceEquals(a, b)) return true;
        if (a is null || b is null) return false;
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }
}
