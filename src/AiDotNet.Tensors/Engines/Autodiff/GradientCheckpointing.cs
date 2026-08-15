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
        => CheckpointCore(functions, input, parameterSourceFactory: null, segmentSize);

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
        return CheckpointCore(functions, input, parameterSourceFactory, segmentSize);
    }

    private static Tensor<T> CheckpointCore(
        IReadOnlyList<Func<Tensor<T>, Tensor<T>>> functions,
        Tensor<T> input,
        Func<IReadOnlyList<Tensor<T>>>? parameterSourceFactory,
        int segmentSize)
    {
        if (functions == null || functions.Count == 0) return input;
        if (segmentSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(segmentSize), "Must be positive.");

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
        int numSegments = (functions.Count + segmentSize - 1) / segmentSize;
        var current2 = input;

        for (int seg = 0; seg < numSegments; seg++)
        {
            int startIdx = seg * segmentSize;
            int endIdx = Math.Min(startIdx + segmentSize, functions.Count);
            var segmentInput = current2;

            // Run segment forward WITHOUT recording (save memory)
            Tensor<T> segmentOutput;
            using (var segmentArena = TensorArena.Create(poolWhenNested: true))
            using (GradientTape<T>.NoGrad())
            {
                segmentOutput = segmentInput;
                for (int i = startIdx; i < endIdx; i++)
                    segmentOutput = functions[i](segmentOutput);
                // Only the segment boundary escapes. Copy it out, then release every internal
                // activation instead of letting the caller's outer training arena retain the
                // no-grad scratch until the end of the whole model step. A nested arena does not
                // return its buffers to the shared pool because layers may cache internal tensors.
                segmentOutput = new Tensor<T>(segmentOutput.AsSpan().ToArray(), segmentOutput.Shape.ToArray());
            }

            // Record a single "checkpoint" op that recomputes during backward. A selective
            // checkpoint records its chosen parameters as logical inputs as well as the segment
            // boundary. Besides making the dependency explicit, this lets requested-source graph
            // pruning retain the checkpoint when a caller asks only for trainable parameters.
            Tensor<T>[]? checkpointInputs = null;
            if (parameterSourceFactory is not null)
            {
                var selected = parameterSourceFactory();
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

                    // Vector-Jacobian product (VJP) seed. The previous
                    // implementation computed gradients with the inner tape's
                    // implicit ones-seed (treating reOutput as if it were the
                    // scalar loss) and then attempted to "chain-rule" by
                    // elementwise-multiplying the resulting dreOutput/dreInput
                    // by gradOutput. That is wrong on two counts:
                    //   (a) the inner gradient's shape matches reInput, not
                    //       reOutput, so the elementwise multiply against
                    //       gradOutput (shape == reOutput's shape) can fail to
                    //       broadcast whenever the segment changes shape
                    //       (Transformer head: [B, L, D] -> [B, D] via
                    //       LastToken slice, or [B, D] -> [B, V] via dense
                    //       projection) — this is the
                    //       ArgumentException("cannot be broadcast") reported
                    //       in AiDotNet#1341 when the head-side gradOutput
                    //       [B, V] meets an encoder segment producing
                    //       [B, L, D];
                    //   (b) even when shapes happen to align (pure unary
                    //       elementwise segment), the chain rule for non-
                    //       elementwise ops (matmul, sum, softmax, etc.) is
                    //       NOT pointwise multiplication, so the gradient is
                    //       numerically wrong.
                    //
                    // Correct construction: define an inner scalar pseudo-loss
                    //     pseudoLoss = sum(reOutput * gradOutput.detach())
                    // whose gradient w.r.t. reInput is, by the chain rule,
                    // exactly the VJP we need:
                    //     d(pseudoLoss)/d(reInput)
                    //         = (d(reOutput)/d(reInput))^T @ gradOutput
                    // which is the contribution this segment owes to dL/dreInput.
                    // ReduceSum reduces over every axis and produces a scalar
                    // tensor — both ops are recorded on the inner tape via the
                    // DifferentiableOps backward registry, so ComputeGradients
                    // walks them and computes the correct VJP into reInput.
                        var weighted = eng.TensorMultiply(reOutput, gradOutput);
                        var pseudoLoss = eng.ReduceSum(weighted);

                    // Differentiate the recomputed segment w.r.t. EVERY leaf it touched — the
                    // (detached) segment input AND every weight/parameter the segment's functions
                    // read — not just the input. PyTorch's torch.utils.checkpoint backpropagates the
                    // recomputed forward through all inputs that require grad, including module
                    // parameters; an earlier version requested only `reInput`, so the WEIGHT gradients
                    // of every checkpointed layer were silently dropped and checkpointed layers never
                    // learned. `sources: null` differentiates the whole recomputed graph.
                    //
                    // Scattering correctness: the input was detached above, so this recompute graph is
                    // self-contained — its only leaves are the detached input and the segment's own
                    // parameters; it does NOT reach into earlier segments, so each leaf is computed
                    // exactly once and the outer reverse walk computes earlier segments exactly once
                    // too (no double counting). The recompute's throwaway intermediates are fresh
                    // instances the caller never queries — harmless. The sole exclusion is gradOutput:
                    // an outer-tape constant folded into the pseudo-loss only to seed the VJP, whose
                    // inner "gradient" (== reOutput) is not a real gradient and must not leak back.
                        Tensor<T>[]? recomputeSources = null;
                        if (checkpointInputs is not null)
                        {
                            recomputeSources = new Tensor<T>[inputs.Length];
                            recomputeSources[0] = reInputDetached;
                            for (int i = 1; i < inputs.Length; i++) recomputeSources[i] = inputs[i];
                        }
                        var segGrads = recomputeTape.ComputeGradients(pseudoLoss, recomputeSources);

                        // Segment-recompute arrays must not escape their bounded arena. Preserve only
                        // the selected VJPs in ordinary owned storage; after this scope disposes, every
                        // other activation/temporary is immediately reusable by the next block.
                        detachedGradients = new List<(Tensor<T>, Tensor<T>)>(segGrads.Count);
                        foreach (var kvp in segGrads)
                        {
                            if (ReferenceEquals(kvp.Key, gradOutput) || kvp.Value is null) continue;
                            var key = ReferenceEquals(kvp.Key, reInputDetached) ? reInput : kvp.Key;
                            var copied = new Tensor<T>(kvp.Value.AsSpan().ToArray(), kvp.Value.Shape.ToArray());
                            detachedGradients.Add((key, copied));
                        }

                        if (detachedGradients.Count == 0 && ReferenceEquals(reOutput, reInputDetached))
                        {
                            var copied = new Tensor<T>(gradOutput.AsSpan().ToArray(), gradOutput.Shape.ToArray());
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
