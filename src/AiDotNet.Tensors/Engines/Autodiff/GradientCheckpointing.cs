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
            using (GradientTape<T>.NoGrad())
            {
                segmentOutput = segmentInput;
                for (int i = startIdx; i < endIdx; i++)
                    segmentOutput = functions[i](segmentOutput);
            }

            // Record a single "checkpoint" op that recomputes during backward
            int capturedStart = startIdx;
            int capturedEnd = endIdx;
            var capturedFunctions = functions;

            DifferentiableOps.RecordUnary<T>(
                $"Checkpoint_seg{seg}",
                segmentOutput,
                segmentInput,
                (gradOutput, inputs, output, savedState, eng, grads) =>
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
                    using var recomputeTape = new GradientTape<T>(
                        new GradientTapeOptions { Persistent = false });
                    var reInput = inputs[0];
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
                    var segGrads = recomputeTape.ComputeGradients(pseudoLoss, sources: null);

                    bool accumulatedAny = false;
                    foreach (var kvp in segGrads)
                    {
                        if (ReferenceEquals(kvp.Key, gradOutput)) continue;
                        if (kvp.Value is null) continue;
                        // The detached input's gradient belongs to the ORIGINAL segment-input tensor
                        // the caller tracks (and that the outer walk hands to the previous segment);
                        // remap it. Every other key is a live parameter the segment read directly.
                        var key = ReferenceEquals(kvp.Key, reInputDetached) ? reInput : kvp.Key;
                        DifferentiableOps.AccumulateGrad(grads, key, kvp.Value, eng);
                        accumulatedAny = true;
                    }

                    // Identity / no-op segment (output IS input by reference, nothing recorded on
                    // the recompute tape): pass the upstream gradient straight through. Reference-
                    // equality is the only safe alias predicate — it covers the empty-segment case
                    // without false positives on shape coincidence. For a genuinely input-
                    // independent segment, leaving grads[input] untouched (zero) is the correct VJP.
                    if (!accumulatedAny && ReferenceEquals(reOutput, reInputDetached))
                    {
                        // Empty / no-op segment (no functions ran): pass the upstream gradient straight
                        // through to the original input.
                        DifferentiableOps.AccumulateGrad(grads, reInput, gradOutput, eng);
                    }
                });

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
