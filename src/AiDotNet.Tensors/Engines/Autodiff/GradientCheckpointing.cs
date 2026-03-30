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
                    // RECOMPUTE: run the segment forward again WITH recording
                    // so backward can flow through the ops
                    using var recomputeTape = new GradientTape<T>();
                    var reInput = inputs[0];
                    var reOutput = reInput;
                    for (int i = capturedStart; i < capturedEnd; i++)
                        reOutput = capturedFunctions[i](reOutput);

                    // Compute gradients through the recomputed segment
                    var segGrads = recomputeTape.ComputeGradients(reOutput, sources: new[] { reInput });

                    if (segGrads.TryGetValue(reInput, out var inputGrad))
                    {
                        // Chain rule: multiply segment gradient by upstream gradient
                        var chained = eng.TensorMultiply(gradOutput, inputGrad);
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], chained, eng);
                    }
                    else
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], gradOutput, eng);
                    }
                });

            current2 = segmentOutput;
        }

        return current2;
    }
}
