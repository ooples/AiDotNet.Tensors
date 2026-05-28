using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Fused multi-layer perceptron forward — runs a stack of dense layers
    /// (<c>activation(x @ Wᵢ + bᵢ)</c>) in a single engine call. Returns the
    /// final layer's output <c>[..., outFeatures]</c>.
    ///
    /// <para>
    /// Equivalent to chaining <see cref="FusedLinear{T}"/> per layer:
    /// </para>
    /// <code>
    /// x = FusedLinear(x, W0, b0, hiddenActivation)   // layer 0
    /// x = FusedLinear(x, W1, b1, hiddenActivation)   // layer 1
    /// ...
    /// x = FusedLinear(x, Wn, bn, outputActivation)   // last layer
    /// </code>
    ///
    /// <para>
    /// Why the wrapper exists: the AIsEval fair-comparison rerun (issue #436 P1)
    /// showed MLP <c>Predict()</c> at <c>bs=128</c> was <b>4.7× slower than
    /// PyTorch</b> (8.94 ms vs 1.91 ms) on
    /// <c>Dense(784→512)→Dense(512→128)→Dense(128→10)</c>. The issue attributes
    /// this to dispatch overhead, not peak compute: the framework's per-layer
    /// <c>DenseLayer.Forward</c> issues a separate <c>MatMul</c> + <c>Add</c> +
    /// activation — nine autograd-tape-aware dispatches for a 3-layer net.
    /// Inference needs none of that tape bookkeeping. This primitive collapses
    /// the stack to one engine call of N fused linear steps (three dispatches
    /// instead of nine) and, by throwing on an active <c>GradientTape</c>,
    /// pays zero recording overhead. It mirrors the
    /// <see cref="LstmSequenceForward{T}(Tensor{T}, Tensor{T}?, Tensor{T}?, Tensor{T}, Tensor{T}, Tensor{T}?, Tensor{T}?, bool)"/>
    /// and <c>MultiHeadAttentionForward</c> primitives so the framework can
    /// adopt it the same way.
    /// </para>
    ///
    /// <para>
    /// <b>Forward-only.</b> Calling this under an active <c>GradientTape</c>
    /// throws — training keeps using the decomposed per-layer path (whose
    /// constituent <c>FusedLinear</c> ops record on the tape).
    /// </para>
    /// </summary>
    /// <param name="input">[..., inFeatures] input. The leading dims are preserved.</param>
    /// <param name="weights">
    /// Per-layer weight matrices, each <c>[inFeatures_i, outFeatures_i]</c>
    /// (the <see cref="FusedLinear{T}"/> layout). <c>weights[i].Shape[0]</c>
    /// must equal the previous layer's output feature count.
    /// </param>
    /// <param name="biases">
    /// Per-layer bias vectors, each <c>[outFeatures_i]</c> or null for no bias.
    /// Must have the same count as <paramref name="weights"/>.
    /// </param>
    /// <param name="hiddenActivation">Activation applied after every layer except the last.</param>
    /// <param name="outputActivation">
    /// Activation applied after the last layer. Defaults to
    /// <see cref="FusedActivationType.None"/> (raw logits — the common
    /// classification-head shape).
    /// </param>
    public virtual Tensor<T> MlpForward<T>(
        Tensor<T> input,
        IReadOnlyList<Tensor<T>> weights,
        IReadOnlyList<Tensor<T>?> biases,
        FusedActivationType hiddenActivation,
        FusedActivationType outputActivation = FusedActivationType.None)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (biases is null) throw new ArgumentNullException(nameof(biases));
        if (weights.Count == 0)
            throw new ArgumentException("MlpForward requires at least one layer.", nameof(weights));
        if (biases.Count != weights.Count)
            throw new ArgumentException(
                $"biases count ({biases.Count}) must equal weights count ({weights.Count}).",
                nameof(biases));

        if (GraphMode.IsActive)
            throw new InvalidOperationException(
                "MlpForward is inference-only and does not yet support GradientTape. " +
                "For training, call the decomposed FusedLinear / DenseLayer path which records each layer.");

        // Issue #436: small-batch inference GEMMs are oversubscribed by the
        // default all-cores native-BLAS thread count. On a 16-core box the
        // AIsEval MLP (bs=128) measured ~1.74 ms at 16 threads vs ~1.20 ms at
        // 8 — the per-GEMM thread-fan-out/sync cost dominates the tiny actual
        // compute (the 128×10×128 head is almost pure overhead). Cap the native
        // BLAS thread count to roughly one thread per 16 output rows (so each
        // thread keeps a cache-resident M-stripe of useful work), clamped to the
        // core count, for the span of the forward. The scope restores the prior
        // count on exit and is reference-counted for nested/concurrent callers.
        // Effective GEMM M is the product of every leading dimension ([..., inFeatures]);
        // using only _shape[0] would collapse a shape like [batch, time, features] to
        // near single-thread and skew the cap badly. Accumulate in a long and stop early
        // once the cap is already saturated (rows/16 >= ProcessorCount) to avoid overflow.
        long rows = 1;
        for (int d = 0; d < Math.Max(0, input.Rank - 1); d++)
        {
            rows *= input._shape[d];
            if (rows >= (long)Environment.ProcessorCount * 16) break;
        }
        // Math.Clamp isn't available on net471 — use Max/Min (see the same
        // pattern in CpuEngine.Geometry.cs / FusedPointwise.cs).
        int blasThreads = Math.Max(1, Math.Min((int)(rows / 16), Environment.ProcessorCount));
        using var _blasScope = Helpers.BlasProvider.ScopeOpenBlasThreads(blasThreads);

        // Chain the fused linear layers. FusedLinear validates each weight /
        // bias shape against the running activation, so a layer-to-layer
        // feature mismatch surfaces as a clear per-layer ArgumentException.
        int last = weights.Count - 1;
        var x = input;
        for (int i = 0; i < weights.Count; i++)
        {
            var activation = i == last ? outputActivation : hiddenActivation;
            x = FusedLinear(x, weights[i], biases[i], activation);
        }

        return x;
    }
}
