using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
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
        FusedActivationType outputActivation = FusedActivationType.None,
        FusedActivationParams? hiddenActivationParams = null,
        FusedActivationParams? outputActivationParams = null)
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

        int last = weights.Count - 1;

        // Phase 3 (compiled-inference): whole-MLP ping-pong fast path for float.
        // Chains every layer through two reused scratch buffers — no per-layer
        // Tensor allocation, no per-layer FusedLinear preamble, intermediates stay
        // cache-resident — with the LAST layer writing straight into the result
        // (no final copy). Each layer's GEMM follows the Phase-1 routing gate
        // (managed cached-B prepack vs native BLAS); bias+activation is applied in
        // place. Output is numerically identical to the per-layer FusedLinear chain.
        if (typeof(T) == typeof(float) && input.Rank == 2 && input.IsContiguous && !input.IsSparse)
        {
            int M = input._shape[0];
            int k0 = input._shape[1];
            bool eligible = true;
            int runK = k0, finalN = 0, maxIntermediate = 0;
            for (int i = 0; i < weights.Count; i++)
            {
                var w = weights[i];
                var bi = biases[i];
                if (w.Rank != 2 || !w.IsContiguous || w.IsSparse || w._shape[0] != runK
                    || (bi is not null && (!bi.IsContiguous || bi.Length < w._shape[1])))
                {
                    eligible = false;
                    break;
                }
                runK = w._shape[1];
                finalN = runK;
                if (i != last && (long)M * runK > maxIntermediate) maxIntermediate = M * runK;
            }

            if (eligible)
            {
                var result = AutoTensorCache.RentOrAllocate<T>(new[] { M, finalN });
                var outArr = (float[])(object)result.GetDataArray();
                var pool = System.Buffers.ArrayPool<float>.Shared;
                float[] bufA = maxIntermediate > 0 ? pool.Rent(maxIntermediate) : System.Array.Empty<float>();
                float[] bufB = maxIntermediate > 0 ? pool.Rent(maxIntermediate) : System.Array.Empty<float>();
                try
                {
                    // #475: read the input/weights through AsSpan (no copy) rather than
                    // GetDataArray. GetDataArray hands back a *copy* for any tensor whose
                    // _device != CPU — and the GPU auto-detect ModuleInitializer makes the
                    // default engine (hence freshly-created tensors) GPU-resident on any box
                    // with a working accelerator, even when the data lives in a CPU array.
                    // That made this CPU primitive re-copy every weight on every call
                    // (~1.5 MB for the 784×512 head alone), dominating per-call allocation
                    // and feeding the GC tail. AsSpan returns the live CPU span after
                    // EnsureMaterialized, so it's a no-op on genuinely CPU-resident tensors
                    // and skips the per-call snapshot on CPU-resident-but-GPU-tagged ones.
                    // (We're inside the typeof(T)==float branch, so the (object) casts are
                    // exact reference casts, not boxing.)
                    var inputF = (Tensor<float>)(object)input;
                    ReadOnlySpan<float> srcSpan = inputF.AsSpan();
                    int curK = k0;
                    int pingToggle = 0;
                    for (int i = 0; i < weights.Count; i++)
                    {
                        var w = weights[i];
                        int n = w._shape[1];
                        // Last layer writes directly into the result buffer (no copy).
                        float[] dst = i == last ? outArr : (pingToggle == 0 ? bufA : bufB);

                        // Per-layer kernel choice: a WIDE layer (large K·N) goes to native
                        // BLAS at EVERY batch — including the small-M latency regime.
                        // The shared PreferManagedInferenceGemm gate keeps wide layers on
                        // managed cached-B at M<16, but measurement (Phase5/Phase7 probes)
                        // shows native BLAS wins the wide classifier-head layer
                        // (e.g. 784→512) even at M=1; routing it to managed made MlpForward
                        // ~6.7× slower than the self-tuned CompiledMlp at M=1. Managed
                        // cached-B still wins the small layers (low K·N), where native
                        // BLAS dispatch overhead dominates. Mirror that split here without
                        // changing PreferManagedInferenceGemm (which FusedLinear also uses).
                        // #475 AsSpan contract (see the comment above the loop): read the weight
                        // as the live CPU span — no GPU-tagged snapshot copy. The JIT and native
                        // paths use this span directly; the managed cached-B path additionally
                        // needs the STABLE backing float[] (GetFlattenedData) for its
                        // identity-keyed pre-pack. AsSpan also materializes lazy data once.
                        var wT = (Tensor<float>)(object)w;
                        ReadOnlySpan<float> wSpan = wT.AsSpan();

                        bool gemmDone = false;
#if !NET471
                        // JIT'd AVX2 kernel first (opt-in) — beats both managed cached-B
                        // and native BLAS on the MLP shapes; this is the path the MLP
                        // actually takes (so without it the JIT never reached the MLP).
                        if (_jitGemm && Simd.JitGemmAvx2.TryMultiply(
                                srcSpan.Slice(0, M * curK), wSpan.Slice(0, curK * n), dst.AsSpan(0, M * n), M, n, curK))
                            gemmDone = true;
#endif
                        bool preferManaged = (long)curK * n <= 200_000 || !BlasProvider.HasRawSgemm;
                        if (gemmDone)
                        {
                            // done by JIT
                        }
                        else if (preferManaged)
                        {
                            // #475: at these small managed shapes (curK*n <= 200K) the direct
                            // serial AVX2 microkernel beats the BlasManaged cached-B dispatch.
                            // For small K/N the pack overhead isn't amortized across the M-loop,
                            // so the packing machinery costs more than it saves — measured on a
                            // Ryzen 9 3950X (min-of-2000, B-pack already warm): the AIsEval
                            // layers L1 128x512x128 +12% (86→97 GF/s) and L2 128x128x10 +81%
                            // (15.6→28.2 GF/s). Same finding as the LSTM routing fix (#503).
                            // These shapes are far below the parallel threshold (20M), so the
                            // dispatch path runs serial anyway — SgemmSequential loses no
                            // parallelism, is concurrency-safe (no shared pack-cache state),
                            // and reads wSpan directly (no GetFlattenedData() float[] copy, no
                            // pack-cache identity concerns). (wT/wSpan hoisted above.)
                            Simd.SimdGemm.SgemmSequential(
                                srcSpan.Slice(0, M * curK), wSpan.Slice(0, curK * n),
                                dst.AsSpan(0, M * n), M, curK, n);
                        }
                        else
                        {
                            unsafe
                            {
                                fixed (float* ps = srcSpan, pw = wSpan, pd = dst)
                                {
                                    if (BlasProvider.HasRawSgemm)
                                        BlasProvider.SgemmRaw(M, n, curK, ps, curK, pw, n, pd, n);
                                    else
                                        // Unreachable in practice (preferManaged already captures
                                        // !HasRawSgemm); kept as a defensive fallback, on the same
                                        // stable-array contract as the preferManaged branch above.
                                        Simd.SimdGemm.SgemmWithCachedB(
                                            srcSpan.Slice(0, M * curK), wT.GetFlattenedData(),
                                            dst.AsSpan(0, M * n), M, curK, n);
                                }
                            }
                        }

                        var act = i == last ? outputActivation : hiddenActivation;
                        var ap = i == last ? outputActivationParams : hiddenActivationParams;
                        var bi = biases[i];
                        var bArr = bi is null ? null : (float[])(object)bi.GetDataArray();
                        if (bArr is not null || act != FusedActivationType.None)
                            CpuFusedOperations.ApplyBiasActivationInPlace(dst, bArr, M, n, act, ap);

                        srcSpan = dst.AsSpan(0, M * n);
                        curK = n;
                        if (i != last) pingToggle ^= 1;
                    }
                    return result;
                }
                finally
                {
                    if (maxIntermediate > 0) { pool.Return(bufA); pool.Return(bufB); }
                }
            }
        }

        // Generic fallback: chain the fused linear layers. FusedLinear validates
        // each weight/bias shape against the running activation, so a layer-to-layer
        // feature mismatch surfaces as a clear per-layer ArgumentException.
        var x = input;
        for (int i = 0; i < weights.Count; i++)
        {
            var activation = i == last ? outputActivation : hiddenActivation;
            var actParams = i == last ? outputActivationParams : hiddenActivationParams;
            x = FusedLinear(x, weights[i], biases[i], activation, actParams);
        }

        return x;
    }
}
