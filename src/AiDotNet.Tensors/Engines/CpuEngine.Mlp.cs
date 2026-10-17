using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // Multiply-adds (M * K * N) at or above which an MLP layer's managed GEMM goes through the parallel Sgemm dispatch
    // instead of the single-threaded SgemmSequential (#475).
    internal const long MlpParallelGemmMinWork = 1L << 20;

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
    [MethodImpl(Hot)]
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
        {
            if (!GraphMode.IsInferenceTrace)
            {
                var trainingValue = input;
                int trainingLast = weights.Count - 1;
                for (int i = 0; i < weights.Count; i++)
                {
                    var activation = i == trainingLast ? outputActivation : hiddenActivation;
                    var activationParams = i == trainingLast ? outputActivationParams : hiddenActivationParams;
                    trainingValue = FusedLinear(trainingValue, weights[i], biases[i], activation, activationParams);
                }
                return trainingValue;
            }

            if (input.Rank < 1)
                throw new ArgumentException("MlpForward input must have at least one dimension.", nameof(input));

            int runningFeatures = input._shape[input.Rank - 1];
            var capturedWeights = new Tensor<T>[weights.Count];
            var capturedBiases = new Tensor<T>?[biases.Count];
            var biasInputIndices = new int[biases.Count];
            var graphInputs = new List<Tensor<T>>(1 + weights.Count + biases.Count) { input };
            for (int i = 0; i < weights.Count; i++)
            {
                var weight = weights[i]
                    ?? throw new ArgumentException($"weights[{i}] cannot be null.", nameof(weights));
                if (weight.Rank != 2 || weight._shape[0] != runningFeatures)
                    throw new ArgumentException(
                        $"weights[{i}] must be [{runningFeatures}, outFeatures]; got [{string.Join(",", weight._shape)}].",
                        nameof(weights));
                var bias = biases[i];
                if (bias is not null && (bias.Rank != 1 || bias._shape[0] != weight._shape[1]))
                    throw new ArgumentException(
                        $"biases[{i}] must be [{weight._shape[1]}].", nameof(biases));

                capturedWeights[i] = weight;
                capturedBiases[i] = bias;
                graphInputs.Add(weight);
                runningFeatures = weight._shape[1];
            }
            for (int i = 0; i < capturedBiases.Length; i++)
            {
                if (capturedBiases[i] is null)
                {
                    biasInputIndices[i] = -1;
                }
                else
                {
                    biasInputIndices[i] = graphInputs.Count;
                    graphInputs.Add(capturedBiases[i]!);
                }
            }

            var outputShape = (int[])input._shape.Clone();
            outputShape[outputShape.Length - 1] = runningFeatures;
            var scope = GraphMode.Current!; scope.BindEngineIfUnset(this);
            var capturedInputs = graphInputs.ToArray();
            object[] savedState =
            {
                biasInputIndices,
                hiddenActivation,
                outputActivation,
                hiddenActivationParams!,
                outputActivationParams!
            };
            return scope.RecordVariadic(
                LazyNodeType.MlpForward,
                "MlpForward",
                capturedInputs,
                outputShape,
                (eng, output) =>
                {
                    var eager = eng.MlpForward(
                        capturedInputs[0], capturedWeights, capturedBiases,
                        hiddenActivation, outputActivation,
                        hiddenActivationParams, outputActivationParams);
                    DirectGpuTensorEngine.CopyResultInto(eng, eager, output);
                }, backwardFn: AiDotNet.Tensors.Engines.Autodiff.BackwardFunctions<T>.ReplayUnderTape(eng => eng.MlpForward(
                        capturedInputs[0], capturedWeights, capturedBiases,
                        hiddenActivation, outputActivation,
                        hiddenActivationParams, outputActivationParams)),
                savedState: savedState);
        }

        // #436: row stripes. Rows of an MLP are independent through every layer, so one dispatch can run each worker's
        // stripe of rows through ALL the layers: no barrier between layers, no full-batch intermediate, and the bias and
        // ReLU applied in registers. The weights are read in place by the unpacked small-row kernel (they stay in cache
        // across calls); the per-layer path below pays a fork/join per GEMM plus separate bias/activation passes.
        if (typeof(T) == typeof(float)
            && TryMlpRowStripes(input, weights, biases, hiddenActivation, outputActivation, out var striped))
            return striped!;

        int last = weights.Count - 1;

        // Phase 3 (compiled-inference): whole-MLP ping-pong fast path for float.
        // Chains every layer through two reused scratch buffers — no per-layer
        // Tensor allocation, no per-layer FusedLinear preamble, intermediates stay
        // cache-resident — with the LAST layer writing straight into the result
        // (no final copy). Bias+activation is applied in place. Output is numerically identical to the per-layer FusedLinear chain.
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
                var outArr = (float[])(object)result.GetDataArrayUnmarked();
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
                    using var srcSpanLease = inputF.Lease();
                    ReadOnlySpan<float> srcSpan = srcSpanLease.Span;
                    int curK = k0;
                    int pingToggle = 0;
                    for (int i = 0; i < weights.Count; i++)
                    {
                        var w = weights[i];
                        int n = w._shape[1];
                        // Last layer writes directly into the result buffer (no copy).
                        float[] dst = i == last ? outArr : (pingToggle == 0 ? bufA : bufB);

                        // #475 AsSpan contract (see the comment above the loop): read the weight
                        // as the live CPU span — no GPU-tagged snapshot copy. The JIT and managed
                        // paths use this span directly. AsSpan also materializes lazy data once.
                        var wT = (Tensor<float>)(object)w;
                        using var wSpanLease = wT.Lease();
                        ReadOnlySpan<float> wSpan = wSpanLease.Span;

                        bool gemmDone = false;
#if NETCOREAPP
                        // JIT'd AVX2 kernel first (opt-in) — beats managed cached-B
                        // on the MLP shapes; this is the path the MLP
                        // actually takes (so without it the JIT never reached the MLP).
                        if (_jitGemm && Simd.JitGemmAvx2.TryMultiply(
                                srcSpan.Slice(0, M * curK), wSpan.Slice(0, curK * n), dst.AsSpan(0, M * n), M, n, curK))
                            gemmDone = true;
#endif
                        if (!gemmDone)
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
                            // #475 (2026-10): every layer lands here, including the wide head. SgemmSequential is single-threaded, so the AIsEval
                            // layer 0 (128x784x512) took ~1.45 ms at any thread budget on a 64-core Threadripper,
                            // while the parallel Sgemm dispatch took 0.17 ms; layer 1 (128x512x128) 160 -> 40 us.
                            // Below MlpParallelGemmMinWork the sequential kernel still wins (layer 2, 128x128x10:
                            // 10.8 us vs 63.6 us at one thread), so only the larger layers dispatch.
                            if ((long)M * curK * n >= MlpParallelGemmMinWork)
                                BlasManaged.BlasManaged.Gemm<float>(
                                    srcSpan.Slice(0, M * curK), lda: curK, transA: false,
                                    wSpan.Slice(0, curK * n), ldb: n, transB: false,
                                    dst.AsSpan(0, M * n), ldc: n,
                                    M, n, curK,
                                    new BlasManaged.BlasOptions<float> { PackingMode = BlasManaged.PackingMode.DisableAutotune });
                            else
                                Simd.SimdGemm.SgemmSequential(
                                    srcSpan.Slice(0, M * curK), wSpan.Slice(0, curK * n),
                                    dst.AsSpan(0, M * n), M, curK, n);
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

    /// <summary>
    /// #436: <see cref="MlpForward{T}"/> as row stripes (see the call site). On by default; AIDOTNET_MLP_ROW_STRIPES=0
    /// keeps the per-layer path.
    /// </summary>
    internal static bool MlpRowStripesEnabled = Environment.GetEnvironmentVariable("AIDOTNET_MLP_ROW_STRIPES") != "0";

    // Rows per stripe at least this (the kernel's register tile is 4 rows), and two stripes at least, or the per-layer
    // path keeps the call.
    private const int MlpRowStripeMinRows = 4;

    // Each stripe streams every weight from cache: keep a layer's weight within a chiplet's share of L3 (8 MB).
    private const long MlpRowStripeMaxWeightFloats = 2L * 1024 * 1024;

    private static unsafe bool TryMlpRowStripes<T>(
        Tensor<T> input, IReadOnlyList<Tensor<T>> weights, IReadOnlyList<Tensor<T>?> biases,
        FusedActivationType hiddenActivation, FusedActivationType outputActivation, out Tensor<T>? result)
    {
        result = null;
        if (!MlpRowStripesEnabled || !Simd.SimdGemm.RowStripeKernelSupported) return false;
        if (hiddenActivation is not (FusedActivationType.ReLU or FusedActivationType.None)
            || outputActivation is not (FusedActivationType.ReLU or FusedActivationType.None)) return false;
        if (input.Rank != 2 || input.IsSparse) return false;
        int m = input._shape[0];
        int layers = weights.Count;
        if (m < 2 * MlpRowStripeMinRows || layers == 0) return false;

        var dims = new int[layers + 1];
        dims[0] = input._shape[1];
        int widest = 0;
        for (int i = 0; i < layers; i++)
        {
            var w = weights[i];
            if (w is null || w.Rank != 2 || w.IsSparse || w._shape[0] != dims[i]) return false;
            int n = w._shape[1];
            if ((long)dims[i] * n > MlpRowStripeMaxWeightFloats) return false;
            var bias = biases[i];
            if (bias is not null && (bias.Rank != 1 || bias.Length < n)) return false;
            dims[i + 1] = n;
            if (i < layers - 1) widest = Math.Max(widest, n);
        }

        int participants = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism,
            PersistentParallelExecutor.Instance.MaxParticipants);
        int stripes = Math.Min(participants, m / MlpRowStripeMinRows);
        if (stripes < 2) return false;
        int rowsPer = (m + stripes - 1) / stripes;
        rowsPer = (rowsPer + 3) / 4 * 4;   // whole register tiles where the rows allow
        stripes = (m + rowsPer - 1) / rowsPer;

        // Read-only arrays without the escape mark: these tensors are kept alive below until the dispatch is done.
        var x = ((Tensor<float>)(object)input).GetReadOnlyDataArrayUnmarked();
        var wArr = new float[layers][];
        var bArr = new float[]?[layers];
        for (int i = 0; i < layers; i++)
        {
            wArr[i] = ((Tensor<float>)(object)weights[i]).GetReadOnlyDataArrayUnmarked();
            bArr[i] = biases[i] is { } b ? ((Tensor<float>)(object)b).GetReadOnlyDataArrayUnmarked() : null;
        }

        var output = AutoTensorCache.RentOrAllocate<float>(new[] { m, dims[layers] });
        using (var outLease = output.LeaseArray())
        {
            float[] outArr = outLease.Array;
            bool hiddenRelu = hiddenActivation == FusedActivationType.ReLU;
            bool outputRelu = outputActivation == FusedActivationType.ReLU;
            PersistentParallelExecutor.Instance.Execute(stripes, [MethodImpl(Hot)] (stripe) =>
            {
                int r0 = stripe * rowsPer;
                int rows = Math.Min(rowsPer, m - r0);
                if (rows <= 0) return;
                var pool = System.Buffers.ArrayPool<float>.Shared;
                float[] ping = widest > 0 ? pool.Rent(rows * widest) : System.Array.Empty<float>();
                float[] pong = widest > 0 ? pool.Rent(rows * widest) : System.Array.Empty<float>();
                try
                {
                    float[] src = x;
                    int srcOffset = r0 * dims[0];
                    for (int i = 0; i < layers; i++)
                    {
                        int k = dims[i], n = dims[i + 1];
                        bool last = i == layers - 1;
                        float[] dst = last ? outArr : (i % 2 == 0 ? ping : pong);
                        int dstOffset = last ? r0 * n : 0;
                        fixed (float* ps = src)
                        fixed (float* pw = wArr[i])
                        fixed (float* pb = bArr[i])
                        fixed (float* pd = dst)
                            Simd.SimdGemm.RowStripeGemmBiasAct(ps + srcOffset, k, pw, n, pd + dstOffset, n,
                                rows, k, n, pb, last ? outputRelu : hiddenRelu);
                        src = dst;
                        srcOffset = 0;
                    }
                }
                finally
                {
                    if (widest > 0) { pool.Return(ping); pool.Return(pong); }
                }
            });
        }
        GC.KeepAlive(input);
        GC.KeepAlive(weights);
        GC.KeepAlive(biases);
        result = (Tensor<T>)(object)output;
        return true;
    }
}
