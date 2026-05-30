using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Zero-toolchain, zero-warmup, near-allocation-free compiled MLP inference
/// primitive (compiled-inference plan, Phase 3). Steady-state <see cref="Run"/>
/// allocates only a small bounded constant (incidental GEMM thread-cache
/// bookkeeping), never a per-call buffer or Tensor — vs the eager path's
/// KB-per-layer. Prepacks every weight once at
/// <see cref="Create"/> (persistent cached-B — no per-call PackB), owns two
/// ping-pong scratch buffers, and <see cref="Run"/> executes the whole forward
/// with no per-call allocation and no per-op dispatch: each layer is a managed
/// cached-B GEMM + in-place bias/activation, ping-ponging through the scratch
/// buffers, then one copy of the final buffer into the caller's output.
///
/// <para>This is the serving-grade analog of a <c>torch.compile</c>'d module, but
/// built entirely in-process with the .NET JIT — no external C++ compiler
/// (TorchInductor CPU requires <c>cl.exe</c>/<c>gcc</c>) and no multi-second
/// first-call compile. <see cref="Create"/> is the only place state is allocated;
/// after that, steady-state inference is allocation-free, so a long-lived serving
/// process keeps prepacked weights + scratch hot across requests.</para>
///
/// <para>Output is numerically identical to <c>CpuEngine.MlpForward</c> (same GEMM
/// kernel + same epilogue) — verified by parity tests.</para>
/// </summary>
internal sealed class CompiledMlp
{
    private readonly float[][] _weights;
    private readonly float[]?[] _biases;
    private readonly int[] _inFeat;   // K per layer
    private readonly int[] _outFeat;  // N per layer
    private readonly FusedActivationType[] _act;
    private readonly FusedActivationParams?[] _actParams;
    private readonly int _maxBatch;
    private readonly float[] _bufA;
    private readonly float[] _bufB;

    /// <summary>Input feature count (K of the first layer).</summary>
    public int InputFeatures => _inFeat[0];

    /// <summary>Output feature count (N of the last layer).</summary>
    public int OutputFeatures => _outFeat[_outFeat.Length - 1];

    /// <summary>Maximum batch size this plan was sized for.</summary>
    public int MaxBatch => _maxBatch;

    private CompiledMlp(float[][] weights, float[]?[] biases, int[] inFeat, int[] outFeat,
        FusedActivationType[] act, FusedActivationParams?[] actParams, int maxBatch,
        float[] bufA, float[] bufB)
    {
        _weights = weights; _biases = biases; _inFeat = inFeat; _outFeat = outFeat;
        _act = act; _actParams = actParams; _maxBatch = maxBatch; _bufA = bufA; _bufB = bufB;
    }

    /// <summary>
    /// Builds a compiled MLP plan and prepacks all weights for batches up to
    /// <paramref name="maxBatch"/>. The hidden activation applies to every layer
    /// except the last, which uses the output activation (default: none / raw
    /// logits — the common classification head).
    /// </summary>
    public static CompiledMlp Create(
        IReadOnlyList<float[]> weights,
        IReadOnlyList<float[]?> biases,
        IReadOnlyList<int> inFeatures,
        IReadOnlyList<int> outFeatures,
        FusedActivationType hiddenActivation,
        FusedActivationType outputActivation = FusedActivationType.None,
        int maxBatch = 256,
        FusedActivationParams? hiddenActivationParams = null,
        FusedActivationParams? outputActivationParams = null)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (biases is null) throw new ArgumentNullException(nameof(biases));
        int layers = weights.Count;
        if (layers == 0) throw new ArgumentException("CompiledMlp requires at least one layer.", nameof(weights));
        if (biases.Count != layers || inFeatures.Count != layers || outFeatures.Count != layers)
            throw new ArgumentException("weights/biases/inFeatures/outFeatures counts must match.");
        if (maxBatch < 1) throw new ArgumentOutOfRangeException(nameof(maxBatch));

        var w = new float[layers][];
        float[]?[] b = new float[layers][];
        var inF = new int[layers];
        var outF = new int[layers];
        var act = new FusedActivationType[layers];
        var ap = new FusedActivationParams?[layers];
        int last = layers - 1;
        int runK = inFeatures[0];
        int maxN = 0;
        for (int i = 0; i < layers; i++)
        {
            int k = inFeatures[i], n = outFeatures[i];
            if (k != runK)
                throw new ArgumentException($"Layer {i} in-features ({k}) must equal previous out-features ({runK}).");
            if (weights[i] is null || weights[i].Length < (long)k * n)
                throw new ArgumentException($"Layer {i} weight must have at least {k}*{n} elements.");
            if (biases[i] is not null && biases[i]!.Length < n)
                throw new ArgumentException($"Layer {i} bias must have at least {n} elements.");
            w[i] = weights[i];
            b[i] = biases[i];
            inF[i] = k; outF[i] = n;
            act[i] = i == last ? outputActivation : hiddenActivation;
            ap[i] = i == last ? outputActivationParams : hiddenActivationParams;
            runK = n;
            if (n > maxN) maxN = n;
        }

        // Two ping-pong buffers, each sized to the largest layer output at maxBatch.
        long bufLen = (long)maxBatch * maxN;
        if (bufLen > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(maxBatch), "maxBatch * maxN exceeds array limit.");
        var bufA = new float[(int)bufLen];
        var bufB = new float[(int)bufLen];
        var plan = new CompiledMlp(w, b, inF, outF, act, ap, maxBatch, bufA, bufB);

        // Warm the persistent prepacked-B cache (keyed by weight-array identity)
        // for every layer at batch 1, so the first real Run hits the cache and
        // pays no PackB. Output discarded.
        var dummyIn = new float[inF[0]];
        var dummyOut = new float[plan.OutputFeatures];
        plan.Run(dummyIn, 1, dummyOut);
        return plan;
    }

    /// <summary>
    /// Runs the forward pass for <paramref name="batch"/> rows. <paramref name="input"/>
    /// is [batch, InputFeatures] row-major; <paramref name="output"/> receives
    /// [batch, OutputFeatures]. Allocation-free in steady state (one final copy
    /// of the result buffer into <paramref name="output"/>).
    /// </summary>
    public void Run(float[] input, int batch, float[] output)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (batch < 1 || batch > _maxBatch)
            throw new ArgumentOutOfRangeException(nameof(batch), $"batch must be in [1, {_maxBatch}].");
        if (input.Length < (long)batch * _inFeat[0])
            throw new ArgumentException($"input must have at least batch*{_inFeat[0]} elements.", nameof(input));
        if (output.Length < (long)batch * OutputFeatures)
            throw new ArgumentException($"output must have at least batch*{OutputFeatures} elements.", nameof(output));

        int layers = _weights.Length;
        float[] src = input;
        int curK = _inFeat[0];
        // Ping-pong: layer i writes into dst, which becomes the next src.
        for (int i = 0; i < layers; i++)
        {
            int n = _outFeat[i];
            float[] dst = (i % 2 == 0) ? _bufA : _bufB;   // src=input(layer0) → bufA → bufB → bufA …

            SimdGemm.SgemmWithCachedB(
                src.AsSpan(0, batch * curK), _weights[i], dst.AsSpan(0, batch * n), batch, curK, n);

            var bArr = _biases[i];
            if (bArr is not null || _act[i] != FusedActivationType.None)
                CpuFusedOperations.ApplyBiasActivationInPlace(dst, bArr, batch, n, _act[i], _actParams[i]);

            src = dst;
            curK = n;
        }

        // src now holds the final layer output in a ping-pong buffer; copy out.
        Array.Copy(src, 0, output, 0, batch * OutputFeatures);
    }
}
