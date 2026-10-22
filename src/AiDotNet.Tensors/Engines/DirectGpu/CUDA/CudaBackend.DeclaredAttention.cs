// Copyright (c) AiDotNet. All rights reserved.
// Scaled dot-product attention on a declared program (#890 phase 4d): CudaBackend.ScaledDotProductAttention -- causal,
// masked (shared or full), grouped-query, with or without the weights stored -- runs
// CodegenBackendPrograms.ScaledDotProductAttention: the scores as an implicit GEMM with the scale and mask bias in its
// epilogue, a row softmax, and the weighted values as a second implicit GEMM. The hand-written kernel it replaces
// recomputes every score once per output dimension (seqK * headDim^2 multiply-adds a row). The backward
// (ScaledDotProductAttentionBackward) runs its six products and the softmax gradient in place of three batched
// transposes, four GEMMs, a softmax backward and a scale. A logit soft-cap (tanh)
// keeps the hand-written kernel: its bias would have to land after the cap, which the GEMM epilogue's order does not
// allow.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    // Slots of an attention call; "weights" last, so a call that does not keep them binds the first six and the program
    // owns the weights as scratch.
    private static readonly string[] s_attentionNames = { "q", "k", "v", "out", "mask", "causal", "weights" };
    private static readonly string[] s_attentionNamesNoWeights = { "q", "k", "v", "out", "mask", "causal" };
    // FlashAttentionV2's contract: no weights, the row log-sum-exp in the weights' slot.
    private static readonly string[] s_attentionNamesStats = { "q", "k", "v", "out", "mask", "causal", "stats" };

    private const int AttentionMaskShift = 20, AttentionCausalBit = 1 << 24, AttentionWeightsBit = 1 << 25,
        AttentionStatsBit = 1 << 26;

    // The programs keep [batch, heads, seqQ, seqK] scratch (the logits; the weights when the caller does not keep them;
    // the backward's dP and dS) for as long as the backend lives, where the hand-written kernels need none: past this
    // many elements (256 MiB, the convolution im2col route's bound) the call keeps the hand-written path.
    private const long AttentionMaxScoreElements = 64L * 1024 * 1024;

    // Keys seen once. A program is compiled per shape, so a shape is routed from its SECOND call: one that recurs
    // (training, a fixed batch) pays one compile, while autoregressive decoding -- seqK one longer every token, every
    // shape new -- keeps the hand-written kernel instead of compiling a program per token.
    private readonly HashSet<DeclaredKey> _attentionSeenOnce = new();

    // Decoding gives this set one new key per token, so it needs a bound of its own: the programs are bounded by the
    // scratch budget, but a key that never recurs costs nothing but a set entry and would still grow without end.
    // At the cap the memo is dropped rather than closed, so a shape that starts recurring afterwards can still be
    // routed - it just needs two more sightings - while the set cannot exceed the cap.
    private const int AttentionSeenOnceCapacity = 4096;

    // Whether a key's program may be built now: true from its second sighting. The caller holds _directPtxLock.
    private bool AttentionShapeRecurs(DeclaredKey key)
    {
        if (_attentionSeenOnce.Contains(key)) return true;
        if (_attentionSeenOnce.Count >= AttentionSeenOnceCapacity) _attentionSeenOnce.Clear();
        _attentionSeenOnce.Add(key);
        return false;
    }

    // The causal bias of each (seqQ, seqK): 0 for a key at or before q + seqK - seqQ, AttentionMaskedLogit after.
    private readonly Dictionary<long, IGpuBuffer> _causalBiases = new();

    private Func<DeclaredKey, CodegenProgram>? _buildAttention;
    private Func<CodegenKernelSpec, PtxPassLowering?>? _lowerAttention, _lowerAttentionUnbounded;

    /// <summary>The key of an attention call; the scale is the key's epsilon slot.</summary>
    internal static DeclaredKey AttentionKey(int batch, int heads, int kvHeads, int seqQ, int seqK, int headDim, float scale,
        CodegenAttentionMask mask, bool causal, bool storeWeights, bool stats = false) =>
        new(DeclaredFamily.ScaledDotProductAttention, batch, heads, seqQ, seqK, scale, 0, null, kvHeads,
            headDim | ((int)mask << AttentionMaskShift) | (causal ? AttentionCausalBit : 0) |
            (storeWeights ? AttentionWeightsBit : 0) | (stats ? AttentionStatsBit : 0));

    private CodegenProgram BuildAttention(DeclaredKey key)
    {
        int headDim = key.D5 & ((1 << AttentionMaskShift) - 1);
        var mask = (CodegenAttentionMask)((key.D5 >> AttentionMaskShift) & 3);
        CodegenLibraryFunction? notEqual = null;
        if (mask != CodegenAttentionMask.None)
            notEqual = (ResolveElementwise("not_equal_scalar")
                ?? throw new NotSupportedException("not_equal_scalar has no derived function: " +
                    ElementwiseRefusal("not_equal_scalar"))).Handle;
        CodegenLibraryFunction? log = null;
        if ((key.D5 & AttentionStatsBit) != 0)
            log = (ResolveElementwise("log_vector")
                ?? throw new NotSupportedException("log_vector has no derived function: " +
                    ElementwiseRefusal("log_vector"))).Handle;
        return CodegenBackendPrograms.ScaledDotProductAttention(key.D0, key.D1, key.D4, key.D2, key.D3, headDim,
            key.Epsilon, mask, (key.D5 & AttentionCausalBit) != 0, notEqual, log);
    }

    // The contractions as implicit GEMMs; the row passes take the planner's own lowering.
    private PtxPassLowering? LowerAttentionPass(CodegenKernelSpec pass, int minBlocks) =>
        pass.Name is "attention_scores" or "attention_output" or "attention_dv" or "attention_dp" or "attention_dq"
            or "attention_dk"
            ? LowerDeclaredConv(pass, new DeclaredConvSchedule(64, 64, 0), minBlocks)
            : null;

    private static readonly string[] s_attentionBackwardNames = { "dout", "q", "k", "v", "weights", "dq", "dk", "dv" };
    private static readonly string[] s_flashBackwardNames =
        { "dout", "q", "k", "v", "out", "stats", "causal", "dq", "dk", "dv" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildFlashBackward = key =>
        CodegenBackendPrograms.FlashAttentionBackward(key.D0, key.D1, key.D2, key.D3, key.D5 & 0xFFFFF, key.Epsilon,
            (key.D5 & AttentionCausalBit) != 0);

    /// <summary>
    /// FlashAttentionBackward through its declared program: the weights recomputed from softmaxStats, then the
    /// attention backward's products. Not with a bias, nor causal across unequal lengths (the forward's routing rule).
    /// </summary>
    private bool TryDeclaredFlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key,
        IGpuBuffer value, IGpuBuffer output, IGpuBuffer softmaxStats, IGpuBuffer gradQuery, IGpuBuffer gradKey,
        IGpuBuffer gradValue, int batch, int heads, int seqQ, int seqK, int headDim, float scale, bool causal)
    {
        if (!DeclaredKernelsEnabled || batch <= 0 || heads <= 0 || seqQ <= 0 || seqK <= 0 || headDim <= 0 ||
            headDim >= 1 << AttentionMaskShift || (causal && seqQ != seqK) ||
            (long)batch * heads * seqQ * seqK > AttentionMaxScoreElements)
            return false;
        var declaredKey = new DeclaredKey(DeclaredFamily.FlashAttentionBackward, batch, heads, seqQ, seqK, scale, 0, null,
            0, headDim | (causal ? AttentionCausalBit : 0));
        IGpuBuffer? causalBias = null;
        DeclaredProgram? program;
        lock (_directPtxLock)
        {
            if (!_declaredPrograms.TryGetValue(declaredKey, out program))
            {
                if (IsStreamCapturing() || !AttentionShapeRecurs(declaredKey)) return false;
                program = BuildAttentionProgram(declaredKey, s_buildFlashBackward, s_flashBackwardNames);
                _declaredPrograms[declaredKey] = program;
            }
            if (program is not null && causal &&
                (causalBias = CausalBiasBuffer(seqQ, seqK, out string? biasDecline)) is null)
            {
                if (biasDecline is not null) _declaredDeclines[declaredKey] = biasDecline;
                return false;
            }
        }
        if (program is null) return false;
        var bound = new DeclaredBindings(gradOutput, query, key, value, output, softmaxStats, causalBias, gradQuery,
            gradKey, gradValue);
        if (!AlignedFor(program, s_flashBackwardNames, bound)) return false;
        if (TryLaunchDeclaredResidentChecked(declaredKey, program, s_flashBackwardNames, bound)) return true;
        RunDeclaredProgram(program, s_flashBackwardNames, bound);
        return true;
    }

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildAttentionBackward = key =>
        CodegenBackendPrograms.ScaledDotProductAttentionBackward(key.D0, key.D1, key.D2, key.D3, key.D5, key.Epsilon);

    /// <summary>The key of an attention backward call; the scale is the key's epsilon slot.</summary>
    internal static DeclaredKey AttentionBackwardKey(int batch, int heads, int seqQ, int seqK, int headDim, float scale) =>
        new(DeclaredFamily.ScaledDotProductAttentionBackward, batch, heads, seqQ, seqK, scale, 0, null, 0, headDim);

    /// <summary>
    /// The attention backward through its declared program. False -- nothing launched -- when the route is off, the
    /// program cannot be built (or is first met under capture), or a buffer is misaligned for its vector loads; the
    /// composite path then serves it.
    /// </summary>
    private bool TryDeclaredAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer weights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue, int batch, int heads, int seqQ,
        int seqK, int headDim, float scale)
    {
        if (!DeclaredKernelsEnabled || batch <= 0 || heads <= 0 || seqQ <= 0 || seqK <= 0 || headDim <= 0 ||
            (long)batch * heads * seqQ * seqK > AttentionMaxScoreElements)
            return false;
        var declaredKey = AttentionBackwardKey(batch, heads, seqQ, seqK, headDim, scale);
        DeclaredProgram? program;
        lock (_directPtxLock)
        {
            if (!_declaredPrograms.TryGetValue(declaredKey, out program))
            {
                if (IsStreamCapturing() || !AttentionShapeRecurs(declaredKey)) return false;
                program = BuildAttentionProgram(declaredKey, s_buildAttentionBackward, s_attentionBackwardNames);
                _declaredPrograms[declaredKey] = program;
            }
        }
        if (program is null) return false;
        var bound = new DeclaredBindings(gradOutput, query, key, value, weights, gradQuery, gradKey, gradValue);
        if (!AlignedFor(program, s_attentionBackwardNames, bound)) return false;
        if (TryLaunchDeclaredResidentChecked(declaredKey, program, s_attentionBackwardNames, bound)) return true;
        RunDeclaredProgram(program, s_attentionBackwardNames, bound);
        return true;
    }

    /// <summary>
    /// The attention call through its declared program. False -- nothing launched -- when the route is off, the call
    /// soft-caps its logits, the program cannot be built (or is first met under capture), or a buffer is misaligned
    /// for the program's vector loads; the hand-written kernel then serves it.
    /// </summary>
    private bool TryDeclaredAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value, IGpuBuffer output,
        IGpuBuffer? weights, IGpuBuffer? mask, int batch, int heads, int kvHeads, int seqQ, int seqK, int headDim,
        float scale, bool causal, int maskMode, float softcap, IGpuBuffer? stats = null)
    {
        if (stats is not null && (weights is not null || mask is not null)) return false;
        // The same extent and grouping guards the two backward routes already carry. Without them a
        // malformed call reaches CodegenBackendPrograms.ScaledDotProductAttention on its second sighting,
        // which throws ArgumentOutOfRangeException or ArgumentException; BuildAttentionProgram catches only
        // NotSupportedException, so that would escape from inside _directPtxLock instead of declining the
        // route and letting the hand-written kernel serve the call.
        if (batch <= 0 || heads <= 0 || kvHeads <= 0 || seqQ <= 0 || seqK <= 0 || headDim <= 0 ||
            heads % kvHeads != 0 || (maskMode != 0 && mask is null))
            return false;
        if (softcap > 0f || !DeclaredKernelsEnabled || headDim >= 1 << AttentionMaskShift ||
            (long)batch * heads * seqQ * seqK > AttentionMaxScoreElements)
            return false;
        var maskKind = maskMode == 0 ? CodegenAttentionMask.None
            : maskMode == 1 ? CodegenAttentionMask.Shared : CodegenAttentionMask.Full;
        bool storeWeights = weights is not null;
        var declaredKey = AttentionKey(batch, heads, kvHeads, seqQ, seqK, headDim, scale, maskKind, causal, storeWeights,
            stats is not null);
        string[] names = stats is not null ? s_attentionNamesStats
            : storeWeights ? s_attentionNames : s_attentionNamesNoWeights;
        IGpuBuffer? causalBias = null;
        DeclaredProgram? program;
        lock (_directPtxLock)
        {
            if (!_declaredPrograms.TryGetValue(declaredKey, out program))
            {
                if (IsStreamCapturing() || !AttentionShapeRecurs(declaredKey)) return false;
                program = BuildAttentionProgram(declaredKey, _buildAttention ??= BuildAttention, names);
                _declaredPrograms[declaredKey] = program;
            }
            if (program is not null && causal &&
                (causalBias = CausalBiasBuffer(seqQ, seqK, out string? biasDecline)) is null)
            {
                if (biasDecline is not null) _declaredDeclines[declaredKey] = biasDecline;
                return false;
            }
        }
        if (program is null) return false;
        var bound = new DeclaredBindings(query, key, value, output, mask, causalBias, stats ?? weights);
        if (!AlignedFor(program, names, bound)) return false;
        // A launch-bound call (every step within one wave) runs as one cooperative launch.
        if (TryLaunchDeclaredResidentChecked(declaredKey, program, names, bound)) return true;
        RunDeclaredProgram(program, names, bound);
        return true;
    }

    // The cached causal bias for (seqQ, seqK), created on first use; null when first needed under a capture or when
    // it does not fit the scratch budget, with the reason in decline so the caller can record it. The caller holds
    // _directPtxLock.
    private IGpuBuffer? CausalBiasBuffer(int seqQ, int seqK, out string? decline)
    {
        decline = null;
        long cacheKey = ((long)seqQ << 32) | (uint)seqK;
        if (_causalBiases.TryGetValue(cacheKey, out var buffer)) return buffer;
        if (IsStreamCapturing())
        {
            decline = "A causal attention stage is first built outside capture.";
            return null;
        }
        // One bias per (seqQ, seqK), held for the backend's lifetime like a program's scratch, so it is admitted
        // against the same budget. Null declines the route and the hand-written kernel serves the call.
        long bytes = checked((long)seqQ * seqK * sizeof(float));
        if (!DeclaredScratchFits(bytes, out decline)) return null;
        try
        {
            buffer = AllocateBuffer(CausalBias(seqQ, seqK));
        }
        catch (InvalidOperationException ex)
        {
            // The budget above is an eighth of the device's TOTAL memory, so a bias that fits it can still exceed the
            // free VRAM (an OOM) or the per-allocation cap (GpuBufferTooLargeException, an InvalidOperationException
            // too). Both mean this shape cannot be served here, and every caller of the two routes reads a false as
            // "the hand-written kernel serves this call": letting the throw out skips that fallback entirely.
            System.Diagnostics.Trace.TraceWarning(
                "[CudaBackend] the causal attention bias for " + seqQ.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                "x" + seqK.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                " could not be allocated; the declared route declines this shape: " + ex);
            decline = "the causal attention bias (" +
                (bytes >> 20).ToString(System.Globalization.CultureInfo.InvariantCulture) +
                " MiB) could not be allocated, so the hand-written kernel, which needs none, serves this shape.";
            return null;
        }
        _causalBiases[cacheKey] = buffer;
        _declaredScratchBytes = checked(_declaredScratchBytes + bytes);
        return buffer;
    }

    /// <summary>
    /// The attention call as a resident-program stage, in ScaledDotProductAttention's contract (no soft-cap); the
    /// stage's program is the one the routed call runs for the same key.
    /// </summary>
    internal DeclaredResidentStage AttentionStage(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value, IGpuBuffer output,
        IGpuBuffer? weights, IGpuBuffer? mask, int batch, int heads, int kvHeads, int seqQ, int seqK, int headDim,
        float scale, bool causal)
    {
        var maskKind = mask is null ? CodegenAttentionMask.None
            : mask.Size >= (long)batch * heads * seqQ * seqK ? CodegenAttentionMask.Full : CodegenAttentionMask.Shared;
        bool storeWeights = weights is not null;
        var declaredKey = AttentionKey(batch, heads, kvHeads, seqQ, seqK, headDim, scale, maskKind, causal, storeWeights);
        IGpuBuffer? causalBias = null;
        lock (_directPtxLock)
        {
            if (causal && (causalBias = CausalBiasBuffer(seqQ, seqK, out string? biasDecline)) is null)
                throw new NotSupportedException(biasDecline ?? "The causal attention bias could not be built.");
            if (!_declaredPrograms.ContainsKey(declaredKey) && !IsStreamCapturing())
                _declaredPrograms[declaredKey] = BuildAttentionProgram(declaredKey, _buildAttention ??= BuildAttention,
                    storeWeights ? s_attentionNames : s_attentionNamesNoWeights);
        }
        return new DeclaredResidentStage(declaredKey, _buildAttention ??= BuildAttention,
            storeWeights ? s_attentionNames : s_attentionNamesNoWeights,
            new DeclaredBindings(query, key, value, output, mask, causalBias, weights),
            _lowerAttention ??= pass => LowerAttentionPass(pass, minBlocks: 2));
    }

    // An attention program with its contractions as implicit GEMMs, retried unbounded where the two-blocks-a-SM bound
    // makes the JIT spill; null (with the reason recorded) when it cannot be built.
    private DeclaredProgram? BuildAttentionProgram(DeclaredKey key, Func<DeclaredKey, CodegenProgram> build, string[] names)
    {
        CodegenProgram source;
        try
        {
            source = build(key);
        }
        catch (NotSupportedException ex)
        {
            _declaredDeclines[key] = ex.Message;
            return null;
        }
        var program = BuildDeclaredProgram(source, key.ToString(), names,
            _lowerAttention ??= pass => LowerAttentionPass(pass, minBlocks: 2), out string? decline);
        // As for the convolutions: a ragged shape's guards can spill under the two-blocks-a-SM bound.
        if (program is null && decline is not null && decline.IndexOf("local bytes", StringComparison.Ordinal) >= 0)
            program = BuildDeclaredProgram(source, key.ToString(), names,
                _lowerAttentionUnbounded ??= pass => LowerAttentionPass(pass, minBlocks: 0), out decline);
        if (decline is not null) _declaredDeclines[key] = decline;
        return program;
    }

    /// <summary>
    /// The causal bias, row-major [seqQ, seqK]: query q sees key k when <c>k &lt;= q + seqK - seqQ</c> (the hand-written
    /// kernel's qPos, which aligns the last query with the last key).
    /// </summary>
    internal static float[] CausalBias(int seqQ, int seqK)
    {
        var bias = new float[checked(seqQ * seqK)];
        for (int q = 0; q < seqQ; q++)
            for (int k = 0; k < seqK; k++)
                bias[q * seqK + k] = k > q + (seqK - seqQ) ? (float)CodegenBackendPrograms.AttentionMaskedLogit : 0f;
        return bias;
    }

    private void DisposeCausalBiases()
    {
        foreach (var buffer in _causalBiases.Values) buffer.Dispose();
        _causalBiases.Clear();
    }
}
