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

    // Keys seen once whose geometry had already appeared with another key length (see AttentionShapeRecurs): such a
    // key is routed from its SECOND call, so autoregressive decoding -- seqK one longer every token, every shape new --
    // keeps the hand-written kernel instead of compiling a program per token.
    private readonly HashSet<DeclaredKey> _attentionSeenOnce = new();

    // Decoding gives this set one new key per token, so it needs a bound of its own: the programs are bounded by the
    // scratch budget, but a key that never recurs costs nothing but a set entry and would still grow without end.
    // At the cap the memo is dropped rather than closed, so a shape that starts recurring afterwards can still be
    // routed - it just needs two more sightings - while the set cannot exceed the cap.
    private const int AttentionSeenOnceCapacity = 4096;

    // The last key length seen for each geometry (the key with seqK cleared). A decode loop shows the same geometry
    // with a new seqK call after call; anything else (training, a fixed batch, one-off shapes) shows it once per length.
    // Bounded like the seen-once set, and dropped rather than closed at the cap for the same reason.
    private readonly Dictionary<DeclaredKey, int> _attentionLastSeqK = new();

    // Whether a key's program may be built now (#1529). A geometry's first length is built at once, so the first call of
    // a shape and every later one take the same route: building only from the second sighting made two identical calls
    // differ in their last bits (the first ran the hand-written kernel, the second the program). Once the geometry has
    // come back with another length (decoding: seqK one longer every token), a length is built only from its second
    // sighting, so decoding still compiles nothing per token. The caller holds _directPtxLock.
    private bool AttentionShapeRecurs(DeclaredKey key)
    {
        if (_attentionSeenOnce.Contains(key)) return true;
        var geometry = new DeclaredKey(key.Family, key.D0, key.D1, key.D2, 0, key.Epsilon, key.Momentum, key.Name,
            key.D4, key.D5);
        bool lengthChanged = _attentionLastSeqK.TryGetValue(geometry, out int lastSeqK) && lastSeqK != key.D3;
        if (_attentionLastSeqK.Count >= AttentionSeenOnceCapacity) _attentionLastSeqK.Clear();
        _attentionLastSeqK[geometry] = key.D3;
        if (!lengthChanged) return true;
        if (_attentionSeenOnce.Count >= AttentionSeenOnceCapacity) _attentionSeenOnce.Clear();
        _attentionSeenOnce.Add(key);
        return false;
    }

    // The causal bias of each (seqQ, seqK): 0 for a key at or before q + seqK - seqQ, AttentionMaskedLogit after.
    private readonly Dictionary<long, IGpuBuffer> _causalBiases = new();

    // The paged key bias of each (pages * blockSize, seqLen): 0 inside the sequence, AttentionMaskedLogit in the
    // last page's unused tail.
    private readonly Dictionary<long, IGpuBuffer> _pagedKeyBiases = new();

    // The paged prefill's causal bias of each (numQueries, pages * blockSize, startPos).
    private readonly Dictionary<(int Queries, int Length, int StartPos), IGpuBuffer> _pagedCausalBiases = new();

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

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildGroupedAttentionBackward = key =>
        CodegenBackendPrograms.ScaledDotProductAttentionBackward(key.D0, key.D1, key.D4, key.D2, key.D3, key.D5,
            key.Epsilon);

    /// <summary>
    /// The grouped-query attention backward through its declared program (#1440): the same products as the attention
    /// backward, with k and v read at head h / g and dk, dv summed over each group's query heads. Only where the group is
    /// the consistent heads / kvHeads (the hand-written kernel honours an explicit, possibly inconsistent, group).
    /// False -- nothing launched -- when declined.
    /// </summary>
    private bool TryDeclaredGroupedAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key,
        IGpuBuffer value, IGpuBuffer weights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue, int batch,
        int heads, int kvHeads, int queriesPerKV, int seqQ, int seqK, int headDim, float scale)
    {
        if (!DeclaredKernelsEnabled || batch <= 0 || heads <= 0 || kvHeads <= 0 || seqQ <= 0 || seqK <= 0 ||
            headDim <= 0 || heads % kvHeads != 0 || queriesPerKV != heads / kvHeads ||
            (long)batch * heads * seqQ * seqK > AttentionMaxScoreElements)
            return false;
        var declaredKey = new DeclaredKey(DeclaredFamily.GroupedQueryAttentionBackward, batch, heads, seqQ, seqK, scale, 0,
            null, kvHeads, headDim);
        DeclaredProgram? program;
        lock (_directPtxLock)
        {
            if (!_declaredPrograms.TryGetValue(declaredKey, out program))
            {
                if (IsStreamCapturing() || !AttentionShapeRecurs(declaredKey)) return false;
                program = BuildAttentionProgram(declaredKey, s_buildGroupedAttentionBackward, s_attentionBackwardNames);
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
        // malformed call reaches CodegenBackendPrograms.ScaledDotProductAttention when its program is built,
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

    // ---- paged attention (#890 phase 4d) --------------------------------------------------------------------------

    private static readonly string[] s_pagedDecodeNames =
        { "block_table", "q", "kcache", "vcache", "key_bias", "out" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildPagedDecode =
        key => CodegenBackendPrograms.PagedAttentionDecode(key.D0, key.D1, key.D2, key.D3, key.D4, key.D5, key.Epsilon);

    /// <summary>
    /// The key of a paged decode: heads, kvHeads, headDim, blockSize, pages, the cache's block count, and the scale.
    /// </summary>
    internal static DeclaredKey PagedAttentionDecodeKey(int heads, int kvHeads, int headDim, int blockSize, int pages,
        int blocks, float scale) =>
        new(DeclaredFamily.PagedAttentionDecode, heads, kvHeads, headDim, blockSize, scale, 0, name: null,
            d4: pages, d5: blocks);

    /// <summary>
    /// Paged attention's decode step through its declared program, in CudaBackend.PagedAttentionDecode's contract.
    /// <paramref name="keyBias"/> is [pages * blockSize]: 0 for a key inside the sequence and the masked logit for
    /// one in the last page's unused tail, which the backend builds because comparing a key against seqLen is
    /// outside the IR. False -- nothing launched -- when the route is off or the shape declines.
    /// </summary>
    internal bool TryDeclaredPagedAttentionDecode(IGpuBuffer blockTable, IGpuBuffer query, IGpuBuffer keyCache,
        IGpuBuffer valueCache, IGpuBuffer keyBias, IGpuBuffer output, int heads, int kvHeads, int headDim,
        int blockSize, int pages, int blocks, float scale)
    {
        if (heads <= 0 || kvHeads <= 0 || headDim <= 0 || blockSize <= 0 || pages <= 0 || blocks <= 0) return false;
        if (heads % kvHeads != 0) return false;
        long cacheElements = (long)blocks * blockSize * kvHeads * headDim;
        if (blockTable.Size < pages || (long)query.Size < (long)heads * headDim ||
            keyCache.Size < cacheElements || valueCache.Size < cacheElements ||
            (long)keyBias.Size < (long)pages * blockSize || (long)output.Size < (long)heads * headDim)
            return false;
        return TryRunDeclaredProgram(
            PagedAttentionDecodeKey(heads, kvHeads, headDim, blockSize, pages, blocks, scale), s_buildPagedDecode,
            s_pagedDecodeNames,
            new DeclaredBindings(blockTable, query, keyCache, valueCache, keyBias, output));
    }

    /// <summary>Paged attention's decode step as a resident-program stage, in the same contract.</summary>
    internal DeclaredResidentStage PagedAttentionDecodeStage(IGpuBuffer blockTable, IGpuBuffer query,
        IGpuBuffer keyCache, IGpuBuffer valueCache, IGpuBuffer keyBias, IGpuBuffer output, int heads, int kvHeads,
        int headDim, int blockSize, int pages, int blocks, float scale) =>
        new(PagedAttentionDecodeKey(heads, kvHeads, headDim, blockSize, pages, blocks, scale), s_buildPagedDecode,
            s_pagedDecodeNames, new DeclaredBindings(blockTable, query, keyCache, valueCache, keyBias, output));

    private static readonly string[] s_pagedPrefillNames =
        { "block_table", "q", "kcache", "vcache", "causal_bias", "out" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildPagedPrefill =
        key => CodegenBackendPrograms.PagedAttentionPrefill(key.D0, key.D1, key.D2, key.D3, key.D4, key.D5,
            (int)key.Momentum, key.Epsilon);

    /// <summary>
    /// The key of a paged prefill: heads, kvHeads, headDim, blockSize, pages, blocks, the scale in the epsilon slot
    /// and the query count in the momentum one -- the six dimension slots are full, and a query count is exact in a
    /// float well past the extent the kernel guards admit.
    /// </summary>
    internal static DeclaredKey PagedAttentionPrefillKey(int heads, int kvHeads, int headDim, int blockSize, int pages,
        int blocks, int numQueries, float scale) =>
        new(DeclaredFamily.PagedAttentionPrefill, heads, kvHeads, headDim, blockSize, scale, numQueries, name: null,
            d4: pages, d5: blocks);

    /// <summary>
    /// Paged attention's prefill step through its declared program, in CudaBackend.PagedAttentionPrefill's contract.
    /// <paramref name="causalBias"/> is [numQueries, pages * blockSize]: 0 where query qi may see key k and the masked
    /// logit elsewhere. False -- nothing launched -- when the route is off or the shape declines.
    /// </summary>
    internal bool TryDeclaredPagedAttentionPrefill(IGpuBuffer blockTable, IGpuBuffer query, IGpuBuffer keyCache,
        IGpuBuffer valueCache, IGpuBuffer causalBias, IGpuBuffer output, int heads, int kvHeads, int headDim,
        int blockSize, int pages, int blocks, int numQueries, float scale)
    {
        if (heads <= 0 || kvHeads <= 0 || headDim <= 0 || blockSize <= 0 || pages <= 0 || blocks <= 0 ||
            numQueries <= 0)
            return false;
        if (heads % kvHeads != 0) return false;
        long cacheElements = (long)blocks * blockSize * kvHeads * headDim;
        long rows = (long)numQueries * heads;
        long seqK = (long)pages * blockSize;
        if (blockTable.Size < pages || (long)query.Size < rows * headDim ||
            keyCache.Size < cacheElements || valueCache.Size < cacheElements ||
            (long)causalBias.Size < (long)numQueries * seqK || (long)output.Size < rows * headDim)
            return false;
        return TryRunDeclaredProgram(
            PagedAttentionPrefillKey(heads, kvHeads, headDim, blockSize, pages, blocks, numQueries, scale),
            s_buildPagedPrefill, s_pagedPrefillNames,
            new DeclaredBindings(blockTable, query, keyCache, valueCache, causalBias, output));
    }

    /// <summary>Paged attention's prefill step as a resident-program stage, in the same contract.</summary>
    internal DeclaredResidentStage PagedAttentionPrefillStage(IGpuBuffer blockTable, IGpuBuffer query,
        IGpuBuffer keyCache, IGpuBuffer valueCache, IGpuBuffer causalBias, IGpuBuffer output, int heads, int kvHeads,
        int headDim, int blockSize, int pages, int blocks, int numQueries, float scale) =>
        new(PagedAttentionPrefillKey(heads, kvHeads, headDim, blockSize, pages, blocks, numQueries, scale),
            s_buildPagedPrefill, s_pagedPrefillNames,
            new DeclaredBindings(blockTable, query, keyCache, valueCache, causalBias, output));

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

    /// <summary>
    /// The paged decode's key bias, [length] where length is <c>pages * blockSize</c>: 0 for a key position inside the
    /// sequence and the masked logit for one in the last page's unused tail. The program reduces over every position
    /// of every page it is given, and comparing a position against seqLen is not a quasi-affine index, so the tail is
    /// masked through an input instead.
    /// </summary>
    internal static float[] PagedKeyBias(int length, int seqLen)
    {
        var bias = new float[length];
        for (int k = seqLen; k < length; k++) bias[k] = (float)CodegenBackendPrograms.AttentionMaskedLogit;
        return bias;
    }

    // The cached paged key bias for (length, seqLen), created on first use; null when first needed under a capture or
    // when it does not fit the scratch budget, with the reason in decline. The caller holds _directPtxLock.
    private IGpuBuffer? PagedKeyBiasBuffer(int length, int seqLen, out string? decline)
    {
        decline = null;
        long cacheKey = ((long)length << 32) | (uint)seqLen;
        if (_pagedKeyBiases.TryGetValue(cacheKey, out var buffer)) return buffer;
        if (IsStreamCapturing())
        {
            decline = "A paged decode stage is first built outside capture.";
            return null;
        }
        long bytes = checked((long)length * sizeof(float));
        if (!DeclaredScratchFits(bytes, out decline)) return null;
        try
        {
            buffer = AllocateBuffer(PagedKeyBias(length, seqLen));
        }
        catch (InvalidOperationException ex)
        {
            // As for the causal bias: the budget is an eighth of the device's TOTAL memory, so a bias that fits it can
            // still exceed the free VRAM or the per-allocation cap, and both throw an InvalidOperationException. Every
            // caller reads a false as "the hand-written kernel serves this call", so the throw must not escape.
            System.Diagnostics.Trace.TraceWarning(
                "[CudaBackend] the paged key bias for " + length.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                " positions could not be allocated; the declared route declines this shape: " + ex);
            decline = "the paged key bias (" + length.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                " positions) could not be allocated, so the hand-written kernel, which needs none, serves this shape.";
            return null;
        }
        _pagedKeyBiases[cacheKey] = buffer;
        _declaredScratchBytes = checked(_declaredScratchBytes + bytes);
        return buffer;
    }

    /// <summary>
    /// The paged decode through its declared program, in CudaBackend.PagedAttentionDecode's contract: the backend
    /// builds the key bias the program needs. False -- nothing launched -- when the route is off, the shape declines
    /// or the bias cannot be built; the hand-written kernel then serves the call.
    /// </summary>
    private bool TryDeclaredPagedDecodeRouted(IGpuBuffer query, IGpuBuffer keyCache, IGpuBuffer valueCache,
        IGpuBuffer blockTable, IGpuBuffer output, int heads, int kvHeads, int headDim, int blockSize, int seqLen,
        float scale)
    {
        if (!DeclaredKernelsEnabled || heads <= 0 || kvHeads <= 0 || headDim <= 0 || blockSize <= 0 || seqLen <= 0 ||
            heads % kvHeads != 0)
            return false;
        int pages = (seqLen + blockSize - 1) / blockSize;
        long blockStride = (long)blockSize * kvHeads * headDim;
        long blocks = keyCache.Size / blockStride;
        if (blocks <= 0 || blocks > int.MaxValue) return false;
        int length = checked(pages * blockSize);
        IGpuBuffer? keyBias;
        lock (_directPtxLock)
        {
            if ((keyBias = PagedKeyBiasBuffer(length, seqLen, out string? biasDecline)) is null)
            {
                if (biasDecline is not null)
                    _declaredDeclines[PagedAttentionDecodeKey(heads, kvHeads, headDim, blockSize, pages, (int)blocks,
                        scale)] = biasDecline;
                return false;
            }
        }
        return TryDeclaredPagedAttentionDecode(blockTable, query, keyCache, valueCache, keyBias, output, heads, kvHeads,
            headDim, blockSize, pages, (int)blocks, scale);
    }

    /// <summary>
    /// The paged prefill's causal bias, [numQueries, length] where length is <c>pages * blockSize</c>: 0 where query
    /// qi (at logical position <c>startPos + qi</c>) may see key k, the masked logit elsewhere. One input carries both
    /// the causal rule and the last page's unused tail, neither of which is a quasi-affine index.
    /// </summary>
    internal static float[] PagedCausalBias(int numQueries, int length, int startPos)
    {
        var bias = new float[checked(numQueries * length)];
        for (int q = 0; q < numQueries; q++)
        {
            int last = startPos + q;
            for (int k = last + 1; k < length; k++) bias[q * length + k] = (float)CodegenBackendPrograms.AttentionMaskedLogit;
        }
        return bias;
    }

    // The cached paged causal bias for (numQueries, length, startPos). The caller holds _directPtxLock.
    private IGpuBuffer? PagedCausalBiasBuffer(int numQueries, int length, int startPos, out string? decline)
    {
        decline = null;
        var cacheKey = (numQueries, length, startPos);
        if (_pagedCausalBiases.TryGetValue(cacheKey, out var buffer)) return buffer;
        if (IsStreamCapturing())
        {
            decline = "A paged prefill stage is first built outside capture.";
            return null;
        }
        long bytes = checked((long)numQueries * length * sizeof(float));
        if (!DeclaredScratchFits(bytes, out decline)) return null;
        try
        {
            buffer = AllocateBuffer(PagedCausalBias(numQueries, length, startPos));
        }
        catch (InvalidOperationException ex)
        {
            // As for the causal bias: fitting the budget (an eighth of TOTAL memory) does not mean the allocation
            // succeeds, and both the OOM and the per-allocation cap arrive as an InvalidOperationException. A throw
            // here would skip the hand-written kernel that every caller falls back to.
            System.Diagnostics.Trace.TraceWarning(
                "[CudaBackend] the paged causal bias for " +
                numQueries.ToString(System.Globalization.CultureInfo.InvariantCulture) + "x" +
                length.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                " could not be allocated; the declared route declines this shape: " + ex);
            decline = "the paged causal bias (" + (bytes >> 20).ToString(System.Globalization.CultureInfo.InvariantCulture) +
                " MiB) could not be allocated, so the hand-written kernel, which needs none, serves this shape.";
            return null;
        }
        _pagedCausalBiases[cacheKey] = buffer;
        _declaredScratchBytes = checked(_declaredScratchBytes + bytes);
        return buffer;
    }

    /// <summary>
    /// The paged prefill through its declared program, in CudaBackend.PagedAttentionPrefill's contract: the backend
    /// builds the causal bias the program needs. False -- nothing launched -- when the route is off, the shape
    /// declines or the bias cannot be built; the hand-written kernel then serves the call.
    /// </summary>
    private bool TryDeclaredPagedPrefillRouted(IGpuBuffer query, IGpuBuffer keyCache, IGpuBuffer valueCache,
        IGpuBuffer blockTable, IGpuBuffer output, int heads, int kvHeads, int headDim, int blockSize, int numQueries,
        int startPos, float scale)
    {
        if (!DeclaredKernelsEnabled || heads <= 0 || kvHeads <= 0 || headDim <= 0 || blockSize <= 0 ||
            numQueries <= 0 || startPos < 0 || heads % kvHeads != 0)
            return false;
        // The key carries the query count in its float momentum slot, so a count past a float's exact integers would
        // alias another shape's program. The kernel guards stop long before this, which makes it a cheap assertion.
        if (numQueries > 1 << 24) return false;
        int seqLen = checked(startPos + numQueries);
        int pages = (seqLen + blockSize - 1) / blockSize;
        long blockStride = (long)blockSize * kvHeads * headDim;
        long blocks = keyCache.Size / blockStride;
        if (pages <= 0 || blocks <= 0 || blocks > int.MaxValue) return false;
        int length = checked(pages * blockSize);
        IGpuBuffer? causalBias;
        lock (_directPtxLock)
        {
            if ((causalBias = PagedCausalBiasBuffer(numQueries, length, startPos, out string? biasDecline)) is null)
            {
                if (biasDecline is not null)
                    _declaredDeclines[PagedAttentionPrefillKey(heads, kvHeads, headDim, blockSize, pages, (int)blocks,
                        numQueries, scale)] = biasDecline;
                return false;
            }
        }
        return TryDeclaredPagedAttentionPrefill(blockTable, query, keyCache, valueCache, causalBias, output, heads,
            kvHeads, headDim, blockSize, pages, (int)blocks, numQueries, scale);
    }

    // The one-entry block table { 0 } FlashDecode reads its contiguous cache through; guarded by _directPtxLock.
    private IGpuBuffer? _identityBlockTable;

    /// <summary>
    /// FlashDecode through the declared paged decode (#1440): a contiguous [seqLen, kvHeads, headDim] cache is a paged one
    /// of a single block seqLen tokens long, read through the block table { 0 }. The split count is the hand-written
    /// kernel's schedule, not part of the result, so it is not consulted. False -- nothing launched -- when declined.
    /// </summary>
    private bool TryDeclaredFlashDecode(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value, IGpuBuffer output, int heads,
        int kvHeads, int headDim, int seqLen, float scale)
    {
        if (!DeclaredKernelsEnabled || seqLen <= 0) return false;
        IGpuBuffer table;
        lock (_directPtxLock)
        {
            if (_identityBlockTable is null)
            {
                if (IsStreamCapturing()) return false;
                var buffer = AllocateByteBuffer(sizeof(int));
                UploadByteBuffer(buffer, new byte[sizeof(int)]);
                _identityBlockTable = buffer;
            }
            table = _identityBlockTable;
        }
        return TryDeclaredPagedDecodeRouted(query, key, value, table, output, heads, kvHeads, headDim, seqLen, seqLen, scale);
    }

    private void DisposeCausalBiases()
    {
        _identityBlockTable?.Dispose();
        _identityBlockTable = null;
        foreach (var buffer in _causalBiases.Values) buffer.Dispose();
        _causalBiases.Clear();
        foreach (var buffer in _pagedKeyBiases.Values) buffer.Dispose();
        _pagedKeyBiases.Clear();
        foreach (var buffer in _pagedCausalBiases.Values) buffer.Dispose();
        _pagedCausalBiases.Clear();
    }
}
