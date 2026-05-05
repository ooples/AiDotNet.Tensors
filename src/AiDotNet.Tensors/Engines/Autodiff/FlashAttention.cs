// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Generic-T rank-N FlashAttention-2 forward + backward (issue #294
/// Phase 3). Implements the Dao 2023 block-tiled online-softmax
/// algorithm — same math as <see cref="FlashAttention2"/> — but
/// generalized two ways:
///
/// <list type="number">
/// <item><b>Generic over <c>T : unmanaged</c></b>: float gets the
/// paper-faithful single-precision kernel; double gets a
/// double-precision kernel for research / scientific computing; other
/// T fall through to a generic <see cref="INumericOperations{T}"/>
/// path. Same shape as <c>MathF</c> vs <c>Math</c> in the BCL.</item>
/// <item><b>Rank-N inputs</b>: accepts any rank ≥ 2. The last two
/// dims of <c>query</c> are <c>[Sq, D]</c>; <c>[Sk, D]</c> for
/// <c>key</c>; <c>[Sk, Dv]</c> for <c>value</c>. All preceding dims
/// are flattened into a single batchProduct and iterated by the
/// outer loop. Subsumes:
/// <list type="bullet">
/// <item>Rank 2: <c>[Sq, D]</c> — single-sequence, batchProduct = 1.</item>
/// <item>Rank 3: <c>[B, Sq, D]</c> — single-head batched.</item>
/// <item>Rank 4: <c>[B, H, Sq, D]</c> — canonical multi-head.</item>
/// <item>Rank 5+: video <c>[B, F, H, Sq, D]</c>, MoE
/// <c>[E, B, H, Sq, D]</c>, Swin <c>[B, W, H, Sq, D]</c>, etc.</item>
/// </list>
/// </item>
/// </list>
///
/// <para><b>Bias broadcast</b>: optional <paramref name="attentionBias"/>
/// has last two dims <c>[Sq, Sk]</c>; leading dims broadcast
/// NumPy-style against query's leading dims. Subsumes ALiBi
/// <c>[Sq, Sk]</c>, per-batch <c>[B, Sq, Sk]</c>, per-head
/// <c>[B, H, Sq, Sk]</c>, head-only-batch-broadcast
/// <c>[1, H, Sq, Sk]</c>, video-per-frame
/// <c>[B, F, H, Sq, Sk]</c> — no special cases in the kernel.</para>
///
/// <para><b>Constraint</b>: query, key, value must share their
/// leading-prefix shape (they don't broadcast against each other).
/// This keeps the inner kernel simple and matches the realistic
/// usage pattern — the NumPy-broadcast surface is the bias, where
/// it's actually needed.</para>
///
/// <para><b>LogSumExp shape</b>: <see cref="Tensor{T}"/> of shape
/// <c>[...prefix..., Sq]</c>, not <c>float[B*H*Sq]</c> like the
/// rank-fixed predecessor. Cleaner, generic-T, naturally rank-N.</para>
/// </summary>
public static class FlashAttention<T> where T : unmanaged
{
    /// <summary>
    /// Tiled forward producing <c>(Output, LogSumExp)</c>. See class
    /// remarks for the rank-N input contract and broadcast rules.
    /// </summary>
    /// <param name="query">Query tensor with last two dims <c>[Sq, D]</c>.</param>
    /// <param name="key">Key tensor with last two dims <c>[Sk, D]</c>.</param>
    /// <param name="value">Value tensor with last two dims <c>[Sk, Dv]</c>.</param>
    /// <param name="blockSizeQ">Row-block size. Default 64.</param>
    /// <param name="blockSizeKV">Col-block size. Default 64.</param>
    /// <param name="scale">Softmax scale. Null → <c>1/sqrt(D)</c>.</param>
    /// <param name="isCausal">Apply upper-triangular mask (queryOffset-aware).</param>
    /// <param name="queryOffset">For KV-cache decode.</param>
    /// <param name="attentionBias">Optional additive bias with last
    /// two dims <c>[Sq, Sk]</c>; leading dims broadcast NumPy-style
    /// against query's leading dims.</param>
    public static (Tensor<T> Output, Tensor<T> LogSumExp) Forward(
        Tensor<T> query, Tensor<T> key, Tensor<T> value,
        int blockSizeQ = 0, int blockSizeKV = 0,
        double? scale = null, bool isCausal = false, int queryOffset = 0,
        Tensor<T>? attentionBias = null)
    {
        ValidateInputs(query, key, value, attentionBias, queryOffset,
            out int batchProduct, out int Sq, out int Sk, out int headDim, out int Dv,
            out int[] prefixShape, out int[] biasPrefixShape, out int[] biasPrefixStrides);

        // #294 Phase 4 wiring: 0 means "use CacheOptimizer-derived
        // L1-aware tile size for T". Previously the defaults were
        // hardcoded to 64, which is right for ~A100-class L1 on
        // single-precision but wrong for AMD Zen / ARM / double-
        // precision workloads. CacheOptimizer.ComputeOptimalTiling<T>
        // is the single source of truth across the codebase.
        if (blockSizeQ == 0) blockSizeQ = ResolveDefaultBlockSize(headDim);
        if (blockSizeKV == 0) blockSizeKV = ResolveDefaultBlockSize(headDim);
        if (blockSizeQ < 0) throw new ArgumentOutOfRangeException(nameof(blockSizeQ));
        if (blockSizeKV < 0) throw new ArgumentOutOfRangeException(nameof(blockSizeKV));

        double scaleVal = scale ?? 1.0 / Math.Sqrt(headDim);

        // Output shape: prefix ⊕ [Sq, Dv]. LogSumExp: prefix ⊕ [Sq].
        var outShape = new int[prefixShape.Length + 2];
        Array.Copy(prefixShape, outShape, prefixShape.Length);
        outShape[prefixShape.Length] = Sq;
        outShape[prefixShape.Length + 1] = Dv;
        var lseShape = new int[prefixShape.Length + 1];
        Array.Copy(prefixShape, lseShape, prefixShape.Length);
        lseShape[prefixShape.Length] = Sq;

        var output = new Tensor<T>(outShape);
        var logsumexp = new Tensor<T>(lseShape);

        var qArr = query.GetDataArray();
        var kArr = key.GetDataArray();
        var vArr = value.GetDataArray();
        var biasArr = attentionBias?.GetDataArray();
        var outArr = output.GetDataArray();
        var lseArr = logsumexp.GetDataArray();

        // Float / double primitive fast paths run the same kernel as
        // the rank-fixed FlashAttention2 — just looped over
        // batchProduct rather than the explicit B,H pair. Other T
        // fall through to the generic INumericOperations<T> path.
        if (typeof(T) == typeof(float))
        {
            ForwardFloat(
                (float[])(object)qArr!, (float[])(object)kArr!, (float[])(object)vArr!,
                biasArr is null ? null : (float[])(object)biasArr,
                (float[])(object)outArr!, (float[])(object)lseArr!,
                batchProduct, Sq, Sk, headDim, Dv,
                blockSizeQ, blockSizeKV, (float)scaleVal, isCausal, queryOffset,
                biasPrefixShape, biasPrefixStrides);
        }
        else if (typeof(T) == typeof(double))
        {
            ForwardDouble(
                (double[])(object)qArr!, (double[])(object)kArr!, (double[])(object)vArr!,
                biasArr is null ? null : (double[])(object)biasArr,
                (double[])(object)outArr!, (double[])(object)lseArr!,
                batchProduct, Sq, Sk, headDim, Dv,
                blockSizeQ, blockSizeKV, scaleVal, isCausal, queryOffset,
                biasPrefixShape, biasPrefixStrides);
        }
        else
        {
            ForwardGeneric(
                qArr, kArr, vArr, biasArr, outArr, lseArr,
                batchProduct, Sq, Sk, headDim, Dv,
                blockSizeQ, blockSizeKV, scaleVal, isCausal, queryOffset,
                biasPrefixShape, biasPrefixStrides);
        }

        return (output, logsumexp);
    }

    /// <summary>
    /// Tiled backward using the saved <paramref name="logsumexp"/>
    /// from <see cref="Forward"/>. Recomputes P per block (O(blockSize²)
    /// memory) rather than materialising the whole attention matrix.
    /// </summary>
    public static (Tensor<T> GradQuery, Tensor<T> GradKey, Tensor<T> GradValue) Backward(
        Tensor<T> gradOutput,
        Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<T> output, Tensor<T> logsumexp,
        int blockSizeQ = 0, int blockSizeKV = 0,
        double? scale = null, bool isCausal = false, int queryOffset = 0,
        Tensor<T>? attentionBias = null)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (logsumexp is null) throw new ArgumentNullException(nameof(logsumexp));

        ValidateInputs(query, key, value, attentionBias, queryOffset,
            out int batchProduct, out int Sq, out int Sk, out int headDim, out int Dv,
            out int[] prefixShape, out int[] biasPrefixShape, out int[] biasPrefixStrides);

        if (blockSizeQ == 0) blockSizeQ = ResolveDefaultBlockSize(headDim);
        if (blockSizeKV == 0) blockSizeKV = ResolveDefaultBlockSize(headDim);
        if (blockSizeQ < 0) throw new ArgumentOutOfRangeException(nameof(blockSizeQ));
        if (blockSizeKV < 0) throw new ArgumentOutOfRangeException(nameof(blockSizeKV));

        double scaleVal = scale ?? 1.0 / Math.Sqrt(headDim);

        var dQ = new Tensor<T>((int[])query._shape.Clone());
        var dK = new Tensor<T>((int[])key._shape.Clone());
        var dV = new Tensor<T>((int[])value._shape.Clone());

        var qArr = query.GetDataArray();
        var kArr = key.GetDataArray();
        var vArr = value.GetDataArray();
        var oArr = output.GetDataArray();
        var dOArr = gradOutput.GetDataArray();
        var lseArr = logsumexp.GetDataArray();
        var biasArr = attentionBias?.GetDataArray();
        var dQArr = dQ.GetDataArray();
        var dKArr = dK.GetDataArray();
        var dVArr = dV.GetDataArray();

        if (typeof(T) == typeof(float))
        {
            BackwardFloat(
                (float[])(object)dOArr!, (float[])(object)qArr!, (float[])(object)kArr!,
                (float[])(object)vArr!, (float[])(object)oArr!, (float[])(object)lseArr!,
                biasArr is null ? null : (float[])(object)biasArr,
                (float[])(object)dQArr!, (float[])(object)dKArr!, (float[])(object)dVArr!,
                batchProduct, Sq, Sk, headDim, Dv,
                blockSizeQ, blockSizeKV, (float)scaleVal, isCausal, queryOffset,
                biasPrefixShape, biasPrefixStrides);
        }
        else if (typeof(T) == typeof(double))
        {
            BackwardDouble(
                (double[])(object)dOArr!, (double[])(object)qArr!, (double[])(object)kArr!,
                (double[])(object)vArr!, (double[])(object)oArr!, (double[])(object)lseArr!,
                biasArr is null ? null : (double[])(object)biasArr,
                (double[])(object)dQArr!, (double[])(object)dKArr!, (double[])(object)dVArr!,
                batchProduct, Sq, Sk, headDim, Dv,
                blockSizeQ, blockSizeKV, scaleVal, isCausal, queryOffset,
                biasPrefixShape, biasPrefixStrides);
        }
        else
        {
            BackwardGeneric(
                dOArr, qArr, kArr, vArr, oArr, lseArr, biasArr,
                dQArr, dKArr, dVArr,
                batchProduct, Sq, Sk, headDim, Dv,
                blockSizeQ, blockSizeKV, scaleVal, isCausal, queryOffset,
                biasPrefixShape, biasPrefixStrides);
        }

        return (dQ, dK, dV);
    }

    /// <summary>
    /// L1-aware default block size for FlashAttention's Q/KV tiles.
    /// Delegates to <see cref="AiDotNet.Tensors.Engines.Optimization.CacheOptimizer.ComputeOptimalTiling{T}"/>
    /// for the float / double primitives — matching the rest of the
    /// codebase's tile-sizing convention. For other T (BFloat16 /
    /// Half) we fall back to 64, which is the historical default and
    /// matches FlashAttention2's hardcoded value.
    /// </summary>
    private static int ResolveDefaultBlockSize(int headDim)
    {
        // Cap by headDim so the tile is never larger than the matrix
        // dim (would dispatch a tile of shape [hd, hd] when only one
        // tile-of-headDim fits).
        if (typeof(T) == typeof(float))
        {
            var (m, _, _) = AiDotNet.Tensors.Engines.Optimization.CacheOptimizer
                .ComputeOptimalTiling<float>(headDim, headDim, headDim);
            return Math.Max(16, Math.Min(m, 128));
        }
        if (typeof(T) == typeof(double))
        {
            var (m, _, _) = AiDotNet.Tensors.Engines.Optimization.CacheOptimizer
                .ComputeOptimalTiling<double>(headDim, headDim, headDim);
            return Math.Max(16, Math.Min(m, 128));
        }
        return 64;
    }

    /// <summary>
    /// Validates rank-N inputs and computes the batch flattening
    /// shape + strides used by all three kernels (float / double /
    /// generic). The bias broadcast strides are computed once here so
    /// the inner kernel just multiplies them — same shape as
    /// NumPy's broadcast iterator but specialized for the
    /// "leading-prefix broadcast against shared q/k/v prefix"
    /// case that attention bias actually uses.
    /// </summary>
    private static void ValidateInputs(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, Tensor<T>? attentionBias,
        int queryOffset,
        out int batchProduct, out int Sq, out int Sk, out int headDim, out int Dv,
        out int[] prefixShape, out int[] biasPrefixShape, out int[] biasPrefixStrides)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));

        int qRank = query.Rank;
        int kRank = key.Rank;
        int vRank = value.Rank;
        if (qRank < 2) throw new ArgumentException($"FlashAttention requires query rank >= 2; got {qRank}.", nameof(query));
        if (kRank != qRank) throw new ArgumentException($"key rank ({kRank}) must equal query rank ({qRank}).", nameof(key));
        if (vRank != qRank) throw new ArgumentException($"value rank ({vRank}) must equal query rank ({qRank}).", nameof(value));

        // Last two dims define [Sq, D] / [Sk, D] / [Sk, Dv]. Leading
        // dims are the shared batch prefix.
        Sq = query._shape[qRank - 2];
        headDim = query._shape[qRank - 1];
        Sk = key._shape[kRank - 2];
        Dv = value._shape[vRank - 1];

        if (key._shape[kRank - 1] != headDim)
            throw new ArgumentException($"query/key headDim mismatch: {headDim} vs {key._shape[kRank - 1]}.", nameof(key));
        if (value._shape[vRank - 2] != Sk)
            throw new ArgumentException($"key/value seq len mismatch: {Sk} vs {value._shape[vRank - 2]}.", nameof(value));
        if (queryOffset < 0 || queryOffset + Sq > Sk)
            throw new ArgumentException($"queryOffset={queryOffset} + Sq={Sq} must be <= Sk={Sk}.", nameof(queryOffset));

        // Build the shared prefix shape and verify q/k/v share it.
        prefixShape = new int[qRank - 2];
        batchProduct = 1;
        for (int i = 0; i < prefixShape.Length; i++)
        {
            int qd = query._shape[i];
            if (key._shape[i] != qd)
                throw new ArgumentException($"key prefix shape mismatch at axis {i}: query={qd} vs key={key._shape[i]}.", nameof(key));
            if (value._shape[i] != qd)
                throw new ArgumentException($"value prefix shape mismatch at axis {i}: query={qd} vs value={value._shape[i]}.", nameof(value));
            prefixShape[i] = qd;
            batchProduct *= qd;
        }

        // Bias broadcast: last two dims are [Sq, Sk]; leading dims
        // broadcast NumPy-style against prefixShape.
        biasPrefixShape = Array.Empty<int>();
        biasPrefixStrides = Array.Empty<int>();
        if (attentionBias is not null)
        {
            int bRank = attentionBias.Rank;
            if (bRank < 2)
                throw new ArgumentException($"attentionBias rank must be >= 2; got {bRank}.", nameof(attentionBias));
            if (attentionBias._shape[bRank - 2] != Sq)
                throw new ArgumentException($"attentionBias last-two[-2] must be Sq={Sq}; got {attentionBias._shape[bRank - 2]}.", nameof(attentionBias));
            if (attentionBias._shape[bRank - 1] != Sk)
                throw new ArgumentException($"attentionBias last-two[-1] must be Sk={Sk}; got {attentionBias._shape[bRank - 1]}.", nameof(attentionBias));

            int biasPrefixRank = bRank - 2;
            biasPrefixShape = new int[biasPrefixRank];
            for (int i = 0; i < biasPrefixRank; i++) biasPrefixShape[i] = attentionBias._shape[i];

            // NumPy-broadcast check: align trailing-prefix-axis-by-axis;
            // bias prefix shape can have fewer dims (left-padded with
            // size-1) and any size-1 axis broadcasts. Compute strides
            // such that each output prefix index maps to a flat bias
            // prefix offset, with stride 0 wherever a dim broadcasts.
            biasPrefixStrides = new int[prefixShape.Length];
            int leftPad = prefixShape.Length - biasPrefixRank;
            int biasStride = Sq * Sk; // last-two product = stride for the
                                       // last bias-prefix axis
            for (int i = biasPrefixRank - 1; i >= 0; i--)
            {
                int outAxis = leftPad + i;
                int bDim = biasPrefixShape[i];
                int oDim = prefixShape[outAxis];
                if (bDim == oDim)
                {
                    biasPrefixStrides[outAxis] = biasStride;
                }
                else if (bDim == 1)
                {
                    biasPrefixStrides[outAxis] = 0; // broadcast along this axis
                }
                else
                {
                    throw new ArgumentException(
                        $"attentionBias prefix dim {i} (size {bDim}) must equal " +
                        $"output prefix dim {outAxis} (size {oDim}) or be 1 to broadcast.",
                        nameof(attentionBias));
                }
                biasStride *= bDim;
            }
            // Left-padded axes of the bias prefix (where bias has fewer
            // leading dims than the output) implicitly broadcast — set
            // their stride to 0.
            for (int i = 0; i < leftPad; i++) biasPrefixStrides[i] = 0;
        }
    }

    /// <summary>
    /// Maps a flat batchProduct index <paramref name="b"/> to the
    /// equivalent flat bias prefix offset using the precomputed
    /// per-axis strides (entries with stride 0 cause broadcasting on
    /// that axis).
    /// </summary>
    private static int BiasPrefixOffsetFor(int b, int[] prefixShape, int[] biasPrefixStrides)
    {
        if (biasPrefixStrides.Length == 0) return 0;
        int offset = 0;
        int rem = b;
        for (int i = prefixShape.Length - 1; i >= 0; i--)
        {
            int axisIdx = rem % prefixShape[i];
            rem /= prefixShape[i];
            offset += axisIdx * biasPrefixStrides[i];
        }
        return offset;
    }

    // ── Float kernel (paper-faithful Dao 2023) ──────────────────────────

    private static void ForwardFloat(
        float[] q, float[] k, float[] v, float[]? bias,
        float[] o, float[] lse,
        int batchProduct, int Sq, int Sk, int headDim, int Dv,
        int blockSizeQ, int blockSizeKV, float scale, bool isCausal, int queryOffset,
        int[] prefixShape, int[] biasPrefixStrides)
    {
        int Br = Math.Min(blockSizeQ, Sq);
        int Bc = Math.Min(blockSizeKV, Sk);
        var mRow = new float[Br];
        var lRow = new float[Br];
        var oBlock = new float[Br * Dv];
        var sBlock = new float[Br * Bc];

        for (int b = 0; b < batchProduct; b++)
        {
            int qBase = b * Sq * headDim;
            int kBase = b * Sk * headDim;
            int vBase = b * Sk * Dv;
            int oBase = b * Sq * Dv;
            int lseBase = b * Sq;
            int biasBase = bias is not null ? BiasPrefixOffsetFor(b, prefixShape, biasPrefixStrides) : 0;

            for (int qStart = 0; qStart < Sq; qStart += Br)
            {
                int qEnd = Math.Min(qStart + Br, Sq);
                int qLen = qEnd - qStart;

                for (int i = 0; i < qLen; i++)
                {
                    mRow[i] = float.NegativeInfinity;
                    lRow[i] = 0f;
                }
                Array.Clear(oBlock, 0, qLen * Dv);

                for (int kStart = 0; kStart < Sk; kStart += Bc)
                {
                    int kEnd = Math.Min(kStart + Bc, Sk);
                    int kLen = kEnd - kStart;
                    if (isCausal && kStart > queryOffset + qEnd - 1) break;

                    // S = Q @ K^T * scale
                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int qRow = qBase + (qStart + ii) * headDim;
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int kRow = kBase + (kStart + jj) * headDim;
                            float acc = 0f;
                            for (int d = 0; d < headDim; d++) acc += q[qRow + d] * k[kRow + d];
                            float score = acc * scale;
                            if (bias is not null)
                                score += bias[biasBase + (qStart + ii) * Sk + (kStart + jj)];
                            if (isCausal && (kStart + jj) > queryOffset + qStart + ii)
                                score = float.NegativeInfinity;
                            sBlock[ii * Bc + jj] = score;
                        }
                    }

                    // Online softmax + accumulate P @ V.
                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int sRowBase = ii * Bc;
                        float rowMax = float.NegativeInfinity;
                        for (int jj = 0; jj < kLen; jj++)
                            if (sBlock[sRowBase + jj] > rowMax) rowMax = sBlock[sRowBase + jj];
                        float mPrev = mRow[ii];
                        float mNew = Math.Max(mPrev, rowMax);
                        float alpha = float.IsNegativeInfinity(mPrev) ? 0f : (float)Math.Exp(mPrev - mNew);
                        int oRowBase = ii * Dv;
                        float lNew = alpha * lRow[ii];
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            float p = (float)Math.Exp(sBlock[sRowBase + jj] - mNew);
                            lNew += p;
                            int vRow = vBase + (kStart + jj) * Dv;
                            if (jj == 0 && alpha != 1f)
                            {
                                for (int d = 0; d < Dv; d++)
                                    oBlock[oRowBase + d] = alpha * oBlock[oRowBase + d] + p * v[vRow + d];
                            }
                            else
                            {
                                for (int d = 0; d < Dv; d++)
                                    oBlock[oRowBase + d] += p * v[vRow + d];
                            }
                        }
                        mRow[ii] = mNew;
                        lRow[ii] = lNew;
                    }
                }

                for (int ii = 0; ii < qLen; ii++)
                {
                    float invL = lRow[ii] == 0f ? 0f : 1f / lRow[ii];
                    int oRow = oBase + (qStart + ii) * Dv;
                    for (int d = 0; d < Dv; d++) o[oRow + d] = oBlock[ii * Dv + d] * invL;
                    lse[lseBase + qStart + ii] = mRow[ii] + (float)Math.Log(lRow[ii] == 0f ? 1e-30f : lRow[ii]);
                }
            }
        }
    }

    private static void BackwardFloat(
        float[] dO, float[] q, float[] k, float[] v, float[] o, float[] lse,
        float[]? bias,
        float[] dQ, float[] dK, float[] dV,
        int batchProduct, int Sq, int Sk, int headDim, int Dv,
        int blockSizeQ, int blockSizeKV, float scale, bool isCausal, int queryOffset,
        int[] prefixShape, int[] biasPrefixStrides)
    {
        // D_i = row-wise sum(dO * O) across Dv — used in dS.
        var D = new float[batchProduct * Sq];
        for (int b = 0; b < batchProduct; b++)
        {
            int oBase = b * Sq * Dv;
            int dBase = b * Sq;
            for (int i = 0; i < Sq; i++)
            {
                int oRow = oBase + i * Dv;
                float acc = 0f;
                for (int d = 0; d < Dv; d++) acc += dO[oRow + d] * o[oRow + d];
                D[dBase + i] = acc;
            }
        }

        int Br = Math.Min(blockSizeQ, Sq);
        int Bc = Math.Min(blockSizeKV, Sk);
        var sBlock = new float[Br * Bc];
        var pBlock = new float[Br * Bc];

        for (int b = 0; b < batchProduct; b++)
        {
            int qBase = b * Sq * headDim;
            int kBase = b * Sk * headDim;
            int vBase = b * Sk * Dv;
            int oBase = b * Sq * Dv;
            int lseBase = b * Sq;
            int biasBase = bias is not null ? BiasPrefixOffsetFor(b, prefixShape, biasPrefixStrides) : 0;

            for (int qStart = 0; qStart < Sq; qStart += Br)
            {
                int qEnd = Math.Min(qStart + Br, Sq);
                int qLen = qEnd - qStart;
                for (int kStart = 0; kStart < Sk; kStart += Bc)
                {
                    int kEnd = Math.Min(kStart + Bc, Sk);
                    int kLen = kEnd - kStart;
                    if (isCausal && kStart > queryOffset + qEnd - 1) break;

                    // Recompute S.
                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int qRow = qBase + (qStart + ii) * headDim;
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int kRow = kBase + (kStart + jj) * headDim;
                            float acc = 0f;
                            for (int d = 0; d < headDim; d++) acc += q[qRow + d] * k[kRow + d];
                            float score = acc * scale;
                            if (bias is not null)
                                score += bias[biasBase + (qStart + ii) * Sk + (kStart + jj)];
                            if (isCausal && (kStart + jj) > queryOffset + qStart + ii)
                                score = float.NegativeInfinity;
                            sBlock[ii * Bc + jj] = score;
                        }
                    }

                    // P = softmax(S) reconstructed from saved LSE.
                    for (int ii = 0; ii < qLen; ii++)
                    {
                        float lseV = lse[lseBase + qStart + ii];
                        for (int jj = 0; jj < kLen; jj++)
                            pBlock[ii * Bc + jj] = (float)Math.Exp(sBlock[ii * Bc + jj] - lseV);
                    }

                    // dV += P^T @ dO
                    for (int jj = 0; jj < kLen; jj++)
                    {
                        int vRow = vBase + (kStart + jj) * Dv;
                        for (int d = 0; d < Dv; d++)
                        {
                            float acc = 0f;
                            for (int ii = 0; ii < qLen; ii++)
                                acc += pBlock[ii * Bc + jj] * dO[oBase + (qStart + ii) * Dv + d];
                            dV[vRow + d] += acc;
                        }
                    }

                    // dQ, dK from dP/dS chain.
                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int dORow = oBase + (qStart + ii) * Dv;
                        float dI = D[lseBase + qStart + ii];
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int vRow = vBase + (kStart + jj) * Dv;
                            float dP = 0f;
                            for (int d = 0; d < Dv; d++) dP += dO[dORow + d] * v[vRow + d];
                            float p = pBlock[ii * Bc + jj];
                            float dS = p * (dP - dI) * scale;
                            int qRow = qBase + (qStart + ii) * headDim;
                            int kRow = kBase + (kStart + jj) * headDim;
                            for (int d = 0; d < headDim; d++)
                            {
                                dQ[qRow + d] += dS * k[kRow + d];
                                dK[kRow + d] += dS * q[qRow + d];
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Double kernel (research / scientific) ──────────────────────────

    private static void ForwardDouble(
        double[] q, double[] k, double[] v, double[]? bias,
        double[] o, double[] lse,
        int batchProduct, int Sq, int Sk, int headDim, int Dv,
        int blockSizeQ, int blockSizeKV, double scale, bool isCausal, int queryOffset,
        int[] prefixShape, int[] biasPrefixStrides)
    {
        int Br = Math.Min(blockSizeQ, Sq);
        int Bc = Math.Min(blockSizeKV, Sk);
        var mRow = new double[Br];
        var lRow = new double[Br];
        var oBlock = new double[Br * Dv];
        var sBlock = new double[Br * Bc];

        for (int b = 0; b < batchProduct; b++)
        {
            int qBase = b * Sq * headDim;
            int kBase = b * Sk * headDim;
            int vBase = b * Sk * Dv;
            int oBase = b * Sq * Dv;
            int lseBase = b * Sq;
            int biasBase = bias is not null ? BiasPrefixOffsetFor(b, prefixShape, biasPrefixStrides) : 0;

            for (int qStart = 0; qStart < Sq; qStart += Br)
            {
                int qEnd = Math.Min(qStart + Br, Sq);
                int qLen = qEnd - qStart;
                for (int i = 0; i < qLen; i++) { mRow[i] = double.NegativeInfinity; lRow[i] = 0.0; }
                Array.Clear(oBlock, 0, qLen * Dv);

                for (int kStart = 0; kStart < Sk; kStart += Bc)
                {
                    int kEnd = Math.Min(kStart + Bc, Sk);
                    int kLen = kEnd - kStart;
                    if (isCausal && kStart > queryOffset + qEnd - 1) break;

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int qRow = qBase + (qStart + ii) * headDim;
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int kRow = kBase + (kStart + jj) * headDim;
                            double acc = 0.0;
                            for (int d = 0; d < headDim; d++) acc += q[qRow + d] * k[kRow + d];
                            double score = acc * scale;
                            if (bias is not null)
                                score += bias[biasBase + (qStart + ii) * Sk + (kStart + jj)];
                            if (isCausal && (kStart + jj) > queryOffset + qStart + ii)
                                score = double.NegativeInfinity;
                            sBlock[ii * Bc + jj] = score;
                        }
                    }

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int sRowBase = ii * Bc;
                        double rowMax = double.NegativeInfinity;
                        for (int jj = 0; jj < kLen; jj++)
                            if (sBlock[sRowBase + jj] > rowMax) rowMax = sBlock[sRowBase + jj];
                        double mPrev = mRow[ii];
                        double mNew = Math.Max(mPrev, rowMax);
                        double alpha = double.IsNegativeInfinity(mPrev) ? 0.0 : Math.Exp(mPrev - mNew);
                        int oRowBase = ii * Dv;
                        double lNew = alpha * lRow[ii];
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            double p = Math.Exp(sBlock[sRowBase + jj] - mNew);
                            lNew += p;
                            int vRow = vBase + (kStart + jj) * Dv;
                            if (jj == 0 && alpha != 1.0)
                            {
                                for (int d = 0; d < Dv; d++)
                                    oBlock[oRowBase + d] = alpha * oBlock[oRowBase + d] + p * v[vRow + d];
                            }
                            else
                            {
                                for (int d = 0; d < Dv; d++)
                                    oBlock[oRowBase + d] += p * v[vRow + d];
                            }
                        }
                        mRow[ii] = mNew;
                        lRow[ii] = lNew;
                    }
                }

                for (int ii = 0; ii < qLen; ii++)
                {
                    double invL = lRow[ii] == 0.0 ? 0.0 : 1.0 / lRow[ii];
                    int oRow = oBase + (qStart + ii) * Dv;
                    for (int d = 0; d < Dv; d++) o[oRow + d] = oBlock[ii * Dv + d] * invL;
                    lse[lseBase + qStart + ii] = mRow[ii] + Math.Log(lRow[ii] == 0.0 ? 1e-300 : lRow[ii]);
                }
            }
        }
    }

    private static void BackwardDouble(
        double[] dO, double[] q, double[] k, double[] v, double[] o, double[] lse,
        double[]? bias,
        double[] dQ, double[] dK, double[] dV,
        int batchProduct, int Sq, int Sk, int headDim, int Dv,
        int blockSizeQ, int blockSizeKV, double scale, bool isCausal, int queryOffset,
        int[] prefixShape, int[] biasPrefixStrides)
    {
        var D = new double[batchProduct * Sq];
        for (int b = 0; b < batchProduct; b++)
        {
            int oBase = b * Sq * Dv;
            int dBase = b * Sq;
            for (int i = 0; i < Sq; i++)
            {
                int oRow = oBase + i * Dv;
                double acc = 0.0;
                for (int d = 0; d < Dv; d++) acc += dO[oRow + d] * o[oRow + d];
                D[dBase + i] = acc;
            }
        }

        int Br = Math.Min(blockSizeQ, Sq);
        int Bc = Math.Min(blockSizeKV, Sk);
        var sBlock = new double[Br * Bc];
        var pBlock = new double[Br * Bc];

        for (int b = 0; b < batchProduct; b++)
        {
            int qBase = b * Sq * headDim;
            int kBase = b * Sk * headDim;
            int vBase = b * Sk * Dv;
            int oBase = b * Sq * Dv;
            int lseBase = b * Sq;
            int biasBase = bias is not null ? BiasPrefixOffsetFor(b, prefixShape, biasPrefixStrides) : 0;

            for (int qStart = 0; qStart < Sq; qStart += Br)
            {
                int qEnd = Math.Min(qStart + Br, Sq);
                int qLen = qEnd - qStart;
                for (int kStart = 0; kStart < Sk; kStart += Bc)
                {
                    int kEnd = Math.Min(kStart + Bc, Sk);
                    int kLen = kEnd - kStart;
                    if (isCausal && kStart > queryOffset + qEnd - 1) break;

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int qRow = qBase + (qStart + ii) * headDim;
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int kRow = kBase + (kStart + jj) * headDim;
                            double acc = 0.0;
                            for (int d = 0; d < headDim; d++) acc += q[qRow + d] * k[kRow + d];
                            double score = acc * scale;
                            if (bias is not null)
                                score += bias[biasBase + (qStart + ii) * Sk + (kStart + jj)];
                            if (isCausal && (kStart + jj) > queryOffset + qStart + ii)
                                score = double.NegativeInfinity;
                            sBlock[ii * Bc + jj] = score;
                        }
                    }

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        double lseV = lse[lseBase + qStart + ii];
                        for (int jj = 0; jj < kLen; jj++)
                            pBlock[ii * Bc + jj] = Math.Exp(sBlock[ii * Bc + jj] - lseV);
                    }

                    for (int jj = 0; jj < kLen; jj++)
                    {
                        int vRow = vBase + (kStart + jj) * Dv;
                        for (int d = 0; d < Dv; d++)
                        {
                            double acc = 0.0;
                            for (int ii = 0; ii < qLen; ii++)
                                acc += pBlock[ii * Bc + jj] * dO[oBase + (qStart + ii) * Dv + d];
                            dV[vRow + d] += acc;
                        }
                    }

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int dORow = oBase + (qStart + ii) * Dv;
                        double dI = D[lseBase + qStart + ii];
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int vRow = vBase + (kStart + jj) * Dv;
                            double dP = 0.0;
                            for (int d = 0; d < Dv; d++) dP += dO[dORow + d] * v[vRow + d];
                            double p = pBlock[ii * Bc + jj];
                            double dS = p * (dP - dI) * scale;
                            int qRow = qBase + (qStart + ii) * headDim;
                            int kRow = kBase + (kStart + jj) * headDim;
                            for (int d = 0; d < headDim; d++)
                            {
                                dQ[qRow + d] += dS * k[kRow + d];
                                dK[kRow + d] += dS * q[qRow + d];
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Generic-T fallback (correct-but-slow via INumericOperations<T>) ──

    private static void ForwardGeneric(
        T[] q, T[] k, T[] v, T[]? bias, T[] o, T[] lse,
        int batchProduct, int Sq, int Sk, int headDim, int Dv,
        int blockSizeQ, int blockSizeKV, double scale, bool isCausal, int queryOffset,
        int[] prefixShape, int[] biasPrefixStrides)
    {
        // Generic path: convert each tile through ToDouble, run the
        // double kernel inline, ToDouble for output. Slower than the
        // typed path but covers BFloat16 / Half / custom numerics.
        var ops = MathHelper.GetNumericOperations<T>();

        int Br = Math.Min(blockSizeQ, Sq);
        int Bc = Math.Min(blockSizeKV, Sk);
        var mRow = new double[Br];
        var lRow = new double[Br];
        var oBlock = new double[Br * Dv];
        var sBlock = new double[Br * Bc];

        for (int b = 0; b < batchProduct; b++)
        {
            int qBase = b * Sq * headDim;
            int kBase = b * Sk * headDim;
            int vBase = b * Sk * Dv;
            int oBase = b * Sq * Dv;
            int lseBase = b * Sq;
            int biasBase = bias is not null ? BiasPrefixOffsetFor(b, prefixShape, biasPrefixStrides) : 0;

            for (int qStart = 0; qStart < Sq; qStart += Br)
            {
                int qEnd = Math.Min(qStart + Br, Sq);
                int qLen = qEnd - qStart;
                for (int i = 0; i < qLen; i++) { mRow[i] = double.NegativeInfinity; lRow[i] = 0.0; }
                Array.Clear(oBlock, 0, qLen * Dv);

                for (int kStart = 0; kStart < Sk; kStart += Bc)
                {
                    int kEnd = Math.Min(kStart + Bc, Sk);
                    int kLen = kEnd - kStart;
                    if (isCausal && kStart > queryOffset + qEnd - 1) break;

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int qRow = qBase + (qStart + ii) * headDim;
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int kRow = kBase + (kStart + jj) * headDim;
                            double acc = 0.0;
                            for (int d = 0; d < headDim; d++)
                                acc += ops.ToDouble(q[qRow + d]) * ops.ToDouble(k[kRow + d]);
                            double score = acc * scale;
                            if (bias is not null)
                                score += ops.ToDouble(bias[biasBase + (qStart + ii) * Sk + (kStart + jj)]);
                            if (isCausal && (kStart + jj) > queryOffset + qStart + ii)
                                score = double.NegativeInfinity;
                            sBlock[ii * Bc + jj] = score;
                        }
                    }

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int sRowBase = ii * Bc;
                        double rowMax = double.NegativeInfinity;
                        for (int jj = 0; jj < kLen; jj++)
                            if (sBlock[sRowBase + jj] > rowMax) rowMax = sBlock[sRowBase + jj];
                        double mPrev = mRow[ii];
                        double mNew = Math.Max(mPrev, rowMax);
                        double alpha = double.IsNegativeInfinity(mPrev) ? 0.0 : Math.Exp(mPrev - mNew);
                        int oRowBase = ii * Dv;
                        double lNew = alpha * lRow[ii];
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            double p = Math.Exp(sBlock[sRowBase + jj] - mNew);
                            lNew += p;
                            int vRow = vBase + (kStart + jj) * Dv;
                            if (jj == 0 && alpha != 1.0)
                            {
                                for (int d = 0; d < Dv; d++)
                                    oBlock[oRowBase + d] = alpha * oBlock[oRowBase + d] + p * ops.ToDouble(v[vRow + d]);
                            }
                            else
                            {
                                for (int d = 0; d < Dv; d++)
                                    oBlock[oRowBase + d] += p * ops.ToDouble(v[vRow + d]);
                            }
                        }
                        mRow[ii] = mNew;
                        lRow[ii] = lNew;
                    }
                }

                for (int ii = 0; ii < qLen; ii++)
                {
                    double invL = lRow[ii] == 0.0 ? 0.0 : 1.0 / lRow[ii];
                    int oRow = oBase + (qStart + ii) * Dv;
                    for (int d = 0; d < Dv; d++) o[oRow + d] = ops.FromDouble(oBlock[ii * Dv + d] * invL);
                    lse[lseBase + qStart + ii] = ops.FromDouble(mRow[ii] + Math.Log(lRow[ii] == 0.0 ? 1e-300 : lRow[ii]));
                }
            }
        }
    }

    private static void BackwardGeneric(
        T[] dO, T[] q, T[] k, T[] v, T[] o, T[] lse, T[]? bias,
        T[] dQ, T[] dK, T[] dV,
        int batchProduct, int Sq, int Sk, int headDim, int Dv,
        int blockSizeQ, int blockSizeKV, double scale, bool isCausal, int queryOffset,
        int[] prefixShape, int[] biasPrefixStrides)
    {
        var ops = MathHelper.GetNumericOperations<T>();

        var D = new double[batchProduct * Sq];
        for (int b = 0; b < batchProduct; b++)
        {
            int oBase = b * Sq * Dv;
            int dBase = b * Sq;
            for (int i = 0; i < Sq; i++)
            {
                int oRow = oBase + i * Dv;
                double acc = 0.0;
                for (int d = 0; d < Dv; d++) acc += ops.ToDouble(dO[oRow + d]) * ops.ToDouble(o[oRow + d]);
                D[dBase + i] = acc;
            }
        }

        int Br = Math.Min(blockSizeQ, Sq);
        int Bc = Math.Min(blockSizeKV, Sk);
        var sBlock = new double[Br * Bc];
        var pBlock = new double[Br * Bc];

        for (int b = 0; b < batchProduct; b++)
        {
            int qBase = b * Sq * headDim;
            int kBase = b * Sk * headDim;
            int vBase = b * Sk * Dv;
            int oBase = b * Sq * Dv;
            int lseBase = b * Sq;
            int biasBase = bias is not null ? BiasPrefixOffsetFor(b, prefixShape, biasPrefixStrides) : 0;

            for (int qStart = 0; qStart < Sq; qStart += Br)
            {
                int qEnd = Math.Min(qStart + Br, Sq);
                int qLen = qEnd - qStart;
                for (int kStart = 0; kStart < Sk; kStart += Bc)
                {
                    int kEnd = Math.Min(kStart + Bc, Sk);
                    int kLen = kEnd - kStart;
                    if (isCausal && kStart > queryOffset + qEnd - 1) break;

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int qRow = qBase + (qStart + ii) * headDim;
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int kRow = kBase + (kStart + jj) * headDim;
                            double acc = 0.0;
                            for (int d = 0; d < headDim; d++)
                                acc += ops.ToDouble(q[qRow + d]) * ops.ToDouble(k[kRow + d]);
                            double score = acc * scale;
                            if (bias is not null)
                                score += ops.ToDouble(bias[biasBase + (qStart + ii) * Sk + (kStart + jj)]);
                            if (isCausal && (kStart + jj) > queryOffset + qStart + ii)
                                score = double.NegativeInfinity;
                            sBlock[ii * Bc + jj] = score;
                        }
                    }

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        double lseV = ops.ToDouble(lse[lseBase + qStart + ii]);
                        for (int jj = 0; jj < kLen; jj++)
                            pBlock[ii * Bc + jj] = Math.Exp(sBlock[ii * Bc + jj] - lseV);
                    }

                    for (int jj = 0; jj < kLen; jj++)
                    {
                        int vRow = vBase + (kStart + jj) * Dv;
                        for (int d = 0; d < Dv; d++)
                        {
                            double acc = 0.0;
                            for (int ii = 0; ii < qLen; ii++)
                                acc += pBlock[ii * Bc + jj] * ops.ToDouble(dO[oBase + (qStart + ii) * Dv + d]);
                            dV[vRow + d] = ops.Add(dV[vRow + d], ops.FromDouble(acc));
                        }
                    }

                    for (int ii = 0; ii < qLen; ii++)
                    {
                        int dORow = oBase + (qStart + ii) * Dv;
                        double dI = D[lseBase + qStart + ii];
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int vRow = vBase + (kStart + jj) * Dv;
                            double dP = 0.0;
                            for (int d = 0; d < Dv; d++) dP += ops.ToDouble(dO[dORow + d]) * ops.ToDouble(v[vRow + d]);
                            double p = pBlock[ii * Bc + jj];
                            double dS = p * (dP - dI) * scale;
                            int qRow = qBase + (qStart + ii) * headDim;
                            int kRow = kBase + (kStart + jj) * headDim;
                            for (int d = 0; d < headDim; d++)
                            {
                                dQ[qRow + d] = ops.Add(dQ[qRow + d], ops.FromDouble(dS * ops.ToDouble(k[kRow + d])));
                                dK[kRow + d] = ops.Add(dK[kRow + d], ops.FromDouble(dS * ops.ToDouble(q[qRow + d])));
                            }
                        }
                    }
                }
            }
        }
    }
}
