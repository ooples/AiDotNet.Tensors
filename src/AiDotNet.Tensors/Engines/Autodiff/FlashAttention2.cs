using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Block-tiled online-softmax FlashAttention-2 forward + backward.
/// Works strictly in <c>float</c> (the precision the paper assumes);
/// higher-precision T is converted on entry. O(seqLen) intermediate
/// memory — per the paper, the full <c>[Sq, Sk]</c> score matrix is
/// never materialised.
///
/// <para>Algorithm (Dao 2023, <i>FlashAttention-2</i>):</para>
/// <list type="bullet">
/// <item>Split Q into row blocks of size <c>Br</c>; K / V into column
/// blocks of size <c>Bc</c>.</item>
/// <item>For each Q-block, iterate over K-blocks maintaining running
/// (m, l) for online softmax: <c>m</c> = running row max,
/// <c>l</c> = running row denominator.</item>
/// <item>Rescale the running output <c>O</c> by
/// <c>exp(m_prev - m_new)</c> whenever the max updates.</item>
/// <item>Save <c>logsumexp = m + log(l)</c> per row for the backward
/// pass — reconstruction cheaper than checkpointing the full P.</item>
/// </list>
/// </summary>
public static class FlashAttention2
{
    /// <summary>
    /// Tiled forward. Produces <c>(O, logsumexp)</c>. Accepts
    /// <c>[B, H, Sq, D]</c> and <c>[B, H, Sk, D]</c> rank-4 tensors.
    /// </summary>
    /// <param name="blockSizeQ">Row-block size. Default 64.</param>
    /// <param name="blockSizeKV">Col-block size. Default 64.</param>
    /// <param name="scale">Softmax scale. Null → 1/sqrt(headDim).</param>
    /// <param name="isCausal">Apply upper-triangular mask (queryOffset-aware).</param>
    /// <param name="queryOffset">For KV-cache decode.</param>
    /// <param name="attentionBias">Optional additive bias
    /// <c>[B, H, Sq, Sk]</c> or broadcastable.</param>
    public static (Tensor<float> Output, float[] LogSumExp) Forward(
        Tensor<float> query, Tensor<float> key, Tensor<float> value,
        int blockSizeQ = 64, int blockSizeKV = 64,
        double? scale = null, bool isCausal = false, int queryOffset = 0,
        Tensor<float>? attentionBias = null)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            throw new ArgumentException("FlashAttention2 requires rank-4 inputs [B, H, Sq/Sk, D].");
        if (blockSizeQ <= 0) throw new ArgumentOutOfRangeException(nameof(blockSizeQ));
        if (blockSizeKV <= 0) throw new ArgumentOutOfRangeException(nameof(blockSizeKV));

        int B = query._shape[0], H = query._shape[1];
        int Sq = query._shape[2], headDim = query._shape[3];
        int Sk = key._shape[2];
        int Dv = value._shape[3];
        if (key._shape[3] != headDim) throw new ArgumentException("query/key headDim mismatch.");
        if (value._shape[2] != Sk) throw new ArgumentException("key/value seq len mismatch.");
        if (queryOffset < 0 || queryOffset + Sq > Sk)
            throw new ArgumentException(
                $"queryOffset={queryOffset} + Sq={Sq} must be <= Sk={Sk}.", nameof(queryOffset));

        double scaleVal = scale ?? 1.0 / Math.Sqrt(headDim);
        float scaleF = (float)scaleVal;

        var q = query.GetDataArray();
        var k = key.GetDataArray();
        var v = value.GetDataArray();
        var bias = attentionBias?.GetDataArray();
        int biasStrideB = 0, biasStrideH = 0;
        if (bias is not null)
        {
            int br = attentionBias!._shape[0];
            int bh = attentionBias._shape[1];
            biasStrideB = (br == 1 ? 0 : bh * Sq * Sk);
            biasStrideH = (bh == 1 ? 0 : Sq * Sk);
        }

        var output = new Tensor<float>(new[] { B, H, Sq, Dv });
        var o = output.GetDataArray();
        var logsumexp = new float[B * H * Sq];

        // Per-Q-block scratch reused for every Q tile.
        int Br = Math.Min(blockSizeQ, Sq);
        int Bc = Math.Min(blockSizeKV, Sk);
        var mRow = new float[Br]; // running max
        var lRow = new float[Br]; // running denom
        var oBlock = new float[Br * Dv];
        var sBlock = new float[Br * Bc];

        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                int qBHBase = ((b * H) + h) * Sq * headDim;
                int kvBHBase = ((b * H) + h) * Sk * headDim;
                int vBHBase = ((b * H) + h) * Sk * Dv;
                int oBHBase = ((b * H) + h) * Sq * Dv;
                int lseBHBase = ((b * H) + h) * Sq;
                int biasBHBase = bias is not null ? b * biasStrideB + h * biasStrideH : 0;

                for (int qStart = 0; qStart < Sq; qStart += Br)
                {
                    int qEnd = Math.Min(qStart + Br, Sq);
                    int qLen = qEnd - qStart;

                    // Init running state for this Q block.
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

                        // Causal mask: we can skip whole K blocks that are
                        // entirely beyond the max visible position. The
                        // max q-row in this block is qStart + qLen - 1;
                        // it sees keys up to queryOffset + qStart + qLen - 1.
                        if (isCausal && kStart > queryOffset + qEnd - 1) break;

                        // Compute S = Q_i @ K_j^T * scale     [qLen, kLen]
                        for (int ii = 0; ii < qLen; ii++)
                        {
                            int qRow = qBHBase + (qStart + ii) * headDim;
                            for (int jj = 0; jj < kLen; jj++)
                            {
                                int kRow = kvBHBase + (kStart + jj) * headDim;
                                float acc = 0f;
                                for (int d = 0; d < headDim; d++)
                                    acc += q[qRow + d] * k[kRow + d];
                                float score = acc * scaleF;
                                if (bias is not null)
                                {
                                    int bIdx = biasBHBase + (qStart + ii) * Sk + (kStart + jj);
                                    score += bias[bIdx];
                                }
                                // Causal mask inside the block: keys with
                                // index > queryOffset + (qStart+ii) are
                                // masked.
                                if (isCausal && (kStart + jj) > queryOffset + qStart + ii)
                                    score = float.NegativeInfinity;
                                sBlock[ii * Bc + jj] = score;
                            }
                        }

                        // Online softmax update per-row.
                        for (int ii = 0; ii < qLen; ii++)
                        {
                            int sRowBase = ii * Bc;
                            // rowmax over this block's row.
                            float rowMax = float.NegativeInfinity;
                            for (int jj = 0; jj < kLen; jj++)
                                if (sBlock[sRowBase + jj] > rowMax) rowMax = sBlock[sRowBase + jj];

                            float mPrev = mRow[ii];
                            float mNew = Math.Max(mPrev, rowMax);
                            float alpha = float.IsNegativeInfinity(mPrev) ? 0f : (float)Math.Exp(mPrev - mNew);

                            // Rescale previous output + accumulate P @ V.
                            int oRowBase = ii * Dv;
                            float lNew = alpha * lRow[ii];
                            for (int jj = 0; jj < kLen; jj++)
                            {
                                float p = (float)Math.Exp(sBlock[sRowBase + jj] - mNew);
                                lNew += p;
                                int vRow = vBHBase + (kStart + jj) * Dv;
                                // oBlock[ii, :] = alpha * oBlock[ii, :] + p * v[kStart+jj, :]
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

                    // Finalize: normalize by l, write output, save logsumexp.
                    for (int ii = 0; ii < qLen; ii++)
                    {
                        float invL = lRow[ii] == 0f ? 0f : 1f / lRow[ii];
                        int oRow = oBHBase + (qStart + ii) * Dv;
                        for (int d = 0; d < Dv; d++)
                            o[oRow + d] = oBlock[ii * Dv + d] * invL;
                        logsumexp[lseBHBase + qStart + ii] = mRow[ii] + (float)Math.Log(lRow[ii] == 0f ? 1e-30f : lRow[ii]);
                    }
                }
            }
        }

        return (output, logsumexp);
    }

    /// <summary>
    /// Tiled backward using the saved <paramref name="logsumexp"/> from
    /// <see cref="Forward"/> — recomputes P per block (O(blockSize²)
    /// memory) rather than materialising the whole attention matrix.
    /// </summary>
    public static (Tensor<float> GradQuery, Tensor<float> GradKey, Tensor<float> GradValue) Backward(
        Tensor<float> gradOutput,
        Tensor<float> query, Tensor<float> key, Tensor<float> value,
        Tensor<float> output, float[] logsumexp,
        int blockSizeQ = 64, int blockSizeKV = 64,
        double? scale = null, bool isCausal = false, int queryOffset = 0,
        Tensor<float>? attentionBias = null)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (logsumexp is null) throw new ArgumentNullException(nameof(logsumexp));

        int B = query._shape[0], H = query._shape[1];
        int Sq = query._shape[2], headDim = query._shape[3];
        int Sk = key._shape[2];
        int Dv = value._shape[3];
        double scaleVal = scale ?? 1.0 / Math.Sqrt(headDim);
        float scaleF = (float)scaleVal;

        var q = query.GetDataArray();
        var k = key.GetDataArray();
        var v = value.GetDataArray();
        var o = output.GetDataArray();
        var dO = gradOutput.GetDataArray();
        var bias = attentionBias?.GetDataArray();
        int biasStrideB = 0, biasStrideH = 0;
        if (bias is not null)
        {
            int br = attentionBias!._shape[0];
            int bh = attentionBias._shape[1];
            biasStrideB = (br == 1 ? 0 : bh * Sq * Sk);
            biasStrideH = (bh == 1 ? 0 : Sq * Sk);
        }

        var dQ = new Tensor<float>(query._shape);
        var dK = new Tensor<float>(key._shape);
        var dV = new Tensor<float>(value._shape);
        var dQArr = dQ.GetDataArray();
        var dKArr = dK.GetDataArray();
        var dVArr = dV.GetDataArray();

        // Precompute D_i = row-wise sum(dO * O) across Dv — used in dS.
        var D = new float[B * H * Sq];
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                int base_ = ((b * H) + h) * Sq;
                for (int i = 0; i < Sq; i++)
                {
                    int oRow = (base_ + i) * Dv;
                    float acc = 0f;
                    for (int d = 0; d < Dv; d++) acc += dO[oRow + d] * o[oRow + d];
                    D[base_ + i] = acc;
                }
            }
        }

        int Br = Math.Min(blockSizeQ, Sq);
        int Bc = Math.Min(blockSizeKV, Sk);
        var sBlock = new float[Br * Bc];
        var pBlock = new float[Br * Bc];

        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                int qBHBase = ((b * H) + h) * Sq * headDim;
                int kvBHBase = ((b * H) + h) * Sk * headDim;
                int vBHBase = ((b * H) + h) * Sk * Dv;
                int oBHBase = ((b * H) + h) * Sq * Dv;
                int lseBHBase = ((b * H) + h) * Sq;
                int biasBHBase = bias is not null ? b * biasStrideB + h * biasStrideH : 0;

                for (int qStart = 0; qStart < Sq; qStart += Br)
                {
                    int qEnd = Math.Min(qStart + Br, Sq);
                    int qLen = qEnd - qStart;

                    for (int kStart = 0; kStart < Sk; kStart += Bc)
                    {
                        int kEnd = Math.Min(kStart + Bc, Sk);
                        int kLen = kEnd - kStart;

                        if (isCausal && kStart > queryOffset + qEnd - 1) break;

                        // Recompute S = Q_i @ K_j^T * scale.
                        for (int ii = 0; ii < qLen; ii++)
                        {
                            int qRow = qBHBase + (qStart + ii) * headDim;
                            for (int jj = 0; jj < kLen; jj++)
                            {
                                int kRow = kvBHBase + (kStart + jj) * headDim;
                                float acc = 0f;
                                for (int d = 0; d < headDim; d++) acc += q[qRow + d] * k[kRow + d];
                                float score = acc * scaleF;
                                if (bias is not null)
                                {
                                    int bIdx = biasBHBase + (qStart + ii) * Sk + (kStart + jj);
                                    score += bias[bIdx];
                                }
                                if (isCausal && (kStart + jj) > queryOffset + qStart + ii)
                                    score = float.NegativeInfinity;
                                sBlock[ii * Bc + jj] = score;
                            }
                        }

                        // P = softmax from saved logsumexp.
                        for (int ii = 0; ii < qLen; ii++)
                        {
                            float lse = logsumexp[lseBHBase + qStart + ii];
                            for (int jj = 0; jj < kLen; jj++)
                                pBlock[ii * Bc + jj] = (float)Math.Exp(sBlock[ii * Bc + jj] - lse);
                        }

                        // dV += P^T @ dO
                        for (int jj = 0; jj < kLen; jj++)
                        {
                            int vRow = vBHBase + (kStart + jj) * Dv;
                            for (int d = 0; d < Dv; d++)
                            {
                                float acc = 0f;
                                for (int ii = 0; ii < qLen; ii++)
                                {
                                    int dORow = oBHBase + (qStart + ii) * Dv;
                                    acc += pBlock[ii * Bc + jj] * dO[dORow + d];
                                }
                                dVArr[vRow + d] += acc;
                            }
                        }

                        // dP = dO @ V^T     [qLen, kLen]
                        // dS = P * (dP - D_i)   where D_i is the row-sum of dO * O
                        // dQ += dS @ K * scale
                        // dK += dS^T @ Q * scale
                        for (int ii = 0; ii < qLen; ii++)
                        {
                            int dORow = oBHBase + (qStart + ii) * Dv;
                            float dI = D[lseBHBase + qStart + ii];
                            for (int jj = 0; jj < kLen; jj++)
                            {
                                int vRow = vBHBase + (kStart + jj) * Dv;
                                float dP = 0f;
                                for (int d = 0; d < Dv; d++) dP += dO[dORow + d] * v[vRow + d];
                                float p = pBlock[ii * Bc + jj];
                                float dS = p * (dP - dI) * scaleF;

                                // Accumulate into dQ[i] += dS * K[j]
                                int qRow = qBHBase + (qStart + ii) * headDim;
                                int kRow = kvBHBase + (kStart + jj) * headDim;
                                for (int d = 0; d < headDim; d++)
                                {
                                    dQArr[qRow + d] += dS * k[kRow + d];
                                    dKArr[kRow + d] += dS * q[qRow + d];
                                }
                            }
                        }
                    }
                }
            }
        }
        return (dQ, dK, dV);
    }
}
