// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 3: Attention, Activations, Loss, Optimizers, FFT, RNN.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    #region Attention Operations

    public void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        EnsureInitialized();
        var q = DownloadBuffer(query);
        var k = DownloadBuffer(key);
        var v = DownloadBuffer(value);
        float[]? m = mask is not null ? DownloadBuffer(mask) : null;
        int totalHeads = batch * numHeads;
        var outp = new float[totalHeads * seqLen * headDim];
        float[]? aw = attentionWeights is not null ? new float[totalHeads * seqLen * seqLen] : null;

        for (int h = 0; h < totalHeads; h++)
        {
            int qOff = h * seqLen * headDim;
            int kOff = h * seqLen * headDim;
            var scores = new float[seqLen * seqLen];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < seqLen; j++)
                {
                    if (isCausal && j > i) { scores[i * seqLen + j] = float.MinValue; continue; }
                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qOff + i * headDim + d] * k[kOff + j * headDim + d];
                    scores[i * seqLen + j] = dot * scale;
                    if (m is not null)
                    {
                        // Support per-head masks: layout [batch*numHeads, seqLen, seqLen] or [seqLen, seqLen]
                        int maskOffset = m.Length >= totalHeads * seqLen * seqLen ? h * seqLen * seqLen : 0;
                        scores[i * seqLen + j] += m[maskOffset + i * seqLen + j];
                    }
                }

            for (int i = 0; i < seqLen; i++)
            {
                float max = float.MinValue;
                for (int j = 0; j < seqLen; j++) if (scores[i * seqLen + j] > max) max = scores[i * seqLen + j];
                float sum = 0;
                for (int j = 0; j < seqLen; j++) { scores[i * seqLen + j] = MathF.Exp(scores[i * seqLen + j] - max); sum += scores[i * seqLen + j]; }
                if (sum > 0) for (int j = 0; j < seqLen; j++) scores[i * seqLen + j] /= sum;
            }

            if (aw is not null) Array.Copy(scores, 0, aw, h * seqLen * seqLen, seqLen * seqLen);

            int vOff = h * seqLen * headDim;
            int oOff = h * seqLen * headDim;
            for (int i = 0; i < seqLen; i++)
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0;
                    for (int j = 0; j < seqLen; j++)
                        sum += scores[i * seqLen + j] * v[vOff + j * headDim + d];
                    outp[oOff + i * headDim + d] = sum;
                }
        }

        UploadToBuffer(outp, output);
        if (attentionWeights is not null && aw is not null) UploadToBuffer(aw, attentionWeights);
    }

    private void AttentionBackwardCore(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var q = DownloadBuffer(query);
        var k = DownloadBuffer(key);
        var v = DownloadBuffer(value);
        var aw = DownloadBuffer(attentionWeights);
        int totalHeads = batch * numHeads;
        var gq = new float[totalHeads * seqLen * headDim];
        var gk = new float[totalHeads * seqLen * headDim];
        var gv = new float[totalHeads * seqLen * headDim];

        for (int h = 0; h < totalHeads; h++)
        {
            int qOff = h * seqLen * headDim;
            int awOff = h * seqLen * seqLen;

            // gradV = attn_weights^T * gradOutput
            for (int j = 0; j < seqLen; j++)
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0;
                    for (int i = 0; i < seqLen; i++)
                        sum += aw[awOff + i * seqLen + j] * go[qOff + i * headDim + d];
                    gv[qOff + j * headDim + d] = sum;
                }

            // gradAttn = gradOutput * V^T
            var gradAttn = new float[seqLen * seqLen];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < seqLen; j++)
                {
                    float sum = 0;
                    for (int d = 0; d < headDim; d++)
                        sum += go[qOff + i * headDim + d] * v[qOff + j * headDim + d];
                    gradAttn[i * seqLen + j] = sum;
                }

            // softmax backward
            var gradScores = new float[seqLen * seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                float dot = 0;
                for (int j = 0; j < seqLen; j++)
                    dot += gradAttn[i * seqLen + j] * aw[awOff + i * seqLen + j];
                for (int j = 0; j < seqLen; j++)
                {
                    if (isCausal && j > i)
                        gradScores[i * seqLen + j] = 0;
                    else
                        gradScores[i * seqLen + j] = aw[awOff + i * seqLen + j] * (gradAttn[i * seqLen + j] - dot);
                }
            }

            // gradQ = gradScores * K * scale
            for (int i = 0; i < seqLen; i++)
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0;
                    for (int j = 0; j < seqLen; j++)
                        sum += gradScores[i * seqLen + j] * k[qOff + j * headDim + d];
                    gq[qOff + i * headDim + d] = sum * scale;
                }

            // gradK = gradScores^T * Q * scale
            for (int j = 0; j < seqLen; j++)
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0;
                    for (int i = 0; i < seqLen; i++)
                        sum += gradScores[i * seqLen + j] * q[qOff + i * headDim + d];
                    gk[qOff + j * headDim + d] = sum * scale;
                }
        }

        UploadToBuffer(gq, gradQuery);
        UploadToBuffer(gk, gradKey);
        UploadToBuffer(gv, gradValue);
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        AttentionBackwardCore(gradOutput, query, key, value, attentionWeights, gradQuery, gradKey, gradValue,
            batch, numHeads, seqLen, headDim, scale, isCausal);
    }

    public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        => ScaledDotProductAttention(query, key, value, output, null, mask, batch, numHeads, seqLen, headDim, scale, isCausal);

    public void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        if (seqQ != seqK)
            throw new NotSupportedException($"Cross-attention (seqQ={seqQ} != seqK={seqK}) is not yet supported by the Vulkan backend. Use CPU engine for cross-attention.");
        ScaledDotProductAttention(query, key, value, output, null, attentionBias, batch, numHeads, seqQ, headDim, scale, isCausal);
        Fill(softmaxStats, 0f, softmaxStats.Size);
    }

    public void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        var awBuffer = AllocateBuffer(batch * numHeads * seqQ * seqQ);
        ScaledDotProductAttention(query, key, value, output, awBuffer, attentionBias, batch, numHeads, seqQ, headDim, scale, isCausal);
        AttentionBackwardCore(gradOutput, query, key, value, awBuffer, gradQuery, gradKey, gradValue,
            batch, numHeads, seqQ, headDim, scale, isCausal);
        awBuffer.Dispose();
    }

    public void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        // GQA: numKVHeads < numQHeads, each KV head is shared by numQHeads/numKVHeads query heads.
        // Fallback treats it as standard MHA with numQHeads.
        ScaledDotProductAttention(query, key, value, output, attentionWeights, null, batch, numQHeads, seqQ, headDim, scale, isCausal);
    }

    public void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
    {
        // GQA backward: using numQHeads for fallback path.
        AttentionBackwardCore(gradOutput, query, key, value, attentionWeights, gradQuery, gradKey, gradValue,
            batch, numQHeads, seqQ, headDim, scale, false);
    }

    #endregion

    #region Activation Functions and Gradients

    public void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
        => CpuUnary(A, B, size, v => v >= 0 ? v : alpha * v);

    public void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
        => CpuBinary(gradOutput, input, gradInput, size, (g, x) => g * (x >= 0 ? 1f : alpha));

    public void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
        => CpuUnary(A, B, size, v => v >= 0 ? v : alpha * (MathF.Exp(v) - 1f));

    public void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var outp = DownloadBuffer(output);
        var gi = new float[size];
        for (int i = 0; i < size; i++)
            gi[i] = go[i] * (inp[i] >= 0 ? 1f : outp[i] + alpha);
        UploadToBuffer(gi, gradInput);
    }

    public void Swish(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => v / (1f + MathF.Exp(-v)));

    public void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var gi = new float[size];
        for (int i = 0; i < size; i++)
        {
            float sig = 1f / (1f + MathF.Exp(-inp[i]));
            gi[i] = go[i] * (sig + inp[i] * sig * (1f - sig));
        }
        UploadToBuffer(gi, gradInput);
    }

    public void Silu(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => v / (1f + MathF.Exp(-v)));

    public void Mish(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => v * MathF.Tanh(MathF.Log(1f + MathF.Exp(v))));

    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => MathF.Log(1f + MathF.Exp(v)));

    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => v <= -3f ? 0f : v >= 3f ? v : v * (v + 3f) / 6f);

    public void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
        => CpuUnary(A, B, size, v => scale * (v >= 0 ? v : alpha * (MathF.Exp(v) - 1f)));

    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => MathF.Max(0f, MathF.Min(1f, v / 6f + 0.5f)));

    public void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
        => CpuUnary(A, B, size, v => MathF.Max(minVal, MathF.Min(maxVal, v)));

    public void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => CpuBinary(gradOutput, input, gradInput, size, (g, x) => x > 0 ? g : 0f);

    public void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
        => CpuBinary(gradOutput, output, gradInput, size, (g, o) => g * o * (1f - o));

    public void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
        => CpuBinary(gradOutput, output, gradInput, size, (g, o) => g * (1f - o * o));

    public void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var gi = new float[size];
        for (int i = 0; i < size; i++)
        {
            float x = inp[i];
            float t = MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x));
            float dtdx = 0.7978845608f * (1f + 0.134145f * x * x) * (1f - t * t);
            gi[i] = go[i] * (0.5f * (1f + t) + 0.5f * x * dtdx);
        }
        UploadToBuffer(gi, gradInput);
    }

    public void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var outp = DownloadBuffer(output);
        var gi = new float[batchSize * features];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * features;
            float dot = 0;
            for (int j = 0; j < features; j++) dot += go[off + j] * outp[off + j];
            for (int j = 0; j < features; j++)
                gi[off + j] = outp[off + j] * (go[off + j] - dot);
        }
        UploadToBuffer(gi, gradInput);
    }

    public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => SwishBackward(gradOutput, input, gradInput, size);

    public void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var gi = new float[size];
        for (int i = 0; i < size; i++)
        {
            float x = inp[i];
            float sp = MathF.Log(1f + MathF.Exp(x));
            float tsp = MathF.Tanh(sp);
            float sig = 1f / (1f + MathF.Exp(-x));
            gi[i] = go[i] * (tsp + x * sig * (1f - tsp * tsp));
        }
        UploadToBuffer(gi, gradInput);
    }

    public void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => CpuBinary(gradOutput, input, gradInput, size, (g, x) => g / (1f + MathF.Exp(-x)));

    public void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => CpuBinary(gradOutput, input, gradInput, size, (g, x) => g * (x <= -3f ? 0f : x >= 3f ? 1f : (2f * x + 3f) / 6f));

    public void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
        => CpuBinary(gradOutput, input, gradInput, size, (g, x) => g * scale * (x >= 0 ? 1f : alpha * MathF.Exp(x)));

    public void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => CpuBinary(gradOutput, input, gradInput, size, (g, x) => g * (x > -3f && x < 3f ? 1f / 6f : 0f));

    public void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
        => CpuBinary(gradOutput, input, gradInput, size, (g, x) => x > minVal && x < maxVal ? g : 0f);

    #endregion

    #region Loss Functions

    private float CpuLossReduce(IGpuBuffer predictions, IGpuBuffer targets, int size, Func<float, float, float> lossPerElement)
    {
        EnsureInitialized();
        if (size == 0) return 0f;
        var p = DownloadBuffer(predictions);
        var t = DownloadBuffer(targets);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += lossPerElement(p[i], t[i]);
        return sum / size;
    }

    private void CpuLossBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, Func<float, float, float> gradPerElement)
    {
        EnsureInitialized();
        if (size == 0) return;
        var p = DownloadBuffer(predictions);
        var t = DownloadBuffer(targets);
        var gi = new float[size];
        for (int i = 0; i < size; i++) gi[i] = gradPerElement(p[i], t[i]) / size;
        UploadToBuffer(gi, gradInput);
    }

    public float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) => (p - t) * (p - t));

    public void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => 2f * (p - t));

    public float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) => MathF.Abs(p - t));

    public void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => MathF.Sign(p - t));

    public float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) =>
        {
            float cp = MathF.Max(1e-7f, MathF.Min(1f - 1e-7f, p));
            return -(t * MathF.Log(cp) + (1f - t) * MathF.Log(1f - cp));
        });

    public void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) =>
        {
            float cp = MathF.Max(1e-7f, MathF.Min(1f - 1e-7f, p));
            return (-t / cp + (1f - t) / (1f - cp));
        });

    public float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        EnsureInitialized();
        var p = DownloadBuffer(predictions);
        var t = DownloadBuffer(targets);
        float loss = 0;
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * numClasses;
            float max = float.MinValue;
            for (int c = 0; c < numClasses; c++) if (p[off + c] > max) max = p[off + c];
            float logSumExp = 0;
            for (int c = 0; c < numClasses; c++) logSumExp += MathF.Exp(p[off + c] - max);
            logSumExp = max + MathF.Log(logSumExp);
            int target = SingleToInt32BitsCompat(t[b]);
            if ((uint)target >= (uint)numClasses)
                throw new ArgumentOutOfRangeException(nameof(targets), $"Target class index {target} at batch {b} is out of range [0, {numClasses}).");
            loss -= p[off + target] - logSumExp;
        }
        return loss / batchSize;
    }

    public void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
    {
        EnsureInitialized();
        var p = DownloadBuffer(predictions);
        var t = DownloadBuffer(targets);
        var gi = new float[batchSize * numClasses];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * numClasses;
            float max = float.MinValue;
            for (int c = 0; c < numClasses; c++) if (p[off + c] > max) max = p[off + c];
            float sum = 0;
            for (int c = 0; c < numClasses; c++) { gi[off + c] = MathF.Exp(p[off + c] - max); sum += gi[off + c]; }
            for (int c = 0; c < numClasses; c++) gi[off + c] /= sum;
            int target = SingleToInt32BitsCompat(t[b]);
            if ((uint)target >= (uint)numClasses)
                throw new ArgumentOutOfRangeException(nameof(targets), $"Target class index {target} at batch {b} is out of range [0, {numClasses}).");
            gi[off + target] -= 1f;
            for (int c = 0; c < numClasses; c++) gi[off + c] /= batchSize;
        }
        UploadToBuffer(gi, gradInput);
    }

    public float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
        => CpuLossReduce(predictions, targets, size, (p, t) => { float d = MathF.Abs(p - t); return d < beta ? 0.5f * d * d / beta : d - 0.5f * beta; });

    public void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { float d = p - t; return MathF.Abs(d) < beta ? d / beta : MathF.Sign(d); });

    public float HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float delta)
        => SmoothL1Loss(predictions, targets, size, delta);

    public void HuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float delta)
        => SmoothL1Backward(predictions, targets, gradInput, size, delta);

    public float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
    {
        EnsureInitialized();
        var a = DownloadBuffer(anchor); var p = DownloadBuffer(positive); var n = DownloadBuffer(negative);
        float loss = 0;
        for (int b = 0; b < batchSize; b++)
        {
            float dp = 0, dn = 0;
            int off = b * embeddingDim;
            for (int d = 0; d < embeddingDim; d++) { float da = a[off + d] - p[off + d]; dp += da * da; da = a[off + d] - n[off + d]; dn += da * da; }
            loss += MathF.Max(0, MathF.Sqrt(dp) - MathF.Sqrt(dn) + margin);
        }
        return loss / batchSize;
    }

    public void TripletLossBackward(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative,
        IGpuBuffer gradAnchor, IGpuBuffer gradPositive, IGpuBuffer gradNegative,
        int batchSize, int embeddingDim, float margin)
    {
        EnsureInitialized();
        var a = DownloadBuffer(anchor);
        var p = DownloadBuffer(positive);
        var n = DownloadBuffer(negative);
        var ga = new float[batchSize * embeddingDim];
        var gp = new float[batchSize * embeddingDim];
        var gn = new float[batchSize * embeddingDim];

        for (int b = 0; b < batchSize; b++)
        {
            int off = b * embeddingDim;
            float dp = 0, dn = 0;
            for (int d = 0; d < embeddingDim; d++)
            {
                float diffP = a[off + d] - p[off + d]; dp += diffP * diffP;
                float diffN = a[off + d] - n[off + d]; dn += diffN * diffN;
            }
            float distP = MathF.Sqrt(dp + 1e-8f);
            float distN = MathF.Sqrt(dn + 1e-8f);

            if (distP - distN + margin > 0)
            {
                for (int d = 0; d < embeddingDim; d++)
                {
                    float diffP = a[off + d] - p[off + d];
                    float diffN = a[off + d] - n[off + d];
                    ga[off + d] += (diffP / distP - diffN / distN) / batchSize;
                    gp[off + d] += (-diffP / distP) / batchSize;
                    gn[off + d] += (diffN / distN) / batchSize;
                }
            }
        }

        UploadToBuffer(ga, gradAnchor);
        UploadToBuffer(gp, gradPositive);
        UploadToBuffer(gn, gradNegative);
    }

    public float FocalLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float alpha, float gamma)
        => CpuLossReduce(predictions, targets, size, (p, t) =>
        {
            const float eps = 1e-7f;
            float cp = MathF.Max(eps, MathF.Min(1f - eps, p));
            float pt = t > 0.5f ? cp : 1f - cp;
            pt = MathF.Max(eps, MathF.Min(1f - eps, pt));
            return -alpha * MathF.Pow(1f - pt, gamma) * MathF.Log(pt);
        });

    public void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) =>
        {
            const float eps = 1e-7f;
            float cp = MathF.Max(eps, MathF.Min(1f - eps, p));
            bool isPositive = t > 0.5f;
            float pt = isPositive ? cp : 1f - cp;
            pt = MathF.Max(eps, MathF.Min(1f - eps, pt));
            float oneMinusPt = 1f - pt;
            float logPt = MathF.Log(pt);
            float sign = isPositive ? -1f : 1f;
            float common = gamma * logPt + 1f / pt;
            return -alpha * sign * MathF.Pow(oneMinusPt, gamma) * common;
        });

    public float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) => MathF.Log(MathF.Cosh(p - t)));

    public void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => MathF.Tanh(p - t));

    public float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile)
        => CpuLossReduce(predictions, targets, size, (p, t) => { float d = t - p; return d >= 0 ? quantile * d : (quantile - 1f) * d; });

    public void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => t - p >= 0 ? -quantile : 1f - quantile);

    public float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) => MathF.Max(0, 1f - t * p));

    public void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => t * p < 1f ? -t : 0f);

    public float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) => { float h = MathF.Max(0, 1f - t * p); return h * h; });

    public void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { float h = MathF.Max(0, 1f - t * p); return h > 0 ? -2f * h * t : 0f; });

    public float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) => MathF.Exp(p) - t * p);

    public void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => MathF.Exp(p) - t);

    public float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) => MathF.Exp(-t * p));

    public void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => -t * MathF.Exp(-t * p));

    public float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) =>
        {
            float yp = t * p;
            return yp >= -1f ? MathF.Max(0, 1f - yp) * MathF.Max(0, 1f - yp) : -4f * yp;
        });

    public void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) =>
        {
            float yp = t * p;
            return yp >= 1f ? 0f : yp >= -1f ? -2f * t * (1f - yp) : -4f * t;
        });

    public float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => CpuLossReduce(predictions, targets, size, (p, t) => -t * MathF.Log(MathF.Max(1e-7f, p)));

    public void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => -t / MathF.Max(1e-7f, p));

    public float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon)
        => CpuLossReduce(predictions, targets, size, (p, t) => MathF.Sqrt((p - t) * (p - t) + epsilon * epsilon));

    public void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => (p - t) / MathF.Sqrt((p - t) * (p - t) + epsilon * epsilon));

    public float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight)
        => CpuLossReduce(predictions, targets, size, (p, t) => l1Weight * MathF.Abs(p - t) + l2Weight * (p - t) * (p - t));

    public void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
        => CpuLossBackward(predictions, targets, gradInput, size, (p, t) => l1Weight * MathF.Sign(p - t) + 2f * l2Weight * (p - t));

    public float ContrastiveLoss(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
    {
        EnsureInitialized();
        var o1 = DownloadBuffer(output1); var o2 = DownloadBuffer(output2); var lab = DownloadBuffer(labels);
        float loss = 0;
        for (int b = 0; b < batchSize; b++)
        {
            float dist = 0;
            int off = b * embeddingDim;
            for (int d = 0; d < embeddingDim; d++) { float diff = o1[off + d] - o2[off + d]; dist += diff * diff; }
            dist = MathF.Sqrt(dist);
            float y = lab[b];
            loss += y * dist * dist + (1f - y) * MathF.Max(0, margin - dist) * MathF.Max(0, margin - dist);
        }
        return loss / (2f * batchSize);
    }

    public void ContrastiveBackward(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels,
        IGpuBuffer gradOutput1, IGpuBuffer gradOutput2,
        int batchSize, int embeddingDim, float margin)
    {
        EnsureInitialized();
        var o1 = DownloadBuffer(output1);
        var o2 = DownloadBuffer(output2);
        var lab = DownloadBuffer(labels);
        var g1 = new float[batchSize * embeddingDim];
        var g2 = new float[batchSize * embeddingDim];

        for (int b = 0; b < batchSize; b++)
        {
            int off = b * embeddingDim;
            float distSq = 0;
            for (int d = 0; d < embeddingDim; d++)
            {
                float diff = o1[off + d] - o2[off + d];
                distSq += diff * diff;
            }
            float dist = MathF.Sqrt(distSq + 1e-8f);
            float y = lab[b];

            for (int d = 0; d < embeddingDim; d++)
            {
                float diff = o1[off + d] - o2[off + d];
                float gradSimilar = y * diff;
                float gradDissimilar = 0f;
                if (margin - dist > 0)
                    gradDissimilar = -(1f - y) * (margin - dist) * diff / dist;
                g1[off + d] = (gradSimilar + gradDissimilar) / batchSize;
                g2[off + d] = -(gradSimilar + gradDissimilar) / batchSize;
            }
        }

        UploadToBuffer(g1, gradOutput1);
        UploadToBuffer(g2, gradOutput2);
    }

    #endregion

    #region Gradient Clipping and Utility

    public void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
        => CpuUnary(A, B, size, v => MathF.Max(min, MathF.Min(max, v)));

    public float L2Norm(IGpuBuffer A, int size)
    {
        float sumSq = CpuReduce(A, size, 0f, (acc, v) => acc + v * v);
        return MathF.Sqrt(sumSq);
    }

    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
        => CpuUnary(A, B, size, v => MathF.Max(-clipValue, MathF.Min(clipValue, v)));

    public void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
    {
        EnsureInitialized();
        float norm = L2Norm(A, size);
        if (norm > maxNorm)
        {
            float scale = maxNorm / norm;
            Scale(A, B, scale, size);
        }
        else
        {
            Copy(A, B, size);
        }
    }

    public void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A); var b = DownloadBuffer(B); var c = DownloadBuffer(C);
        var d = new float[size];
        for (int i = 0; i < size; i++) d[i] = a[i] * b[i] + c[i];
        UploadToBuffer(d, D);
    }

    public void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
    {
        EnsureInitialized();
        var src = DownloadBuffer(source);
        var idx = DownloadBuffer(indices);
        var dst = DownloadBuffer(destination);
        for (int i = 0; i < sourceSize; i++)
        {
            int dstIdx = SingleToInt32BitsCompat(idx[i]);
            if (dstIdx >= 0 && dstIdx < destSize) dst[dstIdx] += src[i];
        }
        UploadToBuffer(dst, destination);
    }

    public void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource, int numIndices, int featureSize)
    {
        EnsureInitialized();
        var gd = DownloadBuffer(gradDestination);
        var idx = DownloadBuffer(indices);
        var gs = new float[numIndices * featureSize];
        int scatterNumRows = gd.Length / featureSize;
        for (int i = 0; i < numIndices; i++)
        {
            int srcIdx = SingleToInt32BitsCompat(idx[i]);
            if ((uint)srcIdx >= (uint)scatterNumRows)
                throw new ArgumentOutOfRangeException(nameof(indices), $"Decoded index {srcIdx} at position {i} is out of range [0, {scatterNumRows}).");
            for (int f = 0; f < featureSize; f++)
                gs[i * featureSize + f] = gd[srcIdx * featureSize + f];
        }
        UploadToBuffer(gs, gradSource);
    }

    public void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        EnsureInitialized();
        var src = DownloadBuffer(source);
        var idx = DownloadBuffer(indices);
        var outp = new float[numIndices * featureSize];
        int gatherNumRows = src.Length / featureSize;
        for (int i = 0; i < numIndices; i++)
        {
            int srcIdx = SingleToInt32BitsCompat(idx[i]);
            if ((uint)srcIdx >= (uint)gatherNumRows)
                throw new ArgumentOutOfRangeException(nameof(indices), $"Decoded index {srcIdx} at position {i} is out of range [0, {gatherNumRows}).");
            Array.Copy(src, srcIdx * featureSize, outp, i * featureSize, featureSize);
        }
        UploadToBuffer(outp, output);
    }

    #endregion

    #region Comparison Operations

    public void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => CpuBinary(A, B, C, size, (a, b) => a > b ? 1f : 0f);

    public void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => CpuBinary(A, B, C, size, (a, b) => a < b ? 1f : 0f);

    public void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => CpuBinary(A, B, C, size, (a, b) => MathF.Abs(a - b) < 1e-7f ? 1f : 0f);

    public void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        EnsureInitialized();
        var cond = DownloadBuffer(condition); var a = DownloadBuffer(A); var b = DownloadBuffer(B);
        var c = new float[size];
        for (int i = 0; i < size; i++) c[i] = cond[i] != 0f ? a[i] : b[i];
        UploadToBuffer(c, C);
    }

    public void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
        => CpuUnary(A, C, size, v => MathF.Abs(v - scalar) > 1e-7f ? 1f : 0f);

    #endregion

    #region Statistics

    public void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        if (reduceSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(reduceSize), "reduceSize must be positive.");
        var a = DownloadBuffer(A);
        var b = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float sum = 0;
            for (int j = 0; j < reduceSize; j++) sum += a[i * reduceSize + j];
            b[i] = sum / reduceSize;
        }
        UploadToBuffer(b, B);
    }

    public void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        if (reduceSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(reduceSize), "reduceSize must be positive.");
        var a = DownloadBuffer(A);
        var m = DownloadBuffer(mean);
        var v = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float sum = 0;
            for (int j = 0; j < reduceSize; j++) { float d = a[i * reduceSize + j] - m[i]; sum += d * d; }
            v[i] = sum / reduceSize;
        }
        UploadToBuffer(v, variance);
    }

    public void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var idx = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float max = float.MinValue; int maxJ = 0;
            for (int j = 0; j < reduceSize; j++)
                if (a[i * reduceSize + j] > max) { max = a[i * reduceSize + j]; maxJ = j; }
            idx[i] = Int32BitsToSingleCompat(maxJ);
        }
        UploadToBuffer(idx, indices);
    }

    public void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        => ArgMax(A, indices, outerSize, reduceSize);

    public void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var idx = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float min = float.MaxValue; int minJ = 0;
            for (int j = 0; j < reduceSize; j++)
                if (a[i * reduceSize + j] < min) { min = a[i * reduceSize + j]; minJ = j; }
            idx[i] = Int32BitsToSingleCompat(minJ);
        }
        UploadToBuffer(idx, indices);
    }

    public void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float max = float.MinValue;
            for (int j = 0; j < reduceSize; j++) if (a[i * reduceSize + j] > max) max = a[i * reduceSize + j];
            b[i] = max;
        }
        UploadToBuffer(b, B);
    }

    public void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var v = new float[outerSize * k];
        var idx = new float[outerSize * k];
        for (int i = 0; i < outerSize; i++)
        {
            var pairs = new (float val, int idx)[reduceSize];
            for (int j = 0; j < reduceSize; j++) pairs[j] = (a[i * reduceSize + j], j);
            Array.Sort(pairs, (x, y) => y.val.CompareTo(x.val));
            for (int j = 0; j < k && j < reduceSize; j++)
            {
                v[i * k + j] = pairs[j].val;
                idx[i * k + j] = Int32BitsToSingleCompat(pairs[j].idx);
            }
        }
        UploadToBuffer(v, values);
        UploadToBuffer(idx, indices);
    }

    public void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A); var b = DownloadBuffer(B);
        var c = new float[outerSize * innerSize];
        for (int i = 0; i < outerSize; i++)
            for (int j = 0; j < innerSize; j++)
                c[i * innerSize + j] = a[i * innerSize + j] * b[j];
        UploadToBuffer(c, C);
    }

    public void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A); var b = DownloadBuffer(B);
        var c = new float[outerSize * innerSize];
        for (int i = 0; i < outerSize; i++)
            for (int j = 0; j < innerSize; j++)
                c[i * innerSize + j] = a[i * innerSize + j] * b[i];
        UploadToBuffer(c, C);
    }

    #endregion
}
