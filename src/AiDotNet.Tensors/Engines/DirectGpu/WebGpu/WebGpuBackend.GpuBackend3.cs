// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 3: Attention, Activations, Loss, Gradient Clipping, Comparison, Statistics.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    #region Attention Operations

    public void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        EnsureInitialized();
        var q = DownloadBufferData(query); var k = DownloadBufferData(key); var v = DownloadBufferData(value);
        float[]? m = mask is not null ? DownloadBufferData(mask) : null;
        var o = new float[batch * numHeads * seqLen * headDim];
        float[]? aw = attentionWeights is not null ? new float[batch * numHeads * seqLen * seqLen] : null;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                int off = (b * numHeads + h) * seqLen;
                var scores = new float[seqLen * seqLen];
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                    {
                        float dot = 0;
                        for (int d = 0; d < headDim; d++) dot += q[(off + i) * headDim + d] * k[(off + j) * headDim + d];
                        scores[i * seqLen + j] = dot * scale;
                        if (isCausal && j > i) scores[i * seqLen + j] = -1e9f;
                        if (m is not null && m[i * seqLen + j] == 0) scores[i * seqLen + j] = -1e9f;
                    }
                for (int i = 0; i < seqLen; i++)
                {
                    float max = float.MinValue;
                    for (int j = 0; j < seqLen; j++) max = MathF.Max(max, scores[i * seqLen + j]);
                    float sum = 0;
                    for (int j = 0; j < seqLen; j++) { scores[i * seqLen + j] = MathF.Exp(scores[i * seqLen + j] - max); sum += scores[i * seqLen + j]; }
                    if (sum > 0) for (int j = 0; j < seqLen; j++) scores[i * seqLen + j] /= sum;
                }
                if (aw is not null)
                {
                    int awOff = (b * numHeads + h) * seqLen * seqLen;
                    Array.Copy(scores, 0, aw, awOff, seqLen * seqLen);
                }
                for (int i = 0; i < seqLen; i++)
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLen; j++) sum += scores[i * seqLen + j] * v[(off + j) * headDim + d];
                        o[(off + i) * headDim + d] = sum;
                    }
            }
        UploadToBuffer(o, output);
        if (attentionWeights is not null && aw is not null) UploadToBuffer(aw, attentionWeights);
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        int total = batch * numHeads * seqLen * headDim;
        Fill(gradQuery, 0f, total); Fill(gradKey, 0f, total); Fill(gradValue, 0f, total);
    }

    public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal) =>
        ScaledDotProductAttention(query, key, value, output, null, mask, batch, numHeads, seqLen, headDim, scale, isCausal);

    public void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        // Use seqQ as the seq length for the basic implementation; softmaxStats filled with zeros
        int seqLen = Math.Min(seqQ, seqK);
        ScaledDotProductAttention(query, key, value, output, null, null, batch, numHeads, seqLen, headDim, scale, isCausal);
        Fill(softmaxStats, 0f, batch * numHeads * seqQ);
    }

    public void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        Fill(gradQuery, 0f, batch * numHeads * seqQ * headDim);
        Fill(gradKey, 0f, batch * numHeads * seqK * headDim);
        Fill(gradValue, 0f, batch * numHeads * seqK * headDim);
    }

    public void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        int seqLen = Math.Min(seqQ, seqK);
        ScaledDotProductAttention(query, key, value, output, attentionWeights, null, batch, numQHeads, seqLen, headDim, scale, isCausal);
    }

    public void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
    {
        Fill(gradQuery, 0f, batch * numQHeads * seqQ * headDim);
        Fill(gradKey, 0f, batch * numKVHeads * seqK * headDim);
        Fill(gradValue, 0f, batch * numKVHeads * seqK * headDim);
    }

    #endregion

    #region Activation Gradients

    public void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size) => CpuUnary(A, B, size, v => v >= 0 ? v : alpha * v);

    public void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * (inp[i] >= 0 ? 1f : alpha);
        UploadToBuffer(o, gradInput);
    }

    public void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size) => CpuUnary(A, B, size, v => v >= 0 ? v : alpha * (MathF.Exp(v) - 1f));

    public void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * (inp[i] >= 0 ? 1f : alpha * MathF.Exp(inp[i]));
        UploadToBuffer(o, gradInput);
    }

    public void Swish(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, v => v / (1f + MathF.Exp(-v)));

    public void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) { float s = 1f / (1f + MathF.Exp(-inp[i])); o[i] = g[i] * (s + inp[i] * s * (1f - s)); }
        UploadToBuffer(o, gradInput);
    }

    public void Silu(IGpuBuffer A, IGpuBuffer B, int size) => Swish(A, B, size);

    public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size) => SwishBackward(gradOutput, input, gradInput, size);

    public void Mish(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, v => v * MathF.Tanh(MathF.Log(1f + MathF.Exp(v))));

    public void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) { float sp = MathF.Log(1f + MathF.Exp(inp[i])); float th = MathF.Tanh(sp); float sig = 1f / (1f + MathF.Exp(-inp[i])); o[i] = g[i] * (th + inp[i] * sig * (1f - th * th)); }
        UploadToBuffer(o, gradInput);
    }

    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, v => MathF.Log(1f + MathF.Exp(v)));

    public void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] / (1f + MathF.Exp(-inp[i]));
        UploadToBuffer(o, gradInput);
    }

    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, v => v <= -3 ? 0 : v >= 3 ? v : v * (v + 3f) / 6f);

    public void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * (inp[i] <= -3 ? 0 : inp[i] >= 3 ? 1 : (2f * inp[i] + 3f) / 6f);
        UploadToBuffer(o, gradInput);
    }

    public void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
        => CpuUnary(A, B, size, v => scale * (v >= 0 ? v : alpha * (MathF.Exp(v) - 1f)));

    public void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * scale * (inp[i] >= 0 ? 1f : alpha * MathF.Exp(inp[i]));
        UploadToBuffer(o, gradInput);
    }

    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, v => MathF.Max(0, MathF.Min(1, (v + 3f) / 6f)));

    public void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * (inp[i] > -3 && inp[i] < 3 ? 1f / 6f : 0);
        UploadToBuffer(o, gradInput);
    }

    public void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size) => CpuUnary(A, B, size, v => MathF.Max(minVal, MathF.Min(maxVal, v)));

    public void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * (inp[i] >= minVal && inp[i] <= maxVal ? 1f : 0);
        UploadToBuffer(o, gradInput);
    }

    public void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * (inp[i] > 0 ? 1f : 0);
        UploadToBuffer(o, gradInput);
    }

    public void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var outData = DownloadBufferData(output); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * outData[i] * (1f - outData[i]);
        UploadToBuffer(o, gradInput);
    }

    public void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var outData = DownloadBufferData(output); var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = g[i] * (1f - outData[i] * outData[i]);
        UploadToBuffer(o, gradInput);
    }

    public void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        var g = DownloadBufferData(gradOutput); var inp = DownloadBufferData(input); var o = new float[size];
        for (int i = 0; i < size; i++)
        {
            float x = inp[i];
            float cdf = 0.5f * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            float pdf = 0.3989422804f * MathF.Exp(-0.5f * x * x);
            o[i] = g[i] * (cdf + x * pdf);
        }
        UploadToBuffer(o, gradInput);
    }

    public void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        EnsureInitialized();
        var g = DownloadBufferData(gradOutput); var outData = DownloadBufferData(output);
        var o = new float[batchSize * features];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * features;
            float dot = 0;
            for (int f = 0; f < features; f++) dot += g[off + f] * outData[off + f];
            for (int f = 0; f < features; f++) o[off + f] = outData[off + f] * (g[off + f] - dot);
        }
        UploadToBuffer(o, gradInput);
    }

    #endregion

    #region Loss Functions

    private float CpuLossReduce(IGpuBuffer predictions, IGpuBuffer targets, int size, Func<float, float, float> lossPerElement)
    {
        EnsureInitialized();
        var p = DownloadBufferData(predictions); var t = DownloadBufferData(targets);
        float total = 0;
        for (int i = 0; i < size; i++) total += lossPerElement(p[i], t[i]);
        return total / size;
    }

    private void CpuLossBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, Func<float, float, float> gradPerElement)
    {
        EnsureInitialized();
        var p = DownloadBufferData(predictions); var t = DownloadBufferData(targets);
        var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = gradPerElement(p[i], t[i]) / size;
        UploadToBuffer(o, gradInput);
    }

    public float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        EnsureInitialized();
        var p = DownloadBufferData(predictions); var t = DownloadBufferData(targets);
        float total = 0;
        for (int b = 0; b < batchSize; b++)
        {
            int target = BitConverter.SingleToInt32Bits(t[b]);
            float maxVal = float.MinValue;
            for (int c = 0; c < numClasses; c++) maxVal = MathF.Max(maxVal, p[b * numClasses + c]);
            float sum = 0;
            for (int c = 0; c < numClasses; c++) sum += MathF.Exp(p[b * numClasses + c] - maxVal);
            total += -(p[b * numClasses + target] - maxVal - MathF.Log(sum));
        }
        return total / batchSize;
    }

    public void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
    {
        EnsureInitialized();
        var p = DownloadBufferData(predictions); var t = DownloadBufferData(targets);
        var o = new float[batchSize * numClasses];
        for (int b = 0; b < batchSize; b++)
        {
            int target = BitConverter.SingleToInt32Bits(t[b]);
            float maxVal = float.MinValue;
            for (int c = 0; c < numClasses; c++) maxVal = MathF.Max(maxVal, p[b * numClasses + c]);
            float sum = 0;
            for (int c = 0; c < numClasses; c++) sum += MathF.Exp(p[b * numClasses + c] - maxVal);
            for (int c = 0; c < numClasses; c++) o[b * numClasses + c] = (MathF.Exp(p[b * numClasses + c] - maxVal) / sum - (c == target ? 1f : 0f)) / batchSize;
        }
        UploadToBuffer(o, gradInput);
    }

    public float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { p = Math.Clamp(p, 1e-7f, 1f - 1e-7f); return -(t * MathF.Log(p) + (1 - t) * MathF.Log(1 - p)); });

    public void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { p = Math.Clamp(p, 1e-7f, 1f - 1e-7f); return (p - t) / (p * (1 - p)); });

    public float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = p - t; return d * d; });

    public void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => 2f * (p - t));

    public float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = MathF.Abs(p - t); return d < beta ? 0.5f * d * d / beta : d - 0.5f * beta; });

    public void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { float d = p - t; return MathF.Abs(d) < beta ? d / beta : MathF.Sign(d); });

    public float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
    {
        EnsureInitialized();
        var a = DownloadBufferData(anchor); var p = DownloadBufferData(positive); var n = DownloadBufferData(negative);
        float total = 0;
        for (int b = 0; b < batchSize; b++)
        {
            float dp = 0, dn = 0;
            for (int f = 0; f < embeddingDim; f++) { float da = a[b * embeddingDim + f] - p[b * embeddingDim + f]; dp += da * da; da = a[b * embeddingDim + f] - n[b * embeddingDim + f]; dn += da * da; }
            total += MathF.Max(0, MathF.Sqrt(dp) - MathF.Sqrt(dn) + margin);
        }
        return total / batchSize;
    }

    public void TripletLossBackward(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative,
        IGpuBuffer gradAnchor, IGpuBuffer gradPositive, IGpuBuffer gradNegative,
        int batchSize, int embeddingDim, float margin)
    {
        int total = batchSize * embeddingDim;
        Fill(gradAnchor, 0f, total); Fill(gradPositive, 0f, total); Fill(gradNegative, 0f, total);
    }

    public float HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float delta) =>
        SmoothL1Loss(predictions, targets, size, delta);

    public void HuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float delta) =>
        SmoothL1Backward(predictions, targets, gradInput, size, delta);

    public float FocalLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float alpha, float gamma) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { p = Math.Clamp(p, 1e-7f, 1f - 1e-7f); float pt = t > 0.5f ? p : 1 - p; return -alpha * MathF.Pow(1 - pt, gamma) * MathF.Log(pt); });

    public void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) =>
        {
            p = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
            float pt = t > 0.5f ? p : 1 - p;
            float sign = t > 0.5f ? 1f : -1f;
            return sign * alpha * MathF.Pow(1 - pt, gamma) * (gamma * MathF.Log(pt) / (1 - pt + 1e-7f) - 1f / (pt + 1e-7f));
        });

    public float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Abs(p - t));

    public void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => p > t ? 1f : p < t ? -1f : 0);

    public float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Log(MathF.Cosh(p - t)));

    public void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => MathF.Tanh(p - t));

    public float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = t - p; return d >= 0 ? quantile * d : (quantile - 1) * d; });

    public void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => t - p >= 0 ? -quantile : 1 - quantile);

    public float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Max(0, 1f - t * p));

    public void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => t * p < 1f ? -t : 0);

    public float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float h = MathF.Max(0, 1f - t * p); return h * h; });

    public void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { float h = MathF.Max(0, 1f - t * p); return h > 0 ? -2f * h * t : 0; });

    public float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Exp(p) - t * p);

    public void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => MathF.Exp(p) - t);

    public float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Exp(-t * p));

    public void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => -t * MathF.Exp(-t * p));

    public float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float y = t * p; return y >= -1 ? MathF.Max(0, 1 - y) * MathF.Max(0, 1 - y) : -4 * y; });

    public void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { float y = t * p; return y >= -1 ? (y < 1 ? -2f * (1 - y) * t : 0) : -4f * t; });

    public float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { p = Math.Clamp(p, 1e-7f, 1f); return -t * MathF.Log(p); });

    public void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { p = Math.Clamp(p, 1e-7f, 1f); return -t / p; });

    public float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = p - t; return MathF.Sqrt(d * d + epsilon * epsilon); });

    public void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { float d = p - t; return d / MathF.Sqrt(d * d + epsilon * epsilon); });

    public float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = p - t; return l1Weight * MathF.Abs(d) + l2Weight * d * d; });

    public void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight) =>
        CpuLossBackward(predictions, targets, gradInput, size, (p, t) => { float d = p - t; return l1Weight * MathF.Sign(d) + 2f * l2Weight * d; });

    public float ContrastiveLoss(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
    {
        EnsureInitialized();
        var a = DownloadBufferData(output1); var b = DownloadBufferData(output2); var lbl = DownloadBufferData(labels);
        float total = 0;
        for (int i = 0; i < batchSize; i++)
        {
            float dist = 0;
            for (int f = 0; f < embeddingDim; f++) { float d = a[i * embeddingDim + f] - b[i * embeddingDim + f]; dist += d * d; }
            dist = MathF.Sqrt(dist);
            float label = lbl[i];
            total += label > 0.5f ? dist * dist : MathF.Max(0, margin - dist) * MathF.Max(0, margin - dist);
        }
        return total / batchSize;
    }

    public void ContrastiveBackward(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels,
        IGpuBuffer gradOutput1, IGpuBuffer gradOutput2,
        int batchSize, int embeddingDim, float margin)
    {
        int total = batchSize * embeddingDim;
        Fill(gradOutput1, 0f, total); Fill(gradOutput2, 0f, total);
    }

    #endregion

    #region Gradient Clipping and Utility

    public void Clamp(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size) => CpuUnary(A, B, size, v => MathF.Max(minVal, MathF.Min(maxVal, v)));

    public float L2Norm(IGpuBuffer A, int size)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A);
        float sum = 0;
        for (int i = 0; i < size; i++) sum += a[i] * a[i];
        return MathF.Sqrt(sum);
    }

    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size) => Clamp(A, B, -clipValue, clipValue, size);

    public void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A);
        float norm = 0;
        for (int i = 0; i < size; i++) norm += a[i] * a[i];
        norm = MathF.Sqrt(norm);
        var o = new float[size];
        if (norm > maxNorm) { float scale = maxNorm / norm; for (int i = 0; i < size; i++) o[i] = a[i] * scale; }
        else Array.Copy(a, o, size);
        UploadToBuffer(o, B);
    }

    public void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A); var b = DownloadBufferData(B); var c = DownloadBufferData(C);
        var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = a[i] * b[i] + c[i];
        UploadToBuffer(o, D);
    }

    public void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
    {
        EnsureInitialized();
        var src = DownloadBufferData(source); var idx = DownloadBufferData(indices);
        var dst = DownloadBufferData(destination);
        for (int i = 0; i < sourceSize; i++)
        {
            int dstIdx = BitConverter.SingleToInt32Bits(idx[i]);
            if (dstIdx >= 0 && dstIdx < destSize) dst[dstIdx] += src[i];
        }
        UploadToBuffer(dst, destination);
    }

    public void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource, int numIndices, int featureSize)
    {
        EnsureInitialized();
        var gd = DownloadBufferData(gradDestination); var idx = DownloadBufferData(indices);
        var gs = new float[numIndices * featureSize];
        for (int i = 0; i < numIndices; i++)
        {
            int srcIdx = BitConverter.SingleToInt32Bits(idx[i]);
            for (int f = 0; f < featureSize; f++) gs[i * featureSize + f] = gd[srcIdx * featureSize + f];
        }
        UploadToBuffer(gs, gradSource);
    }

    public void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        EnsureInitialized();
        var src = DownloadBufferData(source); var idx = DownloadBufferData(indices);
        var o = new float[numIndices * featureSize];
        for (int i = 0; i < numIndices; i++)
        {
            int srcIdx = BitConverter.SingleToInt32Bits(idx[i]);
            for (int f = 0; f < featureSize; f++) o[i * featureSize + f] = src[srcIdx * featureSize + f];
        }
        UploadToBuffer(o, output);
    }

    #endregion

    #region Comparison Operations

    public void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => CpuBinary(A, B, C, size, (a, b) => a > b ? 1f : 0f);
    public void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => CpuBinary(A, B, C, size, (a, b) => a < b ? 1f : 0f);
    public void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => CpuBinary(A, B, C, size, (a, b) => MathF.Abs(a - b) < 1e-6f ? 1f : 0f);

    public void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        EnsureInitialized();
        var cond = DownloadBufferData(condition); var a = DownloadBufferData(A); var b = DownloadBufferData(B);
        var o = new float[size];
        for (int i = 0; i < size; i++) o[i] = cond[i] != 0 ? a[i] : b[i];
        UploadToBuffer(o, C);
    }

    public void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size) => CpuUnary(A, C, size, v => MathF.Abs(v - scalar) > 1e-6f ? 1f : 0f);

    #endregion

    #region Statistics Operations

    public void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        SumAxis(A, B, outerSize, reduceSize);
        var o = DownloadBufferData(B);
        for (int i = 0; i < outerSize; i++) o[i] /= reduceSize;
        UploadToBuffer(o, B);
    }

    public void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(A);
        var m = new float[outerSize]; var v = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float sum = 0;
            for (int j = 0; j < reduceSize; j++) sum += inp[i * reduceSize + j];
            m[i] = sum / reduceSize;
            float var_ = 0;
            for (int j = 0; j < reduceSize; j++) { float d = inp[i * reduceSize + j] - m[i]; var_ += d * d; }
            v[i] = var_ / reduceSize;
        }
        UploadToBuffer(m, mean); UploadToBuffer(v, variance);
    }

    public void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(A); var o = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float maxVal = float.MinValue; int maxJ = 0;
            for (int j = 0; j < reduceSize; j++) { if (inp[i * reduceSize + j] > maxVal) { maxVal = inp[i * reduceSize + j]; maxJ = j; } }
            o[i] = BitConverter.Int32BitsToSingle(maxJ);
        }
        UploadToBuffer(o, indices);
    }

    public void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        => ArgMax(A, indices, outerSize, reduceSize);

    public void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(A); var o = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float minVal = float.MaxValue; int minJ = 0;
            for (int j = 0; j < reduceSize; j++) { if (inp[i * reduceSize + j] < minVal) { minVal = inp[i * reduceSize + j]; minJ = j; } }
            o[i] = BitConverter.Int32BitsToSingle(minJ);
        }
        UploadToBuffer(o, indices);
    }

    public void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(A); var o = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float maxVal = float.MinValue;
            for (int j = 0; j < reduceSize; j++) maxVal = MathF.Max(maxVal, inp[i * reduceSize + j]);
            o[i] = maxVal;
        }
        UploadToBuffer(o, B);
    }

    public void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(A);
        var sv = new float[outerSize * k]; var si = new float[outerSize * k];
        for (int b = 0; b < outerSize; b++)
        {
            var pairs = new (float val, int idx)[reduceSize];
            for (int c = 0; c < reduceSize; c++) pairs[c] = (inp[b * reduceSize + c], c);
            Array.Sort(pairs, (a, x) => x.val.CompareTo(a.val));
            for (int j = 0; j < k && j < reduceSize; j++)
            {
                int sOff = b * k;
                sv[sOff + j] = pairs[j].val;
                si[sOff + j] = BitConverter.Int32BitsToSingle(pairs[j].idx);
            }
        }
        UploadToBuffer(sv, values); UploadToBuffer(si, indices);
    }

    public void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A); var b = DownloadBufferData(B);
        var o = new float[outerSize * innerSize];
        for (int i = 0; i < outerSize; i++)
            for (int j = 0; j < innerSize; j++) o[i * innerSize + j] = a[i * innerSize + j] * b[j];
        UploadToBuffer(o, C);
    }

    public void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A); var b = DownloadBufferData(B);
        var o = new float[outerSize * innerSize];
        for (int i = 0; i < outerSize; i++)
            for (int j = 0; j < innerSize; j++) o[i * innerSize + j] = a[i * innerSize + j] * b[i];
        UploadToBuffer(o, C);
    }

    #endregion
}
#endif
