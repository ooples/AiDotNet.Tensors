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
        // AttentionSource: binding(0)=query, binding(1)=key, binding(2)=value, binding(3)=output, binding(4)=uniform
        // Uniform: batch_heads, seq_len, head_dim, pad
        int batchHeads = batch * numHeads;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchHeads),
            BitConverter.Int32BitsToSingle(seqLen),
            BitConverter.Int32BitsToSingle(headDim),
            0
        };
        int totalElements = batchHeads * seqLen * headDim;
        Dispatch4BufferAsync("Attention", WebGpuKernels.AttentionSource, "scaled_dot_product_attention",
            query, key, value, output, uniforms, totalElements).GetAwaiter().GetResult();
        // attentionWeights not produced by this kernel; fill with zeros if requested
        if (attentionWeights is not null)
            Fill(attentionWeights, 0f, batchHeads * seqLen * seqLen);
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

    public void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
        => LeakyReLUAsync(A, B, size, alpha).GetAwaiter().GetResult();

    public void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "leaky_relu_backward",
            gradOutput, input, gradInput, MakeUniform3(size, alpha, 0), size).GetAwaiter().GetResult();
    }

    public void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
        => ELUAsync(A, B, size, alpha).GetAwaiter().GetResult();

    public void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "elu_backward",
            gradOutput, input, gradInput, MakeUniform3(size, alpha, 0), size).GetAwaiter().GetResult();
    }

    public void Swish(IGpuBuffer A, IGpuBuffer B, int size) => SwishAsync(A, B, size).GetAwaiter().GetResult();

    public void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "swish_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void Silu(IGpuBuffer A, IGpuBuffer B, int size) => Swish(A, B, size);

    public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size) => SwishBackward(gradOutput, input, gradInput, size);

    public void Mish(IGpuBuffer A, IGpuBuffer B, int size) => MishAsync(A, B, size).GetAwaiter().GetResult();

    public void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "mish_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size) => SoftplusAsync(A, B, size).GetAwaiter().GetResult();

    public void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "softplus_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
        => DispatchActivationAsync("hardswish", A, B, size, 0).GetAwaiter().GetResult();

    public void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "hardswish_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
        => DispatchActivationAsync("selu", A, B, size, alpha).GetAwaiter().GetResult();

    public void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "selu_backward",
            gradOutput, input, gradInput, MakeUniform3(size, alpha, scale), size).GetAwaiter().GetResult();
    }

    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
        => DispatchActivationAsync("hardsigmoid", A, B, size, 0).GetAwaiter().GetResult();

    public void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "hardsigmoid_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
    {
        Dispatch2BufferAsync("Clamp", WebGpuKernels.ClampSource, "clamp_op",
            A, B, MakeUniform3(size, minVal, maxVal), size).GetAwaiter().GetResult();
    }

    public void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "hardtanh_backward",
            gradOutput, input, gradInput, MakeUniform3(size, minVal, maxVal), size).GetAwaiter().GetResult();
    }

    public void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "relu_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "sigmoid_backward",
            gradOutput, output, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "tanh_backward",
            gradOutput, output, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "gelu_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        Dispatch3BufferAsync("SoftmaxBackward", WebGpuKernels.SoftmaxBackwardSource, "softmax_backward",
            gradOutput, output, gradInput, MakeUniformInts2(batchSize, features), batchSize).GetAwaiter().GetResult();
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
        // Convert integer-encoded targets to one-hot for GPU kernel
        EnsureInitialized();
        var t = DownloadBufferData(targets);
        var oneHot = new float[batchSize * numClasses];
        for (int b = 0; b < batchSize; b++)
        {
            int target = BitConverter.SingleToInt32Bits(t[b]);
            if (target >= 0 && target < numClasses)
                oneHot[b * numClasses + target] = 1f;
        }
        using var oneHotBuf = (WebGpuBuffer)AllocateBuffer(batchSize * numClasses);
        UploadToBuffer(oneHot, oneHotBuf);
        // CrossEntropySource cross_entropy_backward: softmax(predictions) - targets (one-hot)
        var uniforms = MakeUniformInts2(batchSize, numClasses);
        int total = batchSize * numClasses;
        Dispatch3BufferAsync("CrossEntropy", WebGpuKernels.CrossEntropySource, "cross_entropy_backward",
            predictions, oneHotBuf, gradInput, uniforms, total).GetAwaiter().GetResult();
    }

    public float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { p = Math.Clamp(p, 1e-7f, 1f - 1e-7f); return -(t * MathF.Log(p) + (1 - t) * MathF.Log(1 - p)); });

    public void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "bce_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = p - t; return d * d; });

    public void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "mse_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = MathF.Abs(p - t); return d < beta ? 0.5f * d * d / beta : d - 0.5f * beta; });

    public void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "huber_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, beta, 0), size).GetAwaiter().GetResult();
    }

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

    public void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "focal_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, alpha, gamma), size).GetAwaiter().GetResult();
    }

    public float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Abs(p - t));

    public void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "mae_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Log(MathF.Cosh(p - t)));

    public void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "logcosh_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = t - p; return d >= 0 ? quantile * d : (quantile - 1) * d; });

    public void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "quantile_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, quantile, 0), size).GetAwaiter().GetResult();
    }

    public float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Max(0, 1f - t * p));

    public void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "hinge_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float h = MathF.Max(0, 1f - t * p); return h * h; });

    public void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "squared_hinge_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Exp(p) - t * p);

    public void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "poisson_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => MathF.Exp(-t * p));

    public void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "exponential_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float y = t * p; return y >= -1 ? MathF.Max(0, 1 - y) * MathF.Max(0, 1 - y) : -4 * y; });

    public void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "modified_huber_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { p = Math.Clamp(p, 1e-7f, 1f); return -t * MathF.Log(p); });

    public void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "categorical_ce_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = p - t; return MathF.Sqrt(d * d + epsilon * epsilon); });

    public void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "charbonnier_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, epsilon, 0), size).GetAwaiter().GetResult();
    }

    public float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight) =>
        CpuLossReduce(predictions, targets, size, (p, t) => { float d = p - t; return l1Weight * MathF.Abs(d) + l2Weight * d * d; });

    public void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "elastic_net_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, l1Weight, l2Weight), size).GetAwaiter().GetResult();
    }

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

    public void Clamp(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
    {
        Dispatch2BufferAsync("Clamp", WebGpuKernels.ClampSource, "clamp_op",
            A, B, MakeUniform3(size, minVal, maxVal), size).GetAwaiter().GetResult();
    }

    public float L2Norm(IGpuBuffer A, int size)
    {
        // Use GPU square + sum reduction
        using var squared = (WebGpuBuffer)AllocateBuffer(size);
        SquareAsync(A, squared, size).GetAwaiter().GetResult();
        float sumSq = SumAsync(squared, size).GetAwaiter().GetResult();
        return MathF.Sqrt(sumSq);
    }

    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
    {
        Dispatch2BufferAsync("GradientClip", WebGpuKernels.GradientClipSource, "clip_by_value",
            A, B, MakeUniform2(size, clipValue), size).GetAwaiter().GetResult();
    }

    public void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
    {
        float norm = L2Norm(A, size);
        float scale = norm > maxNorm ? maxNorm / norm : 1f;
        Dispatch2BufferAsync("GradientClip", WebGpuKernels.GradientClipSource, "clip_by_norm",
            A, B, MakeUniform3(size, maxNorm, scale), size).GetAwaiter().GetResult();
    }

    public void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
    {
        Dispatch4BufferAsync("Fma", WebGpuKernels.FmaSource, "fma_op",
            A, B, C, D, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
    {
        // ScatterGatherExtSource scatter_add: gather pattern - for each dest element, scan source for matching indices
        // binding(0)=source, binding(1)=indices, binding(2)=destination
        // Uniform: num_indices=sourceSize, inner_size=1 (flat), dest_outer=destSize, pad
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(sourceSize),
            BitConverter.Int32BitsToSingle(1),
            BitConverter.Int32BitsToSingle(destSize),
            0
        };
        Dispatch3BufferAsync("ScatterGatherExt", WebGpuKernels.ScatterGatherExtSource, "scatter_add",
            source, indices, destination, uniforms, destSize).GetAwaiter().GetResult();
    }

    public void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource, int numIndices, int featureSize)
    {
        // ScatterAddBackward is equivalent to Gather
        Gather(gradDestination, indices, gradSource, numIndices, featureSize);
    }

    public void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        // ScatterGatherExtSource gather_op: binding(0)=source, binding(1)=indices, binding(2)=output(destination)
        // Uniform: num_indices, inner_size=featureSize, dest_outer(unused), pad
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(numIndices),
            BitConverter.Int32BitsToSingle(featureSize),
            0,
            0
        };
        int totalElements = numIndices * featureSize;
        Dispatch3BufferAsync("ScatterGatherExt", WebGpuKernels.ScatterGatherExtSource, "gather_op",
            source, indices, output, uniforms, totalElements).GetAwaiter().GetResult();
    }

    #endregion

    #region Comparison Operations

    public void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        Dispatch3BufferAsync("Comparison", WebGpuKernels.ComparisonSource, "greater_than",
            A, B, C, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        Dispatch3BufferAsync("Comparison", WebGpuKernels.ComparisonSource, "less_than",
            A, B, C, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        Dispatch3BufferAsync("Comparison", WebGpuKernels.ComparisonSource, "equal_to",
            A, B, C, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        // ConditionalSource where_op: A=condition, B=trueVals, C gets falseVals first then selected
        // First copy B (false values) to C, then the kernel writes: C[i] = select(C[i], B[i], A[i]>0.5)
        // Wait - the kernel has bindings: A=condition at binding(0), B=trueVals at binding(1), C at binding(2)
        // So we need to copy the false values (B param) into C first, then dispatch with condition and trueVals (A param)
        Copy(B, C, size);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            0 // scalar not used for where_op
        };
        Dispatch3BufferAsync("Conditional", WebGpuKernels.ConditionalSource, "where_op",
            condition, A, C, uniforms, size).GetAwaiter().GetResult();
    }

    public void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
    {
        // ConditionalSource not_equal_scalar: A at binding(0), C at binding(2)
        using var dummyB = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            scalar
        };
        Dispatch3BufferAsync("Conditional", WebGpuKernels.ConditionalSource, "not_equal_scalar",
            A, dummyB, C, uniforms, size).GetAwaiter().GetResult();
    }

    #endregion

    #region Statistics Operations

    public void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        Dispatch2BufferAsync("Statistics", WebGpuKernels.StatisticsSource, "mean_axis",
            A, B, MakeUniformInts2(outerSize, reduceSize), outerSize).GetAwaiter().GetResult();
    }

    public void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        // First compute mean using existing GPU kernel
        MeanAxis(A, mean, outerSize, reduceSize);
        // Then compute variance using VarAxisSource
        var uniforms = MakeUniformInts2(outerSize, reduceSize);
        Dispatch2BufferAsync("VarAxis", WebGpuKernels.VarAxisSource, "var_axis",
            A, variance, uniforms, outerSize).GetAwaiter().GetResult();
    }

    public void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        Dispatch2BufferAsync("Statistics", WebGpuKernels.StatisticsSource, "argmax_axis",
            A, indices, MakeUniformInts2(outerSize, reduceSize), outerSize).GetAwaiter().GetResult();
    }

    public void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        => ArgMax(A, indices, outerSize, reduceSize);

    public void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        Dispatch2BufferAsync("Statistics", WebGpuKernels.StatisticsSource, "argmin_axis",
            A, indices, MakeUniformInts2(outerSize, reduceSize), outerSize).GetAwaiter().GetResult();
    }

    public void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        Dispatch2BufferAsync("Statistics", WebGpuKernels.StatisticsSource, "max_axis",
            A, B, MakeUniformInts2(outerSize, reduceSize), outerSize).GetAwaiter().GetResult();
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
        int total = outerSize * innerSize;
        Dispatch3BufferAsync("Broadcast", WebGpuKernels.BroadcastSource, "broadcast_mul_last",
            A, B, C, MakeUniformInts2(outerSize, innerSize), total).GetAwaiter().GetResult();
    }

    public void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        int total = outerSize * innerSize;
        Dispatch3BufferAsync("Broadcast", WebGpuKernels.BroadcastSource, "broadcast_mul_first",
            A, B, C, MakeUniformInts2(outerSize, innerSize), total).GetAwaiter().GetResult();
    }

    #endregion
}
#endif
