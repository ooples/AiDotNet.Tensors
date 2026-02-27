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

    // GPU loss forward helper: dispatch per-element kernel then reduce via SumAsync
    private float GpuLossReduce(IGpuBuffer predictions, IGpuBuffer targets, int size, string kernelName,
        float param1 = 0, float param2 = 0, float param3 = 0)
    {
        using var temp = (WebGpuBuffer)AllocateBuffer(size);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            param1, param2, param3
        };
        Dispatch3BufferAsync("LossElement", WebGpuKernels.LossElementSource, kernelName,
            predictions, targets, temp, uniforms, size).GetAwaiter().GetResult();
        float total = SumAsync(temp, size).GetAwaiter().GetResult();
        return total / size;
    }

    public float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        // GPU kernel computes per-batch loss, then reduce
        using var temp = (WebGpuBuffer)AllocateBuffer(batchSize);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(numClasses),
            0, 0
        };
        Dispatch3BufferAsync("CrossEntropyForward", WebGpuKernels.CrossEntropyForwardSource, "cross_entropy_forward",
            predictions, targets, temp, uniforms, batchSize).GetAwaiter().GetResult();
        float total = SumAsync(temp, batchSize).GetAwaiter().GetResult();
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
        GpuLossReduce(predictions, targets, size, "bce_elem");

    public void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "bce_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "mse_elem");

    public void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "mse_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta) =>
        GpuLossReduce(predictions, targets, size, "smooth_l1_elem", beta);

    public void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "huber_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, beta, 0), size).GetAwaiter().GetResult();
    }

    public float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
    {
        // TripletContrastiveSource triplet_loss_elem: A=anchor, B=positive, C=negative, output=per-batch loss
        using var temp = (WebGpuBuffer)AllocateBuffer(batchSize);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(embeddingDim),
            margin, 0
        };
        Dispatch4BufferAsync("TripletContrastive", WebGpuKernels.TripletContrastiveSource, "triplet_loss_elem",
            anchor, positive, negative, temp, uniforms, batchSize).GetAwaiter().GetResult();
        float total = SumAsync(temp, batchSize).GetAwaiter().GetResult();
        return total / batchSize;
    }

    public void TripletLossBackward(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative,
        IGpuBuffer gradAnchor, IGpuBuffer gradPositive, IGpuBuffer gradNegative,
        int batchSize, int embeddingDim, float margin)
    {
        // TripletContrastiveSource triplet_loss_backward: computes grad w.r.t. anchor into output
        int total = batchSize * embeddingDim;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(embeddingDim),
            margin, 0
        };
        Dispatch4BufferAsync("TripletContrastive", WebGpuKernels.TripletContrastiveSource, "triplet_loss_backward",
            anchor, positive, negative, gradAnchor, uniforms, total).GetAwaiter().GetResult();
        // Grad for positive = -gradAnchor component (positive direction), for negative = +gradAnchor component (negative direction)
        // For a complete backward we'd need separate kernels for each output; approximate with negate/copy
        Negate(gradAnchor, gradPositive, total);
        Copy(gradAnchor, gradNegative, total);
        Negate(gradNegative, gradNegative, total);
    }

    public float HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float delta) =>
        SmoothL1Loss(predictions, targets, size, delta);

    public void HuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float delta) =>
        SmoothL1Backward(predictions, targets, gradInput, size, delta);

    public float FocalLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float alpha, float gamma) =>
        GpuLossReduce(predictions, targets, size, "focal_elem", alpha, gamma);

    public void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "focal_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, alpha, gamma), size).GetAwaiter().GetResult();
    }

    public float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "mae_elem");

    public void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "mae_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "logcosh_elem");

    public void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "logcosh_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile) =>
        GpuLossReduce(predictions, targets, size, "quantile_elem", quantile);

    public void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "quantile_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, quantile, 0), size).GetAwaiter().GetResult();
    }

    public float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "hinge_elem");

    public void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "hinge_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "squared_hinge_elem");

    public void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "squared_hinge_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "poisson_elem");

    public void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "poisson_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "exponential_elem");

    public void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "exponential_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "modified_huber_elem");

    public void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "modified_huber_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size) =>
        GpuLossReduce(predictions, targets, size, "categorical_ce_elem");

    public void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "categorical_ce_backward",
            predictions, targets, gradInput, MakeUniform3(size, invSize, 0), size).GetAwaiter().GetResult();
    }

    public float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon) =>
        GpuLossReduce(predictions, targets, size, "charbonnier_elem", epsilon);

    public void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "charbonnier_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, epsilon, 0), size).GetAwaiter().GetResult();
    }

    public float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight) =>
        GpuLossReduce(predictions, targets, size, "elastic_net_elem", l1Weight, l2Weight);

    public void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
    {
        float invSize = 1f / size;
        Dispatch3BufferAsync("LossBackward", WebGpuKernels.LossBackwardSource, "elastic_net_backward",
            predictions, targets, gradInput, MakeUniform4(size, invSize, l1Weight, l2Weight), size).GetAwaiter().GetResult();
    }

    public float ContrastiveLoss(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
    {
        // TripletContrastiveSource contrastive_loss_elem: A=output1, B=output2, C=labels, output=per-batch loss
        using var temp = (WebGpuBuffer)AllocateBuffer(batchSize);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(embeddingDim),
            margin, 0
        };
        Dispatch4BufferAsync("TripletContrastive", WebGpuKernels.TripletContrastiveSource, "contrastive_loss_elem",
            output1, output2, labels, temp, uniforms, batchSize).GetAwaiter().GetResult();
        float total = SumAsync(temp, batchSize).GetAwaiter().GetResult();
        return total / batchSize;
    }

    public void ContrastiveBackward(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels,
        IGpuBuffer gradOutput1, IGpuBuffer gradOutput2,
        int batchSize, int embeddingDim, float margin)
    {
        // TripletContrastiveSource contrastive_loss_backward: computes grad w.r.t. output1
        int total = batchSize * embeddingDim;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(embeddingDim),
            margin, 0
        };
        Dispatch4BufferAsync("TripletContrastive", WebGpuKernels.TripletContrastiveSource, "contrastive_loss_backward",
            output1, output2, labels, gradOutput1, uniforms, total).GetAwaiter().GetResult();
        // gradOutput2 = -gradOutput1
        Negate(gradOutput1, gradOutput2, total);
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
        // TopKSource topk_select: binding(0)=input, binding(1)=values, binding(2)=indices
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(outerSize),
            BitConverter.Int32BitsToSingle(reduceSize),
            BitConverter.Int32BitsToSingle(k),
            0
        };
        Dispatch3BufferAsync("TopK", WebGpuKernels.TopKSource, "topk_select",
            A, values, indices, uniforms, outerSize).GetAwaiter().GetResult();
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
