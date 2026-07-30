// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 3: Attention, Activations, Loss, Optimizers, FFT, RNN.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    /// <summary>Reinterpret float bits as uint32, compatible with net471 (no BitConverter.SingleToUInt32Bits).</summary>
    private static uint FloatToUInt32(float value) => *(uint*)&value;

    #region Attention Operations

    public void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal, float softcap = 0.0f,
        int numKVHeads = 0)
        // numKVHeads <= 0 means MHA; >0 enables Grouped-Query Attention (the core already broadcasts shared K/V heads).
        => AttentionForwardCore(query, key, value, output, attentionWeights, mask,
            batch, numHeads, numKVHeads > 0 ? numKVHeads : numHeads, seqQ, seqK, headDim, scale, isCausal, softcap: softcap);

    public void RopeInterleaved(IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer output,
        int rows, int headDim, int seqLen, int startPosition)
    {
        if (rows <= 0 || headDim <= 0 || seqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "RoPE dimensions must be positive.");
        if ((headDim & 1) != 0)
            throw new ArgumentException("RoPE requires an even head dimension.", nameof(headDim));
        int total = checked(rows * headDim);
        if (input.Size < total || output.Size < total)
            throw new ArgumentException("RoPE input/output buffers are smaller than rows * headDim.");
        int pairs = checked(rows * (headDim / 2));
        GlslNaryOp(VulkanGlslKernels.RopeInterleaved,
            new[] { input, cos, sin, output }, pairs,
            new[] { (uint)rows, (uint)headDim, (uint)seqLen, (uint)startPosition });
    }

    private void AttentionForwardCore(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int queryHeads, int kvHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        int maskBatchStride = 0, float softcap = 0.0f)
    {
        if (queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0)
            throw new ArgumentException("queryHeads must be a positive multiple of kvHeads.");
        IGpuBuffer? dummyWeights = null;
        IGpuBuffer? dummyMask = null;
        try
        {
            IGpuBuffer weights = attentionWeights ?? (dummyWeights = AllocateBuffer(1));
            IGpuBuffer maskBuffer = mask ?? (dummyMask = AllocateBuffer(1));
            uint maskMode = mask is null ? 0u :
                mask.Size >= batch * queryHeads * seqQ * seqK ? 2u : 1u;
            uint effectiveMaskBatchStride = maskMode == 2u
                ? (uint)(maskBatchStride > 0 ? maskBatchStride : queryHeads * seqQ * seqK)
                : 0u;
            GlslNaryOp(VulkanGlslKernels.AttentionForward,
                new[] { query, key, value, output, weights, maskBuffer }, batch * queryHeads * seqQ,
                new[]
                {
                    (uint)batch, (uint)queryHeads, (uint)kvHeads, (uint)seqQ, (uint)seqK, (uint)headDim,
                    FloatBits(scale), isCausal ? 1u : 0u, attentionWeights is null ? 0u : 1u, maskMode,
                    effectiveMaskBatchStride, FloatBits(softcap)
                });
        }
        finally
        {
            dummyWeights?.Dispose();
            dummyMask?.Dispose();
        }
    }

    private void AttentionBackwardCore(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        => AttentionBackwardGeneral(gradOutput, query, key, value, attentionWeights,
            gradQuery, gradKey, gradValue, batch, numHeads, numHeads, seqLen, seqLen, headDim, scale, isCausal);

    private void AttentionBackwardGeneral(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int queryHeads, int kvHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        if (queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0)
            throw new ArgumentException("queryHeads must be a positive multiple of kvHeads.");
        var common = new uint[]
        {
            (uint)batch, (uint)queryHeads, (uint)kvHeads, (uint)seqQ, (uint)seqK, (uint)headDim,
            FloatBits(scale), isCausal ? 1u : 0u, 0
        };
        var buffers = new[] { gradOutput, query, key, value, attentionWeights, gradQuery };
        GlslNaryOp(VulkanGlslKernels.AttentionBackward, buffers, batch * queryHeads * seqQ * headDim, common);
        common[8] = 1;
        buffers[5] = gradKey;
        GlslNaryOp(VulkanGlslKernels.AttentionBackward, buffers, batch * kvHeads * seqK * headDim, common);
        common[8] = 2;
        buffers[5] = gradValue;
        GlslNaryOp(VulkanGlslKernels.AttentionBackward, buffers, batch * kvHeads * seqK * headDim, common);
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        AttentionBackwardGeneral(gradOutput, query, key, value, attentionWeights, gradQuery, gradKey, gradValue,
            batch, numHeads, numHeads, seqQ, seqK, headDim, scale, isCausal);
    }

    public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        => ScaledDotProductAttention(query, key, value, output, null, mask, batch, numHeads, seqLen, seqLen, headDim, scale, isCausal);

    public void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        if (batch <= 0 || numHeads <= 0 || seqQ <= 0 || seqK <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "FlashAttention dimensions must be positive.");
        AttentionForwardCore(query, key, value, output, null, attentionBias,
            batch, numHeads, numHeads, seqQ, seqK, headDim, scale, isCausal, biasBatchStride);
        IGpuBuffer? dummyBias = null;
        try
        {
            IGpuBuffer bias = attentionBias ?? (dummyBias = AllocateBuffer(1));
            int effectiveBiasStride = biasBatchStride > 0 ? biasBatchStride : numHeads * seqQ * seqK;
            GlslQuadOp(VulkanGlslKernels.AttentionStats, query, key, bias, softmaxStats,
                batch * numHeads * seqQ,
                new uint[]
                {
                    (uint)batch, (uint)numHeads, (uint)seqQ, (uint)seqK, (uint)headDim, FloatBits(scale),
                    isCausal ? 1u : 0u, attentionBias is null ? 0u : 1u, (uint)effectiveBiasStride
                }, 9 * sizeof(uint));
        }
        finally
        {
            dummyBias?.Dispose();
        }
    }

    public void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        using var attentionWeights = AllocateBuffer(checked(batch * numHeads * seqQ * seqK));
        AttentionForwardCore(query, key, value, output, attentionWeights, attentionBias,
            batch, numHeads, numHeads, seqQ, seqK, headDim, scale, isCausal, biasBatchStride);
        AttentionBackwardGeneral(gradOutput, query, key, value, attentionWeights, gradQuery, gradKey, gradValue,
            batch, numHeads, numHeads, seqQ, seqK, headDim, scale, isCausal);
    }

    public void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        AttentionForwardCore(query, key, value, output, attentionWeights, null,
            batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scale, isCausal);
    }

    public void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale,
        int numQueriesPerKV)
    {
        if (numQueriesPerKV <= 0)
            throw new ArgumentOutOfRangeException(nameof(numQueriesPerKV), numQueriesPerKV, "numQueriesPerKV must be positive.");
        if (numQHeads <= 0 || numKVHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numQHeads), "GQA head counts must be positive.");
        // True GQA (numQHeads != numKVHeads) needs a KV-head-aware backward (kvh = qh / numQueriesPerKV)
        // that accumulates gradKey/gradValue across the query heads sharing each KV head. The Vulkan backend
        // has no such kernel yet (its attention backward is the host reference AttentionBackwardCore). The
        // previous fallback ran that with numQHeads, which mis-indexed K/V and wrote gradKey/gradValue
        // (sized for numKVHeads) OUT OF BOUNDS — silently wrong gradients. Reject it loudly instead; standard
        // multi-head attention (equal head counts) is correct via the shared backward.
        if (numQueriesPerKV != numQHeads / numKVHeads)
            throw new ArgumentException("numQueriesPerKV does not match numQHeads / numKVHeads.", nameof(numQueriesPerKV));
        AttentionBackwardGeneral(gradOutput, query, key, value, attentionWeights, gradQuery, gradKey, gradValue,
            batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scale, false);
    }

    #endregion

    #region Activation Functions and Gradients

    public void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
        => ResidentUnary(ResidentUnaryOp.LeakyRelu, A, B, size, alpha);

    public void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
        => ResidentBinary(ResidentBinaryOp.LeakyReluBackward, gradOutput, input, gradInput, size, alpha);

    public void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
        => ResidentUnary(ResidentUnaryOp.Elu, A, B, size, alpha);

    public void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        GlslQuadOp(VulkanGlslKernels.EluBackward, gradOutput, input, output, gradInput, size,
            new uint[] { (uint)size, FloatBits(alpha) }, 2 * sizeof(uint));
    }

    public void Swish(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Swish, A, B, size);

    public void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.SwishBackward, gradOutput, input, gradInput, size);

    public void Silu(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Swish, A, B, size);

    public void Mish(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Mish, A, B, size);

    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.Softplus, A, B, size);

    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.HardSwish, A, B, size);

    public void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
        => ResidentUnary(ResidentUnaryOp.Selu, A, B, size, alpha, scale);

    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
        => ResidentUnary(ResidentUnaryOp.HardSigmoid, A, B, size);

    public void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
        => ResidentUnary(ResidentUnaryOp.Clamp, A, B, size, minVal, maxVal);

    public void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.ReluBackward, gradOutput, input, gradInput, size);

    public void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.SigmoidBackward, gradOutput, output, gradInput, size);

    public void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.TanhBackward, gradOutput, output, gradInput, size);

    public void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.GeluBackward, gradOutput, input, gradInput, size);

    public void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        GlslBinaryOp(VulkanGlslKernels.SoftmaxBackward, gradOutput, output, gradInput,
            batchSize, new uint[] { (uint)batchSize, (uint)features }, 2 * sizeof(uint));
    }

    public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => SwishBackward(gradOutput, input, gradInput, size);

    public void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.MishBackward, gradOutput, input, gradInput, size);

    public void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.SoftplusBackward, gradOutput, input, gradInput, size);

    public void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.HardSwishBackward, gradOutput, input, gradInput, size);

    public void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
        => ResidentBinary(ResidentBinaryOp.SeluBackward, gradOutput, input, gradInput, size, alpha, scale);

    public void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => ResidentBinary(ResidentBinaryOp.HardSigmoidBackward, gradOutput, input, gradInput, size);

    public void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
        => ResidentBinary(ResidentBinaryOp.HardTanhBackward, gradOutput, input, gradInput, size, minVal, maxVal);

    public void Relu6(IGpuBuffer A, IGpuBuffer B, int size)
        => GlslUnaryOp(VulkanGlslKernels.Relu6, A, B, size);

    public void Relu6Backward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => GlslBinaryOp(VulkanGlslKernels.Relu6Backward, gradOutput, input, gradInput, size);

    public void PRelu(IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer output, int size, int alphaSize)
    {
        var pushData = new uint[] { (uint)size, (uint)alphaSize };
        GlslBinaryOp(VulkanGlslKernels.PRelu, input, alpha, output, size, pushData, (uint)(2 * sizeof(uint)));
    }

    public void PReluBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer gradInput, int size, int alphaSize)
    {
        var pushData = new uint[] { (uint)size, (uint)alphaSize };
        GlslQuadOp(VulkanGlslKernels.PReluBackwardInput, gradOutput, input, alpha, gradInput, size, pushData, (uint)(2 * sizeof(uint)));
    }

    public void PReluBackwardAlpha(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradAlpha, int size, int alphaSize)
    {
        // Segmented reduction: one thread per alpha channel, loops over its segment
        var pushData = new uint[] { (uint)size, (uint)alphaSize };
        GlslBinaryOp(VulkanGlslKernels.PReluBackwardAlpha, gradOutput, input, gradAlpha, alphaSize, pushData, (uint)(2 * sizeof(uint)));
    }

    public void RRelu(IGpuBuffer input, IGpuBuffer noise, IGpuBuffer output, int size)
        => GlslBinaryOp(VulkanGlslKernels.RRelu, input, noise, output, size);

    public void RReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer noise, IGpuBuffer gradInput, int size)
        => GlslQuadOp(VulkanGlslKernels.RReluBackward, gradOutput, input, noise, gradInput, size);

    public void Threshold(IGpuBuffer input, IGpuBuffer output, float threshold, float value, int size)
    {
        var pushData = new uint[] { (uint)size, FloatToUInt32(threshold), FloatToUInt32(value) };
        GlslUnaryOp(VulkanGlslKernels.Threshold, input, output, size, pushData, (uint)(3 * sizeof(uint)));
    }

    public void ThresholdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float threshold, int size)
    {
        var pushData = new uint[] { (uint)size, FloatToUInt32(threshold) };
        GlslBinaryOp(VulkanGlslKernels.ThresholdBackward, gradOutput, input, gradInput, size, pushData, (uint)(2 * sizeof(uint)));
    }

    public void ReciprocalBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => GlslBinaryOp(VulkanGlslKernels.ReciprocalBackward, gradOutput, input, gradInput, size);

    public void VarBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => GlslQuadOp(VulkanGlslKernels.VarBackwardGlsl, gradOutput, input, mean, gradInput,
            outerSize * reduceSize, new uint[] { (uint)outerSize, (uint)reduceSize }, 2 * sizeof(uint));

    public void StdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer std, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => GlslQuintOp(VulkanGlslKernels.StdBackwardGlsl, gradOutput, input, mean, std, gradInput,
            outerSize * reduceSize, new uint[] { (uint)outerSize, (uint)reduceSize }, 2 * sizeof(uint));

    public void MaskedFillBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size)
        => GlslBinaryOp(VulkanGlslKernels.MaskedFillBackwardGlsl, gradOutput, mask, gradInput, size);

    public void WhereBackward(IGpuBuffer gradOutput, IGpuBuffer condition, IGpuBuffer gradX, IGpuBuffer gradY, int size)
        => GlslQuadOp(VulkanGlslKernels.WhereBackwardGlsl, gradOutput, condition, gradX, gradY, size);

    public void NormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer norm, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => GlslQuadOp(VulkanGlslKernels.NormBackwardGlsl, gradOutput, input, norm, gradInput,
            outerSize * reduceSize, new uint[] { (uint)outerSize, (uint)reduceSize }, 2 * sizeof(uint));

    public void LogSumExpBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer lse, IGpuBuffer gradInput, int outerSize, int reduceSize)
        => GlslQuadOp(VulkanGlslKernels.LogSumExpBackwardGlsl, gradOutput, input, lse, gradInput,
            outerSize * reduceSize, new uint[] { (uint)outerSize, (uint)reduceSize }, 2 * sizeof(uint));

    public void AvgPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        // Use binary op layout (input, output) with push constants for parameters
        GlslUnaryOp(VulkanGlslKernels.AvgPool1DGlsl, input, output, batch * channels * outLength,
            new uint[] { (uint)batch, (uint)channels, (uint)inLength, (uint)outLength, (uint)kernelSize, (uint)stride }, 6 * sizeof(uint));
    }

    public void MaxPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        GlslUnaryOp(VulkanGlslKernels.MaxPool1DGlsl, input, output, batch * channels * outLength,
            new uint[] { (uint)batch, (uint)channels, (uint)inLength, (uint)outLength, (uint)kernelSize, (uint)stride }, 6 * sizeof(uint));
    }

    public void BilinearUpsample2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW)
    {
        GlslUnaryOp(VulkanGlslKernels.BilinearUpsample2DGlsl, input, output, batch * channels * outH * outW,
            new uint[] { (uint)batch, (uint)channels, (uint)inH, (uint)inW, (uint)outH, (uint)outW }, 6 * sizeof(uint));
    }

    public void ScatterMean(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts, int sourceSize, int outputSize, int featureSize)
    {
        // Initialize output and counts to zero before accumulation
        Fill(output, 0f, outputSize * featureSize);
        Fill(counts, 0f, outputSize);
        // Two-pass: scatter-add then divide
        if (GpuDeterminism.IsActive)
        {
            // Issue #382: CAS-loop atomic adds are FP-non-deterministic; use the
            // bit-deterministic variant that scans source rows in fixed order
            // (one work-item per (dstRow, col)).
            int outputFlat = outputSize * featureSize;
            GlslQuadOp(VulkanGlslKernels.ScatterMeanDeterministicGlsl, source, indices, output, counts,
                outputFlat, new uint[] { (uint)sourceSize, (uint)outputSize, (uint)featureSize }, 3 * sizeof(uint));
        }
        else
        {
            GlslQuadOp(VulkanGlslKernels.ScatterMeanGlsl, source, indices, output, counts,
                sourceSize, new uint[] { (uint)sourceSize, (uint)featureSize }, 2 * sizeof(uint));
        }
        // Divide kernel guards `idx < outputSize` and indexes `a[idx]` (flat),
        // so the dispatch count and size limit must be the flat output element
        // count (outputSize * featureSize), not row count.
        int divideElems = outputSize * featureSize;
        GlslBinaryOp(VulkanGlslKernels.ScatterMeanDivideGlsl, output, counts, output,
            divideElems, new uint[] { (uint)divideElems, (uint)featureSize }, 2 * sizeof(uint));
    }

    public void L1Loss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
    {
        var pushData = new uint[] { (uint)batchSize, (uint)numFeatures };
        GlslBinaryOp(VulkanGlslKernels.L1LossBatch, predictions, targets, loss, batchSize, pushData, (uint)(2 * sizeof(uint)));
    }

    public void HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures, float delta)
    {
        var pushData = new uint[] { (uint)batchSize, (uint)numFeatures, FloatToUInt32(delta) };
        GlslBinaryOp(VulkanGlslKernels.HuberLossBatch, predictions, targets, loss, batchSize, pushData, (uint)(3 * sizeof(uint)));
    }

    public void BceWithLogitsLoss(IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer loss, int size)
        => GlslBinaryOp(VulkanGlslKernels.BceWithLogitsElementwise, logits, targets, loss, size);

    public void NllLoss(IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
    {
        var pushData = new uint[] { (uint)batchSize, (uint)numClasses };
        GlslBinaryOp(VulkanGlslKernels.NllLossBatch, logProbs, targets, loss, batchSize, pushData, (uint)(2 * sizeof(uint)));
    }

    public void KlDivLoss(IGpuBuffer input, IGpuBuffer target, IGpuBuffer loss, int size)
        => GlslBinaryOp(VulkanGlslKernels.KlDivElementwise, input, target, loss, size);

    public void MseLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        var pushData = new uint[] { (uint)size, FloatToUInt32(invN) };
        GlslQuadOp(VulkanGlslKernels.MseLossBackward, gradOutput, predictions, targets, gradInput, size, pushData, (uint)(2 * sizeof(uint)));
    }

    public void L1LossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        var pushData = new uint[] { (uint)size, FloatToUInt32(invN) };
        GlslQuadOp(VulkanGlslKernels.L1LossBackward, gradOutput, predictions, targets, gradInput, size, pushData, (uint)(2 * sizeof(uint)));
    }

    public void HuberLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN, float delta)
    {
        var pushData = new uint[] { (uint)size, FloatToUInt32(invN), FloatToUInt32(delta) };
        GlslQuadOp(VulkanGlslKernels.HuberLossBackward, gradOutput, predictions, targets, gradInput, size, pushData, (uint)(3 * sizeof(uint)));
    }

    public void BceWithLogitsBackward(IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        var pushData = new uint[] { (uint)size, FloatToUInt32(invN) };
        GlslQuadOp(VulkanGlslKernels.BceWithLogitsBackward, gradOutput, logits, targets, gradInput, size, pushData, (uint)(2 * sizeof(uint)));
    }

    #endregion

    #region Loss Functions

    private float ResidentLossReduce(IGpuBuffer predictions, IGpuBuffer targets, int size, uint operation,
        float p0 = 0f, float p1 = 0f)
    {
        if (size == 0) return 0f;
        using var elementLoss = AllocateBuffer(size);
        GlslBinaryOp(VulkanGlslKernels.GenericLoss, predictions, targets, elementLoss, size,
            new uint[] { (uint)size, operation, FloatBits(p0), FloatBits(p1) }, 4 * sizeof(uint));
        return Sum(elementLoss, size) / size;
    }

    private void ResidentLossBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput,
        int size, uint operation, float p0 = 0f, float p1 = 0f)
    {
        if (size == 0) return;
        GlslBinaryOp(VulkanGlslKernels.GenericLossBackward, predictions, targets, gradInput, size,
            new uint[] { (uint)size, operation, FloatBits(p0), FloatBits(p1) }, 4 * sizeof(uint));
    }

    public float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 0);

    public void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 0);

    public float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 1);

    public void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 1);

    public float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 2);

    public void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 2);

    public float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        using var rowLoss = AllocateBuffer(batchSize);
        GlslBinaryOp(VulkanGlslKernels.SparseCrossEntropyRows, predictions, targets, rowLoss, batchSize,
            new uint[] { (uint)batchSize, (uint)numClasses }, 2 * sizeof(uint));
        return Sum(rowLoss, batchSize) / batchSize;
    }

    public void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
    {
        GlslBinaryOp(VulkanGlslKernels.SparseCrossEntropyBackward, predictions, targets, gradInput,
            batchSize * numClasses, new uint[] { (uint)batchSize, (uint)numClasses }, 2 * sizeof(uint));
    }

    public float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
        => ResidentLossReduce(predictions, targets, size, 3, beta);

    public void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
        => ResidentLossBackward(predictions, targets, gradInput, size, 3, beta);

    public float HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float delta)
        => SmoothL1Loss(predictions, targets, size, delta);

    public void HuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float delta)
        => SmoothL1Backward(predictions, targets, gradInput, size, delta);

    public float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
    {
        using var rowLoss = AllocateBuffer(batchSize);
        GlslQuadOp(VulkanGlslKernels.TripletLossRows, anchor, positive, negative, rowLoss, batchSize,
            new uint[] { (uint)batchSize, (uint)embeddingDim, FloatBits(margin) }, 3 * sizeof(uint));
        return Sum(rowLoss, batchSize) / batchSize;
    }

    public void TripletLossBackward(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative,
        IGpuBuffer gradAnchor, IGpuBuffer gradPositive, IGpuBuffer gradNegative,
        int batchSize, int embeddingDim, float margin)
    {
        int total = batchSize * embeddingDim;
        IGpuBuffer[] outputs = { gradAnchor, gradPositive, gradNegative };
        for (uint operation = 0; operation < 3; operation++)
            GlslQuadOp(VulkanGlslKernels.TripletLossBackward, anchor, positive, negative, outputs[operation], total,
                new uint[] { (uint)batchSize, (uint)embeddingDim, FloatBits(margin), operation }, 4 * sizeof(uint));
    }

    public float FocalLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float alpha, float gamma)
        => ResidentLossReduce(predictions, targets, size, 4, alpha, gamma);

    public void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
        => ResidentLossBackward(predictions, targets, gradInput, size, 4, alpha, gamma);

    public float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 5);

    public void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 5);

    public float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile)
        => ResidentLossReduce(predictions, targets, size, 6, quantile);

    public void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
        => ResidentLossBackward(predictions, targets, gradInput, size, 6, quantile);

    public float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 7);

    public void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 7);

    public float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 8);

    public void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 8);

    public float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 9);

    public void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 9);

    public float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 10);

    public void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 10);

    public float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 11);

    public void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 11);

    public float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => ResidentLossReduce(predictions, targets, size, 12);

    public void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ResidentLossBackward(predictions, targets, gradInput, size, 12);

    public float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon)
        => ResidentLossReduce(predictions, targets, size, 13, epsilon);

    public void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
        => ResidentLossBackward(predictions, targets, gradInput, size, 13, epsilon);

    public float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight)
        => ResidentLossReduce(predictions, targets, size, 14, l1Weight, l2Weight);

    public void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
        => ResidentLossBackward(predictions, targets, gradInput, size, 14, l1Weight, l2Weight);

    public float ContrastiveLoss(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
    {
        using var rowLoss = AllocateBuffer(batchSize);
        GlslQuadOp(VulkanGlslKernels.ContrastiveLossRows, output1, output2, labels, rowLoss, batchSize,
            new uint[] { (uint)batchSize, (uint)embeddingDim, FloatBits(margin) }, 3 * sizeof(uint));
        return Sum(rowLoss, batchSize) / batchSize;
    }

    public void ContrastiveBackward(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels,
        IGpuBuffer gradOutput1, IGpuBuffer gradOutput2,
        int batchSize, int embeddingDim, float margin)
    {
        int total = batchSize * embeddingDim;
        GlslQuadOp(VulkanGlslKernels.ContrastiveLossBackward, output1, output2, labels, gradOutput1, total,
            new uint[] { (uint)batchSize, (uint)embeddingDim, FloatBits(margin), 0 }, 4 * sizeof(uint));
        GlslQuadOp(VulkanGlslKernels.ContrastiveLossBackward, output1, output2, labels, gradOutput2, total,
            new uint[] { (uint)batchSize, (uint)embeddingDim, FloatBits(margin), 1 }, 4 * sizeof(uint));
    }

    #endregion

    #region Gradient Clipping and Utility

    public void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size)
        => ResidentUnary(ResidentUnaryOp.Clamp, A, B, size, min, max);

    public float L2Norm(IGpuBuffer A, int size)
        => ResidentScalarReduce(A, size, 4);

    public void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
        => ResidentUnary(ResidentUnaryOp.Clamp, A, B, size, -clipValue, clipValue);

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
        GlslQuadOp(VulkanGlslKernels.Fma, A, B, C, D, size);
    }

    public void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
    {
        GlslBinaryOp(VulkanGlslKernels.ScatterAdd, source, indices, destination, destSize,
            new uint[] { (uint)sourceSize, (uint)destSize }, 2 * sizeof(uint));
    }

    public void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource, int numIndices, int featureSize)
    {
        GlslBinaryOp(VulkanGlslKernels.GatherRows, gradDestination, indices, gradSource,
            numIndices * featureSize,
            new uint[] { (uint)numIndices, (uint)featureSize, (uint)(gradDestination.Size / featureSize) },
            3 * sizeof(uint));
    }

    public void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
    {
        GlslBinaryOp(VulkanGlslKernels.GatherRows, source, indices, output,
            numIndices * featureSize,
            new uint[] { (uint)numIndices, (uint)featureSize, (uint)(source.Size / featureSize) },
            3 * sizeof(uint));
    }

    #endregion

    #region Comparison Operations

    public void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => ResidentBinary(ResidentBinaryOp.Greater, A, B, C, size);

    public void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => ResidentBinary(ResidentBinaryOp.Less, A, B, C, size);

    public void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => ResidentBinary(ResidentBinaryOp.Equal, A, B, C, size);

    public void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        GlslQuadOp(VulkanGlslKernels.Where, condition, A, B, C, size);
    }

    public void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
        => ResidentUnary(ResidentUnaryOp.NotEqualScalar, A, C, size, scalar);

    #endregion

    #region Statistics

    public void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        if (reduceSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(reduceSize), "reduceSize must be positive.");
        GlslUnaryOp(VulkanGlslKernels.MeanAxis, A, B, outerSize,
            new uint[] { (uint)outerSize, (uint)reduceSize }, 2 * sizeof(uint));
    }

    public void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
    {
        if (reduceSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(reduceSize), "reduceSize must be positive.");
        GlslBinaryOp(VulkanGlslKernels.VarianceAxisFromMean, A, mean, variance, outerSize,
            new uint[] { (uint)outerSize, (uint)reduceSize }, 2 * sizeof(uint));
    }

    public void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        GlslUnaryOp(VulkanGlslKernels.ArgExtremaAxis, A, indices, outerSize,
            new uint[] { (uint)outerSize, (uint)reduceSize, 1 }, 3 * sizeof(uint));
    }

    /// <inheritdoc/>
    public bool ArgMaxIndicesAreBitReinterpreted => false; // shared contract: numeric float index

    public void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        => ArgMax(A, indices, outerSize, reduceSize);

    public void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        GlslUnaryOp(VulkanGlslKernels.ArgExtremaAxis, A, indices, outerSize,
            new uint[] { (uint)outerSize, (uint)reduceSize, 0 }, 3 * sizeof(uint));
    }

    public void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        GlslUnaryOp(VulkanGlslKernels.MaxAxis, A, B, outerSize,
            new uint[] { (uint)outerSize, (uint)reduceSize }, 2 * sizeof(uint));
    }

    public void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
    {
        if (outerSize <= 0 || reduceSize <= 0 || k <= 0) return;
        if (k > reduceSize) throw new ArgumentOutOfRangeException(nameof(k), "k cannot exceed reduceSize.");
        GlslDispatchN(VulkanGlslKernels.TopKAxis, outerSize * reduceSize,
            new[] { A, values, indices }, new[] { (uint)outerSize, (uint)reduceSize, (uint)k });
    }

    public void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        GlslBinaryOp(VulkanGlslKernels.BroadcastMultiplyLastAxis, A, B, C, outerSize * innerSize,
            new uint[] { (uint)outerSize, (uint)innerSize }, 2 * sizeof(uint));
    }

    public void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
    {
        EnsureInitialized();
        GlslBinaryOp(
            VulkanGlslKernels.BroadcastMultiplyFirstAxis,
            A,
            B,
            C,
            outerSize * innerSize,
            new uint[] { (uint)outerSize, (uint)innerSize },
            2 * sizeof(uint));
    }

    #endregion

    #region StopGradient, Fused Linear, and IoU Operations

    public void CopyBuffer(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        if (size <= 0) return;
        Copy(source, destination, size);
    }

    public void FusedLinearReLU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { GlslQuadOp(VulkanGlslKernels.FusedLinearReLU, input, weight, bias, output, batchSize * outFeatures, new uint[] { (uint)batchSize, (uint)inFeatures, (uint)outFeatures }, 3 * sizeof(uint)); }
    public void FusedLinearSigmoid(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { GlslQuadOp(VulkanGlslKernels.FusedLinearSigmoid, input, weight, bias, output, batchSize * outFeatures, new uint[] { (uint)batchSize, (uint)inFeatures, (uint)outFeatures }, 3 * sizeof(uint)); }
    public void FusedLinearTanh(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { GlslQuadOp(VulkanGlslKernels.FusedLinearTanh, input, weight, bias, output, batchSize * outFeatures, new uint[] { (uint)batchSize, (uint)inFeatures, (uint)outFeatures }, 3 * sizeof(uint)); }
    public void FusedLinearGELU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { GlslQuadOp(VulkanGlslKernels.FusedLinearGELU, input, weight, bias, output, batchSize * outFeatures, new uint[] { (uint)batchSize, (uint)inFeatures, (uint)outFeatures }, 3 * sizeof(uint)); }
    public void FusedLinearSwish(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { GlslQuadOp(VulkanGlslKernels.FusedLinearSwish, input, weight, bias, output, batchSize * outFeatures, new uint[] { (uint)batchSize, (uint)inFeatures, (uint)outFeatures }, 3 * sizeof(uint)); }
    public void FusedLinearReLUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardVulkan(VulkanGlslKernels.FusedLinearReLUBackwardGradInput, gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 0); }
    public void FusedLinearSigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardVulkan(VulkanGlslKernels.FusedLinearSigmoidBackwardGradInput, gradOutput, input, weight, output, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 1); }
    public void FusedLinearTanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardVulkan(VulkanGlslKernels.FusedLinearTanhBackwardGradInput, gradOutput, input, weight, output, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 2); }
    public void FusedLinearGELUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardVulkan(VulkanGlslKernels.FusedLinearGELUBackwardGradInput, gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 3); }
    public void FusedLinearSwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardVulkan(VulkanGlslKernels.FusedLinearSwishBackwardGradInput, gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 4); }
    public void IoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { GlslBinaryOp(VulkanGlslKernels.IoULoss, predicted, target, loss, numBoxes, new uint[] { (uint)numBoxes }, sizeof(uint)); }
    public void GIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { GlslBinaryOp(VulkanGlslKernels.GIoULoss, predicted, target, loss, numBoxes, new uint[] { (uint)numBoxes }, sizeof(uint)); }
    public void DIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { GlslBinaryOp(VulkanGlslKernels.DIoULoss, predicted, target, loss, numBoxes, new uint[] { (uint)numBoxes }, sizeof(uint)); }
    public void CIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { GlslBinaryOp(VulkanGlslKernels.CIoULoss, predicted, target, loss, numBoxes, new uint[] { (uint)numBoxes }, sizeof(uint)); }
    public void IoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { GlslQuadOp(VulkanGlslKernels.IoULossBackward, gradOutput, predicted, target, gradPredicted, numBoxes, new uint[] { (uint)numBoxes }, sizeof(uint)); }
    public void GIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { GlslQuadOp(VulkanGlslKernels.GIoULossBackward, gradOutput, predicted, target, gradPredicted, numBoxes, new uint[] { (uint)numBoxes }, sizeof(uint)); }
    public void DIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { GlslQuadOp(VulkanGlslKernels.DIoULossBackward, gradOutput, predicted, target, gradPredicted, numBoxes, new uint[] { (uint)numBoxes }, sizeof(uint)); }
    public void CIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { GlslQuadOp(VulkanGlslKernels.CIoULossBackward, gradOutput, predicted, target, gradPredicted, numBoxes, new uint[] { (uint)numBoxes }, sizeof(uint)); }

    private void LaunchFusedLinearBackwardVulkan(string gradInputKernel, IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer weight, IGpuBuffer saved, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures, uint activationType)
    {
        // Kernel 1: grad_input[b,i] = sum_j(masked_grad[b,j] * weight[i,j])
        var giPc = new uint[] { (uint)batchSize, (uint)inFeatures, (uint)outFeatures };
        GlslQuadOp(gradInputKernel, gradOutput, weight, saved, gradInput, batchSize * inFeatures, giPc, 3 * sizeof(uint));

        // Kernel 2: weight gradient — gradWeight[i,j] = sum_b(input[b,i] * masked_grad[b,j])
        var wgPc = new uint[] { (uint)batchSize, (uint)inFeatures, (uint)outFeatures, activationType };
        GlslQuadOp(VulkanGlslKernels.FusedLinearWeightGrad, gradOutput, input, saved, gradWeight, inFeatures * outFeatures, wgPc, 4 * sizeof(uint));

        // Kernel 3: bias gradient — gradBias[j] = sum_b(masked_grad[b,j])
        var bgPc = new uint[] { (uint)batchSize, (uint)outFeatures, activationType };
        GlslBinaryOp(VulkanGlslKernels.FusedLinearBiasGrad, gradOutput, saved, gradBias, outFeatures, bgPc, 3 * sizeof(uint));
    }

    #endregion
}
