// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 3: Attention, Activations, Loss, Gradient Clipping, Comparison, Statistics.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    #region Attention Operations

    private static int CheckedAttentionSize(string name, params int[] dimensions)
    {
        int size = 1;
        foreach (int dimension in dimensions)
        {
            if (dimension <= 0)
                throw new ArgumentOutOfRangeException(name, $"Attention dimensions must be positive; got {dimension}.");
            size = checked(size * dimension);
        }
        return size;
    }

    private static void RequireAttentionBuffer(IGpuBuffer buffer, int size, string name)
    {
        if (buffer is null) throw new ArgumentNullException(name);
        if (buffer.Size < size)
            throw new ArgumentException($"{name} requires at least {size} elements, but the buffer has {buffer.Size}.", name);
    }

    private static float[] MakeAttentionParams(
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim,
        int queriesPerKV, float scale, bool isCausal, bool hasBias, int biasBatchStride,
        bool flag0, bool flag1 = false, int booleanMaskMode = 0)
    {
        return new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(numQHeads),
            BitConverter.Int32BitsToSingle(numKVHeads),
            BitConverter.Int32BitsToSingle(seqQ),
            BitConverter.Int32BitsToSingle(seqK),
            BitConverter.Int32BitsToSingle(headDim),
            BitConverter.Int32BitsToSingle(queriesPerKV),
            BitConverter.Int32BitsToSingle(isCausal ? 1 : 0),
            scale,
            BitConverter.Int32BitsToSingle(hasBias ? 1 : 0),
            BitConverter.Int32BitsToSingle(biasBatchStride),
            BitConverter.Int32BitsToSingle(flag0 ? 1 : 0),
            BitConverter.Int32BitsToSingle(flag1 ? 1 : 0),
            BitConverter.Int32BitsToSingle(booleanMaskMode),
            0, 0
        };
    }

    private void AttentionForwardResident(
        IGpuBuffer query, IGpuBuffer key, IGpuBuffer value, IGpuBuffer output,
        IGpuBuffer? attentionWeights, IGpuBuffer? softmaxStats, IGpuBuffer? attentionBias,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim,
        float scale, bool isCausal, int biasBatchStride, int booleanMaskMode = 0)
    {
        if (numKVHeads <= 0 || numQHeads <= 0 || numQHeads % numKVHeads != 0)
            throw new ArgumentException("The number of query heads must be divisible by the number of KV heads.");
        int queriesPerKV = numQHeads / numKVHeads;
        int querySize = CheckedAttentionSize(nameof(query), batch, numQHeads, seqQ, headDim);
        int kvSize = CheckedAttentionSize(nameof(key), batch, numKVHeads, seqK, headDim);
        int rowCount = CheckedAttentionSize(nameof(output), batch, numQHeads, seqQ);
        int outputSize = checked(rowCount * headDim);
        int weightsSize = checked(rowCount * seqK);

        RequireAttentionBuffer(query, querySize, nameof(query));
        RequireAttentionBuffer(key, kvSize, nameof(key));
        RequireAttentionBuffer(value, kvSize, nameof(value));
        RequireAttentionBuffer(output, outputSize, nameof(output));
        if (attentionWeights is not null)
            RequireAttentionBuffer(attentionWeights, weightsSize, nameof(attentionWeights));
        if (softmaxStats is not null)
            RequireAttentionBuffer(softmaxStats, rowCount, nameof(softmaxStats));

        if (attentionBias is not null)
        {
            int biasHeadSize = booleanMaskMode == 1
                ? CheckedAttentionSize(nameof(attentionBias), seqQ, seqK)
                : CheckedAttentionSize(nameof(attentionBias), numQHeads, seqQ, seqK);
            if (biasBatchStride != 0 && biasBatchStride < biasHeadSize)
                throw new ArgumentOutOfRangeException(nameof(biasBatchStride),
                    "A nonzero bias batch stride must span every head and query row.");
            int biasSize = biasBatchStride == 0
                ? biasHeadSize
                : checked((batch - 1) * biasBatchStride + biasHeadSize);
            RequireAttentionBuffer(attentionBias, biasSize, nameof(attentionBias));
        }

        var uniforms = MakeAttentionParams(batch, numQHeads, numKVHeads, seqQ, seqK, headDim,
            queriesPerKV, scale, isCausal, attentionBias is not null, biasBatchStride,
            attentionWeights is not null, softmaxStats is not null, booleanMaskMode);
        IGpuBuffer dummy = SharedDummyBuffer;
        DispatchNBufferAsync("AttentionResident", WebGpuKernels.AttentionSource, "attention_forward_resident",
            new[]
            {
                query, key, value, output,
                attentionBias ?? dummy, attentionWeights ?? dummy, softmaxStats ?? dummy
            }, uniforms, rowCount).GetAwaiter().GetResult();
    }

    private void AttentionBackwardResident(
        IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer? attentionWeights, IGpuBuffer? attentionBias,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim,
        float scale, bool isCausal, int queriesPerKV, int biasBatchStride)
    {
        if (queriesPerKV <= 0 || (numQHeads - 1) / queriesPerKV >= numKVHeads)
            throw new ArgumentOutOfRangeException(nameof(queriesPerKV),
                "queriesPerKV must map every query head to an existing KV head.");
        int querySize = CheckedAttentionSize(nameof(query), batch, numQHeads, seqQ, headDim);
        int kvSize = CheckedAttentionSize(nameof(key), batch, numKVHeads, seqK, headDim);
        int rowCount = CheckedAttentionSize(nameof(gradOutput), batch, numQHeads, seqQ);
        int weightsSize = checked(rowCount * seqK);

        RequireAttentionBuffer(gradOutput, querySize, nameof(gradOutput));
        RequireAttentionBuffer(query, querySize, nameof(query));
        RequireAttentionBuffer(key, kvSize, nameof(key));
        RequireAttentionBuffer(value, kvSize, nameof(value));
        RequireAttentionBuffer(gradQuery, querySize, nameof(gradQuery));
        RequireAttentionBuffer(gradKey, kvSize, nameof(gradKey));
        RequireAttentionBuffer(gradValue, kvSize, nameof(gradValue));
        if (attentionWeights is not null)
            RequireAttentionBuffer(attentionWeights, weightsSize, nameof(attentionWeights));

        if (attentionBias is not null)
        {
            int biasHeadSize = CheckedAttentionSize(nameof(attentionBias), numQHeads, seqQ, seqK);
            if (biasBatchStride != 0 && biasBatchStride < biasHeadSize)
                throw new ArgumentOutOfRangeException(nameof(biasBatchStride));
            int biasSize = biasBatchStride == 0
                ? biasHeadSize
                : checked((batch - 1) * biasBatchStride + biasHeadSize);
            RequireAttentionBuffer(attentionBias, biasSize, nameof(attentionBias));
        }

        var uniforms = MakeAttentionParams(batch, numQHeads, numKVHeads, seqQ, seqK, headDim,
            queriesPerKV, scale, isCausal, attentionBias is not null, biasBatchStride,
            attentionWeights is not null);
        IGpuBuffer dummy = SharedDummyBuffer;
        var queryBuffers = new[]
        {
            gradOutput, query, key, value, attentionWeights ?? dummy, attentionBias ?? dummy,
            gradQuery, dummy
        };
        DispatchNBufferAsync("AttentionBackwardResident", WebGpuKernels.AttentionBackwardResidentSource,
            "attention_backward_resident_query", queryBuffers, uniforms, querySize).GetAwaiter().GetResult();
        var keyValueBuffers = new[]
        {
            gradOutput, query, key, value, attentionWeights ?? dummy, attentionBias ?? dummy,
            gradKey, gradValue
        };
        DispatchNBufferAsync("AttentionBackwardResident", WebGpuKernels.AttentionBackwardResidentSource,
            "attention_backward_resident_key_value", keyValueBuffers, uniforms, kvSize).GetAwaiter().GetResult();
    }
    public void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        int sharedMaskSize = CheckedAttentionSize(nameof(mask), seqQ, seqK);
        int fullMaskSize = CheckedAttentionSize(nameof(mask), batch, numHeads, seqQ, seqK);
        if (mask is not null && mask.Size < sharedMaskSize)
            throw new ArgumentException("The attention mask must contain at least seqQ * seqK elements.", nameof(mask));
        int maskMode = mask is null ? 0 : mask.Size >= fullMaskSize ? 2 : 1;
        int maskBatchStride = maskMode == 2 ? checked(numHeads * seqQ * seqK) : 0;
        AttentionForwardResident(query, key, value, output, attentionWeights, null, mask,
            batch, numHeads, numHeads, seqQ, seqK, headDim, scale, isCausal, maskBatchStride, maskMode);
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        AttentionBackwardResident(gradOutput, query, key, value, attentionWeights, null,
            gradQuery, gradKey, gradValue, batch, numHeads, numHeads, seqQ, seqK, headDim,
            scale, isCausal, 1, 0);
    }

    public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal) =>
        ScaledDotProductAttention(query, key, value, output, null, mask, batch, numHeads, seqLen, seqLen, headDim, scale, isCausal);

    public void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        AttentionForwardResident(query, key, value, output, null, softmaxStats, attentionBias,
            batch, numHeads, numHeads, seqQ, seqK, headDim, scale, isCausal, biasBatchStride);
    }

    public void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        AttentionBackwardResident(gradOutput, query, key, value, null, attentionBias,
            gradQuery, gradKey, gradValue, batch, numHeads, numHeads, seqQ, seqK, headDim,
            scale, isCausal, 1, biasBatchStride);
    }

    public void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        AttentionForwardResident(query, key, value, output, attentionWeights, null, null,
            batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scale, isCausal, 0);
    }

    public void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale,
        int numQueriesPerKV)
    {
        AttentionBackwardResident(gradOutput, query, key, value, attentionWeights, null,
            gradQuery, gradKey, gradValue, batch, numQHeads, numKVHeads, seqQ, seqK, headDim,
            scale, false, numQueriesPerKV, 0);
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

    public void Relu6(IGpuBuffer A, IGpuBuffer B, int size)
        => DispatchActivationAsync("relu6", A, B, size, 0).GetAwaiter().GetResult();

    public void Relu6Backward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "relu6_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void PRelu(IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer output, int size, int alphaSize)
    {
        var uniformParams = new float[] { BitConverter.Int32BitsToSingle(size), BitConverter.Int32BitsToSingle(alphaSize), 0f, 0f };
        Dispatch3BufferAsync("PReluForward", WebGpuKernels.PReluForwardSource, "prelu_forward",
            input, alpha, output, uniformParams, size).GetAwaiter().GetResult();
    }

    public void PReluBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer gradInput, int size, int alphaSize)
    {
        var uniformParams = new float[] { BitConverter.Int32BitsToSingle(size), BitConverter.Int32BitsToSingle(alphaSize), 0f, 0f };
        Dispatch4BufferAsync("PReluBackward", WebGpuKernels.PReluBackwardSource, "prelu_backward_input",
            gradOutput, input, alpha, gradInput, uniformParams, size).GetAwaiter().GetResult();
    }

    public void PReluBackwardAlpha(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradAlpha, int size, int alphaSize)
    {
        var uniformParams = new float[] { BitConverter.Int32BitsToSingle(size), BitConverter.Int32BitsToSingle(alphaSize), 0f, 0f };
        Dispatch3BufferAsync("PReluAlphaBackward", WebGpuKernels.PReluAlphaBackwardSource, "prelu_backward_alpha",
            gradOutput, input, gradAlpha, uniformParams, alphaSize).GetAwaiter().GetResult();
    }

    public void RRelu(IGpuBuffer input, IGpuBuffer noise, IGpuBuffer output, int size)
    {
        var uniformParams = new float[] { BitConverter.Int32BitsToSingle(size), BitConverter.Int32BitsToSingle(size), 0f, 0f };
        Dispatch3BufferAsync("PReluForward", WebGpuKernels.PReluForwardSource, "rrelu_forward",
            input, noise, output, uniformParams, size).GetAwaiter().GetResult();
    }

    public void RReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer noise, IGpuBuffer gradInput, int size)
    {
        var uniformParams = new float[] { BitConverter.Int32BitsToSingle(size), BitConverter.Int32BitsToSingle(size), 0f, 0f };
        Dispatch4BufferAsync("PReluBackward", WebGpuKernels.PReluBackwardSource, "rrelu_backward",
            gradOutput, input, noise, gradInput, uniformParams, size).GetAwaiter().GetResult();
    }

    public void Threshold(IGpuBuffer input, IGpuBuffer output, float threshold, float value, int size)
        => DispatchActivationAsync("threshold_op", input, output, size, threshold).GetAwaiter().GetResult();

    public void ThresholdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float threshold, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "threshold_backward",
            gradOutput, input, gradInput, MakeUniform2(size, threshold), size).GetAwaiter().GetResult();
    }

    public void ReciprocalBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("ActivationBackward", WebGpuKernels.ActivationBackwardSource, "reciprocal_backward",
            gradOutput, input, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    private static float[] MakePoolUniform(int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        return new[] {
            BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(inLength), BitConverter.Int32BitsToSingle(outLength),
            BitConverter.Int32BitsToSingle(kernelSize), BitConverter.Int32BitsToSingle(stride),
            0f, 0f
        };
    }

    public void AvgPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        Dispatch2BufferAsync("Pool1D", WebGpuKernels.Pool1DSource, "avg_pool1d",
            input, output, MakePoolUniform(batch, channels, inLength, outLength, kernelSize, stride),
            batch * channels * outLength).GetAwaiter().GetResult();
    }

    public void MaxPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        Dispatch2BufferAsync("Pool1D", WebGpuKernels.Pool1DSource, "max_pool1d",
            input, output, MakePoolUniform(batch, channels, inLength, outLength, kernelSize, stride),
            batch * channels * outLength).GetAwaiter().GetResult();
    }

    public void BilinearUpsample2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW)
    {
        Dispatch2BufferAsync("BilinearUpsample", WebGpuKernels.BilinearUpsample2DSource, "bilinear_upsample2d",
            input, output, new[] {
                BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(channels),
                BitConverter.Int32BitsToSingle(inH), BitConverter.Int32BitsToSingle(inW),
                BitConverter.Int32BitsToSingle(outH), BitConverter.Int32BitsToSingle(outW), 0f, 0f
            },
            batch * channels * outH * outW).GetAwaiter().GetResult();
    }

    public void ScatterMean(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts, int sourceSize, int outputSize, int featureSize)
    {
        if (sourceSize < 0 || outputSize < 0 || featureSize < 0)
            throw new ArgumentOutOfRangeException(nameof(sourceSize), "ScatterMean dimensions cannot be negative.");
        int total = checked(outputSize * featureSize);
        if (total == 0) return;
        var uniforms = new[]
        {
            BitConverter.Int32BitsToSingle(sourceSize),
            BitConverter.Int32BitsToSingle(outputSize),
            BitConverter.Int32BitsToSingle(featureSize),
            0f
        };
        Dispatch4BufferAsync("ScatterMean", WebGpuKernels.ScatterMeanSource, "scatter_mean_deterministic",
            source, indices, output, counts, uniforms, total).GetAwaiter().GetResult();
    }

    public void VarBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        Dispatch4BufferAsync("ReductionBackward4", WebGpuKernels.ReductionBackward4Source, "var_backward",
            gradOutput, input, mean, gradInput, MakeUniform2(outerSize, reduceSize), outerSize * reduceSize).GetAwaiter().GetResult();
    }

    public void StdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer std, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        Dispatch5BufferAsync("ReductionBackward5", WebGpuKernels.ReductionBackward5Source, "std_backward",
            gradOutput, input, mean, std, gradInput, MakeUniform2(outerSize, reduceSize), outerSize * reduceSize).GetAwaiter().GetResult();
    }

    public void MaskedFillBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size)
    {
        Dispatch3BufferAsync("MaskedFillBackward", WebGpuKernels.MaskedFillBackwardSource, "masked_fill_backward",
            gradOutput, mask, gradInput, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void WhereBackward(IGpuBuffer gradOutput, IGpuBuffer condition, IGpuBuffer gradX, IGpuBuffer gradY, int size)
    {
        Dispatch4BufferAsync("WhereBackward4", WebGpuKernels.WhereBackward4Source, "where_backward",
            gradOutput, condition, gradX, gradY, MakeUniform1(size), size).GetAwaiter().GetResult();
    }

    public void NormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer norm, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        Dispatch4BufferAsync("ReductionBackward4", WebGpuKernels.ReductionBackward4Source, "norm_backward",
            gradOutput, input, norm, gradInput, MakeUniform2(outerSize, reduceSize), outerSize * reduceSize).GetAwaiter().GetResult();
    }

    public void LogSumExpBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer lse, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        Dispatch4BufferAsync("ReductionBackward4", WebGpuKernels.ReductionBackward4Source, "logsumexp_backward",
            gradOutput, input, lse, gradInput, MakeUniform2(outerSize, reduceSize), outerSize * reduceSize).GetAwaiter().GetResult();
    }

    public void L1Loss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
    {
        Dispatch3BufferAsync("LossForward", WebGpuKernels.LossForwardSource, "l1_loss_batch",
            predictions, targets, loss, MakeUniform4(batchSize, BitConverter.Int32BitsToSingle(numFeatures), 0f, 0f), batchSize).GetAwaiter().GetResult();
    }

    public void HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures, float delta)
    {
        Dispatch3BufferAsync("LossForward", WebGpuKernels.LossForwardSource, "huber_loss_batch",
            predictions, targets, loss, MakeUniform4(batchSize, BitConverter.Int32BitsToSingle(numFeatures), delta, 0f), batchSize).GetAwaiter().GetResult();
    }

    public void BceWithLogitsLoss(IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer loss, int size)
    {
        Dispatch3BufferAsync("LossForward", WebGpuKernels.LossForwardSource, "bce_with_logits_forward",
            logits, targets, loss, MakeUniform4(size, 0f, 0f, 0f), size).GetAwaiter().GetResult();
    }

    public void NllLoss(IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
    {
        Dispatch3BufferAsync("LossForward", WebGpuKernels.LossForwardSource, "nll_loss_batch",
            logProbs, targets, loss, MakeUniform4(batchSize, BitConverter.Int32BitsToSingle(numClasses), 0f, 0f), batchSize).GetAwaiter().GetResult();
    }

    public void KlDivLoss(IGpuBuffer input, IGpuBuffer target, IGpuBuffer loss, int size)
    {
        Dispatch3BufferAsync("LossForward", WebGpuKernels.LossForwardSource, "kl_div_forward",
            input, target, loss, MakeUniform4(size, 0f, 0f, 0f), size).GetAwaiter().GetResult();
    }

    public void MseLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        Dispatch4BufferAsync("LossBackward4", WebGpuKernels.LossBackward4Source, "mse_4buf_backward",
            gradOutput, predictions, targets, gradInput, MakeUniform4(size, invN, 0f, 0f), size).GetAwaiter().GetResult();
    }

    public void L1LossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        Dispatch4BufferAsync("LossBackward4", WebGpuKernels.LossBackward4Source, "l1_4buf_backward",
            gradOutput, predictions, targets, gradInput, MakeUniform4(size, invN, 0f, 0f), size).GetAwaiter().GetResult();
    }

    public void HuberLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN, float delta)
    {
        Dispatch4BufferAsync("LossBackward4", WebGpuKernels.LossBackward4Source, "huber_4buf_backward",
            gradOutput, predictions, targets, gradInput, MakeUniform4(size, invN, delta, 0f), size).GetAwaiter().GetResult();
    }

    public void BceWithLogitsBackward(IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        Dispatch4BufferAsync("LossBackward4", WebGpuKernels.LossBackward4Source, "bce_logits_backward",
            gradOutput, logits, targets, gradInput, MakeUniform4(size, invN, 0f, 0f), size).GetAwaiter().GetResult();
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
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");
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
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");
        // Convert integer-encoded targets to one-hot via GPU kernel
        int total = batchSize * numClasses;
        using var oneHotBuf = (WebGpuBuffer)AllocateBuffer(total);
        Fill(oneHotBuf, 0f, total);
        var ohUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(numClasses),
            0, 0
        };
        Dispatch2BufferAsync("OneHot", WebGpuKernels.OneHotSource, "one_hot_encode",
            targets, oneHotBuf, ohUniforms, total).GetAwaiter().GetResult();
        // CrossEntropySource cross_entropy_backward: softmax(predictions) - targets (one-hot)
        var uniforms = MakeUniformInts2(batchSize, numClasses);
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
        int total = batchSize * embeddingDim;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(embeddingDim),
            margin, 0
        };
        // Compute gradAnchor: d(loss)/d(anchor) = (a-p)/||a-p|| - (a-n)/||a-n||
        Dispatch4BufferAsync("TripletContrastive", WebGpuKernels.TripletContrastiveSource, "triplet_loss_backward",
            anchor, positive, negative, gradAnchor, uniforms, total).GetAwaiter().GetResult();
        // Compute gradPositive: d(loss)/d(positive) = -(a-p)/||a-p|| (uses dedicated kernel)
        Dispatch4BufferAsync("TripletContrastive", WebGpuKernels.TripletContrastiveSource, "triplet_grad_positive",
            anchor, positive, negative, gradPositive, uniforms, total).GetAwaiter().GetResult();
        // Compute gradNegative: d(loss)/d(negative) = (a-n)/||a-n|| (uses dedicated kernel)
        Dispatch4BufferAsync("TripletContrastive", WebGpuKernels.TripletContrastiveSource, "triplet_grad_negative",
            anchor, positive, negative, gradNegative, uniforms, total).GetAwaiter().GetResult();
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
        if (sourceSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceSize), "Source size must be positive.");
        if (destSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(destSize), "Destination size must be positive.");
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
        if (numIndices <= 0)
            throw new ArgumentOutOfRangeException(nameof(numIndices), "Number of indices must be positive.");
        if (featureSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(featureSize), "Feature size must be positive.");
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
        // ConditionalSource where_op: binding(0)=condition, binding(1)=trueVals, binding(2)=C (pre-filled with falseVals)
        // Semantics: C[i] = condition[i] ? A[i] : B[i].
        //
        // Implementation strategy:
        //  - Pre-fill the destination buffer with the "false" values (B).
        //  - The kernel then selects between the existing C[i] (false) and A[i] (true) based on condition[i].
        //
        // To avoid corrupting inputs when buffers alias, we must not overwrite 'condition' or 'A' during pre-fill.
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            0 // scalar not used for where_op
        };

        if (ReferenceEquals(C, condition) || ReferenceEquals(C, A))
        {
            // Use a temporary output buffer to preserve aliased inputs during pre-fill.
            using var tempC = (WebGpuBuffer)AllocateBuffer(size);
            Copy(B, tempC, size);
            Dispatch3BufferAsync("Conditional", WebGpuKernels.ConditionalSource, "where_op",
                condition, A, tempC, uniforms, size).GetAwaiter().GetResult();
            Copy(tempC, C, size);
        }
        else
        {
            // Fast path when C does not alias condition or A (including the in-place B==C case).
            Copy(B, C, size);
            Dispatch3BufferAsync("Conditional", WebGpuKernels.ConditionalSource, "where_op",
                condition, A, C, uniforms, size).GetAwaiter().GetResult();
        }
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
        // First compute mean using existing GPU kernel (caller expects mean buffer populated)
        MeanAxis(A, mean, outerSize, reduceSize);
        // The VarAxisSource kernel recomputes the mean internally per workgroup for numerical
        // stability (avoids an extra buffer read and potential sync issues). The mean buffer
        // populated above is for the caller's use, not consumed by the variance kernel.
        var uniforms = MakeUniformInts2(outerSize, reduceSize);
        Dispatch2BufferAsync("VarAxis", WebGpuKernels.VarAxisSource, "var_axis",
            A, variance, uniforms, outerSize).GetAwaiter().GetResult();
    }

    public void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
    {
        Dispatch2BufferAsync("Statistics", WebGpuKernels.StatisticsSource, "argmax_axis",
            A, indices, MakeUniformInts2(outerSize, reduceSize), outerSize).GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public bool ArgMaxIndicesAreBitReinterpreted => false; // shared contract: numeric float index

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
        // The kernel's descending order is also valid when sorted=false, where order is unspecified.
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

    public void Lerp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, float t, int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "Size must be positive.");

        Dispatch3BufferAsync("LerpFused", WebGpuKernels.LerpFusedSource, "lerp_fused",
            a, b, output, MakeUniform2(size, t), size).GetAwaiter().GetResult();
    }

    public void AddScaled(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, float scaleA, float scaleB, int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "Size must be positive.");

        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            BitConverter.Int32BitsToSingle(0),
            scaleA,
            scaleB
        };
        Dispatch3BufferAsync("AddScaled", WebGpuKernels.AddScaledSource, "add_scaled",
            a, b, output, uniforms, size).GetAwaiter().GetResult();
    }

    public float StdDev(IGpuBuffer input, int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "Size must be positive.");
        if (input.Size < size)
            throw new ArgumentException($"Buffer 'input' capacity ({input.Size}) is less than size ({size}).", nameof(input));
        if (size <= 1) return 0.0f;

        float mean = Sum(input, size) / size;

        using var temp = AllocateBuffer(size);
        AddScalarAsync(input, temp, -mean, size).GetAwaiter().GetResult();
        Multiply(temp, temp, temp, size);
        float varianceSum = Sum(temp, size);

        float variance = Math.Max(0, varianceSum / size);
        return MathF.Sqrt(variance);
    }

    #endregion

    #region StopGradient, Fused Linear, and IoU Operations

    public void CopyBuffer(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        Copy(source, destination, size);
    }

    public void FusedLinearReLU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearForward(WebGpuKernels.FusedLinearReLU, input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearSigmoid(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearForward(WebGpuKernels.FusedLinearSigmoid, input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearTanh(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearForward(WebGpuKernels.FusedLinearTanh, input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearGELU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearForward(WebGpuKernels.FusedLinearGELU, input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearSwish(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearForward(WebGpuKernels.FusedLinearSwish, input, weight, bias, output, batchSize, inFeatures, outFeatures); }

    public void FusedLinearReLUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackward(WebGpuKernels.FusedLinearReLUBackwardGradInput, gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 0u); }
    public void FusedLinearSigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackward(WebGpuKernels.FusedLinearSigmoidBackwardGradInput, gradOutput, input, weight, output, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 1u); }
    public void FusedLinearTanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackward(WebGpuKernels.FusedLinearTanhBackwardGradInput, gradOutput, input, weight, output, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 2u); }
    public void FusedLinearGELUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackward(WebGpuKernels.FusedLinearGELUBackwardGradInput, gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 3u); }
    public void FusedLinearSwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackward(WebGpuKernels.FusedLinearSwishBackwardGradInput, gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 4u); }

    public void IoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoUForward(WebGpuKernels.IoULossWgsl, predicted, target, loss, numBoxes); }
    public void GIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoUForward(WebGpuKernels.GIoULossWgsl, predicted, target, loss, numBoxes); }
    public void DIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoUForward(WebGpuKernels.DIoULossWgsl, predicted, target, loss, numBoxes); }
    public void CIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoUForward(WebGpuKernels.CIoULossWgsl, predicted, target, loss, numBoxes); }

    public void IoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackward(WebGpuKernels.IoULossBackward, gradOutput, predicted, target, gradPredicted, numBoxes); }
    public void GIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackward(WebGpuKernels.GIoULossBackward, gradOutput, predicted, target, gradPredicted, numBoxes); }
    public void DIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackward(WebGpuKernels.DIoULossBackward, gradOutput, predicted, target, gradPredicted, numBoxes); }
    public void CIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackward(WebGpuKernels.CIoULossBackward, gradOutput, predicted, target, gradPredicted, numBoxes); }

    private void LaunchFusedLinearForward(string wgslSource, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inFeatures),
            BitConverter.Int32BitsToSingle(outFeatures),
            0
        };
        Dispatch4BufferAsync("FusedLinear", wgslSource, "main", input, weight, bias, output, uniforms, batchSize * outFeatures).GetAwaiter().GetResult();
    }

    private void LaunchFusedLinearBackward(string gradInputSource, IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer weight, IGpuBuffer saved, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures, uint activationType)
    {
        // Kernel 1: grad_input
        var giUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inFeatures),
            BitConverter.Int32BitsToSingle(outFeatures),
            0
        };
        Dispatch4BufferAsync("FusedLinearBwGI", gradInputSource, "main", gradOutput, weight, saved, gradInput, giUniforms, batchSize * inFeatures).GetAwaiter().GetResult();

        // Kernel 2: weight gradient
        var wgUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inFeatures),
            BitConverter.Int32BitsToSingle(outFeatures),
            BitConverter.Int32BitsToSingle((int)activationType)
        };
        Dispatch4BufferAsync("FusedLinearBwWG", WebGpuKernels.FusedLinearWeightGrad, "main", gradOutput, input, saved, gradWeight, wgUniforms, inFeatures * outFeatures).GetAwaiter().GetResult();

        // Kernel 3: bias gradient
        var bgUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(outFeatures),
            BitConverter.Int32BitsToSingle((int)activationType),
            0
        };
        Dispatch3BufferAsync("FusedLinearBwBG", WebGpuKernels.FusedLinearBiasGrad, "main", gradOutput, saved, gradBias, bgUniforms, outFeatures).GetAwaiter().GetResult();
    }

    private void LaunchIoUForward(string wgslSource, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes)
    {
        var uniforms = new float[] { BitConverter.Int32BitsToSingle(numBoxes), 0, 0, 0 };
        Dispatch3BufferAsync("IoULoss", wgslSource, "main", predicted, target, loss, uniforms, numBoxes).GetAwaiter().GetResult();
    }

    private void LaunchIoUBackward(string wgslSource, IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes)
    {
        var uniforms = new float[] { BitConverter.Int32BitsToSingle(numBoxes), 0, 0, 0 };
        Dispatch4BufferAsync("IoULossBw", wgslSource, "main", gradOutput, predicted, target, gradPredicted, uniforms, numBoxes).GetAwaiter().GetResult();
    }

    #endregion
}
#endif
