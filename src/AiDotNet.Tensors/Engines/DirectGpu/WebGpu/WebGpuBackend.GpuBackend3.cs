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
        // AttentionSource: binding(0)=query, binding(1)=key, binding(2)=value, binding(3)=output,
        // binding(4)=mask, binding(5)=uniform
        // Uniform: batch_heads, seq_len, head_dim, is_causal, scale, has_mask, pad, pad
        int batchHeads = batch * numHeads;
        bool hasMask = mask is not null;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchHeads),
            BitConverter.Int32BitsToSingle(seqLen),
            BitConverter.Int32BitsToSingle(headDim),
            BitConverter.Int32BitsToSingle(isCausal ? 1 : 0),
            scale,
            BitConverter.Int32BitsToSingle(hasMask ? 1 : 0),
            0, 0
        };
        int totalElements = batchHeads * seqLen * headDim;
        // Use the mask buffer if provided, otherwise pass the shared dummy buffer for the binding slot
        IGpuBuffer maskBuffer = mask ?? (IGpuBuffer)SharedDummyBuffer;
        Dispatch5BufferAsync("Attention", WebGpuKernels.AttentionSource, "scaled_dot_product_attention",
            query, key, value, output, maskBuffer, uniforms, totalElements).GetAwaiter().GetResult();
        // attentionWeights not produced by this kernel; fill with zeros if requested
        if (attentionWeights is not null)
            Fill(attentionWeights, 0f, batchHeads * seqLen * seqLen);
    }

    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        int batchHeads = batch * numHeads;
        int total = batchHeads * seqLen * headDim;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchHeads),
            BitConverter.Int32BitsToSingle(seqLen),
            BitConverter.Int32BitsToSingle(headDim),
            0
        };
        // Compute gradQuery via attention_backward_query kernel
        // 6-buffer: gradOutput, query, key, value, gradQuery, gradKV_combined
        using var gradKV = (WebGpuBuffer)AllocateBuffer(total * 2);
        Dispatch6BufferAsync("AttentionBackward", WebGpuKernels.AttentionBackwardSource, "attention_backward_query",
            gradOutput, query, key, value, gradQuery, gradKV, uniforms, total).GetAwaiter().GetResult();
        // Compute gradKey and gradValue via attention_backward_kv kernel
        Dispatch6BufferAsync("AttentionBackward", WebGpuKernels.AttentionBackwardSource, "attention_backward_kv",
            gradOutput, query, key, value, gradKey, gradKV, uniforms, total).GetAwaiter().GetResult();
        // Copy gradKV results: first half = gradKey, second half = gradValue
        Copy(gradKV, 0, gradKey, 0, total);
        Copy(gradKV, total, gradValue, 0, total);
    }

    public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal) =>
        ScaledDotProductAttention(query, key, value, output, null, mask, batch, numHeads, seqLen, headDim, scale, isCausal);

    public void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        // Use the full sequence lengths: seqQ for queries, seqK for keys/values.
        // When seqQ == seqK, delegate to standard attention. When they differ,
        // we must handle the asymmetric case.
        int batchHeads = batch * numHeads;
        int totalElements = batchHeads * seqQ * headDim;
        // When seqQ == seqK, use the standard attention path
        if (seqQ == seqK)
        {
            ScaledDotProductAttention(query, key, value, output, null, attentionBias,
                batch, numHeads, seqQ, headDim, scale, isCausal);
        }
        else
        {
            // For asymmetric lengths, use seqK for the kernel's KV loop range.
            // The output still covers seqQ positions, so we dispatch over seqQ.
            var asymUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batchHeads),
                BitConverter.Int32BitsToSingle(seqK),
                BitConverter.Int32BitsToSingle(headDim),
                BitConverter.Int32BitsToSingle(isCausal ? 1 : 0),
                scale,
                BitConverter.Int32BitsToSingle(attentionBias is not null ? 1 : 0),
                0, 0
            };
            IGpuBuffer maskBuffer = attentionBias ?? (IGpuBuffer)SharedDummyBuffer;
            Dispatch5BufferAsync("Attention", WebGpuKernels.AttentionSource, "scaled_dot_product_attention",
                query, key, value, output, maskBuffer, asymUniforms, totalElements).GetAwaiter().GetResult();
        }
        // Fill softmaxStats with per-head logsumexp values (zeros for now, actual stats would require a separate kernel)
        Fill(softmaxStats, 0f, batch * numHeads * seqQ);
    }

    public void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        // Delegate to the standard attention backward with seqQ as the shared length.
        // When seqQ != seqK, we use seqQ for the query gradient and seqK for KV gradients.
        // Flash attention does not materialize attention weights, so pass a dummy buffer
        // for the attentionWeights slot (the backward kernel recomputes weights internally).
        int seqLen = Math.Min(seqQ, seqK);
        IGpuBuffer dummyAttnWeights = SharedDummyBuffer;
        ScaledDotProductAttentionBackward(gradOutput, query, key, value, dummyAttnWeights, gradQuery, gradKey, gradValue,
            batch, numHeads, seqLen, headDim, scale, isCausal);
        // Zero-fill any remaining elements if seqQ > seqK or vice versa
        if (seqQ > seqLen)
        {
            int extraQ = batch * numHeads * (seqQ - seqLen) * headDim;
            if (extraQ > 0)
            {
                Fill(gradQuery, 0f, batch * numHeads * seqQ * headDim);
                ScaledDotProductAttentionBackward(gradOutput, query, key, value, dummyAttnWeights, gradQuery, gradKey, gradValue,
                    batch, numHeads, seqLen, headDim, scale, isCausal);
            }
        }
    }

    public void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        int seqLen = Math.Min(seqQ, seqK);
        if (numQHeads == numKVHeads)
        {
            // Standard multi-head attention: no head grouping needed
            ScaledDotProductAttention(query, key, value, output, attentionWeights, null, batch, numQHeads, seqLen, headDim, scale, isCausal);
        }
        else
        {
            // Grouped Query Attention: map Q-heads to KV-heads via head_ratio
            // Use dedicated GQA kernel that computes kv_head = q_head / (numQHeads / numKVHeads)
            int totalOutput = batch * numQHeads * seqLen * headDim;
            var uniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(numQHeads),
                BitConverter.Int32BitsToSingle(numKVHeads),
                BitConverter.Int32BitsToSingle(seqLen),
                BitConverter.Int32BitsToSingle(headDim),
                0, 0, 0
            };
            Dispatch4BufferAsync("GroupedQueryAttention", WebGpuKernels.GroupedQueryAttentionSource, "grouped_query_attention",
                query, key, value, output, uniforms, totalOutput).GetAwaiter().GetResult();
        }
        // attentionWeights not produced by these kernels; fill with zeros if requested
        if (attentionWeights is not null)
            Fill(attentionWeights, 0f, batch * numQHeads * seqLen * seqLen);
    }

    public void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
    {
        // Compute gradients using the attention backward kernel.
        // For GQA, multiple Q-heads share the same KV-head, so KV gradients must be accumulated.
        int seqLen = Math.Min(seqQ, seqK);
        if (numQHeads == numKVHeads)
        {
            ScaledDotProductAttentionBackward(gradOutput, query, key, value, attentionWeights,
                gradQuery, gradKey, gradValue, batch, numQHeads, seqLen, headDim, scale, false);
        }
        else
        {
            // Grouped query attention: multiple Q-heads share a single KV-head.
            // We compute per-(batch, Q-head) gradients and accumulate K/V gradients
            // into the corresponding shared KV-head.
            int headRatio = numQHeads / numKVHeads;

            // Zero KV gradients; we'll accumulate into them.
            Fill(gradKey, 0f, batch * numKVHeads * seqLen * headDim);
            Fill(gradValue, 0f, batch * numKVHeads * seqLen * headDim);

            // Each (batch=1, head=1) instance covers seqLen * headDim elements.
            int headStride = seqLen * headDim;

            // Temporary buffers for per-(batch, head) slicing and gradient computation.
            // Each slice is for a single (batch=1, numHeads=1) attention operation.
            using var sliceGO = (WebGpuBuffer)AllocateBuffer(headStride);
            using var sliceQ = (WebGpuBuffer)AllocateBuffer(headStride);
            using var sliceK = (WebGpuBuffer)AllocateBuffer(headStride);
            using var sliceV = (WebGpuBuffer)AllocateBuffer(headStride);
            using var sliceAttn = (WebGpuBuffer)AllocateBuffer(headStride);
            using var sliceGQ = (WebGpuBuffer)AllocateBuffer(headStride);
            using var tempGradK = (WebGpuBuffer)AllocateBuffer(headStride);
            using var tempGradV = (WebGpuBuffer)AllocateBuffer(headStride);
            // Reusable buffers for reading current KV grad slices during accumulation
            using var currentGK = (WebGpuBuffer)AllocateBuffer(headStride);
            using var currentGV = (WebGpuBuffer)AllocateBuffer(headStride);

            var goBuffer = (WebGpuBuffer)gradOutput;
            var qBuffer = (WebGpuBuffer)query;
            var kBuffer = (WebGpuBuffer)key;
            var vBuffer = (WebGpuBuffer)value;
            var attnBuffer = (WebGpuBuffer)attentionWeights;
            var gqBuffer = (WebGpuBuffer)gradQuery;
            var gkBuffer = (WebGpuBuffer)gradKey;
            var gvBuffer = (WebGpuBuffer)gradValue;

            // Iterate over batch, KV-head, and Q-heads within each KV group.
            for (int b = 0; b < batch; b++)
            {
                for (int kvh = 0; kvh < numKVHeads; kvh++)
                {
                    // Copy the KV-head slice for this (batch, kvh) into temp buffers
                    int kvBhIndex = b * numKVHeads + kvh;
                    int kvOffset = kvBhIndex * headStride;
                    sliceK.CopyFromBuffer(kBuffer, kvOffset, 0, headStride);
                    sliceV.CopyFromBuffer(vBuffer, kvOffset, 0, headStride);

                    for (int g = 0; g < headRatio; g++)
                    {
                        int qh = kvh * headRatio + g;

                        // Flattened (batch * head) index for this Q-head
                        int qBhIndex = b * numQHeads + qh;
                        int qOffset = qBhIndex * headStride;

                        // Copy per-head slices for Q, gradOutput, and attentionWeights
                        sliceGO.CopyFromBuffer(goBuffer, qOffset, 0, headStride);
                        sliceQ.CopyFromBuffer(qBuffer, qOffset, 0, headStride);
                        sliceAttn.CopyFromBuffer(attnBuffer, qOffset, 0, headStride);

                        // Compute backward gradients for this single (batch=1, head=1) slice
                        ScaledDotProductAttentionBackward(
                            sliceGO, sliceQ, sliceK, sliceV, sliceAttn,
                            sliceGQ, tempGradK, tempGradV,
                            1, 1, seqLen, headDim, scale, false);

                        // Write gradQuery back to the correct Q-head offset
                        gqBuffer.CopyFromBuffer(sliceGQ, 0, qOffset, headStride);

                        // Accumulate KV gradients into the shared KV-head position:
                        // gradKey[kvOffset..] += tempGradK, gradValue[kvOffset..] += tempGradV
                        currentGK.CopyFromBuffer(gkBuffer, kvOffset, 0, headStride);
                        currentGV.CopyFromBuffer(gvBuffer, kvOffset, 0, headStride);
                        Add(currentGK, tempGradK, currentGK, headStride);
                        Add(currentGV, tempGradV, currentGV, headStride);
                        gkBuffer.CopyFromBuffer(currentGK, 0, kvOffset, headStride);
                        gvBuffer.CopyFromBuffer(currentGV, 0, kvOffset, headStride);
                    }
                }
            }
        }
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
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");
        // Validate target indices are in bounds by downloading and checking
        var targetData = DownloadBufferData(targets);
        for (int i = 0; i < batchSize; i++)
        {
            int classIdx = BitConverter.SingleToInt32Bits(targetData[i]);
            if (classIdx < 0 || classIdx >= numClasses)
                throw new ArgumentOutOfRangeException(nameof(targets),
                    $"Target index {classIdx} at position {i} is out of bounds for {numClasses} classes.");
        }
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
        // Validate target indices are in bounds
        var targetData = DownloadBufferData(targets);
        for (int i = 0; i < batchSize; i++)
        {
            int classIdx = BitConverter.SingleToInt32Bits(targetData[i]);
            if (classIdx < 0 || classIdx >= numClasses)
                throw new ArgumentOutOfRangeException(nameof(targets),
                    $"Target index {classIdx} at position {i} is out of bounds for {numClasses} classes.");
        }
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
        // Validate index bounds by downloading indices
        var indexData = DownloadBufferData(indices);
        for (int i = 0; i < sourceSize; i++)
        {
            int idx = BitConverter.SingleToInt32Bits(indexData[i]);
            if (idx < 0 || idx >= destSize)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {idx} at position {i} is out of bounds for destination size {destSize}.");
        }
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
        // Validate index bounds by downloading indices
        var indexData = DownloadBufferData(indices);
        for (int i = 0; i < numIndices; i++)
        {
            int idx = BitConverter.SingleToInt32Bits(indexData[i]);
            if (idx < 0)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {idx} at position {i} is negative.");
        }
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
        // Note: The WGSL kernel uses selection sort which inherently produces sorted (descending) output.
        // When sorted=false, we shuffle the results to remove the ordering guarantee.
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(outerSize),
            BitConverter.Int32BitsToSingle(reduceSize),
            BitConverter.Int32BitsToSingle(k),
            0
        };
        Dispatch3BufferAsync("TopK", WebGpuKernels.TopKSource, "topk_select",
            A, values, indices, uniforms, outerSize).GetAwaiter().GetResult();

        if (!sorted && k > 1)
        {
            // Shuffle the top-K results to remove the descending sort order.
            // Download, shuffle per batch, re-upload.
            int totalK = outerSize * k;
            var valData = DownloadBufferData(values);
            var idxData = DownloadBufferData(indices);
            var rng = new Random();
            for (int b = 0; b < outerSize; b++)
            {
                int baseIdx = b * k;
                // Fisher-Yates shuffle
                for (int i = k - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (valData[baseIdx + i], valData[baseIdx + j]) = (valData[baseIdx + j], valData[baseIdx + i]);
                    (idxData[baseIdx + i], idxData[baseIdx + j]) = (idxData[baseIdx + j], idxData[baseIdx + i]);
                }
            }
            UploadToBuffer(valData, values);
            UploadToBuffer(idxData, indices);
        }
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
}
#endif
