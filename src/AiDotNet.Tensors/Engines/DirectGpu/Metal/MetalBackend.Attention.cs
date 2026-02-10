// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Attention and Spatial Transformer operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Attention Operations

    /// <summary>
    /// Scaled dot-product attention forward pass.
    /// </summary>
    public void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        ThrowIfDisposed();

        var queryData = DownloadBuffer(query);
        var keyData = DownloadBuffer(key);
        var valueData = DownloadBuffer(value);
        float[]? maskData = mask is not null ? DownloadBuffer(mask) : null;

        var outputData = new float[batch * numHeads * seqLen * headDim];
        var weightsData = attentionWeights is not null ? new float[batch * numHeads * seqLen * seqLen] : null;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int headOffset = (b * numHeads + h) * seqLen * headDim;
                int weightsOffset = (b * numHeads + h) * seqLen * seqLen;

                // Compute attention scores: Q @ K^T
                var scores = new float[seqLen * seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        float sum = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            sum += queryData[headOffset + i * headDim + d] * keyData[headOffset + j * headDim + d];
                        }
                        scores[i * seqLen + j] = sum * scale;

                        // Apply causal mask
                        if (isCausal && j > i)
                        {
                            scores[i * seqLen + j] = float.NegativeInfinity;
                        }
                        else if (maskData is not null && maskData[i * seqLen + j] == 0)
                        {
                            scores[i * seqLen + j] = float.NegativeInfinity;
                        }
                    }
                }

                // Softmax over scores
                for (int i = 0; i < seqLen; i++)
                {
                    float maxScore = float.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                    {
                        maxScore = MathF.Max(maxScore, scores[i * seqLen + j]);
                    }

                    float sumExp = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        scores[i * seqLen + j] = MathF.Exp(scores[i * seqLen + j] - maxScore);
                        sumExp += scores[i * seqLen + j];
                    }

                    for (int j = 0; j < seqLen; j++)
                    {
                        scores[i * seqLen + j] /= sumExp;
                    }
                }

                // Copy weights if needed
                if (weightsData is not null)
                {
                    Array.Copy(scores, 0, weightsData, weightsOffset, seqLen * seqLen);
                }

                // Apply attention: scores @ V
                for (int i = 0; i < seqLen; i++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {
                            sum += scores[i * seqLen + j] * valueData[headOffset + j * headDim + d];
                        }
                        outputData[headOffset + i * headDim + d] = sum;
                    }
                }
            }
        }

        UploadToBuffer(output, outputData);
        if (attentionWeights is not null && weightsData is not null)
        {
            UploadToBuffer(attentionWeights, weightsData);
        }
    }

    /// <summary>
    /// Scaled dot-product attention backward pass.
    /// </summary>
    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var queryData = DownloadBuffer(query);
        var keyData = DownloadBuffer(key);
        var valueData = DownloadBuffer(value);
        var weightsData = DownloadBuffer(attentionWeights);

        var gradQueryData = new float[batch * numHeads * seqLen * headDim];
        var gradKeyData = new float[batch * numHeads * seqLen * headDim];
        var gradValueData = new float[batch * numHeads * seqLen * headDim];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int headOffset = (b * numHeads + h) * seqLen * headDim;
                int weightsOffset = (b * numHeads + h) * seqLen * seqLen;

                // Gradient w.r.t. V: gradV = weights^T @ gradOutput
                for (int j = 0; j < seqLen; j++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int i = 0; i < seqLen; i++)
                        {
                            sum += weightsData[weightsOffset + i * seqLen + j] * gradOutputData[headOffset + i * headDim + d];
                        }
                        gradValueData[headOffset + j * headDim + d] = sum;
                    }
                }

                // Gradient w.r.t. attention weights
                var gradWeights = new float[seqLen * seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        float sum = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            sum += gradOutputData[headOffset + i * headDim + d] * valueData[headOffset + j * headDim + d];
                        }
                        gradWeights[i * seqLen + j] = sum;
                    }
                }

                // Gradient through softmax
                var gradScores = new float[seqLen * seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    float dotProduct = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        dotProduct += gradWeights[i * seqLen + j] * weightsData[weightsOffset + i * seqLen + j];
                    }

                    for (int j = 0; j < seqLen; j++)
                    {
                        float w = weightsData[weightsOffset + i * seqLen + j];
                        gradScores[i * seqLen + j] = w * (gradWeights[i * seqLen + j] - dotProduct);

                        // Apply causal mask gradient
                        if (isCausal && j > i)
                        {
                            gradScores[i * seqLen + j] = 0;
                        }
                    }
                }

                // Gradient w.r.t. Q: gradQ = gradScores @ K * scale
                for (int i = 0; i < seqLen; i++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {
                            sum += gradScores[i * seqLen + j] * keyData[headOffset + j * headDim + d];
                        }
                        gradQueryData[headOffset + i * headDim + d] = sum * scale;
                    }
                }

                // Gradient w.r.t. K: gradK = gradScores^T @ Q * scale
                for (int j = 0; j < seqLen; j++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int i = 0; i < seqLen; i++)
                        {
                            sum += gradScores[i * seqLen + j] * queryData[headOffset + i * headDim + d];
                        }
                        gradKeyData[headOffset + j * headDim + d] = sum * scale;
                    }
                }
            }
        }

        UploadToBuffer(gradQuery, gradQueryData);
        UploadToBuffer(gradKey, gradKeyData);
        UploadToBuffer(gradValue, gradValueData);
    }

    /// <summary>
    /// Flash attention forward pass (memory-efficient).
    /// </summary>
    public void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        ThrowIfDisposed();
        // Flash attention is implemented as standard attention with tiling optimization
        // For simplicity, delegate to standard implementation
        ScaledDotProductAttention(query, key, value, output, null, mask, batch, numHeads, seqLen, headDim, scale, isCausal);
    }

    /// <summary>
    /// Flash attention V2 with log-sum-exp statistics.
    /// </summary>
    public void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        ThrowIfDisposed();

        var queryData = DownloadBuffer(query);
        var keyData = DownloadBuffer(key);
        var valueData = DownloadBuffer(value);
        float[]? biasData = attentionBias is not null ? DownloadBuffer(attentionBias) : null;

        var outputData = new float[batch * numHeads * seqQ * headDim];
        var statsData = new float[batch * numHeads * seqQ];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int qOffset = (b * numHeads + h) * seqQ * headDim;
                int kOffset = (b * numHeads + h) * seqK * headDim;
                int statsOffset = (b * numHeads + h) * seqQ;
                int biasHeadOffset = biasData is not null
                    ? b * biasBatchStride + h * seqQ * seqK
                    : 0;

                for (int i = 0; i < seqQ; i++)
                {
                    // Compute attention scores for this query
                    var scores = new float[seqK];
                    float maxScore = float.NegativeInfinity;

                    for (int j = 0; j < seqK; j++)
                    {
                        if (isCausal && j > i)
                        {
                            scores[j] = float.NegativeInfinity;
                        }
                        else
                        {
                            float sum = 0;
                            for (int d = 0; d < headDim; d++)
                            {
                                sum += queryData[qOffset + i * headDim + d] * keyData[kOffset + j * headDim + d];
                            }
                            scores[j] = sum * scale;

                            // Add attention bias (ALiBi / relative position bias)
                            if (biasData is not null)
                            {
                                scores[j] += biasData[biasHeadOffset + i * seqK + j];
                            }

                            maxScore = MathF.Max(maxScore, scores[j]);
                        }
                    }

                    // Softmax with log-sum-exp
                    float sumExp = 0;
                    for (int j = 0; j < seqK; j++)
                    {
                        if (!float.IsNegativeInfinity(scores[j]))
                        {
                            scores[j] = MathF.Exp(scores[j] - maxScore);
                            sumExp += scores[j];
                        }
                        else
                        {
                            scores[j] = 0;
                        }
                    }

                    for (int j = 0; j < seqK; j++)
                    {
                        scores[j] /= sumExp;
                    }

                    statsData[statsOffset + i] = maxScore + MathF.Log(sumExp);

                    // Apply attention
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqK; j++)
                        {
                            sum += scores[j] * valueData[kOffset + j * headDim + d];
                        }
                        outputData[qOffset + i * headDim + d] = sum;
                    }
                }
            }
        }

        UploadToBuffer(output, outputData);
        UploadToBuffer(softmaxStats, statsData);
    }

    /// <summary>
    /// Flash attention backward pass.
    /// </summary>
    public void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        IGpuBuffer? attentionBias = null, int biasBatchStride = 0)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var queryData = DownloadBuffer(query);
        var keyData = DownloadBuffer(key);
        var valueData = DownloadBuffer(value);
        var statsData = DownloadBuffer(softmaxStats);
        float[]? biasData = attentionBias is not null ? DownloadBuffer(attentionBias) : null;

        var gradQueryData = new float[batch * numHeads * seqQ * headDim];
        var gradKeyData = new float[batch * numHeads * seqK * headDim];
        var gradValueData = new float[batch * numHeads * seqK * headDim];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int qOffset = (b * numHeads + h) * seqQ * headDim;
                int kOffset = (b * numHeads + h) * seqK * headDim;
                int statsOffset = (b * numHeads + h) * seqQ;
                int biasHeadOffset = biasData is not null
                    ? b * biasBatchStride + h * seqQ * seqK
                    : 0;

                for (int i = 0; i < seqQ; i++)
                {
                    float lse = statsData[statsOffset + i];

                    // Recompute attention weights
                    var weights = new float[seqK];
                    for (int j = 0; j < seqK; j++)
                    {
                        if (isCausal && j > i)
                        {
                            weights[j] = 0;
                        }
                        else
                        {
                            float score = 0;
                            for (int d = 0; d < headDim; d++)
                            {
                                score += queryData[qOffset + i * headDim + d] * keyData[kOffset + j * headDim + d];
                            }

                            float biasedScore = score * scale;
                            if (biasData is not null)
                            {
                                biasedScore += biasData[biasHeadOffset + i * seqK + j];
                            }

                            weights[j] = MathF.Exp(biasedScore - lse);
                        }
                    }

                    // Gradient w.r.t. V
                    for (int j = 0; j < seqK; j++)
                    {
                        for (int d = 0; d < headDim; d++)
                        {
                            gradValueData[kOffset + j * headDim + d] += weights[j] * gradOutputData[qOffset + i * headDim + d];
                        }
                    }

                    // Gradient through attention
                    float dotProduct = 0;
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqK; j++)
                        {
                            sum += weights[j] * valueData[kOffset + j * headDim + d];
                        }
                        dotProduct += gradOutputData[qOffset + i * headDim + d] * sum;
                    }

                    for (int j = 0; j < seqK; j++)
                    {
                        float gradWeight = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            gradWeight += gradOutputData[qOffset + i * headDim + d] * valueData[kOffset + j * headDim + d];
                        }
                        float gradScore = weights[j] * (gradWeight - dotProduct) * scale;

                        // Accumulate gradients
                        for (int d = 0; d < headDim; d++)
                        {
                            gradQueryData[qOffset + i * headDim + d] += gradScore * keyData[kOffset + j * headDim + d];
                            gradKeyData[kOffset + j * headDim + d] += gradScore * queryData[qOffset + i * headDim + d];
                        }
                    }
                }
            }
        }

        UploadToBuffer(gradQuery, gradQueryData);
        UploadToBuffer(gradKey, gradKeyData);
        UploadToBuffer(gradValue, gradValueData);
    }

    /// <summary>
    /// Grouped query attention forward pass.
    /// </summary>
    public void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        ThrowIfDisposed();

        var queryData = DownloadBuffer(query);
        var keyData = DownloadBuffer(key);
        var valueData = DownloadBuffer(value);

        int headsPerGroup = numQHeads / numKVHeads;
        var outputData = new float[batch * numQHeads * seqQ * headDim];
        var weightsData = attentionWeights is not null ? new float[batch * numQHeads * seqQ * seqK] : null;

        for (int b = 0; b < batch; b++)
        {
            for (int qh = 0; qh < numQHeads; qh++)
            {
                int kvh = qh / headsPerGroup;
                int qOffset = (b * numQHeads + qh) * seqQ * headDim;
                int kvOffset = (b * numKVHeads + kvh) * seqK * headDim;
                int wOffset = (b * numQHeads + qh) * seqQ * seqK;

                for (int i = 0; i < seqQ; i++)
                {
                    // Compute scores
                    var scores = new float[seqK];
                    float maxScore = float.NegativeInfinity;

                    for (int j = 0; j < seqK; j++)
                    {
                        if (isCausal && j > i)
                        {
                            scores[j] = float.NegativeInfinity;
                        }
                        else
                        {
                            float sum = 0;
                            for (int d = 0; d < headDim; d++)
                            {
                                sum += queryData[qOffset + i * headDim + d] * keyData[kvOffset + j * headDim + d];
                            }
                            scores[j] = sum * scale;
                            maxScore = MathF.Max(maxScore, scores[j]);
                        }
                    }

                    // Softmax
                    float sumExp = 0;
                    for (int j = 0; j < seqK; j++)
                    {
                        if (!float.IsNegativeInfinity(scores[j]))
                        {
                            scores[j] = MathF.Exp(scores[j] - maxScore);
                            sumExp += scores[j];
                        }
                        else
                        {
                            scores[j] = 0;
                        }
                    }

                    for (int j = 0; j < seqK; j++)
                    {
                        scores[j] /= sumExp;
                    }

                    if (weightsData is not null)
                    {
                        Array.Copy(scores, 0, weightsData, wOffset + i * seqK, seqK);
                    }

                    // Apply attention
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0;
                        for (int j = 0; j < seqK; j++)
                        {
                            sum += scores[j] * valueData[kvOffset + j * headDim + d];
                        }
                        outputData[qOffset + i * headDim + d] = sum;
                    }
                }
            }
        }

        UploadToBuffer(output, outputData);
        if (attentionWeights is not null && weightsData is not null)
        {
            UploadToBuffer(attentionWeights, weightsData);
        }
    }

    /// <summary>
    /// Grouped query attention backward pass.
    /// </summary>
    public void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var queryData = DownloadBuffer(query);
        var keyData = DownloadBuffer(key);
        var valueData = DownloadBuffer(value);
        var weightsData = DownloadBuffer(attentionWeights);

        int headsPerGroup = numQHeads / numKVHeads;
        var gradQueryData = new float[batch * numQHeads * seqQ * headDim];
        var gradKeyData = new float[batch * numKVHeads * seqK * headDim];
        var gradValueData = new float[batch * numKVHeads * seqK * headDim];

        for (int b = 0; b < batch; b++)
        {
            for (int qh = 0; qh < numQHeads; qh++)
            {
                int kvh = qh / headsPerGroup;
                int qOffset = (b * numQHeads + qh) * seqQ * headDim;
                int kvOffset = (b * numKVHeads + kvh) * seqK * headDim;
                int wOffset = (b * numQHeads + qh) * seqQ * seqK;

                for (int i = 0; i < seqQ; i++)
                {
                    // Gradient w.r.t. V
                    for (int j = 0; j < seqK; j++)
                    {
                        float w = weightsData[wOffset + i * seqK + j];
                        for (int d = 0; d < headDim; d++)
                        {
                            gradValueData[kvOffset + j * headDim + d] += w * gradOutputData[qOffset + i * headDim + d];
                        }
                    }

                    // Gradient w.r.t. weights
                    var gradWeights = new float[seqK];
                    for (int j = 0; j < seqK; j++)
                    {
                        float sum = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            sum += gradOutputData[qOffset + i * headDim + d] * valueData[kvOffset + j * headDim + d];
                        }
                        gradWeights[j] = sum;
                    }

                    // Gradient through softmax
                    float dotProduct = 0;
                    for (int j = 0; j < seqK; j++)
                    {
                        dotProduct += gradWeights[j] * weightsData[wOffset + i * seqK + j];
                    }

                    for (int j = 0; j < seqK; j++)
                    {
                        float w = weightsData[wOffset + i * seqK + j];
                        float gradScore = w * (gradWeights[j] - dotProduct) * scale;

                        for (int d = 0; d < headDim; d++)
                        {
                            gradQueryData[qOffset + i * headDim + d] += gradScore * keyData[kvOffset + j * headDim + d];
                            gradKeyData[kvOffset + j * headDim + d] += gradScore * queryData[qOffset + i * headDim + d];
                        }
                    }
                }
            }
        }

        UploadToBuffer(gradQuery, gradQueryData);
        UploadToBuffer(gradKey, gradKeyData);
        UploadToBuffer(gradValue, gradValueData);
    }

    #endregion

    #region Spatial Transformer Operations

    /// <summary>
    /// Generates an affine sampling grid.
    /// </summary>
    public void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        ThrowIfDisposed();

        var thetaData = DownloadBuffer(theta);
        var gridData = new float[batch * outputHeight * outputWidth * 2];

        for (int b = 0; b < batch; b++)
        {
            int thetaOffset = b * 6;
            float t00 = thetaData[thetaOffset + 0];
            float t01 = thetaData[thetaOffset + 1];
            float t02 = thetaData[thetaOffset + 2];
            float t10 = thetaData[thetaOffset + 3];
            float t11 = thetaData[thetaOffset + 4];
            float t12 = thetaData[thetaOffset + 5];

            for (int h = 0; h < outputHeight; h++)
            {
                for (int w = 0; w < outputWidth; w++)
                {
                    // Normalized coordinates [-1, 1]
                    float y = 2.0f * h / (outputHeight - 1) - 1.0f;
                    float x = 2.0f * w / (outputWidth - 1) - 1.0f;

                    // Apply affine transformation
                    float outX = t00 * x + t01 * y + t02;
                    float outY = t10 * x + t11 * y + t12;

                    int gridOffset = (b * outputHeight * outputWidth + h * outputWidth + w) * 2;
                    gridData[gridOffset + 0] = outX;
                    gridData[gridOffset + 1] = outY;
                }
            }
        }

        UploadToBuffer(grid, gridData);
    }

    /// <summary>
    /// Grid sample with bilinear interpolation.
    /// </summary>
    public void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        ThrowIfDisposed();

        var inputData = DownloadBuffer(input);
        var gridData = DownloadBuffer(grid);
        var outputData = new float[batch * channels * outHeight * outWidth];

        for (int b = 0; b < batch; b++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int gridIdx = (b * outHeight * outWidth + oh * outWidth + ow) * 2;
                    float x = gridData[gridIdx + 0];
                    float y = gridData[gridIdx + 1];

                    // Convert from [-1, 1] to pixel coordinates
                    float ih, iw;
                    if (alignCorners)
                    {
                        iw = (x + 1) * (inWidth - 1) / 2.0f;
                        ih = (y + 1) * (inHeight - 1) / 2.0f;
                    }
                    else
                    {
                        iw = ((x + 1) * inWidth - 1) / 2.0f;
                        ih = ((y + 1) * inHeight - 1) / 2.0f;
                    }

                    // Bilinear interpolation
                    int h0 = (int)MathF.Floor(ih);
                    int w0 = (int)MathF.Floor(iw);
                    int h1 = h0 + 1;
                    int w1 = w0 + 1;

                    float hLerp = ih - h0;
                    float wLerp = iw - w0;

                    for (int c = 0; c < channels; c++)
                    {
                        float v00 = GetPixelWithPadding(inputData, b, c, h0, w0, batch, channels, inHeight, inWidth, paddingMode);
                        float v01 = GetPixelWithPadding(inputData, b, c, h0, w1, batch, channels, inHeight, inWidth, paddingMode);
                        float v10 = GetPixelWithPadding(inputData, b, c, h1, w0, batch, channels, inHeight, inWidth, paddingMode);
                        float v11 = GetPixelWithPadding(inputData, b, c, h1, w1, batch, channels, inHeight, inWidth, paddingMode);

                        float value = (1 - hLerp) * (1 - wLerp) * v00 +
                                     (1 - hLerp) * wLerp * v01 +
                                     hLerp * (1 - wLerp) * v10 +
                                     hLerp * wLerp * v11;

                        int outIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;
                        outputData[outIdx] = value;
                    }
                }
            }
        }

        UploadToBuffer(output, outputData);
    }

    private static float GetPixelWithPadding(float[] data, int b, int c, int h, int w,
        int batch, int channels, int height, int width, int paddingMode)
    {
        if (h < 0 || h >= height || w < 0 || w >= width)
        {
            switch (paddingMode)
            {
                case 0: // Zeros
                    return 0;
                case 1: // Border
                    h = Math.Max(0, Math.Min(h, height - 1));
                    w = Math.Max(0, Math.Min(w, width - 1));
                    break;
                case 2: // Reflection
                    h = ReflectCoord(h, height);
                    w = ReflectCoord(w, width);
                    break;
            }
        }

        int idx = b * channels * height * width + c * height * width + h * width + w;
        return data[idx];
    }

    private static int ReflectCoord(int coord, int size)
    {
        if (size == 1) return 0;

        while (coord < 0 || coord >= size)
        {
            if (coord < 0)
                coord = -coord - 1;
            if (coord >= size)
                coord = 2 * size - coord - 1;
        }
        return coord;
    }

    /// <summary>
    /// Grid sample backward pass.
    /// </summary>
    public void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        ThrowIfDisposed();

        var gradOutputData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gridData = DownloadBuffer(grid);

        var gradInputData = new float[batch * channels * inHeight * inWidth];
        var gradGridData = new float[batch * outHeight * outWidth * 2];

        for (int b = 0; b < batch; b++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int gridIdx = (b * outHeight * outWidth + oh * outWidth + ow) * 2;
                    float x = gridData[gridIdx + 0];
                    float y = gridData[gridIdx + 1];

                    float ih, iw;
                    if (alignCorners)
                    {
                        iw = (x + 1) * (inWidth - 1) / 2.0f;
                        ih = (y + 1) * (inHeight - 1) / 2.0f;
                    }
                    else
                    {
                        iw = ((x + 1) * inWidth - 1) / 2.0f;
                        ih = ((y + 1) * inHeight - 1) / 2.0f;
                    }

                    int h0 = (int)MathF.Floor(ih);
                    int w0 = (int)MathF.Floor(iw);
                    int h1 = h0 + 1;
                    int w1 = w0 + 1;

                    float hLerp = ih - h0;
                    float wLerp = iw - w0;

                    float gradX = 0, gradY = 0;

                    for (int c = 0; c < channels; c++)
                    {
                        int outIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;
                        float gradOut = gradOutputData[outIdx];

                        // Gradient w.r.t. input
                        AccumulateInputGradient(gradInputData, b, c, h0, w0, (1 - hLerp) * (1 - wLerp) * gradOut,
                            batch, channels, inHeight, inWidth, paddingMode);
                        AccumulateInputGradient(gradInputData, b, c, h0, w1, (1 - hLerp) * wLerp * gradOut,
                            batch, channels, inHeight, inWidth, paddingMode);
                        AccumulateInputGradient(gradInputData, b, c, h1, w0, hLerp * (1 - wLerp) * gradOut,
                            batch, channels, inHeight, inWidth, paddingMode);
                        AccumulateInputGradient(gradInputData, b, c, h1, w1, hLerp * wLerp * gradOut,
                            batch, channels, inHeight, inWidth, paddingMode);

                        // Gradient w.r.t. grid
                        float v00 = GetPixelWithPadding(inputData, b, c, h0, w0, batch, channels, inHeight, inWidth, paddingMode);
                        float v01 = GetPixelWithPadding(inputData, b, c, h0, w1, batch, channels, inHeight, inWidth, paddingMode);
                        float v10 = GetPixelWithPadding(inputData, b, c, h1, w0, batch, channels, inHeight, inWidth, paddingMode);
                        float v11 = GetPixelWithPadding(inputData, b, c, h1, w1, batch, channels, inHeight, inWidth, paddingMode);

                        float dValueDw = (1 - hLerp) * (v01 - v00) + hLerp * (v11 - v10);
                        float dValueDh = (1 - wLerp) * (v10 - v00) + wLerp * (v11 - v01);

                        float dwDx = alignCorners ? (inWidth - 1) / 2.0f : inWidth / 2.0f;
                        float dhDy = alignCorners ? (inHeight - 1) / 2.0f : inHeight / 2.0f;

                        gradX += gradOut * dValueDw * dwDx;
                        gradY += gradOut * dValueDh * dhDy;
                    }

                    gradGridData[gridIdx + 0] = gradX;
                    gradGridData[gridIdx + 1] = gradY;
                }
            }
        }

        UploadToBuffer(gradInput, gradInputData);
        UploadToBuffer(gradGrid, gradGridData);
    }

    private static void AccumulateInputGradient(float[] gradInput, int b, int c, int h, int w, float grad,
        int batch, int channels, int height, int width, int paddingMode)
    {
        if (h < 0 || h >= height || w < 0 || w >= width)
        {
            if (paddingMode == 0) return; // zeros - no gradient to accumulate
            if (paddingMode == 1)
            {
                h = Math.Max(0, Math.Min(h, height - 1));
                w = Math.Max(0, Math.Min(w, width - 1));
            }
            else if (paddingMode == 2)
            {
                h = ReflectCoord(h, height);
                w = ReflectCoord(w, width);
            }
        }

        int idx = b * channels * height * width + c * height * width + h * width + w;
        gradInput[idx] += grad;
    }

    #endregion
}
