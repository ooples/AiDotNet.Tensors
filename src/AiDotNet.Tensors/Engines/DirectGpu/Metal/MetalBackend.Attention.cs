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
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal,
        float softcap = 0.0f, int numKVHeads = 0)
    {
        ThrowIfDisposed();
        // numKVHeads <= 0 means MHA (K/V have numHeads); >0 enables Grouped-Query Attention where each KV head is
        // shared by numHeads/numKVHeads query heads and the K/V buffers are sized [batch * numKVHeads * seqK * headDim].
        int kvHeads = numKVHeads > 0 ? numKVHeads : numHeads;
        if (numHeads % kvHeads != 0)
            throw new ArgumentException("numHeads must be an integer multiple of numKVHeads.", nameof(numKVHeads));
        int outputCount = checked(batch * numHeads * seqQ * headDim);
        int weightCount = checked(batch * numHeads * seqQ * seqK);
        if (outputCount <= 0) return;
        int sharedMaskCount = checked(seqQ * seqK);
        if (mask is not null && mask.Size < sharedMaskCount)
            throw new ArgumentException("The attention mask must contain at least seqQ * seqK elements.", nameof(mask));
        uint maskMode = mask is null ? 0u : mask.Size >= weightCount ? 2u : 1u;
        using var outputTemporary = AllocateBuffer(outputCount);
        using var weightsTemporary = attentionWeights is not null ? AllocateBuffer(weightCount) : null;
        using var unusedWeights = attentionWeights is null ? AllocateBuffer(1) : null;
        using var unusedMask = mask is null ? AllocateBuffer(1) : null;
        DispatchResidentMetal("attention_forward_serial", 1,
            new[] { query, key, value, outputTemporary, weightsTemporary ?? unusedWeights!, mask ?? unusedMask! },
            (uint)batch, (uint)numHeads, (uint)seqQ, (uint)seqK, (uint)headDim, unchecked((uint)SingleToInt32BitsCompat(scale)),
            isCausal ? 1u : 0u, attentionWeights is null ? 0u : 1u, maskMode, unchecked((uint)SingleToInt32BitsCompat(softcap)),
            (uint)kvHeads);
        Copy(outputTemporary, output, outputCount);
        if (attentionWeights is not null) Copy(weightsTemporary!, attentionWeights, weightCount);
    }

    /// <summary>
    /// Fused interleaved RoPE (GPT-NeoX / LLaMA / GGML). Rotates each adjacent dim pair of every [rows, headDim] row.
    /// </summary>
    public void RopeInterleaved(IGpuBuffer input, IGpuBuffer cos, IGpuBuffer sin, IGpuBuffer output,
        int rows, int headDim, int seqLen, int startPosition)
    {
        ThrowIfDisposed();
        if (rows <= 0 || headDim <= 0 || seqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "RoPE dimensions must be positive.");
        if ((headDim & 1) != 0)
            throw new ArgumentException("RoPE requires an even head dimension.", nameof(headDim));
        int total = checked(rows * headDim);
        if (input.Size < total || output.Size < total)
            throw new ArgumentException("RoPE input/output buffers are smaller than rows * headDim.");
        int pairs = checked(rows * (headDim / 2));
        DispatchResidentMetal("rope_interleaved", pairs,
            new[] { input, cos, sin, output },
            (uint)rows, (uint)headDim, (uint)seqLen, (uint)startPosition);
    }

    /// <summary>
    /// Scaled dot-product attention backward pass.
    /// </summary>
    public void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights,
            gradQuery, gradKey, gradValue, batch, numHeads, numHeads, seqQ, seqK,
            headDim, scale, 1);
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
        ScaledDotProductAttention(query, key, value, output, null, mask,
            batch, numHeads, seqLen, seqLen, headDim, scale, isCausal);
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
        int outputCount = checked(batch * numHeads * seqQ * headDim);
        int statsCount = checked(batch * numHeads * seqQ);
        if (outputCount <= 0) return;
        using var outputTemporary = AllocateBuffer(outputCount);
        using var statsTemporary = AllocateBuffer(statsCount);
        using var unusedBias = attentionBias is null ? AllocateBuffer(1) : null;
        DispatchResidentMetal("flash_attention_forward_serial", 1,
            new[] { query, key, value, outputTemporary, statsTemporary, attentionBias ?? unusedBias! },
            (uint)batch, (uint)numHeads, (uint)seqQ, (uint)seqK, (uint)headDim, unchecked((uint)SingleToInt32BitsCompat(scale)),
            isCausal ? 1u : 0u, attentionBias is null ? 0u : 1u, (uint)biasBatchStride);
        Copy(outputTemporary, output, outputCount);
        Copy(statsTemporary, softmaxStats, statsCount);
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
        int queryCount = checked(batch * numHeads * seqQ * headDim);
        int keyCount = checked(batch * numHeads * seqK * headDim);
        if (queryCount <= 0 || keyCount <= 0) return;
        using var queryGradient = AllocateBuffer(queryCount);
        using var keyGradient = AllocateBuffer(keyCount);
        using var valueGradient = AllocateBuffer(keyCount);
        using var unusedBias = attentionBias is null ? AllocateBuffer(1) : null;
        DispatchResidentMetal("flash_attention_backward_serial", 1,
            new[] { gradOutput, query, key, value, softmaxStats, attentionBias ?? unusedBias!,
                queryGradient, keyGradient, valueGradient },
            (uint)batch, (uint)numHeads, (uint)seqQ, (uint)seqK, (uint)headDim, unchecked((uint)SingleToInt32BitsCompat(scale)),
            isCausal ? 1u : 0u, attentionBias is null ? 0u : 1u, (uint)biasBatchStride);
        Copy(queryGradient, gradQuery, queryCount);
        Copy(keyGradient, gradKey, keyCount);
        Copy(valueGradient, gradValue, keyCount);
    }

    /// <summary>
    /// Grouped query attention forward pass.
    /// </summary>
    public void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
    {
        ThrowIfDisposed();
        if (numKVHeads <= 0 || numQHeads % numKVHeads != 0)
            throw new ArgumentException("KV heads must be a positive divisor of query heads.", nameof(numKVHeads));
        int outputCount = checked(batch * numQHeads * seqQ * headDim);
        int weightCount = checked(batch * numQHeads * seqQ * seqK);
        if (outputCount <= 0) return;
        using var outputTemporary = AllocateBuffer(outputCount);
        using var weightsTemporary = attentionWeights is not null ? AllocateBuffer(weightCount) : null;
        using var unusedWeights = attentionWeights is null ? AllocateBuffer(1) : null;
        DispatchResidentMetal("grouped_attention_forward_serial", 1,
            new[] { query, key, value, outputTemporary, weightsTemporary ?? unusedWeights! },
            (uint)batch, (uint)numQHeads, (uint)numKVHeads, (uint)seqQ, (uint)seqK, (uint)headDim,
            unchecked((uint)SingleToInt32BitsCompat(scale)), isCausal ? 1u : 0u, attentionWeights is null ? 0u : 1u);
        Copy(outputTemporary, output, outputCount);
        if (attentionWeights is not null) Copy(weightsTemporary!, attentionWeights, weightCount);
    }

    /// <summary>
    /// Grouped query attention backward pass.
    /// </summary>
    public void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale,
        int numQueriesPerKV)
    {
        ThrowIfDisposed();
        if (numQueriesPerKV <= 0)
            throw new ArgumentOutOfRangeException(nameof(numQueriesPerKV));
        int queryCount = checked(batch * numQHeads * seqQ * headDim);
        int keyCount = checked(batch * numKVHeads * seqK * headDim);
        if (queryCount <= 0 || keyCount <= 0) return;
        using var queryGradient = AllocateBuffer(queryCount);
        using var keyGradient = AllocateBuffer(keyCount);
        using var valueGradient = AllocateBuffer(keyCount);
        DispatchResidentMetal("grouped_attention_backward_serial", 1,
            new[] { gradOutput, query, key, value, attentionWeights, queryGradient, keyGradient, valueGradient },
            (uint)batch, (uint)numQHeads, (uint)numKVHeads, (uint)seqQ, (uint)seqK, (uint)headDim,
            unchecked((uint)SingleToInt32BitsCompat(scale)), (uint)numQueriesPerKV);
        Copy(queryGradient, gradQuery, queryCount);
        Copy(keyGradient, gradKey, keyCount);
        Copy(valueGradient, gradValue, keyCount);
    }

    #endregion

    #region Spatial Transformer Operations

    /// <summary>
    /// Generates an affine sampling grid.
    /// </summary>
    public void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        ThrowIfDisposed();
        int count = checked(batch * outputHeight * outputWidth);
        if (count <= 0) return;
        bool aliasesTheta = ReferenceEquals(theta, grid);
        using var temporary = aliasesTheta ? AllocateBuffer(checked(count * 2)) : null;
        IGpuBuffer target = temporary ?? grid;
        DispatchResidentMetal("affine_grid", count, new[] { theta, target },
            (uint)batch, (uint)outputHeight, (uint)outputWidth);
        if (temporary is not null) Copy(temporary, grid, checked(count * 2));
    }

    /// <summary>
    /// Grid sample with bilinear interpolation.
    /// </summary>
    public void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        ThrowIfDisposed();
        int count = checked(batch * channels * outHeight * outWidth);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(output, input) || ReferenceEquals(output, grid);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("grid_sample", count, new[] { input, grid, target },
            (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth, (uint)outHeight, (uint)outWidth,
            (uint)paddingMode, alignCorners ? 1u : 0u);
        if (temporary is not null) Copy(temporary, output, count);
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
        int inputCount = checked(batch * channels * inHeight * inWidth);
        int gridCount = checked(batch * outHeight * outWidth * 2);
        if (inputCount <= 0 || gridCount <= 0) return;
        using var inputGradient = AllocateBuffer(inputCount);
        using var gridGradient = AllocateBuffer(gridCount);
        DispatchResidentMetal("grid_sample_backward_serial", 1,
            new[] { gradOutput, input, grid, inputGradient, gridGradient },
            (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth, (uint)outHeight, (uint)outWidth,
            (uint)paddingMode, alignCorners ? 1u : 0u);
        Copy(inputGradient, gradInput, inputCount);
        Copy(gridGradient, gradGrid, gridCount);
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
