// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Normalization, Dropout, and Embedding operations.

using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Normalization Operations

    /// <summary>
    /// Batch normalization forward pass.
    /// </summary>
    public void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        ThrowIfDisposed();
        if (batch <= 0 || channels <= 0 || spatialSize <= 0) return;
        DispatchResidentMetal("batch_norm_forward_serial_channels", channels,
            new[] { input, output, gamma, beta, runningMean, runningVar, saveMean, saveInvVar },
            (uint)batch, (uint)channels, (uint)spatialSize, unchecked((uint)SingleToInt32BitsCompat(epsilon)), unchecked((uint)SingleToInt32BitsCompat(momentum)), training ? 1u : 0u);
    }

    public bool TryFusedBatchNormActivation(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training,
        FusedActivationType activation) => false;

    /// <summary>
    /// Batch normalization backward pass.
    /// </summary>
    public void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        ThrowIfDisposed();
        if (batch <= 0 || channels <= 0 || spatialSize <= 0) return;
        DispatchResidentMetal("batch_norm_backward_serial_channels", channels,
            new[] { gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta },
            (uint)batch, (uint)channels, (uint)spatialSize);
    }

    /// <summary>
    /// Layer normalization forward pass.
    /// </summary>
    public void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        ThrowIfDisposed();
        if (batchSize <= 0 || normalizedSize <= 0) return;
        DispatchResidentMetal("layer_norm_forward_serial_rows", batchSize,
            new[] { input, output, gamma, beta, saveMean, saveInvVar },
            (uint)batchSize, (uint)normalizedSize, unchecked((uint)SingleToInt32BitsCompat(epsilon)));
    }

    /// <summary>
    /// Layer normalization backward pass.
    /// </summary>
    public void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        ThrowIfDisposed();
        if (batchSize <= 0 || normalizedSize <= 0) return;
        DispatchResidentMetal("layer_norm_backward_serial", 1,
            new[] { gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta },
            (uint)batchSize, (uint)normalizedSize);
    }

    /// <summary>
    /// Group normalization forward pass.
    /// </summary>
    public void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        ThrowIfDisposed();
        if (batch <= 0 || channels <= 0 || spatialSize <= 0) return;
        if (numGroups <= 0 || channels % numGroups != 0)
            throw new ArgumentException("The group count must be a positive divisor of channels.", nameof(numGroups));
        DispatchResidentMetal("group_norm_forward_serial_groups", checked(batch * numGroups),
            new[] { input, output, gamma, beta, saveMean, saveInvVar },
            (uint)batch, (uint)numGroups, (uint)channels, (uint)spatialSize, unchecked((uint)SingleToInt32BitsCompat(epsilon)));
    }

    /// <summary>
    /// Instance normalization forward pass.
    /// </summary>
    public void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        ThrowIfDisposed();
        if (batch <= 0 || channels <= 0 || spatialSize <= 0) return;
        DispatchResidentMetal("instance_norm_forward_serial_channels", checked(batch * channels),
            new[] { input, output, gamma, beta, saveMean, saveInvVar },
            (uint)batch, (uint)channels, (uint)spatialSize, unchecked((uint)SingleToInt32BitsCompat(epsilon)));
    }

    /// <summary>
    /// Instance normalization backward pass.
    /// </summary>
    public void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        ThrowIfDisposed();
        if (batch <= 0 || channels <= 0 || spatialSize <= 0) return;
        DispatchResidentMetal("instance_norm_backward_serial", 1,
            new[] { gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta },
            (uint)batch, (uint)channels, (uint)spatialSize);
    }

    /// <summary>
    /// RMS normalization forward pass.
    /// </summary>
    public void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        ThrowIfDisposed();
        if (batchSize <= 0 || normalizedSize <= 0) return;
        DispatchResidentMetal("rms_norm_forward_serial_rows", batchSize,
            new[] { input, output, gamma, saveRms },
            (uint)batchSize, (uint)normalizedSize, unchecked((uint)SingleToInt32BitsCompat(epsilon)));
    }

    /// <summary>
    /// RMS normalization backward pass.
    /// </summary>
    public void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        ThrowIfDisposed();
        if (batchSize <= 0 || normalizedSize <= 0) return;
        DispatchResidentMetal("rms_norm_backward_serial", 1,
            new[] { gradOutput, input, gamma, saveRms, gradInput, gradGamma },
            (uint)batchSize, (uint)normalizedSize);
    }

    #endregion

    #region Dropout and Regularization

    /// <summary>
    /// Dropout forward pass.
    /// </summary>
    public void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        bool sharedOutputs = ReferenceEquals(output, mask);
        bool maskAliasesInput = ReferenceEquals(mask, input);
        using var outputTemporary = sharedOutputs ? AllocateBuffer(size) : null;
        using var maskTemporary = sharedOutputs || maskAliasesInput ? AllocateBuffer(size) : null;
        IGpuBuffer outputTarget = outputTemporary ?? output;
        IGpuBuffer maskTarget = maskTemporary ?? mask;
        DispatchResidentMetal("dropout_dotnet_random_serial", 1,
            new[] { input, outputTarget, maskTarget }, (uint)size, unchecked((uint)SingleToInt32BitsCompat(dropoutRate)),
            (uint)(seed & 0x7ffffffful), training ? 1u : 0u);
        if (outputTemporary is not null) Copy(outputTemporary, output, size);
        if (maskTemporary is not null) Copy(maskTemporary, mask, size);
    }

    /// <summary>
    /// Dropout backward pass.
    /// </summary>
    public void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
    {
        ThrowIfDisposed();
        Multiply(gradOutput, mask, gradInput, size);
    }

    public bool TryFusedBiasDropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer bias, IGpuBuffer mask,
        int rows, int cols, float dropoutRate, float scale) => false;

    #endregion

    #region Embedding Operations

    /// <summary>
    /// Embedding lookup forward pass.
    /// </summary>
    public void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        ThrowIfDisposed();
        int count = checked(numIndices * embeddingDim);
        if (count <= 0) return;
        int vocabulary = embeddingTable.Size / embeddingDim;
        bool aliasesTable = ReferenceEquals(output, embeddingTable);
        using var temporary = aliasesTable ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("embedding_lookup", count, new[] { indices, embeddingTable, target },
            (uint)numIndices, (uint)embeddingDim, (uint)vocabulary);
        if (temporary is not null) Copy(temporary, output, count);
    }

    /// <summary>
    /// Embedding backward pass.
    /// </summary>
    public void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        ThrowIfDisposed();
        int count = checked(vocabSize * embeddingDim);
        if (count <= 0) return;
        bool aliasesGradient = ReferenceEquals(gradEmbedding, gradOutput);
        using var temporary = aliasesGradient ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? gradEmbedding;
        DispatchResidentMetal("embedding_backward_deterministic", count,
            new[] { gradOutput, indices, target }, (uint)numIndices, (uint)embeddingDim, (uint)vocabSize);
        if (temporary is not null) Copy(temporary, gradEmbedding, count);
    }

    /// <summary>
    /// Downloads an integer buffer.
    /// </summary>

    #endregion
}
