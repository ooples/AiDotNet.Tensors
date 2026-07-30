// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Direct GPU optimizer kernels whose Adam moment state is stored below fp32.
/// </summary>
public interface ICompressedMomentGpuOptimizerBackend : IDirectGpuBackend
{
    /// <summary>
    /// Adam update where the first and second moment buffers are stored as
    /// packed bfloat16 values. Parameters and gradients remain fp32.
    /// </summary>
    void AdamUpdateBf16(
        IGpuBuffer param,
        IGpuBuffer gradient,
        IGpuBuffer m,
        IGpuBuffer v,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        int step,
        int size);

    /// <summary>
    /// AdamW update where the first and second moment buffers are stored as
    /// packed bfloat16 values. Parameters and gradients remain fp32, and weight
    /// decay is applied using AdamW's decoupled update.
    /// </summary>
    void AdamWUpdateBf16(
        IGpuBuffer param,
        IGpuBuffer gradient,
        IGpuBuffer m,
        IGpuBuffer v,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        int step,
        int size);

    /// <summary>
    /// Adam update where first and second moments are stored as one byte per
    /// element plus one scale per quantization block. Implementations keep the
    /// update in place on the supplied buffers; native-kernel backends should not
    /// download state to the host.
    /// </summary>
    void Adam8BitUpdate(
        IGpuBuffer param,
        IGpuBuffer gradient,
        IGpuBuffer mQuant,
        IGpuBuffer vQuant,
        IGpuBuffer mScales,
        IGpuBuffer vScales,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float oneMinusBeta1,
        float oneMinusBeta2,
        float biasCorrection1,
        float biasCorrection2,
        int blockSize,
        int paramLength,
        int numBlocks);
}
