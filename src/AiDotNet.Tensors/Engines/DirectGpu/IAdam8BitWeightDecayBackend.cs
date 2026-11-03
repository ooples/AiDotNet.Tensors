// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The int8 block-quantized Adam update with weight decay, in the same kernel launch: coupled L2 decay for Adam
/// (<c>g + weightDecay * param</c>, as <c>torch.optim.Adam(weight_decay=...)</c>) or AdamW's decoupled decay
/// (<c>param *= 1 - learningRate * weightDecay</c> before the Adam step, as <c>torch.optim.AdamW</c> and
/// bitsandbytes' <c>AdamW8bit</c>).
/// </summary>
/// <remarks>
/// A separate interface rather than a member of <see cref="ICompressedMomentGpuOptimizerBackend"/>, which has
/// shipped: adding a member to it would break implementers outside this assembly. Every in-box backend implements
/// both.
/// </remarks>
public interface IAdam8BitWeightDecayBackend : ICompressedMomentGpuOptimizerBackend
{
    /// <summary>
    /// <see cref="ICompressedMomentGpuOptimizerBackend.Adam8BitUpdate"/> with weight decay applied inside the update.
    /// </summary>
    /// <param name="weightDecay">The decay coefficient; 0 is plain Adam.</param>
    /// <param name="decoupledWeightDecay">True for AdamW's decoupled decay, false for L2 decay added to the gradient.</param>
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
        int numBlocks,
        float weightDecay,
        bool decoupledWeightDecay);
}
