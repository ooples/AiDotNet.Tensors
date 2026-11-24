// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Training BatchNorm followed by a LeakyReLU of positive slope, forward and backward, each in the launches of the
/// BatchNorm+ReLU kernels (the slope replacing ReLU's zero). The output equals BatchNorm then LeakyReLU bit for bit, and
/// the backward masks by the LeakyReLU output's sign (the sign of its input for a positive slope), or, given beta, by
/// the pre-activation rebuilt from x. False -- nothing launched -- for a shape or slope the kernels do not take.
/// </summary>
internal interface IBatchNormLeakyReluBackend
{
    bool TryBatchNormLeakyReluTraining(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveVariance, int batch, int channels, int spatial, float epsilon, float slope);

    bool TryBatchNormLeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer leakyOutput, IGpuBuffer input,
        IGpuBuffer gamma, IGpuBuffer? beta, IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput,
        IGpuBuffer gradGamma, IGpuBuffer gradBeta, int batch, int channels, int spatial, float epsilon, float slope,
        IGpuBuffer? gradOutput2, bool accumulateGradInput);
}
