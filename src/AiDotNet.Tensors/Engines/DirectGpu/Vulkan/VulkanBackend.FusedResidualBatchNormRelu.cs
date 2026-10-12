// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IFusedResidualBatchNormRelu
{
    /// <inheritdoc/>
    /// <remarks>Add, inference BatchNorm and ReLU as three device ops.</remarks>
    bool IFusedResidualBatchNormRelu.TryResidualBatchNormRelu(IGpuBuffer input, IGpuBuffer residual, IGpuBuffer output,
        IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer runningMean, IGpuBuffer runningVar, int batch, int channels,
        int spatialSize, float epsilon)
        => PortableResidualBatchNormRelu.Run(this, input, residual, output, gamma, beta, runningMean, runningVar,
            batch, channels, spatialSize, epsilon);
}
