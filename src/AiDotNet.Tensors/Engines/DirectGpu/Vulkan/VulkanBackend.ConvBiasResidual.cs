namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IConvBiasResidualBackend
{
    bool IConvBiasResidualBackend.TryConvBiasResidual(in ConvBiasResidualArgs args)
        => ConvBiasResidualComposition.TryExecute(this, args);
}

