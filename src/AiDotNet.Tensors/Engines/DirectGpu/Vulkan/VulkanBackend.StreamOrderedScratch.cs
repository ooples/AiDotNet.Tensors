// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IStreamOrderedScratch
{
    /// <remarks>Allocation here is not stream-ordered, so an ordinary buffer serves.</remarks>
    IGpuBuffer IStreamOrderedScratch.AllocateStreamOrderedScratchBuffer(int size) => AllocateBuffer(size);
}
