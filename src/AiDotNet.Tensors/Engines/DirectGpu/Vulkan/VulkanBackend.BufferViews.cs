// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IGpuBufferViews
{
    /// <inheritdoc/>
    // Offsets must be 256-byte multiples, and each view allocates a staging buffer.
    bool IGpuBufferViews.ViewsAreFreeAtAnyOffset => false;

    /// <inheritdoc/>
    /// <remarks>
    /// The view binds the parent's VkBuffer at a byte offset, so kernels, copies and transfers address only its range.
    /// An offset that is not a multiple of 256 bytes (the largest storage-buffer offset alignment a device may
    /// require) yields null, as the contract allows.
    /// </remarks>
    IGpuBuffer? IGpuBufferViews.TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount)
    {
        if (parent is not VulkanGpuBuffer owner) return null;
        if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > parent.Size) return null;
        var storage = VulkanBuffer.CreateView(owner.Storage, elementOffset, elementCount);
        if (storage is null) return null;
        var staging = VulkanBuffer.CreateStagingBuffer(elementCount);
        if (staging is null) return null;
        return new VulkanGpuBuffer(storage, staging, elementCount, owner);
    }
}
