// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IQueuedBufferRelease
{
    /// <inheritdoc/>
    /// <remarks>Every dispatch outside a batch submits and waits on its fence, so no queued work outlives the call.</remarks>
    void IQueuedBufferRelease.ReleaseAfterQueuedWork(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        buffer.Dispose();
    }
}
