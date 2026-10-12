// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IQueuedBufferRelease
{
    /// <inheritdoc/>
    /// <remarks>HIP has no event-ordered release here, and hipFree does not wait for work queued on a non-default stream, so the queue drains before the buffer goes back to the allocator.</remarks>
    void IQueuedBufferRelease.ReleaseAfterQueuedWork(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        Synchronize();
        buffer.Dispose();
    }
}
