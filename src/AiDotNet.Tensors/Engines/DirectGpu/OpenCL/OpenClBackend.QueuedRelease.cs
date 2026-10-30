// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IQueuedBufferRelease
{
    /// <inheritdoc/>
    /// <remarks>clReleaseMemObject defers the free until the commands enqueued on the object complete.</remarks>
    void IQueuedBufferRelease.ReleaseAfterQueuedWork(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        buffer.Dispose();
    }
}
