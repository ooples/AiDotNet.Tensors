// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IQueuedBufferRelease
{
    /// <inheritdoc/>
    /// <remarks>A committed command buffer retains every MTLBuffer it binds.</remarks>
    void IQueuedBufferRelease.ReleaseAfterQueuedWork(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        buffer.Dispose();
    }
}
