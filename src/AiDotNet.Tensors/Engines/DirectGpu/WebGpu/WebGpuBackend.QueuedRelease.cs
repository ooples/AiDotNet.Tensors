#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IQueuedBufferRelease
{
    /// <inheritdoc/>
    /// <remarks>The WebGPU queue keeps a destroyed buffer alive until the submissions that use it finish.</remarks>
    void IQueuedBufferRelease.ReleaseAfterQueuedWork(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        buffer.Dispose();
    }
}
#endif
