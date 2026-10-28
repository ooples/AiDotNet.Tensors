// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IQueuedBufferRelease
{
    /// <inheritdoc/>
    /// <remarks>Freed in stream order: the allocator reuses the memory only after the queued work.</remarks>
    void IQueuedBufferRelease.ReleaseAfterQueuedWork(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        FreeBufferDeferred(buffer);
    }
}
