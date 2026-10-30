// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IQueuedBufferRelease
{
    // Buffers released while a batch was recording: its dispatches are not submitted yet, so they are disposed
    // only after EndBatch has submitted and waited.
    private readonly System.Collections.Generic.List<IGpuBuffer> _releaseAfterBatch = new();

    /// <inheritdoc/>
    /// <remarks>Every dispatch outside a batch submits and waits on its fence, so no queued work outlives the call;
    /// inside a batch the release waits for <see cref="EndBatch"/>.</remarks>
    void IQueuedBufferRelease.ReleaseAfterQueuedWork(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        lock (_releaseAfterBatch)
        {
            if (_batchMode)
            {
                _releaseAfterBatch.Add(buffer);
                return;
            }
        }
        buffer.Dispose();
    }

    private void ReleaseBuffersHeldForBatch()
    {
        IGpuBuffer[] held;
        lock (_releaseAfterBatch)
        {
            held = _releaseAfterBatch.ToArray();
            _releaseAfterBatch.Clear();
        }
        foreach (var buffer in held) buffer.Dispose();
    }
}