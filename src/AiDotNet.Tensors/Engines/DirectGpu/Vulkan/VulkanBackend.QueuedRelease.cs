// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IQueuedBufferRelease
{
    // Buffers released while a batch was recording: its dispatches are not submitted yet, so they are disposed
    // only after EndBatch has submitted and waited.
    private readonly System.Collections.Generic.List<IGpuBuffer> _releaseAfterBatch = new();
    // Buffers whose host->device staging copy was recorded into the open batch command buffer: keep strong references
    // so the GC cannot finalize them mid-batch, and defer pooling/destruction if Dispose() is called before EndBatch.
    private readonly System.Collections.Generic.List<VulkanGpuBuffer> _batchUploadedBuffers = new();
    private readonly System.Collections.Generic.List<VulkanGpuBuffer> _disposedBatchUploads = new();
    private bool _batchUploadActive;

    private void BeginBatchBufferHold()
    {
        lock (_releaseAfterBatch)
        {
            _batchUploadActive = true;
        }
    }

    private void HoldBatchUploadBuffer(VulkanGpuBuffer buffer)
    {
        lock (_releaseAfterBatch)
        {
            if (!_batchUploadActive) return;
            buffer.HoldForBatchUpload(this);
            _batchUploadedBuffers.Add(buffer);
        }
    }

    internal bool TryDeferDisposedBatchUpload(VulkanGpuBuffer buffer)
    {
        lock (_releaseAfterBatch)
        {
            if (_batchUploadActive)
            {
                _disposedBatchUploads.Add(buffer);
                return true;
            }
        }
        return false;
    }

    /// <inheritdoc/>
    /// <remarks>Every dispatch outside a batch submits and waits on its fence, so no queued work outlives the call;
    /// inside a batch the release waits for <see cref="EndBatch"/>.</remarks>
    void IQueuedBufferRelease.ReleaseAfterQueuedWork(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        lock (_releaseAfterBatch)
        {
            if (_batchUploadActive || _batchMode)
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
        VulkanGpuBuffer[] uploaded;
        VulkanGpuBuffer[] disposedUploads;
        lock (_releaseAfterBatch)
        {
            _batchUploadActive = false;
            held = _releaseAfterBatch.ToArray();
            _releaseAfterBatch.Clear();
            uploaded = _batchUploadedBuffers.ToArray();
            _batchUploadedBuffers.Clear();
            disposedUploads = _disposedBatchUploads.ToArray();
            _disposedBatchUploads.Clear();
        }
        foreach (var buffer in uploaded) buffer.ClearBatchUploadHold();
        foreach (var buffer in disposedUploads) VulkanGpuBuffer.CompleteDeferredBatchDispose(buffer);
        foreach (var buffer in held) buffer.Dispose();
    }
}