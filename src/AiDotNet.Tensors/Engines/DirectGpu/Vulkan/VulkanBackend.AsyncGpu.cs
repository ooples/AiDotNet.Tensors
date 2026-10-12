// Copyright (c) AiDotNet. All rights reserved.
// IAsyncGpuBackend over this backend's single in-order queue (#1142).
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend : IAsyncGpuBackend
{
    private IGpuStream? _inOrderDefaultStream;

    public bool SupportsMultiStream => false;
    public bool SupportsEvents => true;
    public bool SupportsAsyncTransfer => false;
    public bool SupportsGraphCapture => false;
    public int MaxConcurrentStreams => 1;

    public IGpuStream CreateStream(GpuStreamType streamType) => InOrderAsyncGpu.CreateStream(this, streamType, 0);
    public IGpuStream CreateStream(GpuStreamType streamType, int priority) => InOrderAsyncGpu.CreateStream(this, streamType, priority);
    public IGpuStream DefaultStream => _inOrderDefaultStream ??= InOrderAsyncGpu.CreateDefaultStream(this);

    public IGpuEvent CreateEvent() => new InOrderAsyncGpu.InOrderGpuEvent();
    public IGpuEvent CreateEvent(bool enableTiming) => new InOrderAsyncGpu.InOrderGpuEvent();
    public void RecordEvent(IGpuEvent gpuEvent, IGpuStream stream) => InOrderAsyncGpu.RecordEvent(this, gpuEvent, stream);
    public void StreamWaitEvent(IGpuStream stream, IGpuEvent gpuEvent) => InOrderAsyncGpu.StreamWaitEvent(stream, gpuEvent);

    public GpuSyncPoint CreateSyncPoint(IGpuStream stream) => InOrderAsyncGpu.CreateSyncPoint(stream);
    public GpuSyncPoint CreateSyncPoint() => InOrderAsyncGpu.CreateSyncPoint(DefaultStream);

    public void UploadBufferAsync(float[] data, IGpuBuffer buffer, IGpuStream stream)
        => InOrderAsyncGpu.UploadBufferAsync(this, data, buffer, stream);
    public void UploadBufferAsync(ReadOnlySpan<float> data, IGpuBuffer buffer, IGpuStream stream)
        => InOrderAsyncGpu.UploadBufferAsync(this, data.ToArray(), buffer, stream);
    public void DownloadBufferAsync(IGpuBuffer buffer, float[] destination, IGpuStream stream)
        => InOrderAsyncGpu.DownloadBufferAsync(this, buffer, destination, stream);
    public IGpuBuffer AllocateBufferAsync(float[] data, IGpuStream stream) => InOrderAsyncGpu.AllocateBufferAsync(this, data, stream);
    public void CopyBufferAsync(IGpuBuffer source, IGpuBuffer destination, int size, IGpuStream stream)
        => InOrderAsyncGpu.CopyBufferAsync(this, source, destination, size, stream);

    public void GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha, float beta, IGpuStream stream)
        => InOrderAsyncGpu.GemmAsync(this, A, B, C, M, N, K, alpha, beta, stream);
    public void FusedGemmBiasActivationAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output,
        int M, int N, int K, FusedActivationType activation, IGpuStream stream)
        => InOrderAsyncGpu.FusedGemmBiasActivationAsync(this, A, B, bias, output, M, N, K, activation, stream);
    public void SynchronizeStream(IGpuStream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        Synchronize();
    }

    public bool QueryStreamComplete(IGpuStream stream) => stream?.Query() ?? throw new ArgumentNullException(nameof(stream));
    public bool QueryEventComplete(IGpuEvent gpuEvent) => gpuEvent?.Query() ?? throw new ArgumentNullException(nameof(gpuEvent));
    public float GetEventElapsedTime(IGpuEvent start, IGpuEvent end) => InOrderAsyncGpu.GetEventElapsedTime(start, end);
}

