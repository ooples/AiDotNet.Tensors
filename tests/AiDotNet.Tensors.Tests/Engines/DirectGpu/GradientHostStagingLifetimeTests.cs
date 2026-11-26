using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using GpuStreamType = AiDotNet.Tensors.Engines.Gpu.GpuStreamType;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A plan's page-locked gradient staging can outlive the backend that made it. Its memory is released only in the
/// context lifetime it was allocated in: while another backend still holds the device's shared primary context it is
/// freed (and the free is counted); a free is never attempted against a destroyed context.
/// </summary>
[Collection("DirectGpuSerial")]
public class GradientHostStagingLifetimeTests
{
    private const int Floats = 1 << 20;   // 4 MB

    private static long PinnedCurrent()
        => GpuMemoryStats.StatsByAllocator().TryGetValue("cuda_pinned_host", out var s) ? s["allocated_bytes.current"] : 0;

    [SkippableFact]
    public void StagingDisposedAfterItsBackend_IsFreedWhileTheContextLives()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var keeper = new CudaBackend();   // holds the primary context
        var owner = new CudaBackend();
        var staging = ((IGradientHostStagingSource)owner).TryCreateGradientHostStaging(Floats);
        Assert.NotNull(staging);
        staging!.Host[Floats - 1] = 1f;
        owner.Dispose();
        long before = PinnedCurrent();
        staging.Dispose();
        Assert.Equal(before - (long)Floats * sizeof(float), PinnedCurrent());
        Assert.Throws<ObjectDisposedException>(() => staging.Host.Length);
    }
    // The owner was the context's last holder: the context is gone, so the release must make no driver call and
    // count nothing (before, it called cuMemFreeHost against the destroyed context and counted the free). When some
    // other engine in the process still holds the primary context, the release frees and counts instead.
    [SkippableFact]
    public void StagingDisposedAfterItsOwner_IsUncounted_AndSkipsTheFreeOnlyWhenTheContextIsGone()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var owner = new CudaBackend();
        IntPtr context = owner.ContextHandle;
        long generation = owner.ContextGeneration;
        var staging = ((IGradientHostStagingSource)owner).TryCreateGradientHostStaging(Floats);
        Assert.NotNull(staging);
        owner.Dispose();
        bool live = CudaBackend.IsLiveContext(context, generation);
        long before = PinnedCurrent();
        long skippedBefore = System.Threading.Interlocked.Read(ref CudaBackend.HostReleasesSkipped);
        staging!.Dispose();
        // Freed or taken by the context, the memory is gone from the process: uncounted either way.
        Assert.Equal(before - (long)Floats * sizeof(float), PinnedCurrent());
        // Gone: the release was skipped before any driver call. Live (another engine holds the primary context): freed.
        Assert.Equal(live ? skippedBefore : skippedBefore + 1, System.Threading.Interlocked.Read(ref CudaBackend.HostReleasesSkipped));
    }

    // A transfer larger than a pooled pinned block (1 GiB) copies synchronously instead of throwing (the pool refuses
    // it), and round-trips bitwise.
    [SkippableFact]
    public void AsyncTransfersOverOneGiB_FallBackToASynchronousCopy()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var backend = new CudaBackend();
        const int n = (1 << 28) + 1024;   // 1 GiB + 4 KB of floats
        Skip.IfNot(GpuMemoryStats.TryGetDeviceMemory(backend, out long free, out _) && free > 3L * (1L << 30),
            "Needs more than 3 GiB of free device memory.");
        var data = new float[n];
        for (int i = 0; i < n; i += 4093) data[i] = i * 0.5f;
        data[n - 1] = -7f;
        using var stream = backend.CreateStream(GpuStreamType.Compute);
        using var buffer = backend.AllocateBuffer(n);
        long before = System.Threading.Interlocked.Read(ref CudaBackend.UnpooledTransferCount);
        backend.UploadBufferAsync(data, buffer, stream);
        var back = new float[n];
        backend.DownloadBufferAsync(buffer, back, stream);
        Assert.Equal(before + 2, System.Threading.Interlocked.Read(ref CudaBackend.UnpooledTransferCount));
        for (int i = 0; i < n; i += 4093) Assert.Equal(data[i], back[i]);
        Assert.Equal(-7f, back[n - 1]);
    }

    // A readback disposed after its backend, while another backend holds the primary context, frees its page-locked
    // bytes (before, it returned without freeing them, and they leaked for the context's life).
    [SkippableFact]
    public void ReadbackDisposedAfterItsBackend_IsFreedWhileTheContextLives()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var keeper = new CudaBackend();
        var owner = new CudaBackend();
        var readback = ((IDeviceReadbackFactory)owner).CreateReadback(4096);
        owner.Dispose();
        Assert.True(CudaBackend.IsLiveContext(keeper.ContextHandle, keeper.ContextGeneration));
        long attempted = System.Threading.Interlocked.Read(ref CudaBackend.HostReleasesAttempted);
        readback.Dispose();
        Assert.Equal(attempted + 1, System.Threading.Interlocked.Read(ref CudaBackend.HostReleasesAttempted));
    }
}
