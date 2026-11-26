using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
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
}
