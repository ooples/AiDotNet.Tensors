using System;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The step I/O, the cache-append fault record and the overlapped-upload slots release their page-locked memory through
/// the live-context guard (counted in HostReleasesAttempted). Before, each called cuMemFreeHost directly (the step I/O
/// not at all once its backend was disposed), so none of these releases reached the guard.
/// </summary>
[Collection("DirectGpuSerial")]
public class PageLockedReleaseRoutingTests
{
    private static long Attempted() => Interlocked.Read(ref CudaBackend.HostReleasesAttempted);

    // Releases a backend's own Dispose makes with nothing extra allocated: the baseline the cases below add to.
    private static long DisposeAttempts(Action<CudaBackend> use)
    {
        var backend = new CudaBackend();
        use(backend);
        long before = Attempted();
        backend.Dispose();
        return Attempted() - before;
    }

    [SkippableFact]
    public void TheCacheAppendFaultRecord_IsReleasedThroughTheGuard()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        long baseline = DisposeAttempts(_ => { });
        long withRecord = DisposeAttempts(backend =>
        {
            using var cache = backend.AllocateBuffer(1 * 4 * 2);
            using var position = backend.AllocateBuffer(new[] { 0f });
            using var value = backend.AllocateBuffer(new[] { 1f, 2f });
            using var mask = backend.AllocateBuffer(4);
            ((IKvCacheAppendBackend)backend).KvCacheAppend(cache, position, value, mask, batch: 1, capacity: 4, width: 2);
            backend.Synchronize();
        });
        Assert.Equal(baseline + 1, withRecord);
    }

    [SkippableFact]
    public void OverlappedUploadSlots_AreReleasedThroughTheGuard()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        long baseline = DisposeAttempts(_ => { });
        bool uploaded = false;
        long withSlot = DisposeAttempts(backend =>
        {
            var destination = backend.AllocateBuffer(256);   // released with the backend
            uploaded = ((IOverlappedUploadBackend)backend).TryUploadOverlapped(new float[256], destination);
            backend.Synchronize();
        });
        Skip.IfNot(uploaded, "The backend declined the overlapped upload.");
        Assert.Equal(baseline + 1, withSlot);   // one slot used: one page-locked block
    }

    [SkippableFact]
    public void StepIoDisposedAfterItsBackend_IsReleasedWhileTheContextLives()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var keeper = new CudaBackend();   // holds the shared primary context
        var owner = new CudaBackend();
        var input = owner.AllocateBuffer(64);
        var stepIo = ((IGpuGraphCapture)owner).CreateStepIo(new[] { input }, new[] { 64 });
        if (stepIo is null)
        {
            owner.Dispose();
            Skip.If(true, "Graph step I/O is unavailable on this device.");
            return;
        }
        owner.Dispose();
        Assert.True(CudaBackend.IsLiveContext(keeper.ContextHandle, keeper.ContextGeneration));
        long before = Attempted();
        stepIo.Dispose();
        // Two staging slots for the input, the results and the masks.
        Assert.True(Attempted() - before >= 4, $"{Attempted() - before} page-locked releases reached the guard");
    }
}
