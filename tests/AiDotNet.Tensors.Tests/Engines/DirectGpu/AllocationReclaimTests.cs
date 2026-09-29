using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Op results release their device buffers from finalizers. CUDA drains finalizers and retries a failed allocation
/// itself; the other backends did not, so an allocation that would fit once unreachable results were collected failed
/// outright. The engine's output allocator now reclaims and retries once on any non-CUDA backend.
/// </summary>
[Collection("DirectGpuSerial")]
public class AllocationReclaimTests
{
    private static volatile bool s_released;

    private sealed class HeldUntilFinalized
    {
        ~HeldUntilFinalized() => s_released = true;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void DropUnreachableHolder() => _ = new HeldUntilFinalized();

    [SkippableFact]
    public void An_allocation_that_fits_after_finalizers_run_succeeds_on_a_non_cuda_backend()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");

        s_released = false;
        DropUnreachableHolder();
        int attempts = 0;
        using var buffer = DirectGpuTensorEngine.AllocateReclaiming(backend, b =>
        {
            attempts++;
            // Out of memory until the unreachable holder has been finalized - as when owned results are what fills
            // the device.
            if (!s_released) throw new InvalidOperationException("out of device memory");
            return b.AllocateBuffer(16);
        });

        Assert.True(s_released);
        Assert.Equal(2, attempts);
        Assert.NotEqual(IntPtr.Zero, buffer.Handle);
    }

    [SkippableFact]
    public void A_failure_that_reclaiming_does_not_fix_still_propagates()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        int attempts = 0;
        Assert.Throws<ArgumentOutOfRangeException>(() => DirectGpuTensorEngine.AllocateReclaiming(backend, _ =>
        {
            attempts++;
            throw new ArgumentOutOfRangeException("size");
        }));
        Assert.Equal(2, attempts);
    }
}
