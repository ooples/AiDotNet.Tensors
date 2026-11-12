// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Op results release their device buffers from finalizers. All 6 built-in DirectGpu backends (CUDA, HIP, OpenCL,
/// Vulkan, Metal, WebGPU) now reclaim unreachable buffers, drain their pools, and retry once internally
/// (<see cref="IDeviceMemoryPressure.ReclaimsOnAllocationFailure"/> == <c>true</c>), while
/// <see cref="DirectGpuTensorEngine.AllocateReclaiming"/> preserves the finalizer-sweep fallback for custom/external
/// <see cref="IDirectGpuBackend"/> implementations that do not reclaim on their own without double-retrying built-in backends.
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

    [Fact]
    public void An_allocation_that_fits_after_finalizers_run_succeeds_on_a_non_reclaiming_backend()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());

        s_released = false;
        DropUnreachableHolder();
        int attempts = 0;
        using var buffer = DirectGpuTensorEngine.AllocateReclaiming(backend, b =>
        {
            attempts++;
            // Out of memory until the unreachable holder has been finalized - as when owned results are what fills
            // the device on an external backend without IDeviceMemoryPressure.ReclaimsOnAllocationFailure.
            if (!s_released) throw new InvalidOperationException("out of device memory");
            return b.AllocateBuffer(16);
        });

        Assert.True(s_released);
        Assert.Equal(2, attempts);
        Assert.NotEqual(IntPtr.Zero, buffer.Handle);
    }

    [Fact]
    public void A_failure_that_reclaiming_does_not_fix_still_propagates()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        int attempts = 0;
        Assert.Throws<ArgumentOutOfRangeException>(() => DirectGpuTensorEngine.AllocateReclaiming(backend, _ =>
        {
            attempts++;
            throw new ArgumentOutOfRangeException("size");
        }));
        Assert.Equal(2, attempts);
    }

    [Fact]
    public void Backend_AllocateWithRetry_collects_unreachable_holders_drains_pool_and_retries_once_on_oom()
    {
        s_released = false;
        DropUnreachableHolder();
        int attempts = 0;
        int drained = 0;

        int result = DeviceMemoryReclaim.AllocateWithRetry(
            () =>
            {
                attempts++;
                if (!s_released) throw new GpuOutOfMemoryException("out of device memory", -4);
                return 42;
            },
            ex => ex is GpuOutOfMemoryException,
            () => drained++);

        Assert.True(s_released);
        Assert.Equal(2, attempts);
        Assert.Equal(1, drained);
        Assert.Equal(42, result);

        int nonOomAttempts = 0;
        int nonOomDrained = 0;
        Assert.Throws<ArgumentOutOfRangeException>(() => DeviceMemoryReclaim.AllocateWithRetry<int>(
            () =>
            {
                nonOomAttempts++;
                throw new ArgumentOutOfRangeException("size");
            },
            ex => ex is GpuOutOfMemoryException,
            () => nonOomDrained++));
        Assert.Equal(1, nonOomAttempts);
        Assert.Equal(0, nonOomDrained);
    }

    [SkippableFact]
    public void Built_in_backend_reclaims_internally_so_engine_does_not_double_retry()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "OpenCL is not available.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");

        Assert.True(((IDeviceMemoryPressure)backend).ReclaimsOnAllocationFailure);
        int attempts = 0;
        Assert.Throws<ArgumentOutOfRangeException>(() => DirectGpuTensorEngine.AllocateReclaiming(backend, _ =>
        {
            attempts++;
            throw new ArgumentOutOfRangeException("size");
        }));
        Assert.Equal(1, attempts);
    }
}
