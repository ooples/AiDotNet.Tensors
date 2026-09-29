// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// With no activation cache evicting live data, device memory of a result nothing references is freed by its
/// buffer's finalizer (queued, run on the next allocation), and an allocation that runs out collects, runs those
/// frees and retries once. Before this, the HIP/OpenCL/Vulkan/Metal/WebGPU buffers had no finalizer at all, so an
/// undisposed buffer's device memory was never freed.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class DeviceMemoryReclaimTests
{
    [Fact]
    public void AllocateWithRetry_ReclaimsOnceThenSucceeds()
    {
        int attempts = 0, reclaims = 0;
        int result = DeviceMemoryReclaim.AllocateWithRetry(() =>
        {
            if (++attempts == 1) throw new GpuOutOfMemoryException("out of memory", -4);
            return 42;
        }, ex => ex is GpuOutOfMemoryException, () => reclaims++);

        Assert.Equal(42, result);
        Assert.Equal(2, attempts);
        Assert.Equal(1, reclaims);
    }

    [Fact]
    public void AllocateWithRetry_SecondFailurePropagates_AndOtherErrorsAreNotRetried()
    {
        int attempts = 0;
        Assert.Throws<GpuOutOfMemoryException>(() => DeviceMemoryReclaim.AllocateWithRetry<int>(() =>
        {
            attempts++;
            throw new GpuOutOfMemoryException("out of memory", -4);
        }, ex => ex is GpuOutOfMemoryException, () => { }));
        Assert.Equal(2, attempts);

        attempts = 0;
        Assert.Throws<InvalidOperationException>(() => DeviceMemoryReclaim.AllocateWithRetry<int>(() =>
        {
            attempts++;
            throw new InvalidOperationException("invalid size");
        }, ex => ex is GpuOutOfMemoryException, () => { }));
        Assert.Equal(1, attempts);
    }

    [Fact]
    public void FreeQueue_RunsEveryFree_EvenWhenOneThrows()
    {
        var queue = new DeviceFreeQueue();
        int ran = 0;
        queue.Enqueue(() => ran++);
        queue.Enqueue(() => throw new InvalidOperationException("context gone"));
        queue.Enqueue(() => ran++);

        Assert.Equal(2, queue.Drain());
        Assert.Equal(2, ran);
        Assert.True(queue.IsEmpty);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AbandonBuffer(IDirectGpuBackend backend) => backend.AllocateBuffer(1024);

    [SkippableFact]
    public void OpenCl_AbandonedBuffer_IsFreedByTheNextAllocation()
    {
        OpenClBackend? backend = null;
        try { backend = new OpenClBackend(); } catch { }
        Skip.If(backend is null || !backend.IsAvailable, "No OpenCL device.");
        using var _ = backend;
        DirectOpenClGpuBuffer.PendingFrees.Drain();

        AbandonBuffer(backend!);
        DeviceMemoryReclaim.CollectUnreachable();
        Assert.False(DirectOpenClGpuBuffer.PendingFrees.IsEmpty);   // the finalizer queued the free

        using var next = backend!.AllocateBuffer(1024);              // the next allocation runs it
        Assert.True(DirectOpenClGpuBuffer.PendingFrees.IsEmpty);
    }

    [SkippableFact]
    public void Vulkan_AbandonedBuffer_IsFreedByTheNextAllocation()
    {
        VulkanBackend? backend = null;
        try { backend = VulkanBackend.Instance; } catch { }
        Skip.If(backend is null || !backend.IsAvailable, "No Vulkan device.");
        VulkanGpuBuffer.PendingFrees.Drain();

        AbandonBuffer(backend!);
        DeviceMemoryReclaim.CollectUnreachable();
        Assert.False(VulkanGpuBuffer.PendingFrees.IsEmpty);

        using var next = backend!.AllocateBuffer(1024);
        Assert.True(VulkanGpuBuffer.PendingFrees.IsEmpty);
    }
}
