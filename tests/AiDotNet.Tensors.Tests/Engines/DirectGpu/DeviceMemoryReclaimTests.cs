// Copyright (c) CrowdTrainer LLC. All rights reserved.

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

    // An inference call has no tape, so before #1432 the due collection ran inside a later call's allocation.
    [Fact]
    public void LeavingTheOutermostInferenceScope_RunsTheDueCollection()
    {
        var tapeOpen = DeviceMemoryReclaim.TapeOpen;
        DeviceMemoryReclaim.TapeOpen = () => false;
        try
        {
            // Bytes other tests left in the process-wide counters could pass CollectNowAfterDeviceBytes, which collects on
            // the spot and clears the flag this test reads.
            DeviceMemoryReclaim.ResetCountsForTests();
            DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);
            Assert.True(DeviceMemoryReclaim.CollectionDue);

            using (new AiDotNet.Tensors.Engines.Autodiff.InferenceModeScope<float>())
            {
                using (new AiDotNet.Tensors.Engines.Autodiff.InferenceModeScope<float>()) { }
                Assert.True(DeviceMemoryReclaim.CollectionDue);   // a nested scope is not the end of the call
            }
            Assert.False(DeviceMemoryReclaim.CollectionDue);
        }
        finally
        {
            DeviceMemoryReclaim.TapeOpen = tapeOpen;
            DeviceMemoryReclaim.CollectIfDueAtStepBoundary();
        }
    }

    [Fact]
    public void LeavingAnInferenceScopeInsideATape_LeavesTheCollectionToTheStepBoundary()
    {
        var tapeOpen = DeviceMemoryReclaim.TapeOpen;
        DeviceMemoryReclaim.TapeOpen = () => true;
        try
        {
            // Bytes other tests left in the process-wide counters could pass CollectNowAfterDeviceBytes, which collects on
            // the spot and clears the flag this test reads.
            DeviceMemoryReclaim.ResetCountsForTests();
            DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);
            using (new AiDotNet.Tensors.Engines.Autodiff.InferenceModeScope<float>()) { }
            Assert.True(DeviceMemoryReclaim.CollectionDue);
        }
        finally
        {
            DeviceMemoryReclaim.TapeOpen = tapeOpen;
            DeviceMemoryReclaim.CollectIfDueAtStepBoundary();
        }
    }

    [Fact]
    public void StepBoundary_SkipsTheCollection_WhenOwnersDisposedWhatWasHandedOut()
    {
        // A TensorScope ending an inference call disposes its intermediates: they are back in the pool and credited, so
        // the collection their allocation asked for would recover nothing (#1432: it ran on every ResNet-50 predict).
        DeviceMemoryReclaim.ResetCountsForTests();
        long epoch = DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);
        Assert.True(DeviceMemoryReclaim.CollectionDue);
        DeviceMemoryReclaim.OnDeviceRelease(DeviceMemoryReclaim.CollectAfterDeviceBytes, epoch);
        Assert.False(DeviceMemoryReclaim.CollectIfDueAtStepBoundary());
        Assert.False(DeviceMemoryReclaim.CollectionDue);
    }

    [Fact]
    public void StepBoundary_Collects_WhenAbandonedBytesRemainPastTheThreshold()
    {
        DeviceMemoryReclaim.ResetCountsForTests();
        long epoch = DeviceMemoryReclaim.OnDeviceAllocation(2 * DeviceMemoryReclaim.CollectAfterDeviceBytes);
        DeviceMemoryReclaim.OnDeviceRelease(DeviceMemoryReclaim.CollectAfterDeviceBytes / 2, epoch);   // some disposed, most abandoned
        Assert.True(DeviceMemoryReclaim.CollectIfDueAtStepBoundary());
        Assert.False(DeviceMemoryReclaim.CollectionDue);
    }

    [Fact]
    public void ADisposeAfterACollection_DoesNotCreditTheChargeTheCollectionAlreadyDropped()
    {
        // A buffer whose charge was counted before a collection reset the count must not be credited against bytes
        // charged after it: that would hide an abandoned allocation from the step-boundary collection.
        DeviceMemoryReclaim.ResetCountsForTests();
        long early = DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);
        Assert.True(DeviceMemoryReclaim.CollectIfDueAtStepBoundary());   // resets the count: a new epoch
        DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);   // abandoned
        DeviceMemoryReclaim.OnDeviceRelease(DeviceMemoryReclaim.CollectAfterDeviceBytes, early);
        Assert.True(DeviceMemoryReclaim.CollectionDue);
        Assert.True(DeviceMemoryReclaim.CollectIfDueAtStepBoundary());
    }

    [Fact]
    public void TheChargeThatSetsOffACollection_IsNotCreditedAfterIt()
    {
        // The allocation that pushes the count past CollectNowAfterDeviceBytes collects at once and zeroes the count;
        // its later dispose has nothing left to credit.
        DeviceMemoryReclaim.ResetCountsForTests();
        long trigger = DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectNowAfterDeviceBytes);
        Assert.False(DeviceMemoryReclaim.CollectionDue);
        DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);   // abandoned
        DeviceMemoryReclaim.OnDeviceRelease(DeviceMemoryReclaim.CollectNowAfterDeviceBytes, trigger);
        Assert.True(DeviceMemoryReclaim.CollectIfDueAtStepBoundary());
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
