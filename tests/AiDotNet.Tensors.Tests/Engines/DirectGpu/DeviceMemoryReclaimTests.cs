// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
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

    // A CUDA allocation still out of memory after reclamation raised a bare InvalidOperationException, so a caller with
    // a lighter fallback (the fused-CE kernel when the chunked forward's scratch does not fit) could not tell it from a
    // real CUDA fault. It now raises the same GpuOutOfMemoryException as the OpenCL and Vulkan allocators.
    [Fact]
    public void CudaAllocationError_ClassifiesOutOfMemory_AndKeepsOtherErrorsGeneric()
    {
        var oom = AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.CudaAllocationError(
            AiDotNet.Tensors.Engines.CudaResult.OutOfMemory, "cuMemAllocAsync failed");
        var gpuOom = Assert.IsType<GpuOutOfMemoryException>(oom);
        Assert.Equal((int)AiDotNet.Tensors.Engines.CudaResult.OutOfMemory, gpuOom.NativeCode);
        Assert.Equal("cuMemAllocAsync failed", gpuOom.Message);

        var other = AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.CudaAllocationError(
            AiDotNet.Tensors.Engines.CudaResult.InvalidValue, "cuMemAlloc failed");
        Assert.IsType<InvalidOperationException>(other);
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
    public void InsideATape_AStepDroppingPastTheInTapeThreshold_CollectsWithoutWaitingForTheBoundary()
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        static WeakReference Abandon() => new WeakReference(new object());

        var tapeOpen = DeviceMemoryReclaim.TapeOpen;
        DeviceMemoryReclaim.TapeOpen = () => true;
        try
        {
            DeviceMemoryReclaim.ResetCountsForTests();
            var dead = Abandon();
            // Below the in-tape threshold a tape still waits for the step boundary.
            DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);
            Assert.True(DeviceMemoryReclaim.CollectionDue);
            // Past it, the step's dead results are collected now: a GPU step can drop gigabytes before its boundary.
            // The collection count proves this call collected; an unrelated GC could clear `dead` on its own.
            int collectionsBefore = GC.CollectionCount(1);
            DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectInsideTapeAfterDeviceBytes - DeviceMemoryReclaim.CollectAfterDeviceBytes);
            Assert.True(GC.CollectionCount(1) > collectionsBefore, "the in-tape threshold did not collect");
            Assert.False(dead.IsAlive);
            // What the tape still holds survived that collection, so the boundary sweep stays requested.
            Assert.True(DeviceMemoryReclaim.CollectionDue);
        }
        finally
        {
            DeviceMemoryReclaim.TapeOpen = tapeOpen;
            DeviceMemoryReclaim.ResetCountsForTests();
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

    [Fact]
    public void CudaBuffersReleasedAfterTheirQuantumDrained_AreCreditedAgainstTheGen2Count()
    {
        // Two 64 MiB buffers fill a 128 MiB quantum, which is forwarded to DeviceMemoryReclaim. Their releases used to
        // be dropped (the drain had moved the epoch on), so four released pairs still forced a Gen-2 collection.
        DeviceMemoryReclaim.ResetCountsForTests();
        CudaBackend.ResetCudaReclaimForTests();
        const long half = 64L << 20;
        for (int pair = 0; pair < 4; pair++)
        {
            long first = CudaBackend.ChargeCudaReclaim(half);
            long second = CudaBackend.ChargeCudaReclaim(half);
            Assert.True(first >= 0 && second >= 0, "a CUDA charge outside a capture is counted");
            CudaBackend.CreditCudaReclaim(half, first);
            CudaBackend.CreditCudaReclaim(half, second);
            // Checked every pair: the fourth uncredited pair would force the Gen-2 collection, which zeroes the count.
            Assert.Equal(0, DeviceMemoryReclaim.BytesSinceGen2CollectionForTests);
            Assert.Equal(0, CudaBackend.CudaReclaimPendingBytesForTests);
        }
    }

    [Fact]
    public void ACudaCreditFromADrainedQuantum_DoesNotCancelANewerCharge()
    {
        DeviceMemoryReclaim.ResetCountsForTests();
        CudaBackend.ResetCudaReclaimForTests();
        const long half = 64L << 20;
        long drained = CudaBackend.ChargeCudaReclaim(half);
        CudaBackend.ChargeCudaReclaim(half);   // fills the quantum: it drains and a new epoch starts
        CudaBackend.ChargeCudaReclaim(32L << 20);   // abandoned in the new epoch
        CudaBackend.CreditCudaReclaim(half, drained);
        Assert.Equal(32L << 20, CudaBackend.CudaReclaimPendingBytesForTests);
    }

    [Fact]
    public void FinalizedBuffers_AreCreditedAgainstTheGen2Count()
    {
        // Young collections that recover every abandoned buffer (their finalizers return them) leave nothing
        // unreleased, so the cumulative Gen-2 count must not keep growing toward a forced Gen-2 collection.
        DeviceMemoryReclaim.ResetCountsForTests();
        for (int step = 0; step < 3; step++)
        {
            long epoch = DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);
            Assert.True(DeviceMemoryReclaim.CollectIfDueAtStepBoundary());
            // The young collection leaves the Gen-2 count alone, so this step's allocation is still charged there.
            Assert.Equal(DeviceMemoryReclaim.CollectAfterDeviceBytes, DeviceMemoryReclaim.BytesSinceGen2CollectionForTests);
            DeviceMemoryReclaim.OnDeviceFinalized(DeviceMemoryReclaim.CollectAfterDeviceBytes, epoch);
        }
        Assert.Equal(0, DeviceMemoryReclaim.BytesSinceGen2CollectionForTests);
    }

    [Fact]
    public void GradientTapeDispose_CollectsStepIntermediatesAfterEntriesReset()
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        static WeakReference RecordUnpooledEntry(
            AiDotNet.Tensors.Engines.Autodiff.GradientTape<float> tape,
            AiDotNet.Tensors.LinearAlgebra.Tensor<float> x)
        {
            var unpooled = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 16 });
            tape.Record(new AiDotNet.Tensors.Engines.Autodiff.TapeEntry<float>
            {
                OperationName = "test_step_boundary",
                Output = unpooled,
                Input0 = x,
                InputCount = 1,
            });
            return new WeakReference(unpooled);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static WeakReference RecordForwardStepAndDispose()
        {
            var x = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 16 });
            WeakReference weakForward;
            using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
            {
                weakForward = RecordUnpooledEntry(tape, x);
                // Device allocations inside an open tape below the in-tape threshold defer to tape.Dispose()
                // rather than running a mid-step GC while _entries still roots unpooled.
                DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectInsideTapeAfterDeviceBytes - 1);
                Assert.True(DeviceMemoryReclaim.CollectionDue);
            }
            return weakForward;
        }

        DeviceMemoryReclaim.ResetCountsForTests();
        var weak = RecordForwardStepAndDispose();
        Assert.False(DeviceMemoryReclaim.CollectionDue);
        Assert.False(weak.IsAlive);
    }

    [Fact]
    public void AnOwnerTheTapeHeldAcrossAnInTapeCollection_IsCollectedAtTheStepBoundary()
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        static WeakReference RecordHeldEntry(
            AiDotNet.Tensors.Engines.Autodiff.GradientTape<float> tape,
            AiDotNet.Tensors.LinearAlgebra.Tensor<float> x)
        {
            var held = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 16 });
            tape.Record(new AiDotNet.Tensors.Engines.Autodiff.TapeEntry<float>
            {
                OperationName = "test_in_tape_survivor",
                Output = held,
                Input0 = x,
                InputCount = 1,
            });
            return new WeakReference(held);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static WeakReference StepWithAnInTapeCollection()
        {
            var x = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 16 });
            WeakReference weakHeld;
            using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
            {
                weakHeld = RecordHeldEntry(tape, x);
                int collectionsBefore = GC.CollectionCount(1);
                DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectInsideTapeAfterDeviceBytes);
                Assert.True(GC.CollectionCount(1) > collectionsBefore, "the in-tape threshold did not collect");
                // The tape still roots its entry: the collection promoted it instead of freeing it.
                Assert.True(weakHeld.IsAlive);
                Assert.True(DeviceMemoryReclaim.CollectionDue);
            }
            return weakHeld;
        }

        DeviceMemoryReclaim.ResetCountsForTests();
        var weak = StepWithAnInTapeCollection();
        // Disposing the tape released the entry and ran the boundary sweep through Gen 2, which reaches the promoted
        // owner.
        Assert.False(DeviceMemoryReclaim.CollectionDue);
        Assert.False(weak.IsAlive);
    }

    [Fact]
    public void StepBoundary_CumulativeMultiStepAllocations_ReclaimsGen2PromotedWrappers()
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        static (WeakReference Weak, object Root) CreateWrapper()
        {
            var obj = new object();
            return (new WeakReference(obj), obj);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static WeakReference CreateGen2PromotedWrapperAndRelease()
        {
            var (weak, root) = CreateWrapper();
            // Two blocking collections while rooted promote the wrapper into Generation 2,
            // mirroring live forward GPU buffers promoted by mid-step ephemeral GCs.
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            Assert.Equal(2, GC.GetGeneration(root));
            GC.KeepAlive(root);
            return weak;
        }

        var tapeOpen = DeviceMemoryReclaim.TapeOpen;
        DeviceMemoryReclaim.TapeOpen = () => true;
        try
        {
            DeviceMemoryReclaim.ResetCountsForTests();
            var weakGen2 = CreateGen2PromotedWrapperAndRelease();

            // Simulate 4 steps each abandoning 64 MB (< 256 MB per step, >= 256 MB cumulative):
            // Steps 1-3 run Gen-1 sweeps (resetting s_driverBytesSinceCollection each time), and Step 4
            // crosses CollectGen2AfterDeviceBytes (256 MB cumulative) and runs a Gen-2 sweep.
            for (int step = 0; step < 4; step++)
            {
                DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectAfterDeviceBytes);
                Assert.True(DeviceMemoryReclaim.CollectIfDueAtStepBoundary());
            }

            Assert.False(weakGen2.IsAlive);
        }
        finally
        {
            DeviceMemoryReclaim.TapeOpen = tapeOpen;
        }
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
