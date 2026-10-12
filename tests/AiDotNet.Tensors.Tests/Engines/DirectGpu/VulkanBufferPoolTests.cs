using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A released Vulkan buffer is kept and handed to the next allocation of exactly the same size instead of being
/// destroyed and created again. Reuse must never change what a caller sees: the right size, the uploaded data, and no
/// sharing between two live buffers.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanBufferPoolTests
{
    // The backend is a process-wide singleton other test classes share, so nothing here disposes it.
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize()) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device.");
        Skip.IfNot(VulkanGpuBuffer.PoolingEnabled, "Vulkan buffer pooling is disabled by the environment.");
        return backend!;
    }

    // Sizes no other test is likely to allocate, so the pool's entries for them are this test's own.
    private const int OddSize = 77_773;

    [SkippableFact]
    public void ReleasedBuffer_IsReused_ForTheSameSizeOnly()
    {
        var backend = BackendOrSkip();
        var first = backend.AllocateBuffer(OddSize);
        IntPtr handle = first.Handle;
        first.Dispose();

        using var other = backend.AllocateBuffer(OddSize + 1);
        Assert.NotEqual(handle, other.Handle);
        Assert.Equal(OddSize + 1, other.Size);

        using var again = backend.AllocateBuffer(OddSize);
        Assert.Equal(handle, again.Handle);
        Assert.Equal(OddSize, again.Size);
    }

    [SkippableFact]
    public void TwoLiveBuffers_OfOneSize_NeverShareStorage()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 11;
        var a = backend.AllocateBuffer(size);
        a.Dispose();
        using var b = backend.AllocateBuffer(size);
        using var c = backend.AllocateBuffer(size);
        Assert.NotEqual(b.Handle, c.Handle);
        backend.Fill(b, 1.5f, size);
        backend.Fill(c, -2.5f, size);
        Assert.All(backend.DownloadBuffer(b), v => Assert.Equal(1.5f, v));
        Assert.All(backend.DownloadBuffer(c), v => Assert.Equal(-2.5f, v));
    }

    [SkippableFact]
    public void ReusedBuffer_CarriesTheNewUpload_NotTheOldContents()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 23;
        var old = backend.AllocateBuffer(Enumerable.Repeat(9f, size).ToArray());
        IntPtr handle = old.Handle;
        old.Dispose();

        float[] data = Enumerable.Range(0, size).Select(i => i * 0.25f).ToArray();
        using var reused = backend.AllocateBuffer(data);
        Assert.Equal(handle, reused.Handle);
        Assert.Equal(data, backend.DownloadBuffer(reused));

        // Int buffers share the pool: the bits must survive the same way.
        reused.Dispose();
        int[] ints = Enumerable.Range(0, size).Select(i => i * 7 - 3).ToArray();
        using var asInts = backend.AllocateIntBuffer(ints);
        float[] bits = backend.DownloadBuffer(asInts);
        for (int i = 0; i < size; i++) Assert.Equal(ints[i], BitConverter.ToInt32(BitConverter.GetBytes(bits[i]), 0));
    }

    [SkippableFact]
    public void DisposingTwice_ParksTheBufferOnce()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 37;
        var buffer = backend.AllocateBuffer(size);
        buffer.Dispose();
        buffer.Dispose();
        using var x = backend.AllocateBuffer(size);
        using var y = backend.AllocateBuffer(size);
        // A double return would hand the same storage to both.
        Assert.NotEqual(x.Handle, y.Handle);
    }

    [SkippableFact]
    public void AStaleReference_CannotReleaseTheAllocationsNextOwner()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 71;
        var stale = backend.AllocateBuffer(size);
        stale.Dispose();
        using var owner = backend.AllocateBuffer(size); // the same allocation, under a new object
        Assert.NotSame(stale, owner);
        stale.Dispose();
        using var other = backend.AllocateBuffer(size);
        // Had the stale reference parked the allocation again, this would be the owner's storage.
        Assert.NotEqual(owner.Handle, other.Handle);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static IntPtr AllocateAndDrop(VulkanBackend backend, int size) => backend.AllocateBuffer(size).Handle;

    [SkippableFact]
    public void ADroppedBuffer_WhoseAllocationSatInThePoolThroughACollection_ReturnsAtTheNextYoungCollection()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 83;
        var first = backend.AllocateBuffer(size);
        IntPtr handle = first.Handle;
        first.Dispose();
        // The parked allocation survives collections; whatever object carries it is now in the oldest generation.
        GC.Collect();
        GC.Collect();

        // A result nobody disposes: only its finalizer can return the allocation.
        Assert.Equal(handle, AllocateAndDrop(backend, size));
        GC.Collect(0, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();

        // The queued return runs on the next allocation, so this one already finds it.
        using var again = backend.AllocateBuffer(size);
        Assert.Equal(handle, again.Handle);
    }

    // Runs a loop that drops every result, with the tape check answered by the test (other test classes open tapes
    // in parallel, and the real check is process-wide). Returns how many of the allocations went to the driver.
    private static long DriverAllocationsOfADroppingLoop(VulkanBackend backend, int size, int allocations, bool tapeOpen)
    {
        var realCheck = DeviceMemoryReclaim.TapeOpen;
        DeviceMemoryReclaim.TapeOpen = () => tapeOpen;
        try
        {
            long before = VulkanGpuBuffer.DriverAllocations;
            for (int i = 0; i < allocations; i++) AllocateAndDrop(backend, size);
            return VulkanGpuBuffer.DriverAllocations - before;
        }
        finally
        {
            DeviceMemoryReclaim.TapeOpen = realCheck;
        }
    }

    [SkippableFact]
    public void ALoopWithNoTape_ThatDropsItsResults_StopsGoingToTheDriverAfterAFewHundredAllocations()
    {
        var backend = BackendOrSkip();
        const int allocations = 1500;
        long driver = DriverAllocationsOfADroppingLoop(backend, OddSize + 97, allocations, tapeOpen: false);
        // A pool miss collects once 256 buffers have gone out, and the collected ones serve the allocations after it.
        // Without that the loop is collected only by the byte threshold (about 2,000 allocations of this size in), so
        // nearly all 1,500 would be driver allocations.
        Assert.InRange(driver, 1, 2 * DeviceMemoryReclaim.CollectOnMissAfterAllocations + 64);
    }

    [SkippableFact]
    public void WithATapeOpen_APoolMissDoesNotCollect()
    {
        BackendOrSkip();
        var realCheck = DeviceMemoryReclaim.TapeOpen;
        try
        {
            // Enough allocations counted that only the tape check can be what declines.
            for (int i = 0; i < 2 * DeviceMemoryReclaim.CollectOnMissAfterAllocations; i++) DeviceMemoryReclaim.OnDeviceAllocation(0);
            DeviceMemoryReclaim.TapeOpen = () => true;
            int collections = GC.CollectionCount(1);
            Assert.False(DeviceMemoryReclaim.CollectOnPoolMiss());
            DeviceMemoryReclaim.TapeOpen = () => false;
            Assert.True(DeviceMemoryReclaim.CollectOnPoolMiss());
            Assert.True(GC.CollectionCount(1) > collections);
            // The count starts again after a collection.
            Assert.False(DeviceMemoryReclaim.CollectOnPoolMiss());
        }
        finally
        {
            DeviceMemoryReclaim.TapeOpen = realCheck;
        }
    }

    [SkippableFact]
    public void OpIntoAPooledBuffer_GivesTheSameResultAsIntoAFreshOne()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 41;
        float[] data = Enumerable.Range(0, size).Select(i => (i % 13) - 6.5f).ToArray();
        using var input = backend.AllocateBuffer(data);
        var dirty = backend.AllocateBuffer(size);
        backend.Fill(dirty, float.NaN, size);
        dirty.Dispose();
        using var output = backend.AllocateBuffer(size); // the NaN-filled storage, reused
        backend.Abs(input, output, size);
        float[] result = backend.DownloadBuffer(output);
        for (int i = 0; i < size; i++) Assert.Equal(Math.Abs(data[i]), result[i]);
    }

    [SkippableFact]
    public void WithPoolingOff_AReleasedBufferIsFreed()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 53;
        int parkedBefore = VulkanGpuBuffer.Pool.PooledCount;
        VulkanGpuBuffer.PoolingEnabled = false;
        try
        {
            backend.AllocateBuffer(size).Dispose();
            Assert.Equal(parkedBefore, VulkanGpuBuffer.Pool.PooledCount);
        }
        finally
        {
            VulkanGpuBuffer.PoolingEnabled = true;
        }
        backend.AllocateBuffer(size).Dispose();
        Assert.Equal(parkedBefore + 1, VulkanGpuBuffer.Pool.PooledCount);
    }

    [SkippableFact]
    public void DrainAll_FreesParkedBuffers_AndAllocationStillWorks()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 67;
        backend.AllocateBuffer(size).Dispose();
        Assert.True(VulkanGpuBuffer.Pool.PooledCount >= 1);
        VulkanGpuBuffer.Pool.DrainAll();
        Assert.Equal(0, VulkanGpuBuffer.Pool.PooledCount);
        using var fresh = backend.AllocateBuffer(size);
        backend.Fill(fresh, 3f, size);
        Assert.Equal(3f, backend.DownloadBuffer(fresh)[size - 1]);
    }

    // The backend's dispose releases its long-lived pooled buffers (the unused-binding placeholder) through
    // ReleaseAndDrain, which must leave none of them parked: a parked buffer would outlive the device.
    [SkippableFact]
    public void ReleaseAndDrain_LeavesNoReleasedBufferParked()
    {
        var backend = BackendOrSkip();
        const int size = OddSize + 79;

        // Control: released after a drain, a pooled buffer is parked again.
        var late = backend.AllocateBuffer(size);
        VulkanGpuBuffer.ReleaseAndDrain();
        late.Dispose();
        Assert.Equal(1, VulkanGpuBuffer.Pool.PooledCount);

        var last = backend.AllocateBuffer(size);   // rents the parked buffer back
        VulkanGpuBuffer.ReleaseAndDrain(last);
        Assert.Equal(0, VulkanGpuBuffer.Pool.PooledCount);
    }
}
