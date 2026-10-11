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
}
