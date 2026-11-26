using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The pinned pool's power-of-two buckets stop at 1 GiB. Above it the int rounding used to wrap to int.MinValue, and
/// the allocation then asked the driver for ~1.8e19 bytes; a request that large is refused before any allocation.
/// No device is needed: the refusal happens before the driver is called.
/// </summary>
[Collection("DirectGpuSerial")]
public class CudaPinnedBufferPoolSizingTests
{
    [Theory]
    [InlineData(1, 1L)]
    [InlineData(4097, 8192L)]
    [InlineData(1 << 30, 1L << 30)]
    [InlineData((1 << 30) + 1, 1L << 31)]
    [InlineData(int.MaxValue, 1L << 31)]
    public void BucketSize_RoundsUpToAPowerOfTwo_WithoutWrapping(int request, long bucket)
        => Assert.Equal(bucket, CudaPinnedBufferPool.BucketSize(request));

    [Fact]
    public void Rent_OverOneGiB_IsRefusedBeforeAnyAllocation()
    {
        using var pool = new CudaPinnedBufferPool();
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => pool.Rent((1 << 30) + 1));
        Assert.Contains("own allocation", error.Message);
    }
    // An int byte count that overflowed is negative; zero is no block. Both are refused (BucketSize rounded them to 1).
    [Theory]
    [InlineData(0)]
    [InlineData(-4)]
    [InlineData(int.MinValue)]
    public void Rent_OfANonPositiveSize_IsRefused(int request)
    {
        using var pool = new CudaPinnedBufferPool();
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.Rent(request));
    }

    // After the pool is disposed, a returned block is freed only in a live context lifetime. A pool bound to none (or
    // to one that was destroyed) makes no driver call and changes no accounting: the context's destruction took the
    // memory back. Before, it called cuMemFreeHost regardless and recorded the free.
    [Fact]
    public void Return_AfterDispose_WithoutALiveContext_MakesNoDriverCall()
    {
        var pool = new CudaPinnedBufferPool();
        pool.BindContext((IntPtr)0x1234, generation: long.MaxValue);   // never a registered generation
        pool.Dispose();
        long before = PinnedCurrent();
        pool.Return((IntPtr)0x1000, 4096);   // not a real block: a driver free of it would fail or fault
        Assert.Equal(before, PinnedCurrent());
        Assert.Null(CudaBackend.ReleaseHostInLiveContext((IntPtr)0x1000, (IntPtr)0x1234, long.MaxValue, unregister: false));
    }

    private static long PinnedCurrent()
        => AiDotNet.Tensors.Engines.DevicePrimitives.GpuMemoryStats.StatsByAllocator().TryGetValue("cuda_pinned_host", out var s)
            ? s["allocated_bytes.current"] : 0;
}
