using System;
using AiDotNet.Tensors.Engines;
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

    // After the pool is disposed, a returned block is freed only in a live context lifetime; when that context is
    // gone it took the block, so the block is no longer counted either way. Before, a skipped release left it counted.
    [SkippableFact]
    public void Return_AfterDispose_UncountsTheBlock_WhetherFreedOrTakenByTheContext()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var owner = new CudaBackend();
        IntPtr context = owner.ContextHandle;
        long generation = owner.ContextGeneration;
        var pool = new CudaPinnedBufferPool();
        pool.BindContext(context, generation);
        IntPtr block;
        Assert.Equal(CudaResult.Success, CuBlasNative.cuCtxPushCurrent(context));
        try { block = pool.Rent(4096); }
        finally { CuBlasNative.cuCtxPopCurrent(out _); }
        pool.Dispose();
        owner.Dispose();
        long before = PinnedCurrent();
        pool.Return(block, 4096);
        Assert.Equal(before - 4096, PinnedCurrent());
    }

    // A free that fails (here: an address that is no block) leaves the block counted, in the full-bucket branch.
    [SkippableFact]
    public void Return_ToAFullBucket_CountsAFreeOnlyWhenItSucceeded()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var keeper = new CudaBackend();
        using var pool = new CudaPinnedBufferPool();
        pool.BindContext(keeper.ContextHandle, keeper.ContextGeneration);
        Assert.Equal(CudaResult.Success, CuBlasNative.cuCtxPushCurrent(keeper.ContextHandle));
        try
        {
            var blocks = new IntPtr[8];
            for (int i = 0; i < blocks.Length; i++) blocks[i] = pool.Rent(4096);
            foreach (var b in blocks) pool.Return(b, 4096);   // the bucket is full
            long before = PinnedCurrent();
            pool.Return((IntPtr)0x1000, 4096);   // not a block: its free fails
            Assert.Equal(before, PinnedCurrent());
        }
        finally { CuBlasNative.cuCtxPopCurrent(out _); }
    }

    private static long PinnedCurrent()
        => AiDotNet.Tensors.Engines.DevicePrimitives.GpuMemoryStats.StatsByAllocator().TryGetValue("cuda_pinned_host", out var s)
            ? s["allocated_bytes.current"] : 0;
}
