using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The pinned pool's power-of-two buckets stop at 1 GiB. Above it the int rounding used to wrap to int.MinValue, and
/// the allocation then asked the driver for ~1.8e19 bytes; a request that large is refused before any allocation.
/// No device is needed: the refusal happens before the driver is called.
/// </summary>
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
}
