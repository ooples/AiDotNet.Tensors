using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class GpuBufferPoolAffinityTests
{
    [Fact]
    public void Rent_DoesNotCrossAffinityDomains()
    {
        using var pool = new GpuBufferPool<FakeBuffer>(maxPerSize: 4, maxSize: 1024);
        var queueA = GpuBufferPoolAffinity.ForNativeQueue(new IntPtr(1));
        var queueB = GpuBufferPoolAffinity.ForNativeQueue(new IntPtr(2));
        var buffer = new FakeBuffer(64);

        pool.Return(buffer, queueA);

        Assert.False(pool.TryRent(64, queueB, out _));
        Assert.True(pool.TryRent(64, queueA, out var rented));
        Assert.Same(buffer, rented);
    }

    [Fact]
    public void Capacity_RemainsSharedAcrossAffinityDomains()
    {
        using var pool = new GpuBufferPool<FakeBuffer>(maxPerSize: 1, maxSize: 1024);
        var queueA = GpuBufferPoolAffinity.ForNativeQueue(new IntPtr(1));
        var queueB = GpuBufferPoolAffinity.ForNativeQueue(new IntPtr(2));
        var first = new FakeBuffer(64);
        var excess = new FakeBuffer(64);

        pool.Return(first, queueA);
        pool.Return(excess, queueB);

        Assert.Equal(1, excess.ReleaseCount);
        Assert.False(pool.TryRent(64, queueB, out _));
        Assert.True(pool.TryRent(64, queueA, out var rented));
        Assert.Same(first, rented);
    }

    private sealed class FakeBuffer : IGpuBuffer, IPoolableGpuBuffer
    {
        internal FakeBuffer(int size) => Size = size;
        public int Size { get; }
        public long SizeInBytes => Size * sizeof(float);
        public IntPtr Handle => new(1);
        internal int ReleaseCount { get; private set; }
        public void MarkRented() { }
        public void Release() => ReleaseCount++;
        public void Dispose() => Release();
    }
}
