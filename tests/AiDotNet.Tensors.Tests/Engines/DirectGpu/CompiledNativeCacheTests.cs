using System;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompiledNativeCacheTests
{
    private sealed class Cell : IGpuBuffer, IPoolableGpuBuffer
    {
        private static int s_next;
        internal Cell(int capacity) { Capacity = Size = capacity; Handle = (IntPtr)Interlocked.Increment(ref s_next); }
        public int Capacity { get; }
        public int Size { get; private set; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle { get; private set; }
        public void MarkRented(int size) { Assert.InRange(size, 1, Capacity); Size = size; }
        public void Release() => Handle = IntPtr.Zero;
        public void Dispose() => Release();
    }

    [Fact]
    public void OnlyAnExplicitMatchingBackendScopeBorrowsTheLargerClass()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        var foreign = MockDirectGpuBackend.Create(new MockBackendState());
        using var pool = new GpuBufferPool<Cell>(4, 1024);
        var cell = new Cell(32); pool.Return(cell);
        Assert.False(pool.TryRentForBackend(16, backend, out _));
        using (CompiledNativeCacheScope.Enter(foreign)) Assert.False(pool.TryRentForBackend(16, backend, out _));
        using (CompiledNativeCacheScope.Enter(backend))
        {
            Assert.True(pool.TryRentForBackend(16, backend, out var rented));
            Assert.Same(cell, rented);
            Assert.Equal(16, rented!.Size);
            Assert.Equal(32, rented.Capacity);
            Assert.Equal(1, pool.CompiledBestFitRents);
            pool.Return(rented);
        }
        Assert.False(pool.TryRentForBackend(16, backend, out _));
        Assert.True(pool.TryRent(32, out var restored));
        Assert.Same(cell, restored); restored!.Release();
    }

    [Fact]
    public void TwofoldCapacityLimitAndExactSizePoolsRemainEnforced()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var scope = CompiledNativeCacheScope.Enter(backend);
        using var pool = new GpuBufferPool<Cell>(4, 1024);
        pool.Return(new Cell(64));
        Assert.False(pool.TryRentForBackend(16, backend, out _));
        Assert.Equal(0, pool.CompiledBestFitRents);
        using var exact = new GpuBufferPool<Cell>(4, 1024, exactSizes: true);
        exact.Return(new Cell(32));
        Assert.False(exact.TryRentForBackend(16, backend, out _));
    }

    [Fact]
    public void AForeignQueueCannotBorrowAnotherQueuesAllocation()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var scope = CompiledNativeCacheScope.Enter(backend);
        var a = GpuBufferPoolAffinity.ForNativeQueue((IntPtr)11);
        var b = GpuBufferPoolAffinity.ForNativeQueue((IntPtr)12);
        using var pool = new GpuBufferPool<Cell>(4, 1024);
        var cell = new Cell(32); pool.Return(cell, a);
        Assert.False(pool.TryRentForBackend(16, b, backend, out _));
        Assert.True(pool.TryRentForBackend(16, a, backend, out var rented));
        Assert.Same(cell, rented); rented!.Release();
    }

    [Fact]
    public void NestedSuppressionRestoresItsCallersBackend()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        var previous = CompiledNativeCacheScope.Current;
        using (CompiledNativeCacheScope.Enter(backend))
        {
            using (CompiledNativeCacheScope.Enter(null)) Assert.Null(CompiledNativeCacheScope.Current);
            Assert.Same(backend, CompiledNativeCacheScope.Current);
        }
        Assert.Same(previous, CompiledNativeCacheScope.Current);
    }

    [Fact]
    public void CrossClassLoansKeepTheOriginalPhysicalCapacityBudget()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var scope = CompiledNativeCacheScope.Enter(backend);
        using var pool = new GpuBufferPool<Cell>(4, 1024) { MaxPooledElements = 32 };
        var cell = new Cell(32); pool.Return(cell);
        Assert.True(pool.TryRentForBackend(16, backend, out var rented));
        pool.Return(rented!);
        var excess = new Cell(1); pool.Return(excess);
        Assert.Equal(IntPtr.Zero, excess.Handle);
        Assert.Equal(1, pool.PooledCount);
        Assert.Equal(32, pool.MaxPooledElements);
    }
}
