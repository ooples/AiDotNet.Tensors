using System;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class InferenceBufferBestFitTests
{
    [Fact]
    public void SmallestLargerClassPreservesCapacityAndReturnsUnderItsPhysicalClass()
    {
        using var pool = new GpuBufferPool<Cell>(4, 1024) { MaxPooledElements = 448 };
        var smallest = new Cell(128);
        var largest = new Cell(256);
        pool.Return(largest); pool.Return(smallest);
        Assert.True(pool.TryRentLargerIdleBuffer(64, default, out var rented));
        Assert.Same(smallest, rented);
        Assert.Equal(64, rented!.Size);
        Assert.Equal(128, rented.Capacity);
        pool.Return(rented);
        Assert.True(pool.TryRent(128, out var restored));
        Assert.Same(smallest, restored);
        restored!.Release();
        Assert.True(pool.TryRent(256, out var other));
        Assert.Same(largest, other);
        other!.Release();
        // Repeated cross-class loans must debit the physical capacity, rather than the smaller logical size.
        for (int i = 0; i < 20; i++)
        {
            pool.Return(new Cell(128));
            Assert.True(pool.TryRentLargerIdleBuffer(64, default, out var loan));
            loan!.Release();
        }
        pool.Return(new Cell(256)); pool.Return(new Cell(128)); pool.Return(new Cell(64));
        Assert.Equal(3, pool.PooledCount);
    }

    [Theory]
    [InlineData(64, 256, true)]
    [InlineData(64, 512, false)]
    [InlineData(1500, 6272, false)]
    [InlineData(2048, 6272, true)]
    public void ReuseIsBoundedByPhysicalCapacity(int requested, int capacity, bool expected)
    {
        using var pool = new GpuBufferPool<Cell>(4, 16384);
        var cell = new Cell(capacity); pool.Return(cell);
        Assert.Equal(expected, pool.TryRentLargerIdleBuffer(requested, default, out var rented));
        if (expected) { Assert.Same(cell, rented); rented!.Release(); }
        else { Assert.True(pool.TryRent(capacity, out var retained)); Assert.Same(cell, retained); retained!.Release(); }
    }

    [Fact]
    public void ExactSizePoolsAndForeignQueuesKeepTheirIdleCells()
    {
        var a = GpuBufferPoolAffinity.ForNativeQueue((IntPtr)11);
        var b = GpuBufferPoolAffinity.ForNativeQueue((IntPtr)12);
        using var exact = new GpuBufferPool<Cell>(4, 1024, exactSizes: true);
        exact.Return(new Cell(128), a);
        Assert.False(exact.TryRentLargerIdleBuffer(64, a, out _));
        Assert.True(exact.TryRent(128, a, out var kept)); kept!.Release();
        using var pool = new GpuBufferPool<Cell>(4, 1024);
        var cell = new Cell(128); pool.Return(cell, a);
        Assert.False(pool.TryRentLargerIdleBuffer(64, b, out _));
        Assert.True(pool.TryRentLargerIdleBuffer(64, a, out var rented));
        Assert.Same(cell, rented); rented!.Release();
    }

    [Fact]
    public void InvalidOrDisposedRequestsNeverRent()
    {
        using var pool = new GpuBufferPool<Cell>(4, int.MaxValue);
        var cell = new Cell(128); pool.Return(cell);
        Assert.False(pool.TryRentLargerIdleBuffer(0, default, out _));
        Assert.False(pool.TryRentLargerIdleBuffer(-1, default, out _));
        Assert.False(pool.TryRentLargerIdleBuffer(int.MaxValue, default, out _));
        pool.Dispose();
        Assert.False(pool.TryRentLargerIdleBuffer(64, default, out _));
        Assert.Equal(1, cell.Releases);
    }

    [Fact]
    public void ReuseRequiresEnabledMatchingInferenceAndNeverCrossesAnOutstandingLoan()
    {
        bool previous = InferenceWorkspaceDemandScope.BestFitIdleBuffersEnabled;
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        var foreign = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var pool = new GpuBufferPool<Cell>(4, 1024);
        var cell = new Cell(128); pool.Return(cell);
        try
        {
            InferenceWorkspaceDemandScope.BestFitIdleBuffersEnabled = true;
            Assert.False(pool.TryRentForBackend(64, backend, out _));
            using var scope = InferenceWorkspaceDemandScope.Enter(engine);
            InferenceWorkspaceDemandScope.BestFitIdleBuffersEnabled = false;
            Assert.False(pool.TryRentForBackend(64, backend, out _));
            InferenceWorkspaceDemandScope.BestFitIdleBuffersEnabled = true;
            Assert.False(pool.TryRentForBackend(64, foreign, out _));
            using (InferenceWorkspaceDemandScope.Enter(null)) Assert.False(pool.TryRentForBackend(64, backend, out _));
            Exception? failure = null;
            var thread = new Thread(() => { try { Assert.False(pool.TryRentForBackend(64, backend, out _)); } catch (Exception e) { failure = e; } });
            thread.Start(); Assert.True(thread.Join(TimeSpan.FromSeconds(5))); Assert.Null(failure);
            using (var tape = new GradientTape<float>()) Assert.False(pool.TryRentForBackend(64, backend, out _));
            engine.SuspendActivationEviction();
            try { using (engine.EnterCompiledCapturePath()) Assert.False(pool.TryRentForBackend(64, backend, out _)); }
            finally { engine.ResumeActivationEviction(); }
            Assert.True(pool.TryRentForBackend(64, backend, out var rented));
            Assert.Same(cell, rented); Assert.Equal(64, rented!.Size); Assert.Equal(0, cell.Releases);
            Assert.False(pool.TryRentForBackend(64, backend, out _));
            Assert.NotEqual(IntPtr.Zero, rented.Handle);
            pool.Dispose(); Assert.Equal(0, cell.Releases);
            rented.Release(); Assert.Equal(1, cell.Releases);
        }
        finally { InferenceWorkspaceDemandScope.BestFitIdleBuffersEnabled = previous; }
    }

    private sealed class Cell : IGpuBuffer, IPoolableGpuBuffer
    {
        public Cell(int capacity) { Capacity = Size = capacity; }
        public int Capacity { get; }
        public int Size { get; private set; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle => Releases == 0 ? new IntPtr(1) : IntPtr.Zero;
        internal int Releases;
        public void MarkRented(int size) { Assert.InRange(size, 1, Capacity); Assert.Equal(0, Releases); Size = size; }
        public void Release() => Releases++;
        public void Dispose() => Release();
    }
}
