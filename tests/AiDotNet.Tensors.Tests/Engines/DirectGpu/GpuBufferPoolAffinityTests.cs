using System;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Serializes the pool tests against kernel tuning elsewhere in the process.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GpuBufferPoolSerialCollection
{
    public const string Name = "GpuBufferPoolSerial";
}

// A pool drains everything it holds on the first rent after any kernel-tuning measurement in the process
// (TunedKernelRegistry.MeasurementGeneration). GPU tests tuning in parallel therefore emptied these pools between a
// Return and the TryRent that expects it back (measured on the CUDA lane), so they run alone.
[Collection(GpuBufferPoolSerialCollection.Name)]
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

    // A pool created after kernel tuning ran holds nothing tuning parked, so its first rent must not drain it.
    [Fact]
    public void Rent_AfterEarlierTuning_KeepsWhatWasJustReturned()
    {
        AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelRegistry.NoteMeasurement();
        using var pool = new GpuBufferPool<FakeBuffer>(maxPerSize: 4, maxSize: 1024);
        var buffer = new FakeBuffer(64);

        pool.Return(buffer);

        Assert.True(pool.TryRent(64, out var rented));
        Assert.Same(buffer, rented);
        Assert.Equal(0, buffer.ReleaseCount);
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

    [Fact]
    public void Return_RacingDispose_ReleasesEveryBufferExactlyOnce()
    {
        const int iterations = 512;
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            var pool = new GpuBufferPool<FakeBuffer>(maxPerSize: 1, maxSize: 1024);
            var buffer = new FakeBuffer(64);
            using var start = new ManualResetEventSlim();

            System.Threading.Tasks.Parallel.Invoke(
                () =>
                {
                    start.Wait();
                    pool.Return(buffer);
                },
                () =>
                {
                    start.Set();
                    pool.Dispose();
                });

            Assert.Equal(1, buffer.ReleaseCount);
        }
    }

    [Fact]
    public void Rent_ExposesRequestedSizeWhilePreservingPhysicalCapacity()
    {
        using var pool = new GpuBufferPool<FakeBuffer>(maxPerSize: 1, maxSize: 8192);
        var buffer = new FakeBuffer(8192);

        pool.Return(buffer);

        Assert.True(pool.TryRent(5000, out var smaller));
        Assert.Same(buffer, smaller);
        Assert.Equal(5000, smaller.Size);
        Assert.Equal(8192, smaller.Capacity);

        pool.Return(smaller);

        Assert.True(pool.TryRent(8000, out var larger));
        Assert.Same(buffer, larger);
        Assert.Equal(8000, larger.Size);
        Assert.Equal(8192, larger.Capacity);
    }

    [Fact]
    public void Rent_RejectsSameBucketBufferWithInsufficientPhysicalCapacity()
    {
        using var pool = new GpuBufferPool<FakeBuffer>(maxPerSize: 1, maxSize: 8192);
        var buffer = new FakeBuffer(6272);

        pool.Return(buffer);

        Assert.False(pool.TryRent(8192, out _));
        Assert.Equal(6272, buffer.Size);
        Assert.Equal(6272, buffer.Capacity);
        Assert.True(pool.TryRent(6272, out var restored));
        Assert.Same(buffer, restored);
    }

    [Fact]
    public void Rent_SkipsUndersizedCandidateAndReusesSufficientBufferInSameBucket()
    {
        using var pool = new GpuBufferPool<FakeBuffer>(maxPerSize: 2, maxSize: 8192);
        var sufficient = new FakeBuffer(8192);
        var undersized = new FakeBuffer(6272);

        pool.Return(sufficient);
        pool.Return(undersized);

        Assert.True(pool.TryRent(8192, out var rented));
        Assert.Same(sufficient, rented);
        Assert.True(pool.TryRent(6272, out var restored));
        Assert.Same(undersized, restored);
    }

    [Fact]
    public void ExactSizePool_TakesOnlyAnExactMatch_AndKeepsTheOthersPooled()
    {
        using var pool = new GpuBufferPool<FakeBuffer>(maxPerSize: 4, maxSize: 1024, exactSizes: true);
        var larger = new FakeBuffer(100);
        var exact = new FakeBuffer(70);
        pool.Return(larger);

        // 100 would serve a request for 70 by capacity, but not exactly.
        Assert.False(pool.TryRent(70, out _));
        Assert.Equal(1, pool.PooledCount);
        Assert.Equal(0, larger.ReleaseCount);

        pool.Return(exact);
        Assert.True(pool.TryRent(70, out var rented));
        Assert.Same(exact, rented);
        Assert.Equal(1, pool.PooledCount);
        Assert.True(pool.TryRent(100, out var rentedLarger));
        Assert.Same(larger, rentedLarger);
    }

    private sealed class FakeBuffer : IGpuBuffer, IPoolableGpuBuffer
    {
        private int _size;
        internal FakeBuffer(int size)
        {
            _size = size;
            Capacity = size;
        }
        public int Size => Volatile.Read(ref _size);
        public int Capacity { get; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle => new(1);
        private int _releaseCount;
        internal int ReleaseCount => Volatile.Read(ref _releaseCount);
        public void MarkRented(int size)
        {
            if (size <= 0 || size > Capacity)
                throw new ArgumentOutOfRangeException(nameof(size));
            Volatile.Write(ref _size, size);
        }
        public void Release() => Interlocked.Increment(ref _releaseCount);
        public void Dispose() => Release();
    }
}
