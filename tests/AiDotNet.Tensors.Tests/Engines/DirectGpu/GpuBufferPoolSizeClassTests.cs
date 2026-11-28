// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection(GpuBufferPoolSerialCollection.Name)]
public sealed class GpuBufferPoolSizeClassTests
{
    [Fact]
    public void MonotonicallyGrowingRequests_513To1024_ZeroUndersizedMissesWhenRoundedToBucketCapacity()
    {
        using var pool = new GpuBufferPool<SizeClassFakeBuffer>(maxPerSize: 64, maxSize: 1 << 20);
        int physicalAllocations = 0;
        int poolHits = 0;

        for (int requested = 513; requested <= 1024; requested++)
        {
            SizeClassFakeBuffer rented;
            if (pool.TryRent(requested, out var candidate) && candidate is not null)
            {
                rented = candidate;
                poolHits++;
            }
            else
            {
                int physicalCapacity = pool.RoundUpToBucketCapacity(requested);
                rented = new SizeClassFakeBuffer(requested, physicalCapacity);
                physicalAllocations++;
            }

            Assert.Equal(requested, rented.Size);
            Assert.Equal((long)requested * sizeof(float), rented.SizeInBytes);
            Assert.True(rented.Capacity >= requested);

            if (requested <= 768)
            {
                Assert.Equal(768, rented.Capacity);
            }
            else
            {
                Assert.Equal(1024, rented.Capacity);
            }

            pool.Return(rented);
        }

        // 513..768 allocates physical 768 once and hits 255 times; 769..1024 allocates physical 1024 once and hits 255 times.
        Assert.Equal(2, physicalAllocations);
        Assert.Equal(510, poolHits);
        Assert.Equal(0, pool.UndersizedCandidateSkipCount);
        Assert.Equal(0, pool.RestoreLockAcquisitions);
        Assert.Equal(2, pool.PooledCount);
    }

    [Fact]
    public void SizeClassCeiling_BoundsInternalFragmentationAtMostFiftyPercent_AndNeverOverflows()
    {
        for (int s = 1; s <= 262144; s++)
        {
            int ceiling = GpuBufferPool<SizeClassFakeBuffer>.ComputeSizeClassCeiling(s);
            Assert.True(ceiling >= s, $"Size class ceiling {ceiling} was smaller than requested {s}.");

            // Worst-case internal fragmentation for half-step size classes (2^k and 1.5 * 2^k) is <= 50%.
            long excess = ceiling - s;
            Assert.True(excess * 2 <= s, $"Internal fragmentation exceeded 50% at s={s}: ceiling={ceiling}.");
        }

        Assert.Equal(1_073_741_824, GpuBufferPool<SizeClassFakeBuffer>.ComputeSizeClassCeiling(1_073_741_824));
        Assert.Equal(1_610_612_736, GpuBufferPool<SizeClassFakeBuffer>.ComputeSizeClassCeiling(1_073_741_825));
        Assert.Equal(1_610_612_736, GpuBufferPool<SizeClassFakeBuffer>.ComputeSizeClassCeiling(1_610_612_736));
        Assert.Equal(int.MaxValue, GpuBufferPool<SizeClassFakeBuffer>.ComputeSizeClassCeiling(1_610_612_737));
        Assert.Equal(int.MaxValue, GpuBufferPool<SizeClassFakeBuffer>.ComputeSizeClassCeiling(int.MaxValue));
        Assert.Equal(long.MaxValue, GpuBufferPool<SizeClassFakeBuffer>.NextSizeClassAfter(int.MaxValue));
    }

    [Fact]
    public void RestoreUndersizedCandidates_AcquiresLifecycleGateOnce_AndPreservesStackOrder()
    {
        using var pool = new GpuBufferPool<SizeClassFakeBuffer>(maxPerSize: 256, maxSize: 8192);
        var sufficient = new SizeClassFakeBuffer(768, 768);
        pool.Return(sufficient);

        // Push 128 legacy unrounded buffers into bucket 768 on top of `sufficient`.
        for (int cap = 513; cap < 513 + 128; cap++)
        {
            pool.Return(new SizeClassFakeBuffer(cap, cap));
        }

        Assert.True(pool.TryRent(768, out var rented));
        Assert.Same(sufficient, rented);
        Assert.Equal(128, pool.UndersizedCandidateSkipCount);
        // Must acquire _lifecycleGate ONCE for all 128 candidates, never 128 times.
        Assert.Equal(1, pool.RestoreLockAcquisitions);
        Assert.Equal(128, pool.PooledCount);

        // Verify LIFO stack order of restored candidates was preserved (top is 513 + 127 = 640).
        Assert.True(pool.TryRent(640, out var topRestored));
        Assert.NotNull(topRestored);
        if (topRestored is not null)
        {
            Assert.Equal(640, topRestored.Capacity);
        }
    }

    [Fact]
    public async Task ConcurrentRentAndFinalizerLikeReturns_WithMaxPooledElementsEviction_NeverOrphansOrDoubleReleases()
    {
        using var pool = new GpuBufferPool<SizeClassFakeBuffer>(maxPerSize: 8, maxSize: 16384)
        {
            MaxPooledElements = 8 * 1024
        };

        var allCreated = new ConcurrentBag<SizeClassFakeBuffer>();
        var returnQueue = new BlockingCollection<SizeClassFakeBuffer>(boundedCapacity: 256);

        // Simulate GC finalizer / deferred-free worker threads returning buffers concurrently with mutator threads.
        Task[] returnWorkers = new Task[2];
        for (int w = 0; w < returnWorkers.Length; w++)
        {
            returnWorkers[w] = Task.Run(() =>
            {
                foreach (var buf in returnQueue.GetConsumingEnumerable())
                {
                    pool.Return(buf);
                }
            });
        }

        Parallel.For(0, 6, workerId =>
        {
            for (int iter = 0; iter < 200; iter++)
            {
                int requested = 513 + ((workerId * 37 + iter * 13) % 512);
                SizeClassFakeBuffer buffer;
                if (pool.TryRent(requested, out var reused) && reused is not null)
                {
                    buffer = reused;
                }
                else
                {
                    int cap = pool.RoundUpToBucketCapacity(requested);
                    buffer = new SizeClassFakeBuffer(requested, cap);
                    allCreated.Add(buffer);
                }

                Assert.Equal(requested, buffer.Size);
                Assert.True(buffer.Capacity >= requested);
                returnQueue.Add(buffer);
            }
        });

        returnQueue.CompleteAdding();
        await Task.WhenAll(returnWorkers);

        Assert.Equal(0, pool.UndersizedCandidateSkipCount);
        pool.Dispose();

        foreach (var created in allCreated)
        {
            Assert.Equal(1, created.ReleaseCount);
        }
    }

    [Fact]
    public void RestoreUndersizedCandidates_RacingDispose_ReleasesEveryBufferExactlyOnce()
    {
        for (int iteration = 0; iteration < 128; iteration++)
        {
            var pool = new GpuBufferPool<SizeClassFakeBuffer>(maxPerSize: 32, maxSize: 4096);
            var buffers = new SizeClassFakeBuffer[16];
            for (int i = 0; i < buffers.Length; i++)
            {
                buffers[i] = new SizeClassFakeBuffer(520 + i, 520 + i);
                pool.Return(buffers[i]);
            }

            using var gate = new ManualResetEventSlim();
            Parallel.Invoke(
                () =>
                {
                    gate.Wait();
                    pool.TryRent(768, out _);
                },
                () =>
                {
                    gate.Set();
                    pool.Dispose();
                });

            pool.Dispose();
            for (int i = 0; i < buffers.Length; i++)
            {
                Assert.Equal(1, buffers[i].ReleaseCount);
            }
        }
    }

    [Fact]
    public void CudaGpuBuffer_CapacityVsLogicalSize_PreservesExactLogicalSizeAcrossPoolDetachAndRent()
    {
        CudaBackend.CudaGpuBuffer? capturedLease = null;
        var initial = new CudaBackend.CudaGpuBuffer(
            context: IntPtr.Zero,
            devicePtr: new IntPtr(0x10000),
            size: 513,
            returnToPool: returned => capturedLease = returned.DetachForPool(),
            capacity: 768);

        Assert.Equal(513, initial.Size);
        Assert.Equal(513L * sizeof(float), initial.SizeInBytes);
        Assert.Equal(768, initial.Capacity);

        initial.Dispose();
        Assert.True(initial.IsReleased);
        Assert.Equal(IntPtr.Zero, initial.RawDevicePointer);
        Assert.NotNull(capturedLease);

        if (capturedLease is not null)
        {
            Assert.Equal(768, capturedLease.Capacity);
            Assert.True(capturedLease.IsReleased);

            capturedLease.MarkRented(700);
            Assert.False(capturedLease.IsReleased);
            Assert.Equal(700, capturedLease.Size);
            Assert.Equal(700L * sizeof(float), capturedLease.SizeInBytes);
            Assert.Equal(768, capturedLease.Capacity);

            // Prevent finalizer enqueue since this uses a synthetic non-driver pointer.
            _ = capturedLease.DetachForPool();
            GC.SuppressFinalize(capturedLease);
        }
    }

    private sealed class SizeClassFakeBuffer : IGpuBuffer, IPoolableGpuBuffer
    {
        private int _size;
        private int _releaseCount;

        internal SizeClassFakeBuffer(int logicalSize, int physicalCapacity)
        {
            if (logicalSize <= 0 || physicalCapacity < logicalSize)
                throw new ArgumentOutOfRangeException(nameof(physicalCapacity));
            _size = logicalSize;
            Capacity = physicalCapacity;
        }

        public GpuElementType ElementType => GpuElementType.Float32;
        public int Size => Volatile.Read(ref _size);
        public int Capacity { get; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle => Volatile.Read(ref _releaseCount) == 0 ? new IntPtr(1) : IntPtr.Zero;
        internal int ReleaseCount => Volatile.Read(ref _releaseCount);

        public void MarkRented(int size)
        {
            if (size <= 0 || size > Capacity)
                throw new ArgumentOutOfRangeException(nameof(size));
            if (Volatile.Read(ref _releaseCount) != 0)
                throw new InvalidOperationException("Cannot rent a released buffer.");
            Volatile.Write(ref _size, size);
        }

        public void Release() => Interlocked.Increment(ref _releaseCount);
        public void Dispose() => Release();
    }
}
