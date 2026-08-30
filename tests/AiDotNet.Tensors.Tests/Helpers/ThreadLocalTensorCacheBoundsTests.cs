// Copyright (c) AiDotNet. All rights reserved.
// ThreadLocalTensorCache<T> caps buffers PER SIZE BUCKET and its doc used to call that "prevents
// unbounded memory growth". It did not: the key is an exact element count, nothing capped how many
// distinct sizes a thread accumulated, and this cache pools at ANY size. A process that keeps
// meeting new shapes grew a bucket per size and released none until someone called Clear() by hand.
//
// Downstream symptom that led here (AiDotNet): RecurrentGemma's finite-difference gradcheck runs in
// 11-13 s in a clean process and blew its whole 120 s budget once 29 sibling model tests had filled
// this cache on the same thread. With retention bounded it completes inside the budget.

using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// Serialized with the arena tests because these tests poke [ThreadStatic] caches. The allocation
// assertion below deliberately uses the current-thread counter so unrelated xUnit collections
// cannot contaminate the measurement.
[Collection(nameof(TensorArenaPinnedTests))]
public class ThreadLocalTensorCacheBoundsTests
{
    /// <summary>
    /// Returning buffers across a long tail of DISTINCT sizes must not retain them all. Before the
    /// byte budget this grew without limit — one bucket per size, up to MaxBuffersPerSize buffers
    /// each, none ever released.
    /// </summary>
    [Fact]
    public void ManyDistinctSizes_RetentionStaysWithinBudget()
    {
        ThreadLocalTensorCache<float>.Clear();

        // 4000 distinct sizes around 64Ki floats (256 KB each). Unbounded retention would be on the
        // order of a gigabyte; the budget is 64 MiB.
        for (int i = 0; i < 4000; i++)
            ThreadLocalTensorCache<float>.TryReturn(new float[(64 * 1024) + i]);

        long retained = ThreadLocalTensorCache<float>.RetainedBytes;
        Assert.True(retained <= 64L * 1024 * 1024,
            $"retained {retained} bytes, above the 64 MiB budget");

        // And the bookkeeping must agree with reality: no empty buckets left behind, no negative
        // counter from a mismatched rent/return.
        Assert.True(retained >= 0, $"retained byte counter went negative: {retained}");

        ThreadLocalTensorCache<float>.Clear();
        Assert.Equal(0, ThreadLocalTensorCache<float>.RetainedBytes);
        Assert.Equal(0, ThreadLocalTensorCache<float>.BucketCount);
    }

    /// <summary>
    /// The other half of the contract: bounding retention must not stop the cache doing its job. A
    /// hot size returned and re-rented in a loop is touched constantly, so it stays cached and every
    /// rent after the first hits.
    /// </summary>
    [Fact]
    public void HotSize_StillServedFromCache()
    {
        ThreadLocalTensorCache<float>.Clear();
        const int size = 4096;

        Assert.True(ThreadLocalTensorCache<float>.TryReturn(new float[size]));

        for (int i = 0; i < 100; i++)
        {
            var rented = ThreadLocalTensorCache<float>.TryRent(size);
            Assert.NotNull(rented);
            Assert.Equal(size, rented!.Length);
            Assert.True(ThreadLocalTensorCache<float>.TryReturn(rented));
        }

        ThreadLocalTensorCache<float>.Clear();
    }

    /// <summary>
    /// Renting a bucket's last buffer must not leave the bucket behind forever. Return-then-rent for
    /// each distinct size keeps RetainedBytes at zero, so the byte budget never fires — and if the
    /// dictionary entry survives, _cache grows with the number of sizes the thread has ever seen.
    /// That is the same "bounded per bucket, unbounded in buckets" defect this class was fixing,
    /// one level up.
    /// </summary>
    [Fact]
    public void ReturnThenRentEachSize_DoesNotGrowTheDictionary()
    {
        ThreadLocalTensorCache<float>.Clear();

        for (int i = 0; i < 4000; i++)
        {
            int size = (64 * 1024) + i;
            ThreadLocalTensorCache<float>.TryReturn(new float[size]);
            Assert.NotNull(ThreadLocalTensorCache<float>.TryRent(size));
        }

        // Every buffer was rented straight back, so nothing is retained...
        Assert.Equal(0, ThreadLocalTensorCache<float>.RetainedBytes);
        // ...and the bookkeeping must not have kept 4,000 entries alive to say so.
        Assert.InRange(ThreadLocalTensorCache<float>.BucketCount, 0, 1024);

        ThreadLocalTensorCache<float>.Clear();
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// The single hottest pattern this cache serves is one buffer of a size cycling return -> rent
    /// -> return. That must stay allocation-free in steady state: the whole point of the class is
    /// that <c>TensorAllocator.Rent</c> costs nothing after warmup. Pins it directly, because the
    /// obvious way to stop empty buckets accumulating — dropping the dictionary entry the moment its
    /// last buffer is rented — reintroduces a Bucket + Stack allocation on every cycle.
    /// </summary>
    [Fact]
    public async Task HotSize_ReturnRentCycle_DoesNotAllocate()
    {
        await Task.Yield();

        ThreadLocalTensorCache<float>.Clear();
        const int size = 4096;

        var buffer = new float[size];
        // Warm up: create the bucket and let the JIT settle.
        for (int i = 0; i < 50; i++)
        {
            ThreadLocalTensorCache<float>.TryReturn(buffer);
            buffer = ThreadLocalTensorCache<float>.TryRent(size)!;
        }

        const int cycles = 50_000;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < cycles; i++)
        {
            ThreadLocalTensorCache<float>.TryReturn(buffer);
            buffer = ThreadLocalTensorCache<float>.TryRent(size)!;
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.NotNull(buffer);

        // The cache's own steady-state path allocates nothing here — a dictionary lookup and a
        // Stack push/pop. Allow only instrumentation-scale slack. Dropping the dictionary entry
        // when its last buffer is rented instead costs a Bucket + Stack per cycle, measured at
        // 1,212,984 bytes over 10,000 cycles (~121 B/cycle), or roughly 6 MB here.
        Assert.True(allocated < 128,
            $"steady-state return/rent cycle allocated {allocated} bytes over {cycles} iterations");

        ThreadLocalTensorCache<float>.Clear();
    }
#endif

    /// <summary>
    /// A buffer larger than the entire budget is not worth evicting the whole cache for, so it is
    /// refused outright rather than trimming everything else away.
    /// </summary>
    [Fact]
    public void BufferLargerThanBudget_IsNotCached()
    {
        ThreadLocalTensorCache<float>.Clear();

        // 32M floats = 128 MB, past the 64 MiB budget.
        Assert.False(ThreadLocalTensorCache<float>.TryReturn(new float[32 * 1024 * 1024]));
        Assert.Equal(0, ThreadLocalTensorCache<float>.RetainedBytes);
    }
}
