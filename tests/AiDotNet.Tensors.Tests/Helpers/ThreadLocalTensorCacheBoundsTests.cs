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
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

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
