using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class SizeClassPoolTests
{
    [Fact]
    public void Rent_ReturnsExactSize()
    {
        var pool = new SizeClassPool<float>();
        var buf = pool.Rent(1000);
        Assert.Equal(1000, buf.Length);
    }

    [Fact]
    public void RentReturn_ReusesBuffer()
    {
        var pool = new SizeClassPool<float>();
        var buf1 = pool.Rent(1000);
        pool.Return(buf1);

        var buf2 = pool.Rent(1000);
        Assert.Same(buf1, buf2); // exact same object reused
        Assert.Equal(1, pool.Hits);
    }

    [Fact]
    public void Rent_DifferentSize_AllocatesNew()
    {
        var pool = new SizeClassPool<float>();
        var buf1 = pool.Rent(1000);
        pool.Return(buf1);

        var buf2 = pool.Rent(2000); // different size
        Assert.NotSame(buf1, buf2);
        Assert.Equal(2000, buf2.Length);
    }

    [Fact]
    public void MaxBuffersPerSize_Enforced()
    {
        var pool = new SizeClassPool<float>(maxBuffersPerSize: 2);

        // Return 5 buffers of the same size
        for (int i = 0; i < 5; i++)
        {
            pool.Return(new float[100]);
        }

        // Only 2 should be cached
        Assert.Equal(2, pool.TotalCachedBuffers);
    }

    [Fact]
    public void Clear_RemovesAllBuffers()
    {
        var pool = new SizeClassPool<float>();
        pool.Return(new float[100]);
        pool.Return(new float[200]);
        pool.Return(new float[300]);

        Assert.Equal(3, pool.TotalCachedBuffers);

        pool.Clear();
        Assert.Equal(0, pool.TotalCachedBuffers);
        Assert.Equal(0, pool.ActiveSizeClasses);
    }

    [Fact]
    public void Warmup_PrePopulatesPool()
    {
        var pool = new SizeClassPool<float>();
        pool.Warmup([100, 200, 300], count: 2);

        Assert.Equal(6, pool.TotalCachedBuffers);
        Assert.Equal(3, pool.ActiveSizeClasses);

        // Rent should hit the cache
        var buf = pool.Rent(200);
        Assert.Equal(200, buf.Length);
        Assert.Equal(1, pool.Hits);
    }

    [Fact]
    public void HitRatio_ComputedCorrectly()
    {
        var pool = new SizeClassPool<float>();

        // 2 misses (first allocations)
        pool.Rent(100);
        pool.Rent(100);
        Assert.Equal(0.0, pool.HitRatio);

        // Return and rent again — 1 hit
        pool.Return(new float[100]);
        pool.Rent(100);
        Assert.True(pool.HitRatio > 0);
    }

    [Fact]
    public void ThreadSafety_ConcurrentRentReturn()
    {
        var pool = new SizeClassPool<float>();
        const int threads = 8;
        const int opsPerThread = 1000;

        var tasks = new Task[threads];
        for (int t = 0; t < threads; t++)
        {
            tasks[t] = Task.Run(() =>
            {
                for (int i = 0; i < opsPerThread; i++)
                {
                    var buf = pool.Rent(256);
                    buf[0] = 1f; // use the buffer
                    pool.Return(buf);
                }
            });
        }

        Task.WaitAll(tasks);

        // No crashes, pool is consistent
        Assert.True(pool.Hits + pool.Misses == threads * opsPerThread);
    }

    [Fact]
    public void Return_Null_DoesNotThrow()
    {
        var pool = new SizeClassPool<float>();
        pool.Return(null);
        Assert.Equal(0, pool.TotalCachedBuffers);
    }

    [Fact]
    public void MultipleSizes_IndependentPools()
    {
        var pool = new SizeClassPool<float>();

        // Return buffers of 3 different sizes
        pool.Return(new float[100]);
        pool.Return(new float[200]);
        pool.Return(new float[300]);

        Assert.Equal(3, pool.ActiveSizeClasses);

        // Rent each size — should be hits
        var b1 = pool.Rent(100); Assert.Equal(100, b1.Length);
        var b2 = pool.Rent(200); Assert.Equal(200, b2.Length);
        var b3 = pool.Rent(300); Assert.Equal(300, b3.Length);

        Assert.Equal(3, pool.Hits);
    }
}
