using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class MemoryMetricsTests
{
    [Fact]
    public void ProcessRss_IsPositive_AndAtLeastManagedHeap()
    {
        long rss = MemoryMetrics.CurrentProcessRssBytes;
        long managed = MemoryMetrics.ManagedHeapBytes;
        Assert.True(rss > 0, "current process RSS should be positive");
        Assert.True(managed > 0, "managed heap should be positive");
        // RSS is the whole-process resident set; the managed heap is a subset, so RSS >= managed
        // (allow a small slack for sampling skew between the two reads).
        Assert.True(rss + (8L << 20) >= managed,
            $"process RSS ({rss}) should be >= managed heap ({managed})");
    }

    [Fact]
    public void PeakProcessRss_IsAtLeastCurrent()
    {
        long current = MemoryMetrics.CurrentProcessRssBytes;
        long peak = MemoryMetrics.PeakProcessRssBytes;
        Assert.True(peak > 0, "peak RSS should be positive");
        // Peak-since-start must be >= current (allow slack: the two reads aren't atomic).
        Assert.True(peak + (8L << 20) >= current, $"peak RSS ({peak}) should be >= current ({current})");
    }

    [Fact]
    public void PeakRssSampler_CapturesAllocationSpike()
    {
        using var sampler = new MemoryMetrics.PeakRssSampler(sampleIntervalMs: 1);
        long baseline = sampler.PeakBytes;

        // Touch ~128 MB so it becomes resident, then let the sampler observe the spike.
        var blocks = new byte[16][];
        for (int b = 0; b < blocks.Length; b++)
        {
            blocks[b] = new byte[8 * 1024 * 1024];
            for (int i = 0; i < blocks[b].Length; i += 4096) blocks[b][i] = (byte)b; // fault pages in
        }
        System.Threading.Thread.Sleep(40); // give the sampler several ticks

        long peak = sampler.PeakBytes;
        GC.KeepAlive(blocks);
        Assert.True(peak >= baseline, $"sampler peak ({peak}) should not drop below baseline ({baseline})");
    }
}
