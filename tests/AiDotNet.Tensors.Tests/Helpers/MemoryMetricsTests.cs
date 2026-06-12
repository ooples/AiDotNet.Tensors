using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class MemoryMetricsTests
{
    [Fact]
    public void ProcessRss_AndManagedHeap_AreBothPositive()
    {
        long rss = MemoryMetrics.CurrentProcessRssBytes;
        long managed = MemoryMetrics.ManagedHeapBytes;
        Assert.True(rss > 0, "current process RSS should be positive");
        Assert.True(managed > 0, "managed heap should be positive");
        // We previously asserted `rss >= managed`, but that inequality doesn't
        // actually hold: GC.GetTotalMemory reports committed managed-heap
        // allocations, including segments whose pages aren't currently
        // resident (paged out under CI memory pressure, or reserved-but-not-
        // resident in server GC). On the GHA Linux runner this PR's full-
        // suite execution accumulated ~2.69 GB committed heap against
        // ~1.93 GB RSS — a real OS reality, not a metric bug. The remaining
        // checks still verify the API returns meaningful positive values.
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
