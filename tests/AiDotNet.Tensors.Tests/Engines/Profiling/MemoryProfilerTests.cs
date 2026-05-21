// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines.Profiling.Memory;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Profiling;

/// <summary>
/// Acceptance tests for the #220 Phase 2 memory profiler. Tests run serial
/// via the same <c>AiDotNetProfiler</c> collection because
/// <see cref="MemoryProfiler"/> uses process-wide static state (mode +
/// counters); parallel test classes would interleave the counter updates and
/// the assertions would race.
/// </summary>
[Collection("AiDotNetProfiler")]
public class MemoryProfilerTests
{
    [Fact]
    public void RecordHistory_Off_RecordsNothing()
    {
        MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.Off);
        MemoryProfiler.Reset();

        var t = TensorAllocator.Rent<float>(new[] { 64 });
        // The allocator hook short-circuits when mode is Off, so events stay empty.
        Assert.Empty(MemoryProfiler.Events);
    }

    [Fact]
    public void RecordHistory_State_LogsAllocationEvent()
    {
        try
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.State);
            MemoryProfiler.Reset();

            var t = TensorAllocator.Rent<float>(new[] { 64 });

            var allocs = MemoryProfiler.Events.Where(e => e.Kind == MemoryEventKind.Alloc).ToList();
            Assert.NotEmpty(allocs);
            // Last allocation is the one we just made.
            var last = allocs[^1];
            Assert.Equal("TensorAllocator", last.Allocator);
            Assert.Equal(64 * sizeof(float), last.Bytes);
            Assert.Null(last.Stack); // mode=State doesn't capture stacks
        }
        finally
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.Off);
            MemoryProfiler.Reset();
        }
    }

    [Fact]
    public void RecordHistory_All_CapturesAllocationStack()
    {
        try
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.All);
            MemoryProfiler.Reset();

            var t = TensorAllocator.Rent<float>(new[] { 32 });

            var ev = MemoryProfiler.Events.LastOrDefault(e => e.Kind == MemoryEventKind.Alloc);
            Assert.NotEqual(default, ev);
            Assert.NotNull(ev.Stack);
            Assert.Contains("TensorAllocator", ev.Stack!,
                StringComparison.Ordinal); // the alloc site is on the captured stack
        }
        finally
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.Off);
            MemoryProfiler.Reset();
        }
    }

    [Fact]
    public void PeakBytes_TracksHighWaterMark()
    {
        try
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.State);
            MemoryProfiler.Reset();

            // Allocate a few tensors of varying sizes; PeakBytes must grow
            // monotonically until we explicitly free.
            long beforePeak = MemoryProfiler.PeakBytes;
            var a = TensorAllocator.Rent<float>(new[] { 1000 });
            long afterPeak = MemoryProfiler.PeakBytes;
            Assert.True(afterPeak >= beforePeak + 1000 * sizeof(float),
                $"Peak should grow by ~{1000 * sizeof(float)} bytes, was {afterPeak - beforePeak}.");
        }
        finally
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.Off);
            MemoryProfiler.Reset();
        }
    }

    [Fact]
    public void RecordAllocation_AndFree_BalancesCurrentBytes()
    {
        try
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.State);
            MemoryProfiler.Reset();

            long before = MemoryProfiler.CurrentBytes;
            long id = MemoryProfiler.RecordAllocation("TestAllocator", 1024);
            Assert.Equal(before + 1024, MemoryProfiler.CurrentBytes);

            MemoryProfiler.RecordFree(id);
            Assert.Equal(before, MemoryProfiler.CurrentBytes);
        }
        finally
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.Off);
            MemoryProfiler.Reset();
        }
    }

    [Fact]
    public void GetLargestLiveAllocations_ReturnsDescendingBySize()
    {
        try
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.State);
            MemoryProfiler.Reset();

            long id1 = MemoryProfiler.RecordAllocation("A", 100);
            long id2 = MemoryProfiler.RecordAllocation("A", 5000);
            long id3 = MemoryProfiler.RecordAllocation("A", 2000);

            var top = MemoryProfiler.GetLargestLiveAllocations(10).ToList();
            Assert.Equal(3, top.Count);
            Assert.Equal(5000, top[0].Bytes);
            Assert.Equal(2000, top[1].Bytes);
            Assert.Equal(100, top[2].Bytes);

            MemoryProfiler.RecordFree(id1);
            MemoryProfiler.RecordFree(id2);
            MemoryProfiler.RecordFree(id3);
        }
        finally
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.Off);
            MemoryProfiler.Reset();
        }
    }

    [Fact]
    public void DumpSnapshot_WritesReadableSummary()
    {
        try
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.State);
            MemoryProfiler.Reset();

            // Generate one alloc that will appear in the live set.
            long id = MemoryProfiler.RecordAllocation("TestAllocator", 4096);

            string path = Path.Combine(Path.GetTempPath(), $"snap-{Guid.NewGuid():N}.txt");
            try
            {
                MemoryProfiler.DumpSnapshot(path);
                string content = File.ReadAllText(path);

                Assert.Contains("AiDotNet.Tensors Memory Snapshot", content);
                Assert.Contains("PeakBytes:", content);
                Assert.Contains("4,096", content); // formatted bytes
                Assert.Contains("TestAllocator", content);
            }
            finally
            {
                MemoryProfiler.RecordFree(id);
                try { File.Delete(path); } catch { /* ignore */ }
            }
        }
        finally
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.Off);
            MemoryProfiler.Reset();
        }
    }

    [Fact]
    public void ResetPeakStats_ClearsPeakToCurrent()
    {
        try
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.State);
            MemoryProfiler.Reset();

            long id = MemoryProfiler.RecordAllocation("A", 10000);
            Assert.True(MemoryProfiler.PeakBytes >= 10000);

            MemoryProfiler.RecordFree(id);
            // Peak still 10000 here (high-water mark); current is 0.
            Assert.True(MemoryProfiler.PeakBytes >= 10000);
            Assert.Equal(0, MemoryProfiler.CurrentBytes);

            MemoryProfiler.ResetPeakStats();
            Assert.Equal(0, MemoryProfiler.PeakBytes);
        }
        finally
        {
            MemoryProfiler.RecordHistory(MemoryProfiler.RecordMode.Off);
            MemoryProfiler.Reset();
        }
    }
}
