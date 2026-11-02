using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>Serializes the tests that read GpuMemoryStats' process-wide counters.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GpuMemoryStatsGlobalStateCollection
{
    public const string Name = "GpuMemoryStatsGlobalState";
}

/// <summary>
/// GpuMemoryStats is process-wide, and the engine records into it too (CudaPinnedBufferPool records every pinned
/// host allocation). These tests run alone (the collection disables parallelization) and assert DELTAS on their own
/// allocator label. They never call <see cref="GpuMemoryStats.Reset"/>: it zeroes every allocator's live counters,
/// and a live CUDA backend's pinned pool then freed more than its zeroed counter held ("Mismatched free for
/// allocator 'cuda_pinned_host'") when a later test released the engine.
/// </summary>
[Collection(GpuMemoryStatsGlobalStateCollection.Name)]
public class GpuMemoryStatsTests
{
    [Fact]
    public void GpuMemoryStats_RecordAllocFree_ReflectInCounters()
    {
        string label = "test_alloc_" + Guid.NewGuid().ToString("N");
        GpuMemoryStats.ResetPeakStats();
        long current0 = GpuMemoryStats.CurrentBytes, total0 = GpuMemoryStats.TotalAllocatedBytes;
        long peak0 = GpuMemoryStats.PeakBytes, active0 = GpuMemoryStats.ActiveAllocations;
        GpuMemoryStats.RecordAllocation(label, 1024);
        GpuMemoryStats.RecordAllocation(label, 2048);
        try
        {
            Assert.Equal(3072, GpuMemoryStats.CurrentBytes - current0);
            Assert.Equal(3072, GpuMemoryStats.PeakBytes - peak0);
            Assert.Equal(3072, GpuMemoryStats.TotalAllocatedBytes - total0);
            Assert.Equal(2, GpuMemoryStats.ActiveAllocations - active0);

            GpuMemoryStats.RecordFree(label, 1024);
            Assert.Equal(2048, GpuMemoryStats.CurrentBytes - current0);
            Assert.Equal(3072, GpuMemoryStats.PeakBytes - peak0); // peak is sticky
            Assert.Equal(1, GpuMemoryStats.ActiveAllocations - active0);

            GpuMemoryStats.ResetPeakStats();
            Assert.Equal(2048, GpuMemoryStats.PeakBytes - current0);
        }
        finally
        {
            GpuMemoryStats.RecordFree(label, 2048);
        }
    }

    [Fact]
    public void GpuMemoryStats_Stats_ExposesTorchParityKeys()
    {
        string label = "test_alloc_" + Guid.NewGuid().ToString("N");
        GpuMemoryStats.ResetPeakStats();
        var before = GpuMemoryStats.Stats();
        GpuMemoryStats.RecordAllocation(label, 512);
        try
        {
            var stats = GpuMemoryStats.Stats();
            Assert.Equal(512L, stats["allocated_bytes.current"] - before["allocated_bytes.current"]);
            Assert.Equal(512L, stats["allocated_bytes.peak"] - before["allocated_bytes.peak"]);
            Assert.Equal(512L, stats["allocated_bytes.total"] - before["allocated_bytes.total"]);
            Assert.Equal(1L, stats["active.current"] - before["active.current"]);
        }
        finally
        {
            GpuMemoryStats.RecordFree(label, 512);
        }
    }

}
