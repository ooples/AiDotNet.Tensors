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
/// host allocation). These tests assert exact counter values after a Reset, so they must not run alongside tests
/// that use a GPU: measured, a concurrent pinned allocation made allocated_bytes.current 528 instead of 512.
/// </summary>
[Collection(GpuMemoryStatsGlobalStateCollection.Name)]
public class GpuMemoryStatsTests
{
    [Fact]
    public void GpuMemoryStats_RecordAllocFree_ReflectInCounters()
    {
        GpuMemoryStats.Reset();
        GpuMemoryStats.RecordAllocation("test_alloc", 1024);
        GpuMemoryStats.RecordAllocation("test_alloc", 2048);
        Assert.Equal(3072, GpuMemoryStats.CurrentBytes);
        Assert.Equal(3072, GpuMemoryStats.PeakBytes);
        Assert.Equal(3072, GpuMemoryStats.TotalAllocatedBytes);
        Assert.Equal(2, GpuMemoryStats.ActiveAllocations);

        GpuMemoryStats.RecordFree("test_alloc", 1024);
        Assert.Equal(2048, GpuMemoryStats.CurrentBytes);
        Assert.Equal(3072, GpuMemoryStats.PeakBytes); // peak is sticky
        Assert.Equal(1, GpuMemoryStats.ActiveAllocations);

        GpuMemoryStats.ResetPeakStats();
        Assert.Equal(2048, GpuMemoryStats.PeakBytes);

        GpuMemoryStats.Reset();
    }

    [Fact]
    public void GpuMemoryStats_Stats_ExposesTorchParityKeys()
    {
        GpuMemoryStats.Reset();
        GpuMemoryStats.RecordAllocation("test_alloc", 512);
        var stats = GpuMemoryStats.Stats();

        Assert.Equal(512L, stats["allocated_bytes.current"]);
        Assert.Equal(512L, stats["allocated_bytes.peak"]);
        Assert.Equal(512L, stats["allocated_bytes.total"]);
        Assert.Equal(1L, stats["active.current"]);

        GpuMemoryStats.Reset();
    }

}
