using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The offload allocator diagnostics name each way the process-wide allocator can be out of service. These run without
/// a GPU: the registry takes any allocator, so a fake stands in for each state.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuOffloadAllocatorStateTests
{
    private sealed class FakeAllocator : IGpuOffloadAllocator
    {
        public bool IsAvailable { get; set; } = true;
        public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme) => throw new NotSupportedException();
        public void Free(GpuOffloadHandle handle) { }
        public void Dispose() { }
    }

    private sealed class FakeWrappingAllocator : IGpuOffloadAllocator, IGpuDevicePointerWrapper
    {
        public bool IsAvailable { get; set; }
        public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme) => throw new NotSupportedException();
        public void Free(GpuOffloadHandle handle) { }
        public void Dispose() { }
        public IGpuBuffer? WrapDevicePointerAsBuffer(IntPtr devicePointer, int elementCount, int elementByteSize) => null;
    }

    [Fact]
    public void Ensure_rejects_a_null_engine()
    {
        Assert.Throws<ArgumentNullException>(() => AiDotNetEngine.EnsureGpuOffloadAllocator(null!));
    }

    [Fact]
    public void Describe_names_a_missing_allocator()
    {
        WeightRegistry.Reset();
        try
        {
            Assert.Null(WeightRegistry.OffloadAllocator);
            Assert.Equal("no offload allocator is registered", AiDotNetEngine.DescribeGpuOffloadAllocatorState());
        }
        finally { WeightRegistry.Reset(); }
    }

    [SkippableFact]
    public void Describe_names_an_allocator_that_cannot_wrap_or_is_unavailable()
    {
        Skip.If(AiDotNetEngine.GpuCircuitBroken, "The GPU circuit breaker answers first when it is open.");
        WeightRegistry.Reset();
        try
        {
            WeightRegistry.Configure(new GpuOffloadOptions(), new FakeAllocator());
            Assert.Equal(nameof(FakeAllocator) + " cannot wrap a device pointer", AiDotNetEngine.DescribeGpuOffloadAllocatorState());

            // Configure accepts only an available allocator; its context can go away afterwards.
            var wrapping = new FakeWrappingAllocator { IsAvailable = true };
            WeightRegistry.Configure(new GpuOffloadOptions(), wrapping);
            Assert.Equal("usable (" + nameof(FakeWrappingAllocator) + ")", AiDotNetEngine.DescribeGpuOffloadAllocatorState());

            wrapping.IsAvailable = false;
            Assert.StartsWith(nameof(FakeWrappingAllocator) + " reports unavailable", AiDotNetEngine.DescribeGpuOffloadAllocatorState());
        }
        finally { WeightRegistry.Reset(); }
    }
}
