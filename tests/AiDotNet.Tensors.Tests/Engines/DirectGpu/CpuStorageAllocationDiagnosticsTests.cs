using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[CollectionDefinition("CPU storage allocation diagnostics", DisableParallelization = true)]
public sealed class CpuStorageAllocationDiagnosticsCollection { }

[Collection("CPU storage allocation diagnostics")]
public sealed class CpuStorageAllocationDiagnosticsTests
{
    [Fact]
    public void RequestBytesRespectElementWidthAndInvalidShapeWithoutChangingAllocatorValidation()
    {
        Assert.Equal(24, CpuStorageAllocationDiagnostics.ShapeBytes<float>(new[] { 2, 3 }));
        Assert.Equal(48, CpuStorageAllocationDiagnostics.ShapeBytes<double>(new[] { 2, 3 }));
        Assert.Equal(0, CpuStorageAllocationDiagnostics.ShapeBytes<float>(new[] { 0, 3 }));
        Assert.Equal(0, CpuStorageAllocationDiagnostics.ShapeBytes<float>(new[] { -1, 3 }));
        Assert.Equal(0, CpuStorageAllocationDiagnostics.ShapeBytes<float>(new[] { int.MaxValue, 3 }));
    }

    [Fact]
    public void FailedRequestAndNestedDelegateCannotInflateCounts()
    {
        var recorder = new CpuStorageAllocationDiagnostics.RequestRecorder();
        Assert.Throws<OutOfMemoryException>(() => recorder.Observe<int, object>("tensor", 64, 0,
            static _ => throw new OutOfMemoryException()));
        Assert.Empty(recorder.Snapshot());
        var value = recorder.Observe("tensor", 64, recorder,
            static nested => nested.Observe("workspace", 128, 7, static result => result));
        Assert.Equal(7, value);
        Assert.Equal(64, recorder.Snapshot()["tensor"].Bytes);
        Assert.Equal(1, recorder.Snapshot()["tensor"].Count);
        Assert.Single(recorder.Snapshot());
        Assert.False(CpuStorageAllocationDiagnostics.RequestRecorder.IsInside);
    }

    [SkippableFact]
    public void ActualTensorAllocatorAndVectorEntriesCountStorageOnceAndExcludeViews()
    {
        Skip.IfNot(CpuStorageAllocationDiagnostics.Enabled, "Set AIDOTNET_BUFFER_ALLOC_DIAG=1 before process launch.");
        static long Counter(string unit) => Convert.ToInt64(CpuStorageAllocationDiagnostics.Snapshot()[unit]);
        long beforeBytes = Counter("bytes"), beforeCount = Counter("count");
        using var uninitialized = TensorAllocator.RentUninitialized<float>(new[] { 2, 3 });
        using var zeroed = TensorAllocator.Rent<float>(new[] { 2, 3 });
        var vector = new Vector<float>(6);
        using var tensor = new Tensor<float>(new[] { 2, 3 }, vector);
        using var view = tensor.Reshape(new[] { 3, 2 });
        Assert.Equal(72, Counter("bytes") - beforeBytes);
        Assert.Equal(3, Counter("count") - beforeCount);
        Assert.Equal("partial_numeric_storage_request_boundaries", CpuStorageAllocationDiagnostics.Snapshot()["coverage"]);
    }

    [SkippableFact]
    public void WorkspacePoolCountsRequestedBytesRatherThanRoundedArrayCapacity()
    {
        Skip.IfNot(CpuStorageAllocationDiagnostics.Enabled, "Set AIDOTNET_BUFFER_ALLOC_DIAG=1 before process launch.");
        static long Counter(string unit) => Convert.ToInt64(CpuStorageAllocationDiagnostics.Snapshot()[unit]);
        long beforeBytes = Counter("bytes"), beforeCount = Counter("count");
        float[] workspace = CpuStorageAllocationDiagnostics.RentWorkspace<float>(6, "workspace.test");
        try
        {
            Assert.True(workspace.Length >= 6);
            Assert.Equal(24, Counter("bytes") - beforeBytes);
            Assert.Equal(1, Counter("count") - beforeCount);
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                CpuStorageAllocationDiagnostics.RentWorkspace<float>(-1, "workspace.test"));
            Assert.Equal(1, Counter("count") - beforeCount);
        }
        finally { ArrayPool<float>.Shared.Return(workspace); }
    }
}
