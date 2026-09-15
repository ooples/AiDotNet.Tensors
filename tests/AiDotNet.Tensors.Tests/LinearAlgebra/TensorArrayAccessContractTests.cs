using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

[Collection("EngineCurrentGlobalState")]
public sealed class TensorArrayAccessContractTests
{
    [Fact]
    public void FullContiguousView_ReadOnlyAccessIsZeroCopy()
    {
        float[] values = { 1, 2, 3, 4, 5, 6 };
        using var source = new Tensor<float>(values, new[] { 2, 3 });
        using var view = source.Reshape(new[] { 3, 2 });
        Assert.True(view.IsView && view.IsContiguous);

        Assert.Same(values, view.GetLiveBackingArrayOrNull());
        for (int i = 0; i < 10; i++)
            Assert.Same(values, view.GetReadOnlyDataArray());
        Assert.Equal(values, view.ToArray());
    }

    [Fact]
    public void FullContiguousView_WritableAccessStillWritesThrough()
    {
        float[] values = { 1, 2, 3, 4, 5, 6 };
        using var source = new Tensor<float>(values, new[] { 2, 3 });
        using var view = source.Reshape(new[] { 3, 2 });

        float[] writable = view.GetDataArray();
        Assert.Same(values, writable);
        writable[2] = 30f;
        view.IncrementVersion();

        Assert.Equal(30f, source[2]);
        Assert.Equal(30f, view[2]);
    }

    [Fact]
    public void CowView_ReadOnlyAccessDoesNotDetach_ThenWriteDetachesRequestedFamily()
    {
        float[] values = { 1, 2, 3, 4, 5, 6 };
        using var source = new Tensor<float>(values, new[] { 2, 3 });
        using var clone = (Tensor<float>)source.CloneShared();
        using var view = clone.Reshape(new[] { 3, 2 });

        Assert.Same(values, view.GetReadOnlyDataArray());
        Assert.Same(source._storage, clone._storage);
        Assert.True(source.IsCowShared && clone.IsCowShared && view.IsCowShared);

        float[] writable = view.GetDataArray();
        Assert.NotSame(values, writable);
        Assert.Same(writable, clone.GetReadOnlyDataArray());
        writable[0] = 99f;
        view.IncrementVersion();

        Assert.Equal(1f, source[0]);
        Assert.Equal(99f, clone[0]);
        Assert.Equal(99f, view[0]);
        Assert.Same(values, source.GetReadOnlyDataArray());
        Assert.NotSame(source._storage, clone._storage);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void PartialContiguousView_ReturnsExactLogicalSnapshot(int row)
    {
        float[] values = { 1, 2, 3, 4, 5, 6 };
        using var source = new Tensor<float>(values, new[] { 2, 3 });
        using var view = source.Slice(row);
        Assert.True(view.IsContiguous);
        Assert.Null(view.GetLiveBackingArrayOrNull());

        float[] snapshot = view.GetReadOnlyDataArray();
        Assert.NotSame(values, snapshot);
        Assert.Equal(row == 0 ? new float[] { 1, 2, 3 } : new float[] { 4, 5, 6 }, snapshot);
        source[row * 3] = 99f;
        Assert.Equal(row == 0 ? 1f : 4f, snapshot[0]);
        Assert.Equal(99f, view.GetReadOnlyDataArray()[0]);
    }

    [Fact]
    public void NonContiguousView_ReadOnlySnapshotFollowsLogicalOrder()
    {
        float[] values = { 1, 2, 3, 4, 5, 6 };
        using var source = new Tensor<float>(values, new[] { 2, 3 });
        using var view = source.Transpose(new[] { 1, 0 });
        Assert.False(view.IsContiguous);
        Assert.Null(view.GetLiveBackingArrayOrNull());

        float[] snapshot = view.GetReadOnlyDataArray();
        Assert.NotSame(values, snapshot);
        Assert.Equal(new float[] { 1, 4, 2, 5, 3, 6 }, snapshot);
        source[0] = 99f;
        Assert.Equal(1f, snapshot[0]);
        Assert.Equal(new float[] { 99, 4, 2, 5, 3, 6 }, view.GetReadOnlyDataArray());
    }

    [Fact]
    public void DeferredStorage_ReadOnlyAccessMaterializesValuesOnce()
    {
        int initializations = 0;
        using var deferred = Tensor<float>.CreateDeferred(new[] { 3 }, tensor =>
        {
            initializations++;
            tensor.CopyFromArray(new float[] { 10, 20, 30 });
        });
        Assert.True(deferred.IsStorageDeferred);

        Assert.Equal(new float[] { 10, 20, 30 }, deferred.GetReadOnlyDataArray());
        Assert.Equal(new float[] { 10, 20, 30 }, deferred.GetReadOnlyDataArray());
        Assert.Equal(1, initializations);
        Assert.False(deferred.IsStorageDeferred);
    }

    [Fact]
    public void LazyGraph_ReadOnlyAccessRealizesValueInsteadOfReturningPlaceholder()
    {
        var engine = new CpuEngine();
        using var input = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3 });
        using var graph = GraphMode.Enable();
        using var deferred = engine.TensorMultiplyScalar(input, 2f);
        Assert.NotNull(deferred.LazySource);

        Assert.Equal(new float[] { 2, 4, 6 }, deferred.GetReadOnlyDataArray());
    }
}
