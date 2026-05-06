using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue305DestinationAwareEngineTests
{
    [Fact]
    public void TensorRandomUniformRangeInto_FillsExistingDestination()
    {
        IEngine engine = new CpuEngine();
        var destination = new Tensor<float>(new[] { 128 });

        engine.TensorRandomUniformRangeInto(destination, -2f, 3f, seed: 305);

        var data = destination.AsSpan().ToArray();
        Assert.All(data, value => Assert.InRange(value, -2f, 3f));
        Assert.Contains(data, value => Math.Abs(value) > 1e-6f);
    }

    [Fact]
    public void TensorRandomUniformRangeInto_SeededFillIsDeterministic()
    {
        IEngine engine = new CpuEngine();
        var first = new Tensor<float>(new[] { 64 });
        var second = new Tensor<float>(new[] { 64 });

        engine.TensorRandomUniformRangeInto(first, -1f, 1f, seed: 17);
        engine.TensorRandomUniformRangeInto(second, -1f, 1f, seed: 17);

        Assert.Equal(first.AsSpan().ToArray(), second.AsSpan().ToArray());
    }

    [Fact]
    public void TensorRandomNormalInto_ZeroStdDevWritesMeanWithoutTempResult()
    {
        IEngine engine = new CpuEngine();
        var destination = new Tensor<float>(new[] { 32 });

        engine.TensorRandomNormalInto(destination, 7f, 0f);

        Assert.All(destination.AsSpan().ToArray(), value => Assert.Equal(7f, value));
    }

    [Fact]
    public void TensorMultiplyScalarInto_DestinationLastOverloadWritesDestination()
    {
        IEngine engine = new CpuEngine();
        var source = new Tensor<float>(new[] { 1f, -2f, 3f }, new[] { 3 });
        var destination = new Tensor<float>(new[] { 3 });

        engine.TensorMultiplyScalarInto(source, 2f, destination);

        Assert.Equal(new[] { 2f, -4f, 6f }, destination.AsSpan().ToArray());
    }

    [Fact]
    public void TensorSubtractInto_DestinationFirstOverloadWritesDestination()
    {
        IEngine engine = new CpuEngine();
        var a = new Tensor<float>(new[] { 5f, 2f, -1f }, new[] { 3 });
        var b = new Tensor<float>(new[] { 3f, -4f, 2f }, new[] { 3 });
        var destination = new Tensor<float>(new[] { 3 });

        engine.TensorSubtractInto(destination, a, b);

        Assert.Equal(new[] { 2f, 6f, -3f }, destination.AsSpan().ToArray());
    }

    [Fact]
    public void UnregisterPersistentTensor_SparseTensor_DoesNotMaterializeDenseStorage()
    {
        IEngine engine = new CpuEngine();
        var sparse = SparseTensor<float>.FromCsr(
            rows: 2,
            columns: 3,
            rowPointers: new[] { 0, 1, 2 },
            columnIndices: new[] { 0, 2 },
            values: new[] { 1f, 4f });

        var exception = Record.Exception(() => engine.UnregisterPersistentTensor(sparse));

        Assert.Null(exception);
    }
}
