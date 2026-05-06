using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue305DestinationAwareInitTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void TensorRandomUniformRangeInto_FillsExistingDestination()
    {
        var destination = new Tensor<float>(Enumerable.Repeat(-1f, 64).ToArray(), [64]);

        _engine.TensorRandomUniformRangeInto(destination, -0.25f, 0.25f, seed: 305);

        var values = destination.ToArray();
        Assert.All(values, value => Assert.InRange(value, -0.25f, 0.25f));
        Assert.Contains(values, value => value != -1f);
    }

    [Fact]
    public void TensorRandomNormalInto_FillsExistingDestination()
    {
        var destination = new Tensor<float>(Enumerable.Repeat(float.NaN, 128).ToArray(), [128]);

        _engine.TensorRandomNormalInto(destination, 0f, 1f);

        var values = destination.ToArray();
        Assert.All(values, value => Assert.False(float.IsNaN(value)));
        Assert.Contains(values, value => value != 0f);
    }

    [Fact]
    public void TensorRandomNormalInto_FillsLargeDestinationWithoutTempTensor()
    {
        var destination = new Tensor<float>(Enumerable.Repeat(float.NaN, 65_536).ToArray(), [65_536]);

        _engine.TensorRandomNormalInto(destination, 0f, 1f);

        var values = destination.ToArray();
        Assert.All(values, value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
        Assert.Contains(values, value => value < 0f);
        Assert.Contains(values, value => value > 0f);
    }

    [Fact]
    public void TensorRandomUniformRangeInto_UsesStrideAwareDestination()
    {
        var backing = new Tensor<float>(Enumerable.Repeat(-7f, 12).ToArray(), [3, 4]);
        var destination = backing.Transpose(new[] { 1, 0 });
        Assert.False(destination.IsContiguous);

        _engine.TensorRandomUniformRangeInto(destination, 1f, 2f, seed: 305);

        var values = destination.ToArray();
        Assert.All(values, value => Assert.InRange(value, 1f, 2f));
        Assert.DoesNotContain(-7f, backing.ToArray());
    }

    [Fact]
    public void TensorSubtractInto_KeepsOriginalDestinationFirstSignature()
    {
        var destination = new Tensor<float>(new float[3], [3]);
        var a = new Tensor<float>(new[] { 10f, 20f, 30f }, [3]);
        var b = new Tensor<float>(new[] { 1f, 2f, 3f }, [3]);

        _engine.TensorSubtractInto(destination, a, b);

        Assert.Equal(new[] { 9f, 18f, 27f }, destination.ToArray());
    }

    [Fact]
    public void TensorSubtractInto_UsesStrideAwareDestination()
    {
        var backing = new Tensor<float>(new float[6], [2, 3]);
        var destination = backing.Transpose(new[] { 1, 0 });
        var a = new Tensor<float>(new[] { 10f, 20f, 30f, 40f, 50f, 60f }, [3, 2]);
        var b = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, [3, 2]);

        _engine.TensorSubtractInto(destination, a, b);

        Assert.Equal(new[] { 9f, 18f, 27f, 36f, 45f, 54f }, destination.ToArray());
    }

    [Fact]
    public void TensorMultiplyScalarInto_UsesStrideAwareSourceAndDestination()
    {
        var sourceBacking = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, [2, 3]);
        var source = sourceBacking.Transpose(new[] { 1, 0 });
        var destinationBacking = new Tensor<float>(new float[6], [2, 3]);
        var destination = destinationBacking.Transpose(new[] { 1, 0 });

        _engine.TensorMultiplyScalarInto(destination, source, 3f);

        Assert.Equal(new[] { 3f, 12f, 6f, 15f, 9f, 18f }, destination.ToArray());
    }

    [Fact]
    public void PersistentTensorManagement_DoesNotDensifySparseTensor()
    {
        var sparse = new SparseTensor<float>(
            2,
            2,
            new[] { 0, 1 },
            new[] { 1, 0 },
            new[] { 3f, 4f });

        _engine.RegisterPersistentTensor(sparse, PersistentTensorRole.Weights);
        _engine.UnregisterPersistentTensor(sparse);
        _engine.InvalidatePersistentTensor(sparse);
    }
}
