using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

public class ZeroCopyConversionTests
{
    [Fact]
    public void VectorAsTensor_SharesMemory()
    {
        var vec = new Vector<double>([1.0, 2.0, 3.0, 4.0]);
        var tensor = vec.AsTensor();

        Assert.Equal(1, tensor.Shape.Length);
        Assert.Equal(4, tensor.Shape[0]);
        Assert.Equal(1.0, tensor[0]);
        Assert.Equal(4.0, tensor[3]);

        // Mutation through tensor is visible in vector
        tensor[2] = 99.0;
        Assert.Equal(99.0, vec[2]);
    }

    [Fact]
    public void TensorAsVector_SharesMemory()
    {
        var tensor = new Tensor<double>([4]);
        tensor[0] = 10.0;
        tensor[1] = 20.0;
        tensor[2] = 30.0;
        tensor[3] = 40.0;

        var vec = tensor.AsVector();
        Assert.Equal(4, vec.Length);
        Assert.Equal(10.0, vec[0]);
        Assert.Equal(40.0, vec[3]);

        // Mutation through vector is visible in tensor
        vec[1] = 99.0;
        Assert.Equal(99.0, tensor[1]);
    }

    [Fact]
    public void VectorAsTensor_RoundTrip_PreservesData()
    {
        var original = new Vector<double>([5.0, 10.0, 15.0]);
        var tensor = original.AsTensor();
        var roundTrip = tensor.AsVector();

        Assert.Equal(original.Length, roundTrip.Length);
        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], roundTrip[i]);
    }

    [Fact]
    public void TensorAsVector_ThrowsOnMultiDimensional()
    {
        var tensor = new Tensor<double>([2, 3]);
        Assert.Throws<InvalidOperationException>(() => tensor.AsVector());
    }

    [Fact]
    public void VectorAsTensor_Float_Works()
    {
        var vec = new Vector<float>([1f, 2f, 3f]);
        var tensor = vec.AsTensor();
        Assert.Equal(3, tensor.Length);
        Assert.Equal(2f, tensor[1]);
    }

    [Fact]
    public void TensorAsVector_DenseRank1_ReturnsCorrectLength()
    {
        var tensor = new Tensor<double>([4]);
        for (int i = 0; i < 4; i++) tensor[i] = i;
        var vec = tensor.AsVector();
        Assert.Equal(4, vec.Length);
        Assert.Equal(2.0, vec[2]);
    }

    [Fact]
    public void AsTensor_AsVector_RoundTrip_SharesMemory()
    {
        var vec = new Vector<double>([10.0, 20.0, 30.0, 40.0, 50.0]);
        var tensor = vec.AsTensor();
        var roundTrip = tensor.AsVector();
        Assert.Equal(5, roundTrip.Length);
        Assert.Equal(30.0, roundTrip[2]);
        roundTrip[0] = 999.0;
        Assert.Equal(999.0, vec[0]);
    }
}
