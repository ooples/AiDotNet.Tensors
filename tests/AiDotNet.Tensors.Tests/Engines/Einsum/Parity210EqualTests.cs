using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210EqualTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void Equal_SameShapeSameValues_True()
    {
        var a = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var b = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        Assert.True(E.TensorEqual(a, b));
    }

    [Fact]
    public void Equal_DifferentValues_False()
    {
        var a = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var b = new Tensor<float>(new[] { 1f, 2f, 4f }, new[] { 3 });
        Assert.False(E.TensorEqual(a, b));
    }

    [Fact]
    public void Equal_DifferentShapes_False()
    {
        var a = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var b = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 1, 3 });
        Assert.False(E.TensorEqual(a, b));
    }

    [Fact]
    public void Equal_NaNVsNaN_False_MatchesTorch()
    {
        var a = new Tensor<float>(new[] { float.NaN }, new[] { 1 });
        var b = new Tensor<float>(new[] { float.NaN }, new[] { 1 });
        Assert.False(E.TensorEqual(a, b));
    }

    [Fact]
    public void Eq_ElementwiseProducesBitTensor()
    {
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var b = new Tensor<float>(new[] { 1f, 5f, 3f, 4f }, new[] { 4 });
        var r = E.TensorEq(a, b);
        Assert.Equal(new[] { 4 }, r.Shape.ToArray());
        Assert.True((bool)r[0]);
        Assert.False((bool)r[1]);
        Assert.True((bool)r[2]);
        Assert.True((bool)r[3]);
    }

    [Fact]
    public void EqScalar_MatchesComparison()
    {
        var a = new Tensor<float>(new[] { 1f, 2f, 2f, 3f }, new[] { 4 });
        var r = E.TensorEqScalar(a, 2f);
        Assert.False((bool)r[0]);
        Assert.True((bool)r[1]);
        Assert.True((bool)r[2]);
        Assert.False((bool)r[3]);
    }
}
