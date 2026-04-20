using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210NanToNumTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void NanToNum_DefaultReplacesNaNWithZero()
    {
        var x = T(new[] { 1f, float.NaN, 3f }, 3);
        var r = E.TensorNanToNum(x);
        Assert.Equal(1f, r[0]);
        Assert.Equal(0f, r[1]);
        Assert.Equal(3f, r[2]);
    }

    [Fact]
    public void NanToNum_CustomNanValue()
    {
        var x = T(new[] { float.NaN }, 1);
        var r = E.TensorNanToNum(x, nan: 42.0);
        Assert.Equal(42f, r[0]);
    }

    [Fact]
    public void NanToNum_PositiveInf_ReplacedWithLarge()
    {
        var x = T(new[] { float.PositiveInfinity }, 1);
        var r = E.TensorNanToNum(x);
        Assert.False(float.IsInfinity(r[0]));
    }

    [Fact]
    public void NanToNum_CustomInfinities()
    {
        var x = T(new[] { float.PositiveInfinity, float.NegativeInfinity }, 2);
        var r = E.TensorNanToNum(x, posinf: 100.0, neginf: -100.0);
        Assert.Equal(100f, r[0]);
        Assert.Equal(-100f, r[1]);
    }

    [Fact]
    public void NanToNum_FiniteUnchanged()
    {
        var x = T(new[] { -5f, 0f, 5f, 1e10f }, 4);
        var r = E.TensorNanToNum(x);
        Assert.Equal(x.AsSpan().ToArray(), r.AsSpan().ToArray());
    }
}
