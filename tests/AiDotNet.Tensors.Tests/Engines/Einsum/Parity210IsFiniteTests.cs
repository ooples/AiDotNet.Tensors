using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210IsFiniteTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void IsFinite_IdentifiesFiniteValues()
    {
        var x = T(new[] { 1f, float.NaN, 3f, float.PositiveInfinity, -5f, float.NegativeInfinity }, 6);
        var r = E.TensorIsFinite(x);
        Assert.Equal(Bit.True, r[0]);
        Assert.Equal(Bit.False, r[1]);
        Assert.Equal(Bit.True, r[2]);
        Assert.Equal(Bit.False, r[3]);
        Assert.Equal(Bit.True, r[4]);
        Assert.Equal(Bit.False, r[5]);
    }

    [Fact]
    public void IsNan_IdentifiesOnlyNan()
    {
        var x = T(new[] { 1f, float.NaN, float.PositiveInfinity }, 3);
        var r = E.TensorIsNan(x);
        Assert.Equal(Bit.False, r[0]);
        Assert.Equal(Bit.True, r[1]);
        Assert.Equal(Bit.False, r[2]);
    }

    [Fact]
    public void IsInf_IdentifiesBothPositiveAndNegativeInf()
    {
        var x = T(new[] { 1f, float.PositiveInfinity, float.NegativeInfinity, float.NaN }, 4);
        var r = E.TensorIsInf(x);
        Assert.Equal(Bit.False, r[0]);
        Assert.Equal(Bit.True, r[1]);
        Assert.Equal(Bit.True, r[2]);
        Assert.Equal(Bit.False, r[3]);  // NaN is not Inf
    }
}
