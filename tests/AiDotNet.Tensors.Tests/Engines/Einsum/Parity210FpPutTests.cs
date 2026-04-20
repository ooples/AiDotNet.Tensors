using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210FpPutTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);

    [Fact]
    public void Ldexp_MultipliesByPowerOfTwo()
    {
        var x = T(new[] { 1f, 1.5f, -2f }, 3);
        var e = I(new[] { 2, 3, 1 }, 3);
        var r = E.TensorLdexp(x, e);
        Assert.Equal(4f, r[0]);    // 1 * 4
        Assert.Equal(12f, r[1]);   // 1.5 * 8
        Assert.Equal(-4f, r[2]);   // -2 * 2
    }

    [Fact]
    public void NextAfter_PositiveToHigher_IncreasesByUlp()
    {
        var a = T(new[] { 1f }, 1);
        var b = T(new[] { 2f }, 1);
        var r = E.TensorNextAfter(a, b);
        Assert.True(r[0] > 1f);
        Assert.True(r[0] < 1f + 1e-6f, $"next-after-1-toward-2 should be just above 1: got {r[0]}");
    }

    [Fact]
    public void NextAfter_EqualValues_ReturnsB()
    {
        var a = T(new[] { 3f }, 1);
        var b = T(new[] { 3f }, 1);
        var r = E.TensorNextAfter(a, b);
        Assert.Equal(3f, r[0]);
    }

    [Fact]
    public void Put_WritesFlatIndices()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var idx = I(new[] { 1, 3 }, 2);
        var src = T(new[] { 20f, 40f }, 2);
        var r = E.TensorPut(x, idx, src);
        Assert.Equal(new[] { 1f, 20f, 3f, 40f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void Put_OutOfRange_Throws()
    {
        var x = T(new[] { 1f, 2f }, 2);
        var idx = I(new[] { 5 }, 1);
        var src = T(new[] { 99f }, 1);
        Assert.Throws<IndexOutOfRangeException>(() => E.TensorPut(x, idx, src));
    }
}
