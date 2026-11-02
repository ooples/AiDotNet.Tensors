using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class StridedReluNaNTests
{
    [Fact]
    public void FloatTransposePreservesNaNAndNormalizesNegativeZero()
    {
        var input = new Tensor<float>(new[] { float.NaN, -0f, -2f, 3f, float.PositiveInfinity, float.NegativeInfinity }, new[] { 2, 3 }).Transpose();
        Assert.False(input.IsContiguous);
        var result = new CpuEngine().ReLU(input).ToArray();
        Assert.True(float.IsNaN(result[0]));
        Assert.Equal(3f, result[1]);
        Assert.Equal(float.PositiveInfinity, result[3]);
        foreach (int index in new[] { 2, 4, 5 }) Assert.Equal(0, BitConverter.ToInt32(BitConverter.GetBytes(result[index]), 0));
    }

    [Fact]
    public void DoubleTransposePreservesNaN()
    {
        var input = new Tensor<double>(new[] { double.NaN, -0d, -2d, 3d, double.PositiveInfinity, double.NegativeInfinity }, new[] { 2, 3 }).Transpose();
        Assert.False(input.IsContiguous);
        var result = new CpuEngine().ReLU(input).ToArray();
        Assert.True(double.IsNaN(result[0]));
        Assert.Equal(3d, result[1]);
        Assert.Equal(double.PositiveInfinity, result[3]);
        foreach (int index in new[] { 2, 4, 5 }) Assert.Equal(0L, BitConverter.DoubleToInt64Bits(result[index]));
    }
}
