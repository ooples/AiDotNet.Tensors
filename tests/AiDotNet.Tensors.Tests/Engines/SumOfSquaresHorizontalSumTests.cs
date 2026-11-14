using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// TensorSumOfSquares against a scalar loop on both sides of the SIMD block length.
/// </summary>
/// <remarks>
/// The double horizontal sum shuffled with an shufps-style mask; shufpd reads only two bits, so the
/// mask selected lanes 0 and 1 unchanged and the result doubled lane 0 instead of adding lane 1. Every
/// double tensor of 16 or more elements was affected (Adafactor's parameter RMS read 21.29 for 17.17).
/// </remarks>
public class SumOfSquaresHorizontalSumTests
{
    private readonly CpuEngine _engine = new();

    // Distinct, sign-varying values so a dropped or doubled lane changes the sum.
    private static double Value(int i) => Math.Sin(0.7 * i + 0.3) * (1 + 0.01 * i);

    [Theory]
    [InlineData(1)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(24)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(100)]
    public void Double_MatchesScalarSum(int length)
    {
        var data = new double[length];
        double expected = 0.0;
        for (int i = 0; i < length; i++)
        {
            data[i] = Value(i);
            expected += data[i] * data[i];
        }

        double actual = _engine.TensorSumOfSquares(new Tensor<double>(new[] { length }, new Vector<double>(data)));

        Assert.True(Math.Abs(actual - expected) <= 1e-12 * expected, $"length {length}: {actual} vs {expected}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(48)]
    [InlineData(64)]
    [InlineData(100)]
    public void Float_MatchesScalarSum(int length)
    {
        var data = new float[length];
        double expected = 0.0;
        for (int i = 0; i < length; i++)
        {
            data[i] = (float)Value(i);
            expected += (double)data[i] * data[i];
        }

        float actual = _engine.TensorSumOfSquares(new Tensor<float>(new[] { length }, new Vector<float>(data)));

        Assert.True(Math.Abs(actual - expected) <= 1e-5 * expected, $"length {length}: {actual} vs {expected}");
    }
}
