using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class TensorFinitePredicateTests
{
    private static readonly bool[] FloatingFinite = { false, true, true, true, true, true, false, false };
    private static readonly bool[] FloatingInfinite = { true, false, false, false, false, false, true, false };

    private static void AssertPredicates<T>(T[] values, bool[] expectedFinite, bool[] expectedInfinite, bool transpose)
    {
        var engine = new CpuEngine();
        var tensor = new Tensor<T>(values, new[] { 2, 4 });
        if (transpose)
        {
            tensor = tensor.Transpose(new[] { 1, 0 });
            Assert.False(tensor.IsContiguous);
        }

        var finite = engine.TensorIsFinite(tensor);
        var infinite = engine.TensorIsInf(tensor);

        Assert.Equal(tensor.Shape.ToArray(), finite.Shape.ToArray());
        Assert.Equal(tensor.Shape.ToArray(), infinite.Shape.ToArray());
        for (int i = 0; i < values.Length; i++)
        {
            int sourceIndex = transpose ? (i % 2) * 4 + i / 2 : i;
            Assert.Equal(expectedFinite[sourceIndex], (bool)finite[i]);
            Assert.Equal(expectedInfinite[sourceIndex], (bool)infinite[i]);
        }
    }

#if NET5_0_OR_GREATER
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Half_DoesNotRequireIConvertible(bool transpose)
    {
        AssertPredicates(new[]
        {
            Half.NegativeInfinity, Half.MinValue, (Half)(-0.0f), Half.Epsilon,
            (Half)0, Half.MaxValue, Half.PositiveInfinity, Half.NaN,
        }, FloatingFinite, FloatingInfinite, transpose);
    }
#endif

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Double_MaxValueRemainsFinite(bool transpose)
    {
        AssertPredicates(new[]
        {
            double.NegativeInfinity, double.MinValue, -0.0, double.Epsilon,
            0.0, double.MaxValue, double.PositiveInfinity, double.NaN,
        }, FloatingFinite, FloatingInfinite, transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Float_MaxValueRemainsFinite(bool transpose)
    {
        AssertPredicates(new[]
        {
            float.NegativeInfinity, float.MinValue, -0.0f, float.Epsilon,
            0.0f, float.MaxValue, float.PositiveInfinity, float.NaN,
        }, FloatingFinite, FloatingInfinite, transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decimal_ValuesAreFinite(bool transpose)
    {
        AssertPredicates(new[] { decimal.MinValue, -1m, 0m, 1m, 0.25m, -0.25m, 1000m, decimal.MaxValue },
            Enumerable.Repeat(true, 8).ToArray(), new bool[8], transpose);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Int64_ValuesAreFinite(bool transpose)
    {
        AssertPredicates(new[] { long.MinValue, -1L, 0L, 1L, 2L, 1L << 54, -(1L << 54), long.MaxValue },
            Enumerable.Repeat(true, 8).ToArray(), new bool[8], transpose);
    }
}
