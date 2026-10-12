using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// #1211: MaxOverAxis / SumOverAxisDirect / MeanOverAxisDirect reduced each contiguous run of Shape[axis] elements, which
/// is the last-axis reduction whatever the axis. Every axis of ranks 1-4 is checked here against a loop over the indexer.
/// </summary>
public class AxisReductionTests
{
    public static IEnumerable<object[]> ShapesAndAxes()
    {
        int[][] shapes = { new[] { 7 }, new[] { 3, 5 }, new[] { 2, 3, 4 }, new[] { 2, 3, 4, 5 }, new[] { 1, 6, 1 } };
        foreach (var shape in shapes)
            for (int axis = 0; axis < shape.Length; axis++)
                yield return new object[] { shape, axis };
    }

    // Values that make every position's max differ from its neighbours' (no ties), signs mixed.
    private static Tensor<double> Make(int[] shape)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        var data = Enumerable.Range(0, n).Select(i => Math.Sin(i * 1.7) * 10.0 + i * 0.01).ToArray();
        return new Tensor<double>(data, shape);
    }

    private static int[] Unravel(int flat, int[] shape)
    {
        var idx = new int[shape.Length];
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            idx[d] = flat % shape[d];
            flat /= shape[d];
        }
        return idx;
    }

    // The reduction over `axis` at every result position, by walking the source indexer.
    private static (double[] max, double[] sum) Reference(Tensor<double> t, int axis)
    {
        int[] shape = t.Shape.ToArray();
        int[] outShape = shape.Where((_, d) => d != axis).ToArray();
        int outLength = outShape.Aggregate(1, (a, b) => a * b);
        var max = new double[outLength];
        var sum = new double[outLength];
        for (int o = 0; o < outLength; o++)
        {
            int[] outIdx = outShape.Length == 0 ? Array.Empty<int>() : Unravel(o, outShape);
            double m = double.NegativeInfinity, s = 0;
            for (int k = 0; k < shape[axis]; k++)
            {
                var idx = new int[shape.Length];
                for (int d = 0, j = 0; d < shape.Length; d++)
                    idx[d] = d == axis ? k : outIdx[j++];
                double v = t[idx];
                m = Math.Max(m, v);
                s += v;
            }
            max[o] = m;
            sum[o] = s;
        }
        return (max, sum);
    }

    private static void AssertReductions(Tensor<double> t, int axis)
    {
        var (max, sum) = Reference(t, axis);
        int[] expectedShape = t.Shape.ToArray().Where((_, d) => d != axis).ToArray();

        var gotMax = t.MaxOverAxis(axis);
        var gotSum = t.SumOverAxisDirect(axis);
        var gotMean = t.MeanOverAxisDirect(axis);
        Assert.Equal(expectedShape, gotMax.Shape.ToArray());
        Assert.Equal(expectedShape, gotSum.Shape.ToArray());
        Assert.Equal(expectedShape, gotMean.Shape.ToArray());

        double[] mx = gotMax.ToArray(), sm = gotSum.ToArray(), mn = gotMean.ToArray();
        int axisSize = t.Shape[axis];
        for (int i = 0; i < max.Length; i++)
        {
            Assert.Equal(max[i], mx[i]);
            Assert.Equal(sum[i], sm[i], 10);
            Assert.Equal(sum[i] / axisSize, mn[i], 10);
        }
    }

    [Theory]
    [MemberData(nameof(ShapesAndAxes))]
    public void EveryAxis_MatchesLoopReference(int[] shape, int axis) => AssertReductions(Make(shape), axis);

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void TransposedView_MatchesLoopReference(int axis) =>
        AssertReductions(Make(new[] { 3, 4, 5 }).Transpose(new[] { 2, 0, 1 }), axis);

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void OffsetSlice_MatchesLoopReference(int axis) =>
        AssertReductions(Make(new[] { 4, 3, 5 }).Slice(0, 1, 3), axis);

    [Fact]
    public void Float_MaxAndSumOverMiddleAxis_MatchLoop()
    {
        var data = Enumerable.Range(0, 24).Select(i => (float)Math.Cos(i * 0.9) * 5f).ToArray();
        var t = new Tensor<float>(data, new[] { 2, 3, 4 });
        var max = t.MaxOverAxis(1).ToArray();
        var sum = t.SumOverAxisDirect(1).ToArray();
        for (int i = 0; i < 2; i++)
            for (int k = 0; k < 4; k++)
            {
                var column = Enumerable.Range(0, 3).Select(j => data[i * 12 + j * 4 + k]).ToArray();
                Assert.Equal(column.Max(), max[i * 4 + k]);
                Assert.Equal(column.Sum(), sum[i * 4 + k], 4);
            }
    }

    [Fact]
    public void EmptyAxis_SumIsZero_MaxThrows()
    {
        var t = new Tensor<double>(new[] { 3, 0, 2 });
        var sum = t.SumOverAxisDirect(1);
        Assert.Equal(new[] { 3, 2 }, sum.Shape.ToArray());
        Assert.All(sum.ToArray(), v => Assert.Equal(0.0, v));
        Assert.Throws<InvalidOperationException>(() => t.MaxOverAxis(1));
    }

    [Fact]
    public void InvalidAxis_Throws()
    {
        var t = Make(new[] { 2, 3 });
        Assert.Throws<ArgumentOutOfRangeException>(() => t.MaxOverAxis(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.SumOverAxisDirect(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.MeanOverAxisDirect(5));
    }
}
