using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// CPU TensorIndexSelect on any rank and axis. It used to throw NotSupportedException for anything but a 2-D tensor
/// (a 1-D bias gather, a [B, S, E] gather along S), although the GPU kernel and IndexSelectBackward were already
/// rank-general, and a GPU engine falls back to this path whenever a gradient tape is active.
/// </summary>
public sealed class IndexSelectAnyRankTests
{
    public static TheoryData<int[], int> Cases => new()
    {
        { new[] { 11 }, 0 },
        { new[] { 4, 5, 6 }, 0 },
        { new[] { 4, 5, 6 }, 1 },
        { new[] { 4, 5, 6 }, 2 },
        { new[] { 4, 5, 6 }, -1 },
        { new[] { 2, 3, 7, 5 }, 2 },
        { new[] { 6, 9 }, 0 },
        { new[] { 6, 9 }, 1 },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Selects_slices_along_the_axis(int[] shape, int axis)
    {
        var eng = new CpuEngine();
        int ax = axis < 0 ? shape.Length + axis : axis;
        var rng = new Random(shape.Sum() + ax);
        int n = shape.Aggregate(1, (a, b) => a * b);
        var data = Enumerable.Range(0, n).Select(_ => (float)rng.NextDouble()).ToArray();
        var x = new Tensor<float>(shape, new Vector<float>(data));
        var idx = new[] { shape[ax] - 1, 0, 2 % shape[ax], 0 };

        var y = eng.TensorIndexSelect(x, new Tensor<int>(idx, new[] { idx.Length }), axis);

        var outShape = (int[])shape.Clone();
        outShape[ax] = idx.Length;
        Assert.Equal(outShape, y.Shape.ToArray());
        int outer = shape.Take(ax).Aggregate(1, (a, b) => a * b), inner = shape.Skip(ax + 1).Aggregate(1, (a, b) => a * b);
        var actual = y.ToArray();
        for (int o = 0; o < outer; o++)
            for (int i = 0; i < idx.Length; i++)
                for (int k = 0; k < inner; k++)
                    Assert.Equal(data[(o * shape[ax] + idx[i]) * inner + k], actual[(o * idx.Length + i) * inner + k]);
    }

    [Fact]
    public void Rank3_inner_axis_gradient_accumulates_repeated_indices()
    {
        var eng = new CpuEngine();
        const int a = 3, b = 5, c = 4;
        var rng = new Random(7);
        var x = new Tensor<double>(new[] { a, b, c }, new Vector<double>(Enumerable.Range(0, a * b * c).Select(_ => rng.NextDouble()).ToArray()));
        var idx = new[] { 1, 4, 1, 0 };
        var wData = Enumerable.Range(0, a * idx.Length * c).Select(_ => rng.NextDouble()).ToArray();
        var w = new Tensor<double>(new[] { a, idx.Length, c }, new Vector<double>(wData));
        Tensor<double> g;
        using (var tape = new GradientTape<double>())
        {
            var y = eng.TensorIndexSelect(x, new Tensor<int>(idx, new[] { idx.Length }), 1);
            var loss = eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false);
            g = tape.ComputeGradients(loss, new[] { x })[x];
        }

        var expected = new double[a * b * c];
        for (int o = 0; o < a; o++)
            for (int i = 0; i < idx.Length; i++)
                for (int k = 0; k < c; k++)
                    expected[(o * b + idx[i]) * c + k] += wData[(o * idx.Length + i) * c + k];
        var actual = g.ToArray();
        for (int j = 0; j < expected.Length; j++) Assert.Equal(expected[j], actual[j], 12);
    }

    [Fact]
    public void Out_of_range_index_throws()
    {
        var eng = new CpuEngine();
        var x = new Tensor<float>(new[] { 2, 3, 4 });
        Assert.Throws<IndexOutOfRangeException>(() => eng.TensorIndexSelect(x, new Tensor<int>(new[] { 0, 3 }, new[] { 2 }), 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => eng.TensorIndexSelect(x, new Tensor<int>(new[] { 0 }, new[] { 1 }), 3));
    }
}
