using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// CPU TensorScatter with torch.scatter semantics on any rank and axis, and its gradient. It used to accept only a 2-D
/// tensor with axis 1, wrote through the per-element indexer, and recorded no backward, so nothing flowed to the
/// destination or the source in training.
/// </summary>
public sealed class ScatterAnyRankTests
{
    public static TheoryData<int[], int, int> Cases => new()
    {
        { new[] { 9 }, 0, 4 },
        { new[] { 3, 8 }, 1, 3 },
        { new[] { 6, 5 }, 0, 2 },
        { new[] { 2, 7, 4 }, 1, 3 },
        { new[] { 2, 3, 6 }, 2, 2 },
        { new[] { 2, 3, 6 }, -1, 2 },
        { new[] { 5, 2, 3 }, 0, 2 },
    };

    // Indices along the axis: distinct within each fibre, so the reference is order-independent.
    private static Tensor<int> Indices(int[] shape, int ax, int count, Random rng)
    {
        var idxShape = (int[])shape.Clone();
        idxShape[ax] = count;
        int outer = shape.Take(ax).Aggregate(1, (a, b) => a * b), inner = shape.Skip(ax + 1).Aggregate(1, (a, b) => a * b);
        var data = new int[outer * count * inner];
        for (int o = 0; o < outer; o++)
            for (int k = 0; k < inner; k++)
            {
                var pick = Enumerable.Range(0, shape[ax]).OrderBy(_ => rng.Next()).Take(count).ToArray();
                for (int j = 0; j < count; j++) data[(o * count + j) * inner + k] = pick[j];
            }
        return new Tensor<int>(data, idxShape);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Scatters_along_the_axis_and_back_propagates(int[] shape, int axis, int count)
    {
        var eng = new CpuEngine();
        int ax = axis < 0 ? shape.Length + axis : axis;
        var rng = new Random(shape.Sum() * 10 + ax);
        int n = shape.Aggregate(1, (a, b) => a * b);
        var idx = Indices(shape, ax, count, rng);
        var dstData = Enumerable.Range(0, n).Select(_ => rng.NextDouble()).ToArray();
        var srcData = Enumerable.Range(0, idx.Length).Select(_ => rng.NextDouble()).ToArray();
        var wData = Enumerable.Range(0, n).Select(_ => rng.NextDouble()).ToArray();
        var dst = new Tensor<double>(shape, new Vector<double>(dstData));
        var src = new Tensor<double>(idx.Shape.ToArray(), new Vector<double>(srcData));
        var w = new Tensor<double>(shape, new Vector<double>(wData));

        Tensor<double> y;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> g;
        using (var tape = new GradientTape<double>())
        {
            y = eng.TensorScatter(dst, idx, src, axis);
            var loss = eng.ReduceSum(eng.TensorMultiply(y, w), null, keepDims: false);
            g = tape.ComputeGradients(loss, new[] { dst, src });
        }

        int outer = shape.Take(ax).Aggregate(1, (a, b) => a * b), inner = shape.Skip(ax + 1).Aggregate(1, (a, b) => a * b);
        var expected = (double[])dstData.Clone();
        var gDst = (double[])wData.Clone();
        var gSrc = new double[idx.Length];
        var idxData = idx.ToArray();
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < count; j++)
                for (int k = 0; k < inner; k++)
                {
                    int p = (o * count + j) * inner + k, q = (o * shape[ax] + idxData[p]) * inner + k;
                    expected[q] = srcData[p];
                    gDst[q] = 0;
                    gSrc[p] = wData[q];
                }

        Assert.Equal(shape, y.Shape.ToArray());
        Assert.Equal(expected, y.ToArray());
        Assert.Equal(gDst, g[dst].ToArray());
        Assert.Equal(gSrc, g[src].ToArray());
        Assert.Equal(dstData, dst.ToArray()); // the destination is not modified
    }

    [Fact]
    public void Rejects_bad_indices_and_shapes()
    {
        var eng = new CpuEngine();
        var dst = new Tensor<float>(new[] { 2, 4 });
        Assert.Throws<IndexOutOfRangeException>(() =>
            eng.TensorScatter(dst, new Tensor<int>(new[] { 0, 4, 1, 1 }, new[] { 2, 2 }), new Tensor<float>(new[] { 2, 2 }), 1));
        Assert.Throws<ArgumentException>(() =>
            eng.TensorScatter(dst, new Tensor<int>(new[] { 0, 1 }, new[] { 1, 2 }), new Tensor<float>(new[] { 1, 2 }), 1));
        Assert.Throws<ArgumentException>(() =>
            eng.TensorScatter(dst, new Tensor<int>(new[] { 0, 1, 2, 3 }, new[] { 2, 2 }), new Tensor<float>(new[] { 2, 3 }), 1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            eng.TensorScatter(dst, new Tensor<int>(new[] { 0, 1 }, new[] { 2, 1 }), new Tensor<float>(new[] { 2, 1 }), 2));
    }
}
