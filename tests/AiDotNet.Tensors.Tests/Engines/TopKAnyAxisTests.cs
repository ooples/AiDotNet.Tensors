using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// TopK / TensorTopK on any rank and axis (torch.topk): every slice along the axis gives its k best values best-first,
/// ties to the lower index, compared against a brute-force stable sort of the same slice.
/// </summary>
public class TopKAnyAxisTests
{
    internal static (float[] Values, int[] Indices) Reference(float[] data, int[] shape, int axis, int k, bool largest)
    {
        int outer = 1, inner = 1, n = shape[axis];
        for (int i = 0; i < axis; i++) outer *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        var values = new float[outer * k * inner];
        var indices = new int[outer * k * inner];
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < inner; j++)
            {
                var order = Enumerable.Range(0, n)
                    .OrderBy(a => largest ? -data[o * n * inner + a * inner + j] : data[o * n * inner + a * inner + j])
                    .ThenBy(a => a).Take(k).ToArray();
                for (int r = 0; r < k; r++)
                {
                    values[o * k * inner + r * inner + j] = data[o * n * inner + order[r] * inner + j];
                    indices[o * k * inner + r * inner + j] = order[r];
                }
            }
        return (values, indices);
    }

    internal static float[] Random(int length, int seed, int distinct)
    {
        var rng = new Random(seed);
        // A small value range forces ties, which pin the lower-index rule.
        return Enumerable.Range(0, length).Select(_ => (float)rng.Next(distinct)).ToArray();
    }

    [Theory]
    [InlineData(new[] { 2, 3, 50 }, 2, 5, true)]
    [InlineData(new[] { 2, 3, 50 }, 2, 5, false)]
    [InlineData(new[] { 4, 37, 3 }, 1, 4, true)]
    [InlineData(new[] { 9, 2, 2 }, 0, 9, true)]
    [InlineData(new[] { 3, 1000 }, 1, 32, true)]
    [InlineData(new[] { 64 }, 0, 1, false)]
    public void TopK_matches_a_stable_sort_on_every_slice(int[] shape, int axis, int k, bool largest)
    {
        var engine = new CpuEngine();
        var data = Random(shape.Aggregate(1, (a, b) => a * b), 7, 11);
        var (values, indices) = engine.TopK(new Tensor<float>(data, shape), k, axis, largest);
        var (ev, ei) = Reference(data, shape, axis, k, largest);
        var outShape = (int[])shape.Clone();
        outShape[axis] = k;
        Assert.Equal(outShape, values.Shape.ToArray());
        Assert.Equal(ev, values.ToArray());
        Assert.Equal(ei, indices.ToArray());
    }

    [Fact]
    public void TensorTopK_takes_any_rank_and_a_negative_axis()
    {
        var engine = new CpuEngine();
        int[] shape = { 2, 4, 300 };
        var data = Random(2 * 4 * 300, 3, 50);
        var values = engine.TensorTopK(new Tensor<float>(data, shape), 16, -1, out var indices);
        var (ev, ei) = Reference(data, shape, 2, 16, largest: true);
        Assert.Equal(new[] { 2, 4, 16 }, values.Shape.ToArray());
        Assert.Equal(ev, values.ToArray());
        Assert.Equal(ei, indices.ToArray());
    }

    [Fact]
    public void TensorTopK_clamps_k_to_the_axis_length()
    {
        var engine = new CpuEngine();
        var values = engine.TensorTopK(new Tensor<float>(new float[] { 3, 1, 2, 5, 4, 6 }, new[] { 2, 3 }), 10, 1, out var indices);
        Assert.Equal(new float[] { 3, 2, 1, 6, 5, 4 }, values.ToArray());
        Assert.Equal(new[] { 0, 2, 1, 2, 0, 1 }, indices.ToArray());
    }
}
