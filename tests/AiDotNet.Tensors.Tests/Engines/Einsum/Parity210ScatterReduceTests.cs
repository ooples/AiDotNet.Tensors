using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210ScatterReduceTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);

    [Fact]
    public void ScatterReduce_Sum_IncludeSelf_AccumulatesIntoBase()
    {
        var x = T(new[] { 10f, 20f, 30f }, 3);
        var idx = I(new[] { 0, 2, 0, 2 }, 4);
        var src = T(new[] { 1f, 2f, 3f, 4f }, 4);
        var r = E.TensorScatterReduce(x, 0, idx, src, ScatterReduceMode.Sum, includeSelf: true);
        // position 0 = 10 + 1 + 3 = 14; position 1 unchanged = 20; position 2 = 30 + 2 + 4 = 36.
        Assert.Equal(new[] { 14f, 20f, 36f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ScatterReduce_Sum_ExcludeSelf_StartsFromZero()
    {
        var x = T(new[] { 10f, 20f, 30f }, 3);
        var idx = I(new[] { 0, 2 }, 2);
        var src = T(new[] { 1f, 2f }, 2);
        var r = E.TensorScatterReduce(x, 0, idx, src, ScatterReduceMode.Sum, includeSelf: false);
        // position 0 = 0 + 1 = 1 (self wiped); position 1 = 20 (untouched); position 2 = 0 + 2 = 2.
        Assert.Equal(new[] { 1f, 20f, 2f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ScatterReduce_Prod_IncludeSelf()
    {
        var x = T(new[] { 2f, 1f, 3f }, 3);
        var idx = I(new[] { 0, 0 }, 2);
        var src = T(new[] { 3f, 4f }, 2);
        var r = E.TensorScatterReduce(x, 0, idx, src, ScatterReduceMode.Prod);
        // position 0 = 2 * 3 * 4 = 24; others unchanged.
        Assert.Equal(new[] { 24f, 1f, 3f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ScatterReduce_Mean_IncludesSelf()
    {
        var x = T(new[] { 10f, 0f, 0f }, 3);
        var idx = I(new[] { 0, 0 }, 2);
        var src = T(new[] { 4f, 6f }, 2);
        var r = E.TensorScatterReduce(x, 0, idx, src, ScatterReduceMode.Mean, includeSelf: true);
        // position 0: (10 + 4 + 6) / 3 = 20/3 ≈ 6.6667.
        Assert.True(System.MathF.Abs(r[0] - 20f / 3f) < 1e-5f, $"got {r[0]}");
    }

    [Fact]
    public void ScatterReduce_AMin_TakesMinAcrossSlot()
    {
        var x = T(new[] { 5f, 5f }, 2);
        var idx = I(new[] { 0, 0, 1 }, 3);
        var src = T(new[] { 2f, 7f, 10f }, 3);
        var r = E.TensorScatterReduce(x, 0, idx, src, ScatterReduceMode.AMin);
        // position 0 = min(5, 2, 7) = 2; position 1 = min(5, 10) = 5.
        Assert.Equal(new[] { 2f, 5f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ScatterReduce_AMax_TakesMaxAcrossSlot()
    {
        var x = T(new[] { 5f, 5f }, 2);
        var idx = I(new[] { 0, 0, 1 }, 3);
        var src = T(new[] { 2f, 7f, 10f }, 3);
        var r = E.TensorScatterReduce(x, 0, idx, src, ScatterReduceMode.AMax);
        Assert.Equal(new[] { 7f, 10f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void ScatterReduce_Shape_2D_AlongAxis0()
    {
        var x = T(new[] { 0f, 0f, 0f, 0f, 0f, 0f }, 2, 3);
        var idx = I(new[] { 0, 1, 0 }, 1, 3);
        var src = T(new[] { 1f, 2f, 3f }, 1, 3);
        var r = E.TensorScatterReduce(x, 0, idx, src, ScatterReduceMode.Sum, includeSelf: true);
        // row 0: [+1, 0, +3]; row 1: [0, +2, 0].
        Assert.Equal(new[] { 1f, 0f, 3f, 0f, 2f, 0f }, r.AsSpan().ToArray());
    }
}
