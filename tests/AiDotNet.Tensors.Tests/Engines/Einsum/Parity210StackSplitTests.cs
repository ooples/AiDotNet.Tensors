using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210StackSplitTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void HStack_1D_ConcatsAlongAxis0()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 3f, 4f, 5f }, 3);
        var r = E.TensorHStack(new[] { a, b });
        Assert.Equal(new[] { 5 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void HStack_2D_ConcatsAlongAxis1()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 5f, 6f }, 2, 1);
        var r = E.TensorHStack(new[] { a, b });
        Assert.Equal(new[] { 2, 3 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 5f, 3f, 4f, 6f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void VStack_1D_BecomesRows()
    {
        var a = T(new[] { 1f, 2f, 3f }, 3);
        var b = T(new[] { 4f, 5f, 6f }, 3);
        var r = E.TensorVStack(new[] { a, b });
        Assert.Equal(new[] { 2, 3 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void VStack_2D_ConcatsAlongAxis0()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 5f, 6f }, 1, 2);
        var r = E.TensorVStack(new[] { a, b });
        Assert.Equal(new[] { 3, 2 }, r.Shape.ToArray());
    }

    [Fact]
    public void DStack_2D_PromotesTo3DAndConcatsAlongAxis2()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 5f, 6f, 7f, 8f }, 2, 2);
        var r = E.TensorDStack(new[] { a, b });
        Assert.Equal(new[] { 1, 2, 4 }, r.Shape.ToArray());
    }

    [Fact]
    public void ColumnStack_1D_BecomesColumns()
    {
        var a = T(new[] { 1f, 2f, 3f }, 3);
        var b = T(new[] { 4f, 5f, 6f }, 3);
        var r = E.TensorColumnStack(new[] { a, b });
        Assert.Equal(new[] { 3, 2 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 4f, 2f, 5f, 3f, 6f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void RowStack_IsAliasForVStack()
    {
        var a = T(new[] { 1f, 2f }, 2);
        var b = T(new[] { 3f, 4f }, 2);
        var r1 = E.TensorRowStack(new[] { a, b });
        var r2 = E.TensorVStack(new[] { a, b });
        Assert.Equal(r2.AsSpan().ToArray(), r1.AsSpan().ToArray());
    }

    [Fact]
    public void HSplit_2D_SplitsAlongAxis1()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var parts = E.TensorHSplit(x, 3);
        Assert.Equal(3, parts.Length);
        Assert.Equal(new[] { 2, 1 }, parts[0].Shape.ToArray());
    }

    [Fact]
    public void VSplit_RequiresRank2_Throws_OnRank1()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        Assert.Throws<System.ArgumentException>(() => E.TensorVSplit(x, 3));
    }

    [Fact]
    public void DSplit_RequiresRank3_Throws_OnRank2()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        Assert.Throws<System.ArgumentException>(() => E.TensorDSplit(x, 2));
    }
}
