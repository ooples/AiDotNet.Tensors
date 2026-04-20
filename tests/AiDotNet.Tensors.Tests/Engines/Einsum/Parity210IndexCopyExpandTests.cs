using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210IndexCopyExpandTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static Tensor<int> I(int[] data, params int[] shape) => new Tensor<int>(data, shape);

    [Fact]
    public void IndexCopy_OverwritesAtIndices()
    {
        var x = T(new[] { 10f, 20f, 30f, 40f }, 4);
        var idx = I(new[] { 0, 3 }, 2);
        var src = T(new[] { 100f, 400f }, 2);
        var r = E.TensorIndexCopy(x, 0, idx, src);
        Assert.Equal(new[] { 100f, 20f, 30f, 400f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void IndexCopy_2D_Axis1_CopiesColumns()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, 2, 3);
        var idx = I(new[] { 1 }, 1);
        var src = T(new[] { 99f, 88f }, 2, 1);
        var r = E.TensorIndexCopy(x, 1, idx, src);
        // Column 1 replaced with src.
        Assert.Equal(new[] { 1f, 99f, 3f, 4f, 88f, 6f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void IndexCopy_DuplicateIndices_KeepsLast()
    {
        var x = T(new[] { 0f, 0f }, 2);
        var idx = I(new[] { 0, 0 }, 2);
        var src = T(new[] { 5f, 7f }, 2);
        var r = E.TensorIndexCopy(x, 0, idx, src);
        Assert.Equal(7f, r[0]);  // last write wins
    }

    [Fact]
    public void ExpandAs_MatchesTargetShape()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var target = T(new float[6], 2, 3);
        var r = E.TensorExpandAs(x, target);
        Assert.Equal(new[] { 2, 3 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 1f, 2f, 3f }, r.AsSpan().ToArray());
    }
}
