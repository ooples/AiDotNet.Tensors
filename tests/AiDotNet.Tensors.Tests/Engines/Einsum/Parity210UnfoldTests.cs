using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210UnfoldTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void Unfold_1D_Size3_Step1_SlidesFullOverlap()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f }, new[] { 5 });
        var r = E.TensorUnfold(x, dim: 0, size: 3, step: 1);
        // Windows: [1,2,3], [2,3,4], [3,4,5] → shape [3, 3]
        Assert.Equal(new[] { 3, 3 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 2f, 3f, 4f, 3f, 4f, 5f }, r.GetDataArray());
    }

    [Fact]
    public void Unfold_1D_Size2_Step2_NoOverlap()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 6 });
        var r = E.TensorUnfold(x, dim: 0, size: 2, step: 2);
        Assert.Equal(new[] { 3, 2 }, r.Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, r.GetDataArray());
    }

    [Fact]
    public void Unfold_2D_AlongAxis1_PreservesLeadingDim()
    {
        // shape [2, 4] — unfold axis 1 with size=2, step=1 → [2, 3, 2]
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 10f, 20f, 30f, 40f }, new[] { 2, 4 });
        var r = E.TensorUnfold(x, dim: 1, size: 2, step: 1);
        Assert.Equal(new[] { 2, 3, 2 }, r.Shape.ToArray());
        // Row 0 windows: [1,2], [2,3], [3,4]
        // Row 1 windows: [10,20], [20,30], [30,40]
        Assert.Equal(new[] {
            1f, 2f, 2f, 3f, 3f, 4f,
            10f, 20f, 20f, 30f, 30f, 40f,
        }, r.GetDataArray());
    }

    [Fact]
    public void Unfold_SizeExceedsDim_Throws()
    {
        var x = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        Assert.Throws<ArgumentException>(() => E.TensorUnfold(x, dim: 0, size: 3, step: 1));
    }

    [Fact]
    public void Unfold_NegativeStep_Throws()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        Assert.Throws<ArgumentOutOfRangeException>(() => E.TensorUnfold(x, dim: 0, size: 2, step: 0));
    }
}
