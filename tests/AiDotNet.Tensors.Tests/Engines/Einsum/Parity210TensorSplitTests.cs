using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210TensorSplitTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void TensorSplit_ByCount_EvenDivision()
    {
        // len 6, sections 3 → three chunks of 2
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 6 });
        var parts = E.TensorTensorSplit(x, sections: 3, dim: 0);
        Assert.Equal(3, parts.Length);
        Assert.Equal(new[] { 1f, 2f }, parts[0].GetDataArray());
        Assert.Equal(new[] { 3f, 4f }, parts[1].GetDataArray());
        Assert.Equal(new[] { 5f, 6f }, parts[2].GetDataArray());
    }

    [Fact]
    public void TensorSplit_ByCount_UnevenDivision()
    {
        // len 7, sections 3 → first `7%3 = 1` chunks get ceil = 3, rest get floor = 2.
        // Expected split sizes: [3, 2, 2]
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f }, new[] { 7 });
        var parts = E.TensorTensorSplit(x, sections: 3, dim: 0);
        Assert.Equal(3, parts.Length);
        Assert.Equal(new[] { 1f, 2f, 3f }, parts[0].GetDataArray());
        Assert.Equal(new[] { 4f, 5f }, parts[1].GetDataArray());
        Assert.Equal(new[] { 6f, 7f }, parts[2].GetDataArray());
    }

    [Fact]
    public void TensorSplit_ByIndices_VariableChunks()
    {
        var x = new Tensor<float>(new[] { 10f, 20f, 30f, 40f, 50f, 60f }, new[] { 6 });
        // indices [2, 5] → chunks [0..2], [2..5], [5..6]
        var parts = E.TensorTensorSplit(x, indices: new[] { 2, 5 }, dim: 0);
        Assert.Equal(3, parts.Length);
        Assert.Equal(new[] { 10f, 20f }, parts[0].GetDataArray());
        Assert.Equal(new[] { 30f, 40f, 50f }, parts[1].GetDataArray());
        Assert.Equal(new[] { 60f }, parts[2].GetDataArray());
    }

    [Fact]
    public void TensorSplit_ByIndices_OutOfBoundsClampsToEmpty()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        // index 10 > size → last chunks empty
        var parts = E.TensorTensorSplit(x, indices: new[] { 2, 10 }, dim: 0);
        Assert.Equal(3, parts.Length);
        Assert.Equal(new[] { 1f, 2f }, parts[0].GetDataArray());
        Assert.Equal(new[] { 3f }, parts[1].GetDataArray());
        Assert.Empty(parts[2].GetDataArray());
    }

    [Fact]
    public void TensorSplit_2D_AlongAxis1()
    {
        // shape [2, 4] split by count=2 along axis 1 → two [2, 2]
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 10f, 20f, 30f, 40f }, new[] { 2, 4 });
        var parts = E.TensorTensorSplit(x, sections: 2, dim: 1);
        Assert.Equal(2, parts.Length);
        Assert.Equal(new[] { 2, 2 }, parts[0].Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 10f, 20f }, parts[0].GetDataArray());
        Assert.Equal(new[] { 3f, 4f, 30f, 40f }, parts[1].GetDataArray());
    }
}
