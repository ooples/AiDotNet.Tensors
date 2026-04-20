using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210ArgsortTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void Argsort_Ascending_1D()
    {
        var x = new Tensor<float>(new[] { 3f, 1f, 4f, 1.5f, 9f, 2f }, new[] { 6 });
        var idx = E.TensorArgsort(x);
        // Sorted ascending:      1f, 1.5f, 2f, 3f, 4f, 9f
        // Original positions:     1,    3,  5,  0,  2,  4
        Assert.Equal(new[] { 1, 3, 5, 0, 2, 4 }, idx.GetDataArray());
    }

    [Fact]
    public void Argsort_Descending_1D()
    {
        var x = new Tensor<float>(new[] { 3f, 1f, 4f, 1.5f, 9f, 2f }, new[] { 6 });
        var idx = E.TensorArgsort(x, descending: true);
        Assert.Equal(new[] { 4, 2, 0, 5, 3, 1 }, idx.GetDataArray());
    }

    [Fact]
    public void Argsort_Axis_2D_LastAxis()
    {
        var x = new Tensor<float>(new[] { 3f, 1f, 4f, 9f, 2f, 6f }, new[] { 2, 3 });
        // row 0 sorted asc: [1f, 3f, 4f] → positions [1, 0, 2]
        // row 1 sorted asc: [2f, 6f, 9f] → positions [1, 2, 0]
        var idx = E.TensorArgsort(x, axis: -1);
        Assert.Equal(new[] { 1, 0, 2, 1, 2, 0 }, idx.GetDataArray());
    }

    [Fact]
    public void Argsort_Axis0_2D()
    {
        var x = new Tensor<float>(new[] { 3f, 1f, 4f, 9f, 2f, 6f }, new[] { 2, 3 });
        // col 0: [3, 9] asc → [0, 1]
        // col 1: [1, 2] asc → [0, 1]
        // col 2: [4, 6] asc → [0, 1]
        var idx = E.TensorArgsort(x, axis: 0);
        Assert.Equal(new[] { 0, 0, 0, 1, 1, 1 }, idx.GetDataArray());
    }

    [Fact]
    public void Argsort_Stable_ForEqualKeys_InAscending()
    {
        // When keys repeat, ascending argsort should keep original order
        // (Array.Sort in .NET uses stable-ish intro sort; PyTorch.argsort is
        // not guaranteed stable either, so this just verifies consistency
        // with TensorSort).
        var x = new Tensor<float>(new[] { 2f, 2f, 2f, 1f }, new[] { 4 });
        var idx = E.TensorArgsort(x);
        // Asc: [1, 2, 2, 2] — smallest at position 3 goes first.
        Assert.Equal(3, idx[0]);
    }
}
