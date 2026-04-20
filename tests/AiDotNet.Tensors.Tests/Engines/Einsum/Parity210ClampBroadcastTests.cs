using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210ClampBroadcastTests
{
    private static CpuEngine E => new CpuEngine();

    [Fact]
    public void ClampTensor_ExactShape_StillWorks()
    {
        var x = new Tensor<float>(new[] { 1f, 5f, 10f, 2f }, new[] { 4 });
        var lo = new Tensor<float>(new[] { 3f, 3f, 3f, 3f }, new[] { 4 });
        var hi = new Tensor<float>(new[] { 8f, 8f, 8f, 8f }, new[] { 4 });
        var r = E.TensorClampTensor(x, lo, hi);
        Assert.Equal(new[] { 3f, 5f, 8f, 3f }, r.GetDataArray());
    }

    [Fact]
    public void ClampTensor_BroadcastsScalarBounds()
    {
        // min as [1], max as [1] broadcast to the whole tensor.
        var x = new Tensor<float>(new[] { -1f, 2f, 7f, 15f }, new[] { 4 });
        var lo = new Tensor<float>(new[] { 0f }, new[] { 1 });
        var hi = new Tensor<float>(new[] { 10f }, new[] { 1 });
        var r = E.TensorClampTensor(x, lo, hi);
        Assert.Equal(new[] { 0f, 2f, 7f, 10f }, r.GetDataArray());
    }

    [Fact]
    public void ClampTensor_BroadcastsLowerRankBounds()
    {
        // Tensor is [2, 3], min is [3] (broadcasts to each row).
        var x = new Tensor<float>(new[] { -1f, 2f, 5f, -2f, 3f, 6f }, new[] { 2, 3 });
        var lo = new Tensor<float>(new[] { 0f, 1f, 2f }, new[] { 3 });
        var r = E.TensorClampTensor(x, lo, null);
        // Row 0: [-1, 2, 5] clamped by [0, 1, 2] → [0, 2, 5]
        // Row 1: [-2, 3, 6] clamped by [0, 1, 2] → [0, 3, 6]
        Assert.Equal(new[] { 0f, 2f, 5f, 0f, 3f, 6f }, r.GetDataArray());
    }

    [Fact]
    public void ClampTensor_BroadcastsWithDimOne()
    {
        // Tensor [2, 3], min [2, 1] → min[i, :] = min[i, 0] for all columns.
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var lo = new Tensor<float>(new[] { 2.5f, 5.5f }, new[] { 2, 1 });
        var r = E.TensorClampTensor(x, lo, null);
        // Row 0 min=2.5: [2.5, 2.5, 3]
        // Row 1 min=5.5: [5.5, 5.5, 6]
        Assert.Equal(new[] { 2.5f, 2.5f, 3f, 5.5f, 5.5f, 6f }, r.GetDataArray());
    }

    [Fact]
    public void ClampTensor_IncompatibleShape_Throws()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var lo = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });  // right-align: [2] vs [3] — not 1-compatible
        Assert.Throws<ArgumentException>(() => E.TensorClampTensor(x, lo, null));
    }

    [Fact]
    public void ClampTensor_BoundsHigherRankThanTensor_Throws()
    {
        var x = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var lo = new Tensor<float>(new[] { 0f, 0f }, new[] { 1, 2 });
        Assert.Throws<ArgumentException>(() => E.TensorClampTensor(x, lo, null));
    }
}
