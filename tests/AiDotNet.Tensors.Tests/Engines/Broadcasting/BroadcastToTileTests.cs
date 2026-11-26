using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Broadcasting;

/// <summary>
/// The CPU engine's <c>TensorBroadcastTo</c> expands size-1 axes with a tile (after left-padding the rank), not by
/// adding a zero tensor of the target shape. These check the values on the CPU for the tile's cases: an expanded
/// middle axis, a padded rank with expanded leading axes, and a size-1 axis broadcast to 0, which has no tile multiple.
/// </summary>
public class BroadcastToTileTests
{
    private static Tensor<float> Iota(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, i + 1);
        return t;
    }

    [Fact]
    public void A_size_one_middle_axis_expands_by_repeating_its_slice()
    {
        var x = Iota(2, 1, 3);
        var y = new CpuEngine().TensorBroadcastTo(x, new[] { 2, 4, 3 });
        Assert.Equal(new[] { 2, 4, 3 }, y.Shape.ToArray());
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 4; b++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(x[a, 0, c], y[a, b, c]);
    }

    [Fact]
    public void A_lower_rank_input_is_padded_and_expanded_on_the_new_and_size_one_axes()
    {
        var x = Iota(3, 1);
        var y = new CpuEngine().TensorBroadcastTo(x, new[] { 2, 3, 4 });
        Assert.Equal(new[] { 2, 3, 4 }, y.Shape.ToArray());
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 3; b++)
                for (int c = 0; c < 4; c++)
                    Assert.Equal(x[b, 0], y[a, b, c]);
    }

    [Fact]
    public void A_size_one_axis_broadcast_to_zero_gives_an_empty_tensor_of_the_target_shape()
    {
        var y = new CpuEngine().TensorBroadcastTo(Iota(1, 3), new[] { 0, 3 });
        Assert.Equal(new[] { 0, 3 }, y.Shape.ToArray());
        Assert.Equal(0, y.Length);
    }
}
