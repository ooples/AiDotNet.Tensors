using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210MeshgridTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);

    [Fact]
    public void Meshgrid_IJ_XShapeFollowsFirstInput()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var y = T(new[] { 4f, 5f }, 2);
        var grids = E.TensorMeshgrid(new[] { x, y }, "ij");
        Assert.Equal(2, grids.Length);
        Assert.Equal(new[] { 3, 2 }, grids[0].Shape.ToArray());
        Assert.Equal(new[] { 3, 2 }, grids[1].Shape.ToArray());
        // X_ij[i,j] = x[i]
        Assert.Equal(new[] { 1f, 1f, 2f, 2f, 3f, 3f }, grids[0].AsSpan().ToArray());
        // Y_ij[i,j] = y[j]
        Assert.Equal(new[] { 4f, 5f, 4f, 5f, 4f, 5f }, grids[1].AsSpan().ToArray());
    }

    [Fact]
    public void Meshgrid_XY_SwapsFirstTwoAxes()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var y = T(new[] { 4f, 5f }, 2);
        var grids = E.TensorMeshgrid(new[] { x, y }, "xy");
        // XY: X.shape = (2, 3), Y.shape = (2, 3)
        Assert.Equal(new[] { 2, 3 }, grids[0].Shape.ToArray());
        Assert.Equal(new[] { 2, 3 }, grids[1].Shape.ToArray());
        // X_xy[i,j] = x[j]
        Assert.Equal(new[] { 1f, 2f, 3f, 1f, 2f, 3f }, grids[0].AsSpan().ToArray());
        // Y_xy[i,j] = y[i]
        Assert.Equal(new[] { 4f, 4f, 4f, 5f, 5f, 5f }, grids[1].AsSpan().ToArray());
    }

    [Fact]
    public void Meshgrid_SingleInput_ReturnsOneGrid()
    {
        var x = T(new[] { 1f, 2f, 3f }, 3);
        var grids = E.TensorMeshgrid(new[] { x });
        Assert.Single(grids);
        Assert.Equal(new[] { 3 }, grids[0].Shape.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f }, grids[0].AsSpan().ToArray());
    }

    [Fact]
    public void Meshgrid_ThreeInputs_ReturnsThreeGrids()
    {
        var x = T(new[] { 1f, 2f }, 2);
        var y = T(new[] { 3f, 4f }, 2);
        var z = T(new[] { 5f, 6f }, 2);
        var grids = E.TensorMeshgrid(new[] { x, y, z });
        Assert.Equal(3, grids.Length);
        foreach (var g in grids)
            Assert.Equal(new[] { 2, 2, 2 }, g.Shape.ToArray());
    }

    [Fact]
    public void Meshgrid_InvalidIndexing_Throws()
    {
        var x = T(new[] { 1f, 2f }, 2);
        Assert.Throws<System.ArgumentException>(
            () => E.TensorMeshgrid(new[] { x }, "invalid"));
    }

    [Fact]
    public void Meshgrid_NonRank1_Throws()
    {
        var x = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        Assert.Throws<System.ArgumentException>(() => E.TensorMeshgrid(new[] { x }));
    }
}
