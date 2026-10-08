using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Geometry;

/// <summary>
/// Pins <c>GridSample</c>'s tensor layout: input and output are NCHW, the grid is
/// <c>[N, outH, outW, 2]</c> (issue #1032).
/// </summary>
/// <remarks>
/// The <c>IEngine</c> docs used to say NHWC and told callers to transpose around the call, while every
/// implementation (CPU and all six GPU kernels) has always read <c>((n * C + c) * H + y) * W + x</c>.
/// Each channel here holds a distinct constant, so a channel-last reading of the same buffer would mix
/// channels and fail the value check as well as the shape check.
/// </remarks>
public class GridSampleLayoutTests
{
    private const int N = 1, C = 6, H = 5, W = 3, OutH = 6, OutW = 6;

    private readonly CpuEngine _engine = new();

    private static Tensor<double> ChannelConstantInput()
    {
        var t = new Tensor<double>([N, C, H, W]);
        for (int c = 0; c < C; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    t[0, c, y, x] = c + 1;
        return t;
    }

    /// <summary>Coordinates in [-0.5, 0.5], so every bilinear tap is in bounds under either corner mode.</summary>
    private static Tensor<double> InteriorGrid()
    {
        var g = new Tensor<double>([N, OutH, OutW, 2]);
        for (int y = 0; y < OutH; y++)
            for (int x = 0; x < OutW; x++)
            {
                g[0, y, x, 0] = -0.5 + x / (double)(OutW - 1);
                g[0, y, x, 1] = -0.5 + y / (double)(OutH - 1);
            }
        return g;
    }

    [Fact]
    public void Forward_IsNchwInAndOut()
    {
        var output = _engine.GridSample(ChannelConstantInput(), InteriorGrid());

        Assert.Equal(new[] { N, C, OutH, OutW }, output.Shape.ToArray());
        for (int c = 0; c < C; c++)
            for (int y = 0; y < OutH; y++)
                for (int x = 0; x < OutW; x++)
                    Assert.Equal(c + 1, output[0, c, y, x], 12);
    }

    [Fact]
    public void ExplicitOverload_IsNchwInAndOut()
    {
        var output = _engine.GridSample(ChannelConstantInput(), InteriorGrid(),
            GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: true);

        Assert.Equal(new[] { N, C, OutH, OutW }, output.Shape.ToArray());
        for (int c = 0; c < C; c++)
            Assert.Equal(c + 1, output[0, c, OutH / 2, OutW / 2], 12);
    }

    [Fact]
    public void Backward_GradientsHaveNchwInputShapeAndGridShape()
    {
        var input = ChannelConstantInput();
        var grid = InteriorGrid();
        var gradOutput = new Tensor<double>([N, C, OutH, OutW]);
        for (int i = 0; i < gradOutput.Length; i++) gradOutput[i] = 1.0;

        var gradInput = _engine.GridSampleBackwardInput(gradOutput, grid, new[] { N, C, H, W });
        var gradGrid = _engine.GridSampleBackwardGrid(gradOutput, input, grid);

        Assert.Equal(new[] { N, C, H, W }, gradInput.Shape.ToArray());
        Assert.Equal(new[] { N, OutH, OutW, 2 }, gradGrid.Shape.ToArray());

        // Bilinear weights sum to one per sample, so each channel receives exactly OutH * OutW of
        // gradient mass. A channel-last reading would spread it across the wrong channel count.
        for (int c = 0; c < C; c++)
        {
            double mass = 0;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    mass += gradInput[0, c, y, x];
            Assert.Equal(OutH * OutW, mass, 9);
        }
    }
}