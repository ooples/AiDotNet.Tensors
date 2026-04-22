// Copyright (c) AiDotNet. All rights reserved.
// CPU reference tests for the geometry / sampling ops added by Issue #217:
// Interpolate (6 modes), PadNd (4 modes), GridSample (mode + padding +
// align_corners matrix), AffineGrid3D. Ground-truth values are computed
// by hand or cross-checked against the closed-form convention used by
// torchvision.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Geometry;

public class GeometryOpsTests
{
    private readonly CpuEngine _cpu = new();

    private static Tensor<float> Arange2D(int h, int w, int start = 0)
    {
        var data = new float[h * w];
        for (int i = 0; i < data.Length; i++) data[i] = start + i;
        return new Tensor<float>(data, new[] { 1, 1, h, w });
    }

    private static void AssertClose(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(expected[i] - actual[i]) > tol * (1 + Math.Abs(expected[i])))
                throw new Xunit.Sdk.XunitException($"mismatch [{i}]: exp={expected[i]}, act={actual[i]}");
        }
    }

    // ------------------------------------------------------------------
    // Interpolate
    // ------------------------------------------------------------------

    [Fact]
    public void Interpolate_Nearest_2D_UpsampleBy2()
    {
        var input = Arange2D(2, 2);        // [[0, 1], [2, 3]]
        var output = _cpu.Interpolate(input, new[] { 4, 4 }, InterpolateMode.Nearest);
        // Each source pixel maps to a 2×2 block.
        var expected = new float[] {
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3,
            2, 2, 3, 3,
        };
        AssertClose(expected, output.AsSpan().ToArray());
    }

    [Fact]
    public void Interpolate_Bilinear_2D_UpsampleBy2_AlignCorners()
    {
        var input = Arange2D(2, 2);
        var output = _cpu.Interpolate(input, new[] { 3, 3 }, InterpolateMode.Bilinear, alignCorners: true);
        // With align_corners the corners of input and output coincide. Output
        // is evaluated at normalized coords {0, 0.5, 1} along each axis.
        var expected = new float[] {
            0.0f, 0.5f, 1.0f,
            1.0f, 1.5f, 2.0f,
            2.0f, 2.5f, 3.0f,
        };
        AssertClose(expected, output.AsSpan().ToArray());
    }

    [Fact]
    public void Interpolate_Bilinear_2D_SameSize_IsIdentity()
    {
        var input = Arange2D(3, 4);
        var output = _cpu.Interpolate(input, new[] { 3, 4 }, InterpolateMode.Bilinear);
        AssertClose(input.AsSpan().ToArray(), output.AsSpan().ToArray());
    }

    [Fact]
    public void Interpolate_Area_2D_Downsample2x_EqualsAvgPool()
    {
        // 4×4 avg pool with 2×2 window.
        var input = Arange2D(4, 4);
        var output = _cpu.Interpolate(input, new[] { 2, 2 }, InterpolateMode.Area);
        // Hand-compute: block means of 2x2 blocks of arange(16).
        // Block (0,0) = mean(0,1,4,5) = 2.5; (0,1) = mean(2,3,6,7) = 4.5
        // Block (1,0) = mean(8,9,12,13) = 10.5; (1,1) = mean(10,11,14,15) = 12.5
        var expected = new float[] { 2.5f, 4.5f, 10.5f, 12.5f };
        AssertClose(expected, output.AsSpan().ToArray());
    }

    [Fact]
    public void Interpolate_Bicubic_2D_SameSize_IsIdentity()
    {
        var input = Arange2D(4, 4);
        var output = _cpu.Interpolate(input, new[] { 4, 4 }, InterpolateMode.Bicubic, alignCorners: true);
        // Bicubic with same input/output size + align_corners should be identity.
        AssertClose(input.AsSpan().ToArray(), output.AsSpan().ToArray(), tol: 1e-3f);
    }

    [Fact]
    public void Interpolate_1D_Linear()
    {
        // [N=1, C=1, W=3] -> W=5
        var input = new Tensor<float>(new float[] { 0, 10, 20 }, new[] { 1, 1, 3 });
        var output = _cpu.Interpolate(input, new[] { 5 }, InterpolateMode.Linear, alignCorners: true);
        // align_corners=true: sample positions {0, 0.5, 1, 1.5, 2} in src -> {0, 5, 10, 15, 20}
        AssertClose(new float[] { 0, 5, 10, 15, 20 }, output.AsSpan().ToArray());
    }

    [Fact]
    public void InterpolateByScale_Factor2_MatchesInterpolate()
    {
        var input = Arange2D(2, 3);
        var a = _cpu.InterpolateByScale(input, new[] { 2.0, 2.0 }, InterpolateMode.Nearest);
        var b = _cpu.Interpolate(input, new[] { 4, 6 }, InterpolateMode.Nearest);
        AssertClose(a.AsSpan().ToArray(), b.AsSpan().ToArray());
    }

    // ------------------------------------------------------------------
    // PadNd
    // ------------------------------------------------------------------

    [Fact]
    public void PadNd_Constant_2D()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        // pad=(1, 1, 0, 0) → 1 on each side of the innermost (width) axis.
        var output = _cpu.PadNd(input, new[] { 1, 1, 0, 0 }, PadMode.Constant, -1f);
        // Expected shape [2, 4] = [[−1, 1, 2, −1], [−1, 3, 4, −1]]
        Assert.Equal(new[] { 2, 4 }, output.Shape.ToArray());
        var expected = new float[] { -1, 1, 2, -1, -1, 3, 4, -1 };
        AssertClose(expected, output.AsSpan().ToArray());
    }

    [Fact]
    public void PadNd_Reflect_1D()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var output = _cpu.PadNd(input, new[] { 2, 2 }, PadMode.Reflect);
        // Reflect (excluding boundary): [3, 2, | 1, 2, 3, 4 | 3, 2]
        var expected = new float[] { 3, 2, 1, 2, 3, 4, 3, 2 };
        AssertClose(expected, output.AsSpan().ToArray());
    }

    [Fact]
    public void PadNd_Replicate_1D()
    {
        var input = new Tensor<float>(new float[] { 5, 6, 7 }, new[] { 3 });
        var output = _cpu.PadNd(input, new[] { 2, 3 }, PadMode.Replicate);
        var expected = new float[] { 5, 5, 5, 6, 7, 7, 7, 7 };
        AssertClose(expected, output.AsSpan().ToArray());
    }

    [Fact]
    public void PadNd_Circular_1D()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3 });
        var output = _cpu.PadNd(input, new[] { 2, 2 }, PadMode.Circular);
        var expected = new float[] { 2, 3, 1, 2, 3, 1, 2 };
        AssertClose(expected, output.AsSpan().ToArray());
    }

    // ------------------------------------------------------------------
    // GridSample (extended)
    // ------------------------------------------------------------------

    [Fact]
    public void GridSample_Bilinear_Zeros_CentreSamplesInputCentre()
    {
        // 2×2 NHWC input, sample at centre.
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 1, 2, 2, 1 });
        // grid at normalised (0, 0) — that's the centre, interpolates all four corners equally.
        var grid = new Tensor<float>(new float[] { 0f, 0f }, new[] { 1, 1, 1, 2 });
        var output = _cpu.GridSample(input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: false);
        // align_corners=false centre maps to pixel coord (0.5, 0.5) — that's
        // the corner between the 4 pixels — equal weights → mean(1,2,3,4) = 2.5.
        Assert.Equal(2.5f, output.AsSpan()[0], 3);
    }

    [Fact]
    public void GridSample_Nearest_OutOfRangeWithZerosPadding()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 1, 2, 2, 1 });
        // (2, 2) normalised is way outside [-1, 1], padding=Zeros → 0.
        var grid = new Tensor<float>(new float[] { 2f, 2f }, new[] { 1, 1, 1, 2 });
        var output = _cpu.GridSample(input, grid, GridSampleMode.Nearest, GridSamplePadding.Zeros, alignCorners: false);
        Assert.Equal(0f, output.AsSpan()[0]);
    }

    [Fact]
    public void GridSample_Nearest_OutOfRangeWithBorderPadding()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 1, 2, 2, 1 });
        var grid = new Tensor<float>(new float[] { 2f, 2f }, new[] { 1, 1, 1, 2 });
        // padding=Border should clamp to the nearest corner — here pixel (1, 1) = 4.
        var output = _cpu.GridSample(input, grid, GridSampleMode.Nearest, GridSamplePadding.Border, alignCorners: false);
        Assert.Equal(4f, output.AsSpan()[0]);
    }

    // ------------------------------------------------------------------
    // AffineGrid3D
    // ------------------------------------------------------------------

    [Fact]
    public void AffineGrid3D_Identity_ProducesCenteredGrid()
    {
        // Identity affine: last row all zero, rotation part = 3x3 identity.
        var theta = new Tensor<float>(new float[] {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
        }, new[] { 1, 3, 4 });
        var grid = _cpu.AffineGrid3D(theta, 2, 2, 2, alignCorners: true);
        // With D=H=W=2 and align_corners=true, coords hit the corners → each
        // output location returns exactly (±1, ±1, ±1).
        Assert.Equal(new[] { 1, 2, 2, 2, 3 }, grid.Shape.ToArray());
        var g = grid.AsSpan();
        // Last element (d=1, h=1, w=1) should be (1, 1, 1).
        int tail = g.Length - 3;
        Assert.Equal(1f, g[tail]);
        Assert.Equal(1f, g[tail + 1]);
        Assert.Equal(1f, g[tail + 2]);
    }
}
