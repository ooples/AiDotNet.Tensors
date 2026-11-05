using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Geometry;

/// <summary>
/// The mode-aware <c>GridSample</c> gradient for every mode, padding and align_corners convention, checked
/// element by element against central finite differences of the forward.
/// </summary>
/// <remarks>
/// <para>
/// Before #1680 the backward threw <see cref="NotSupportedException"/> for anything except Bilinear + Zeros.
/// </para>
/// <para>
/// The sampling points lie between -3 and S+2 pixels, so every padding rule is exercised: taps outside the
/// image read 0 (Zeros), the clamped edge (Border) or the reflected pixel (Reflection). Each point keeps its
/// fractional part at least 0.15 from a tap boundary (and from the .5 rounding tie for Nearest), so the
/// finite-difference step never crosses a discontinuity. The loss weights every output by a distinct
/// random factor, so a gradient routed to the wrong pixel or the wrong axis cannot cancel out.
/// </para>
/// </remarks>
public class GridSampleModeBackwardTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public GridSampleModeBackwardTests(ITestOutputHelper o) => _out = o;

    private const int N = 2, C = 2, H = 4, W = 5, OutH = 3, OutW = 4;

    public static IEnumerable<object[]> Combinations()
    {
        foreach (GridSampleMode mode in Enum.GetValues(typeof(GridSampleMode)))
        foreach (GridSamplePadding padding in Enum.GetValues(typeof(GridSamplePadding)))
        foreach (bool align in new[] { false, true })
            yield return new object[] { mode, padding, align };
    }

    private static Tensor<double> RandomTensor(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    /// <summary>Inverse of the forward's pixel mapping, so points can be placed in pixel space.</summary>
    private static double PixelToNormalized(double pixel, int size, bool align) =>
        align ? pixel * 2.0 / (size - 1) - 1.0 : (2.0 * pixel + 1.0) / size - 1.0;

    private static Tensor<double> SafeGrid(GridSampleMode mode, bool align, int seed)
    {
        var rng = new Random(seed);
        var grid = new Tensor<double>([N, OutH, OutW, 2]);
        for (int p = 0; p < grid.Length / 2; p++)
        {
            for (int axis = 0; axis < 2; axis++)
            {
                int size = axis == 0 ? W : H;
                int whole = rng.Next(-3, size + 2);
                double frac;
                do frac = 0.15 + rng.NextDouble() * 0.7;
                while (mode == GridSampleMode.Nearest && Math.Abs(frac - 0.5) < 0.15);
                grid[p * 2 + axis] = PixelToNormalized(whole + frac, size, align);
            }
        }
        return grid;
    }

    [Theory]
    [MemberData(nameof(Combinations))]
    public void TapeGradient_MatchesFiniteDifferences(GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        var input = RandomTensor([N, C, H, W], 11);
        var grid = SafeGrid(mode, alignCorners, 12);
        var weights = RandomTensor([N, C, OutH, OutW], 13);

        double Loss()
        {
            var y = _engine.GridSample(input, grid, mode, padding, alignCorners);
            double s = 0;
            for (int i = 0; i < y.Length; i++) s += y[i] * weights[i];
            return s;
        }

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            tape.BindEngineIfUnset(_engine);
            var y = _engine.GridSample(input, grid, mode, padding, alignCorners);
            var loss = _engine.ReduceSum(_engine.TensorMultiply(y, weights), null);
            grads = tape.ComputeGradients(loss, [input, grid]);
        }

        const double h = 1e-6;
        double worst = 0;
        foreach (var (name, t) in new[] { ("input", input), ("grid", grid) })
        {
            Assert.True(grads.TryGetValue(t, out var g) && g is not null, $"no gradient for {name}");
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + h; double lp = Loss();
                t[i] = orig - h; double lm = Loss();
                t[i] = orig;
                double numerical = (lp - lm) / (2 * h);
                double err = Math.Abs(g![i] - numerical) / Math.Max(1.0, Math.Abs(numerical));
                worst = Math.Max(worst, err);
                Assert.True(err < 1e-6,
                    $"{mode}/{padding}/align={alignCorners}: d/d{name}[{i}] analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
        _out.WriteLine($"{mode}/{padding}/align={alignCorners}: worst relative error {worst:E3}");
    }

    /// <summary>
    /// The engine entry points the tape and compiled backward call, checked directly, and Nearest's grid
    /// gradient is exactly zero (it is piecewise constant).
    /// </summary>
    [Theory]
    [InlineData(GridSamplePadding.Zeros)]
    [InlineData(GridSamplePadding.Border)]
    [InlineData(GridSamplePadding.Reflection)]
    public void Nearest_GridGradientIsZero_InputGradientCountsEachRead(GridSamplePadding padding)
    {
        var input = RandomTensor([N, C, H, W], 21);
        var grid = SafeGrid(GridSampleMode.Nearest, align: false, 22);
        var gradOut = new Tensor<double>([N, C, OutH, OutW]);
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = 1.0;

        var dGrid = _engine.GridSampleBackwardGrid(gradOut, input, grid, GridSampleMode.Nearest, padding, false);
        for (int i = 0; i < dGrid.Length; i++) Assert.Equal(0.0, dGrid[i]);

        // With an all-ones upstream gradient, each input pixel's gradient is the number of output points
        // that read it, and the total is the number of points that read a real pixel.
        var dInput = _engine.GridSampleBackwardInput(gradOut, grid, input.Shape.ToArray(),
            GridSampleMode.Nearest, padding, false);
        double total = 0;
        for (int i = 0; i < dInput.Length; i++)
        {
            Assert.Equal(Math.Round(dInput[i]), dInput[i]);
            total += dInput[i];
        }
        double expected = padding == GridSamplePadding.Zeros ? double.NaN : N * C * OutH * OutW;
        _out.WriteLine($"Nearest/{padding}: total input gradient {total}");
        if (!double.IsNaN(expected)) Assert.Equal(expected, total);
        else Assert.True(total <= N * C * OutH * OutW);
    }
}
