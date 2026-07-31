using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Geometry;

/// <summary>
/// Both <c>GridSample</c> overloads must share one sampling convention: torchvision's defaults,
/// <c>align_corners=False</c> with <c>padding_mode='zeros'</c>.
/// </summary>
/// <remarks>
/// <para>
/// The narrow <c>GridSample(input, grid)</c> overload used to compute its pixel mapping as
/// <c>(size - 1) / 2</c> — the <c>align_corners=TRUE</c> convention — and clamp its four sample indices
/// into <c>[0, size-1]</c>, which is BORDER padding. So despite <c>CpuEngine.Geometry.cs</c>'s header
/// describing it as a "torchvision-default shim", it implemented neither default, and disagreed with
/// <c>GridSample(..., Bilinear, Zeros, alignCorners:false)</c> by 5.553e-2 on interior coordinates.
/// </para>
/// <para>
/// It was found because the mode-aware overload's gradient could not be wired up: finite differences
/// rejected the <c>GridSampleBackwardInput</c>/<c>GridSampleBackwardGrid</c> kernels even for
/// Bilinear + Zeros + alignCorners=false (analytical 0.0390 vs numerical 0.1862). Those kernels were
/// the adjoint of the narrow overload's convention, not of the forward they were being asked to serve.
/// </para>
/// </remarks>
public class GridSampleForwardConventionTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public GridSampleForwardConventionTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Input()
    {
        var rng = new Random(3);
        var t = new Tensor<double>([1, 2, 4, 4]);
        for (int i = 0; i < t.Length; i++) t[i] = 0.35 + rng.NextDouble() * 0.6;
        return t;
    }

    /// <summary>Grid coordinates well inside [-1, 1], so padding mode cannot explain any difference.</summary>
    private static Tensor<double> InteriorGrid()
    {
        var rng = new Random(4);
        var g = new Tensor<double>([1, 3, 3, 2]);
        for (int i = 0; i < g.Length; i++) g[i] = -0.5 + rng.NextDouble();
        return g;
    }

    /// <summary>
    /// The narrow overload must equal the explicit overload at torchvision's documented defaults.
    /// Interior coordinates isolate the align-corners mapping from padding entirely.
    /// </summary>
    [Fact]
    public void NarrowOverload_MatchesExplicitTorchvisionDefaults()
    {
        var input = Input();
        var grid = InteriorGrid();

        var narrow = _engine.GridSample(input, grid);
        var explicitDefaults = _engine.GridSample(
            input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: false);

        double worst = 0;
        for (int i = 0; i < narrow.Length; i++)
            worst = Math.Max(worst, Math.Abs(narrow[i] - explicitDefaults[i]));
        _out.WriteLine($"narrow[0]={narrow[0]:G15} explicitDefaults[0]={explicitDefaults[0]:G15} worstAbsDiff={worst:E3}");

        Assert.True(worst < 1e-12,
            $"GridSample(input, grid) differs from GridSample(..., Bilinear, Zeros, alignCorners:false) " +
            $"by {worst:E3} — the two overloads must share one convention.");
    }

    /// <summary>
    /// Coordinates that fall OUTSIDE [-1, 1] must read zero, not the clamped border pixel. This is the
    /// half of the fix that padding mode governs, which interior coordinates cannot exercise.
    /// </summary>
    [Fact]
    public void OutOfRangeCoordinates_ReadZeroNotTheBorderPixel()
    {
        var input = Input();
        // Far outside the sampling domain on both axes: every bilinear corner is out of bounds.
        var grid = new Tensor<double>([1, 1, 1, 2]);
        grid[0] = -3.0; grid[1] = -3.0;

        var sampled = _engine.GridSample(input, grid);
        _out.WriteLine($"far out-of-range sample = {sampled[0]:G15} (channel 0)");

        for (int c = 0; c < input.Shape[1]; c++)
            Assert.Equal(0.0, sampled[c]);
    }

    /// <summary>
    /// The gradient of the narrow overload must agree with central finite differences. This is what
    /// failed before: the backward kernels implemented a different convention from the forward.
    /// </summary>
    [Fact]
    public void NarrowOverload_GradientMatchesFiniteDifferences()
    {
        var input = Input();
        var grid = InteriorGrid();

        Func<Tensor<double>> fwd = () => _engine.GridSample(input, grid);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(fwd(), null);
        var grads = tape.ComputeGradients(loss, [input, grid]);

        const double h = 1e-6;
        foreach (var (name, t) in new[] { ("input", input), ("grid", grid) })
        {
            Assert.True(grads.TryGetValue(t, out var g) && g is not null, $"no gradient for {name}");
            int probes = Math.Min(8, t.Length);
            for (int i = 0; i < probes; i++)
            {
                double orig = t[i];
                t[i] = orig + h; double lp = _engine.TensorSum(fwd());
                t[i] = orig - h; double lm = _engine.TensorSum(fwd());
                t[i] = orig;
                double numerical = (lp - lm) / (2 * h);
                double denom = Math.Max(1.0, Math.Max(Math.Abs(g![i]), Math.Abs(numerical)));
                double rel = Math.Abs(g[i] - numerical) / denom;
                _out.WriteLine($"d/d{name}[{i}] analytical={g[i]:G10} numerical={numerical:G10} rel={rel:E3}");
                Assert.True(rel < 1e-5,
                    $"d/d{name}[{i}] analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
    }

    /// <summary>The mode-aware overload's gradient must match finite differences at the defaults too.</summary>
    [Fact]
    public void ModeAwareOverload_DefaultCombination_GradientMatchesFiniteDifferences()
    {
        var input = Input();
        var grid = InteriorGrid();

        Func<Tensor<double>> fwd = () => _engine.GridSample(
            input, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners: false);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(fwd(), null);
        var grads = tape.ComputeGradients(loss, [input, grid]);

        const double h = 1e-6;
        foreach (var (name, t) in new[] { ("input", input), ("grid", grid) })
        {
            Assert.True(grads.TryGetValue(t, out var g) && g is not null, $"no gradient for {name}");
            int probes = Math.Min(6, t.Length);
            for (int i = 0; i < probes; i++)
            {
                double orig = t[i];
                t[i] = orig + h; double lp = _engine.TensorSum(fwd());
                t[i] = orig - h; double lm = _engine.TensorSum(fwd());
                t[i] = orig;
                double numerical = (lp - lm) / (2 * h);
                double denom = Math.Max(1.0, Math.Max(Math.Abs(g![i]), Math.Abs(numerical)));
                Assert.True(Math.Abs(g[i] - numerical) / denom < 1e-5,
                    $"d/d{name}[{i}] analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
    }

    /// <summary>
    /// Sampling parameters the backward does not implement must throw rather than silently return the
    /// bilinear-zeros gradient.
    /// </summary>
    /// <remarks>
    /// alignCorners=true USED to be listed here. It is now implemented — the two conventions differ by
    /// the per-axis rescale g_false = g_true * (S-1)/S, so no second gradient implementation was needed —
    /// and GridSampleBackwardAlignCornersTests pins CPU against GPU for both conventions. Mode and
    /// padding remain genuinely unimplemented and must still throw.
    /// </remarks>
    [Theory]
    [InlineData(GridSampleMode.Nearest, GridSamplePadding.Zeros, false)]
    [InlineData(GridSampleMode.Bilinear, GridSamplePadding.Border, false)]
    public void UnsupportedCombination_ThrowsRatherThanReturningAWrongGradient(
        GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        var input = Input();
        var grid = InteriorGrid();

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(_engine.GridSample(input, grid, mode, padding, alignCorners), null);

        var ex = Record.Exception(() => tape.ComputeGradients(loss, [input, grid]));
        Assert.NotNull(ex);
        _out.WriteLine($"{mode}/{padding}/align={alignCorners} -> {ex!.GetType().Name}");
    }
}
