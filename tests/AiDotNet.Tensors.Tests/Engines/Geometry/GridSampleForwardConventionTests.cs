using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Geometry;

/// <summary>
/// The two <c>GridSample</c> overloads disagree on their sampling convention, and the narrow one does
/// not implement the torchvision defaults it is documented as providing.
/// </summary>
/// <remarks>
/// <para>
/// <c>CpuEngine.Geometry.cs</c>'s header states the pre-existing <c>GridSample(input, grid)</c> API
/// "stays as [a] torchvision-default shim that routes through here". PyTorch/torchvision defaults are
/// <c>align_corners=False</c> and <c>padding_mode='zeros'</c>. But that overload computes its pixel
/// mapping as <c>(size - 1) / 2</c> — the <c>align_corners=TRUE</c> convention — and clamps its four
/// sample indices into <c>[0, size-1]</c>, which is BORDER padding. So it is neither a shim (it has its
/// own implementation) nor torchvision-default.
/// </para>
/// <para>
/// This was found while wiring a gradient for the mode-aware overload. Finite differences rejected the
/// existing <c>GridSampleBackwardInput</c>/<c>GridSampleBackwardGrid</c> kernels for
/// Bilinear + Zeros + alignCorners=false — the one combination their guard claims to support
/// (analytical 0.0390 vs numerical 0.1862). Those kernels are the adjoint of the NARROW overload's
/// convention, so their guard message is wrong about which forward they match. That is why the
/// mode-aware overload is left unrecorded rather than pointed at them.
/// </para>
/// <para>
/// These tests fail until the conventions are reconciled. Fixing it changes forward numerics for
/// existing callers of the 2-argument overload, so which one moves is a deliberate call, not something
/// to decide inside a gradient fix.
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
    /// Coordinates are interior, so this isolates the align-corners mapping from padding entirely.
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
            $"by {worst:E3}. The narrow overload uses (size-1)/2 (align_corners=TRUE) and clamps sample " +
            $"indices (BORDER padding), so it does not implement the torchvision defaults its own file " +
            $"header claims.");
    }

    /// <summary>
    /// Conversely, the narrow overload currently behaves like align_corners=true + Border. If the
    /// reconciliation moves the narrow overload to the documented defaults, this test should be
    /// deleted; if it moves the DOCUMENTATION instead, this becomes the specification.
    /// </summary>
    [Fact]
    public void NarrowOverload_CurrentlyBehavesAsAlignCornersTrueWithBorderPadding()
    {
        var input = Input();
        var grid = InteriorGrid();

        var narrow = _engine.GridSample(input, grid);
        var alignTrueBorder = _engine.GridSample(
            input, grid, GridSampleMode.Bilinear, GridSamplePadding.Border, alignCorners: true);

        double worst = 0;
        for (int i = 0; i < narrow.Length; i++)
            worst = Math.Max(worst, Math.Abs(narrow[i] - alignTrueBorder[i]));
        _out.WriteLine($"worstAbsDiff vs align_corners=true + Border = {worst:E3}");

        Assert.True(worst < 1e-12,
            $"Expected the narrow overload to match Bilinear + Border + alignCorners=true, but it " +
            $"differs by {worst:E3} — the convention is something else again.");
    }
}
