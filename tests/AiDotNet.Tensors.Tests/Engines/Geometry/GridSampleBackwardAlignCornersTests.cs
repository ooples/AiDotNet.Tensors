using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Geometry;

/// <summary>
/// Pins the GPU GridSample backward passes against CpuEngine for BOTH align_corners conventions.
/// </summary>
/// <remarks>
/// <para>
/// The GPU kernels hardcode the align_corners=TRUE mapping, and both overrides used to bail whenever
/// alignCorners was false — which is the torchvision default and the convention this engine standardised
/// on, so the kernels never ran (GpuResidencyProbeTests measured 0/1 launches).
/// </para>
/// <para>
/// They now pre-scale the grid by g' = g * S/(S-1) per axis, which makes an align-true kernel evaluate
/// the align-false mapping exactly, and GridSampleBackwardGrid rescales its OUTPUT by the same factor
/// because the kernel differentiated with respect to g'. That chain-rule step is easy to omit and would
/// leave the sampling positions right while the gradient magnitudes were wrong, so it is pinned here
/// rather than inferred from the forward matching.
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public class GridSampleBackwardAlignCornersTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;

    public GridSampleBackwardAlignCornersTests(ITestOutputHelper o)
    {
        _out = o;
        try { _gpu = new DirectGpuTensorEngine(); _available = _gpu.IsGpuAvailable; }
        catch { _available = false; }
    }

    public void Dispose() { _gpu?.Dispose(); GC.SuppressFinalize(this); }

    private const int N = 1, C = 2, H = 4, W = 4, OutH = 3, OutW = 3;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static double MaxAbsDiff(Tensor<float> a, Tensor<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs((double)a[i] - b[i]));
        return worst;
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void BackwardInput_GpuMatchesCpu(bool alignCorners)
    {
        Skip.If(!_available, "GPU backend not available");
        var gradOut = Rand([N, C, OutH, OutW], 3);
        var grid = Rand([N, OutH, OutW, 2], 5);
        int[] inputShape = [N, C, H, W];

        var c = _cpu.GridSampleBackwardInput(gradOut, grid, inputShape,
            GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners);
        var g = ((IEngine)_gpu!).GridSampleBackwardInput(gradOut, grid, inputShape,
            GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners);

        double worst = MaxAbsDiff(c, g);
        _out.WriteLine($"alignCorners={alignCorners} backwardInput maxAbsDiff={worst:E3}");
        Assert.True(worst < 1e-5,
            $"GPU GridSampleBackwardInput differs from CpuEngine by {worst:E3} at alignCorners={alignCorners}.");
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void BackwardGrid_GpuMatchesCpu(bool alignCorners)
    {
        Skip.If(!_available, "GPU backend not available");
        var gradOut = Rand([N, C, OutH, OutW], 7);
        var input = Rand([N, C, H, W], 11);
        var grid = Rand([N, OutH, OutW, 2], 13);

        var c = _cpu.GridSampleBackwardGrid(gradOut, input, grid,
            GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners);
        var g = ((IEngine)_gpu!).GridSampleBackwardGrid(gradOut, input, grid,
            GridSampleMode.Bilinear, GridSamplePadding.Zeros, alignCorners);

        double worst = MaxAbsDiff(c, g);
        _out.WriteLine($"alignCorners={alignCorners} backwardGrid maxAbsDiff={worst:E3}");
        Assert.True(worst < 1e-5,
            $"GPU GridSampleBackwardGrid differs from CpuEngine by {worst:E3} at alignCorners={alignCorners}. " +
            $"If only the align_corners=false case fails, the grid pre-scale landed but the chain-rule " +
            $"rescale of the OUTPUT by S/(S-1) did not.");
    }
}
