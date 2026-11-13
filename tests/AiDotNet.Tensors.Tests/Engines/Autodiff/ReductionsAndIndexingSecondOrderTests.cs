using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// First- and second-order gradients the sweeps' fixed arguments do not reach: scatter reductions along an axis other
/// than 0, spherical softmax on rows of non-unit norm, bicubic interpolation and grid sampling, pairwise distances
/// between two point sets, and global max pooling. Each was wrong or missing before (see the comments).
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class ReductionsAndIndexingSecondOrderTests : IDisposable
{
    private readonly IEngine _previous;
    private readonly CpuEngine _engine = new();

    public ReductionsAndIndexingSecondOrderTests()
    {
        _previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _engine;
    }

    public void Dispose() => AiDotNetEngine.Current = _previous;

    private static Tensor<double> Values(int[] shape, int seed, double low = 0.35, double span = 0.6)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = low + rng.NextDouble() * span;
        return t;
    }

    private GradCheckOptions Options(params int[] inputs) => new()
    {
        Engine = _engine,
        InputsToCheck = inputs.Length == 0 ? null : inputs,
    };

    private static void AssertPassed(GradCheckResult result) => Assert.True(result.Passed, result.ToString());

    // One index per source slice along dim 1, every group 0..2 hit (no empty group).
    private static Tensor<int> Dim1Index() => new(new[] { 0, 2, 0, 1 }, new[] { 4 });

    [Fact]
    public void ScatterReductions_AlongDim1_AreDifferentiableTwice()
    {
        // The backwards assumed dim 0: ScatterMean/Max/Softmax saved no axis and routed every gradient along dim 0.
        var source = Values(new[] { 2, 4 }, 1);
        var index = Dim1Index();
        AssertPassed(GradCheck.GradGradCheck<double>(s => _engine.ScatterMean(s, index, out _, 1, 3), source, Options()));
        AssertPassed(GradCheck.GradGradCheck<double>(s => _engine.ScatterMax(s, index, out _, 1, 3), source, Options()));
        AssertPassed(GradCheck.GradGradCheck<double>(s => _engine.ScatterSoftmax(s, index, 1, 3), source, Options()));
        AssertPassed(GradCheck.GradGradCheck<double>(s => _engine.ScatterAdd(s, index, 1, 3), source, Options()));
    }

    [Fact]
    public void ScatterReductions_WithAnIndexLongerThanTheScatterAxis_DifferentiateWhatTheForwardReads()
    {
        // The forward reads index[d % index.Length] per source slice (an index shaped like the source, as torch_scatter
        // callers pass, is read by its leading entries); ScatterAdd's backward gathered with the whole index, giving a
        // gradient of the wrong shape.
        var source = Values(new[] { 3, 2 }, 2);
        var index = new Tensor<int>(new[] { 1, 0, 1, 2, 0, 1 }, new[] { 3, 2 });
        AssertPassed(GradCheck.GradGradCheck<double>(s => _engine.ScatterAdd(s, index, 0, 2), source, Options()));
        AssertPassed(GradCheck.GradGradCheck<double>(s => _engine.ScatterMean(s, index, out _, 0, 2), source, Options()));
    }

    [Fact]
    public void SphericalSoftmax_GradientIsCorrectOffTheUnitSphere()
    {
        // The projection term of d(x/‖x‖)/dx was missing a 1/‖x‖: right only for unit-norm rows.
        var x = Values(new[] { 2, 3 }, 3, low: 0.5, span: 2.0);
        AssertPassed(GradCheck.GradGradCheck<double>(v => _engine.SphericalSoftmax(v, -1), x, Options()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BicubicInterpolate_GradientIsTheForwardsAdjoint(bool alignCorners)
    {
        // The backward used area-interpolation weights for bicubic.
        var x = Values(new[] { 1, 2, 3, 3 }, 4);
        AssertPassed(GradCheck.GradGradCheck<double>(
            v => _engine.Interpolate(v, new[] { 5, 4 }, InterpolateMode.Bicubic, alignCorners), x, Options()));
    }

    [Theory]
    [InlineData(GridSamplePadding.Zeros, false)]
    [InlineData(GridSamplePadding.Border, true)]
    [InlineData(GridSamplePadding.Reflection, false)]
    public void BicubicGridSample_IsDifferentiableTwiceInInputAndGrid(GridSamplePadding padding, bool alignCorners)
    {
        // The grid's double backward threw NotSupportedException; it is now a recorded composite of the 4x4 taps.
        var input = Values(new[] { 1, 2, 4, 4 }, 5);
        var grid = Values(new[] { 1, 3, 2, 2 }, 6, low: -0.8, span: 1.6);
        AssertPassed(GradCheck.GradGradCheckMany<double>(
            v => _engine.GridSample(v[0], v[1], GridSampleMode.Bicubic, padding, alignCorners),
            new[] { input, grid }, Options()));
    }

    [Fact]
    public void PairwiseDistances_GiveBothPointSetsGradients_DifferentiableTwice()
    {
        // Neither op recorded anything: no gradient reached x or y.
        var x = Values(new[] { 3, 2 }, 7);
        var y = Values(new[] { 2, 2 }, 8, low: 1.2, span: 0.5);
        AssertPassed(GradCheck.GradGradCheckMany<double>(v => _engine.PairwiseDistanceSquared(v[0], v[1]), new[] { x, y }, Options()));
        AssertPassed(GradCheck.GradGradCheckMany<double>(v => _engine.PairwiseDistance(v[0], v[1]), new[] { x, y }, Options()));
    }

    [Fact]
    public void GlobalMaxPool2D_GivesItsInputAGradient()
    {
        // The kernel recorded nothing.
        var x = Values(new[] { 1, 2, 3, 3 }, 9);
        var check = GradCheck.GradGradCheck<double>(v => _engine.GlobalMaxPool2D(v), x, Options());
        AssertPassed(check);
        Assert.Empty(check.InputsWithoutGradient);
    }
}
