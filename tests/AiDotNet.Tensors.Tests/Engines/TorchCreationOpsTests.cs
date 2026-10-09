using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>The PyTorch creation, window, index, shape and statistics ops against PyTorch float64 reference values.</summary>
[Collection("EngineCurrentGlobalState")]
public class TorchCreationOpsTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static readonly double[] XValues = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9 };

    private static Tensor<double> X() => new Tensor<double>((double[])XValues.Clone(), new[] { 3, 5 });

    private static void Close(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-12)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i], a = actual.GetFlat(i);
            if (double.IsNaN(e)) { Assert.True(double.IsNaN(a), $"{what}[{i}]: expected NaN, got {a}"); continue; }
            Assert.True(Math.Abs(e - a) <= tolerance * Math.Max(1, Math.Abs(e)), $"{what}[{i}]: torch {e:R}, ours {a:R}");
        }
    }

    [Fact]
    public void Ranges_MatchTorch()
    {
        Close(new[] { -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0 }, _engine.TensorArange<double>(-1.5, 2.2, 0.5), "arange");
        Close(new[] { 1.0, 1.5, 2.0, 2.5, 3.0 }, _engine.TensorRange<double>(1, 3, 0.5), "range");
        Close(new[] { 0.1, 0.5623413251903491, 3.1622776601683795, 17.78279410038923, 100.0 }, _engine.TensorLogspace<double>(-1, 2, 5), "logspace");
    }

    [Theory]
    [InlineData("hann", true, new[] { 0.0, 0.1882550990706332, 0.6112604669781572, 0.9504844339512095, 0.9504844339512095, 0.6112604669781573, 0.1882550990706333 })]
    [InlineData("hann", false, new[] { 0.0, 0.24999999999999994, 0.7499999999999999, 1.0, 0.7500000000000002, 0.25000000000000033, 0.0 })]
    [InlineData("hamming", true, new[] { 0.08000000000000002, 0.25319469114498255, 0.6423596296199047, 0.9544456792351128, 0.9544456792351128, 0.6423596296199048, 0.25319469114498266 })]
    [InlineData("hamming", false, new[] { 0.08000000000000002, 0.30999999999999994, 0.7699999999999999, 1.0, 0.7700000000000002, 0.31000000000000033, 0.08000000000000002 })]
    [InlineData("blackman", true, new[] { 0.0, 0.09045342435412806, 0.4591829575459636, 0.9203636180999082, 0.9203636180999082, 0.4591829575459638, 0.09045342435412812 })]
    [InlineData("blackman", false, new[] { 0.0, 0.12999999999999995, 0.6299999999999999, 1.0, 0.6300000000000003, 0.13000000000000023, 0.0 })]
    [InlineData("bartlett", true, new[] { 0.0, 0.2857142857142857, 0.5714285714285714, 0.8571428571428571, 0.8571428571428572, 0.5714285714285716, 0.2857142857142858 })]
    [InlineData("bartlett", false, new[] { 0.0, 0.3333333333333333, 0.6666666666666666, 1.0, 0.6666666666666667, 0.3333333333333335, 0.0 })]
    public void Windows_MatchTorch(string kind, bool periodic, double[] expected)
    {
        var window = kind switch
        {
            "hann" => _engine.TensorHannWindow<double>(7, periodic),
            "hamming" => _engine.TensorHammingWindow<double>(7, periodic),
            "blackman" => _engine.TensorBlackmanWindow<double>(7, periodic),
            _ => _engine.TensorBartlettWindow<double>(7, periodic),
        };
        Close(expected, window, kind);
    }

    [Fact]
    public void Kaiser_And_TriangleIndices_And_Combinations_MatchTorch()
    {
        Close(new[] { 5.2773441320097665e-05, 0.03276828844175176, 0.33089863665707414, 0.8888676640466582, 0.8888676640466582, 0.33089863665707414, 0.03276828844175176 }, _engine.TensorKaiserWindow<double>(7), "kaiser periodic", 1e-10);
        Close(new[] { 0.036710892271286676, 0.3282019573723212, 0.7753221044454067, 1.0, 0.7753221044454067, 0.3282019573723212, 0.036710892271286676 }, _engine.TensorKaiserWindow<double>(7, false, 5.0), "kaiser symmetric", 1e-10);
        Close(new[] { 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0 }, _engine.TensorTrilIndices<double>(3, 4, 1), "tril_indices");
        Close(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 2.0 }, _engine.TensorTriuIndices<double>(4, 3, -1), "triu_indices");
        Close(new[] { 10.0, 20.0, 30.0, 10.0, 20.0, 40.0, 10.0, 30.0, 40.0, 20.0, 30.0, 40.0 }, _engine.TensorCombinations(new Tensor<double>(new[] { 10.0, 20, 30, 40 }, new[] { 4 }), 3), "combinations");
        Close(new[] { 10.0, 10.0, 10.0, 20.0, 10.0, 30.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0 }, _engine.TensorCombinations(new Tensor<double>(new[] { 10.0, 20, 30 }, new[] { 3 }), 2, true), "combinations with replacement");
    }

    [Theory]
    [InlineData(QuantileInterpolation.Linear, new[] { 2.2, 4.2, 7.6 })]
    [InlineData(QuantileInterpolation.Lower, new[] { 1.0, 3.0, 7.0 })]
    [InlineData(QuantileInterpolation.Higher, new[] { 3.0, 5.0, 8.0 })]
    [InlineData(QuantileInterpolation.Nearest, new[] { 3.0, 5.0, 8.0 })]
    [InlineData(QuantileInterpolation.Midpoint, new[] { 2.0, 4.0, 7.5 })]
    public void Quantile_MatchesTorch(QuantileInterpolation interpolation, double[] expected)
        => Close(expected, _engine.TensorQuantile(X(), 0.4, 1, false, interpolation), interpolation.ToString());

    [Fact]
    public void Quantile_OverAll_NanQuantile_AndGradient_MatchTorch()
    {
        Close(new[] { 7.5 }, _engine.TensorQuantile(X(), 0.75), "quantile all");
        var xn = new Tensor<double>(new[] { 1.0, double.NaN, 3, 2, double.NaN, double.NaN, double.NaN, double.NaN, 4, 5, double.NaN, 6 }, new[] { 3, 4 });
        Close(new[] { 2.0, double.NaN, 5.0 }, _engine.TensorNanQuantile(xn, 0.5, 1), "nanquantile");
        var x = X();
        using var tape = new GradientTape<double>();
        var loss = _engine.ReduceSum(_engine.TensorQuantile(x, 0.4, 1), null, false);
        Close(new[] { 0.6000000000000001, 0.0, 0.0, 0.3999999999999999, 0.0, 0.0, 0.0, 0.0, 0.6000000000000001, 0.3999999999999999, 0.0, 0.6000000000000001, 0.0, 0.3999999999999999, 0.0 }, tape.ComputeGradients(loss, new[] { x })[x], "quantile gradient");
    }

    [Fact]
    public void Statistics_MatchTorch()
    {
        var (std, mean) = _engine.TensorStdMean(X(), new[] { 1 });
        Close(new[] { 1.7888543819998317, 2.7386127875258306, 1.6733200530681513 }, std, "std");
        Close(new[] { 2.8, 5.0, 7.6 }, mean, "mean");
        Close(new[] { 6.222222222222222, 9.555555555555557, 4.222222222222222, 6.222222222222222, 6.222222222222221 }, _engine.TensorVarMean(X(), new[] { 0 }, 0).Var, "var correction 0");
        Close(new[] { 3.2, 0.75, 1.15, 0.75, 7.5, -3.25, 1.15, -3.25, 2.8 }, _engine.TensorCov(X()), "cov");
        Close(new[] { 1.0, 0.15309310892394865, 0.38418803524911005, 0.15309310892394862, 1.0, -0.7092081432669752, 0.38418803524911, -0.7092081432669752, 0.9999999999999999 }, _engine.TensorCorrcoef(X()), "corrcoef");
        Close(new[] { 5.0, -6.0, 7.0, 11.0, -5.0, -1.0, -2.0, -3.0, 4.0 }, _engine.TensorDiff(X(), 2, 1), "diff");
        Close(new[] { 5.0, 9.5, 15.5 }, _engine.TensorTrapezoid(X(), 0.5, 1), "trapezoid");
        Close(new[] { 1.0, 2.25, 3.5, 5.0, 2.75, 4.75, 7.5, 9.5, 3.25, 7.5, 11.5, 15.5 }, _engine.TensorCumulativeTrapezoid(X(), 0.5, 1), "cumulative trapezoid");
    }

    [Fact]
    public void ShapeAndTruthOps_BehaveAsInTorch()
    {
        var x = X();
        Assert.Equal(new[] { 2, 2, 1 }, _engine.TensorChunk(x, 3, 1).Select(t => t.Shape[1]).ToArray());
        Assert.Equal(new[] { 1, 4 }, _engine.TensorSplitWithSizes(x, new[] { 1, 4 }, 1).Select(t => t.Shape[1]).ToArray());
        Assert.Equal(new[] { 3, 5, 1 }, _engine.TensorUnflatten(x, 1, new[] { 5, -1 }).Shape.ToArray());
        Close(new[] { 9.0, 2, 6, 5, 3 }, _engine.TensorSelect(x, 0, -2), "select");
        Close(new[] { 17.0, 11, 19, 13, 17 }, _engine.TensorSumToSize(x, new[] { 1, 5 }), "sum_to_size");
        Close(new[] { 1.0, 2, 5 }, _engine.TensorAmin(x, new[] { 1 }), "amin");
        var mixed = new Tensor<double>(new[] { 1.0, 0, 2, 3, 4, 5 }, new[] { 2, 3 });
        Close(new[] { 0.0, 1 }, _engine.TensorAll(mixed, new[] { 1 }), "all");
        Close(new[] { 1.0, 1 }, _engine.TensorAny(mixed, new[] { 1 }), "any");
        var coords = _engine.TensorUnravelIndex(new Tensor<double>(new[] { 0.0, 7, 11 }, new[] { 3 }), new[] { 3, 4 });
        Close(new[] { 0.0, 1, 2 }, coords[0], "unravel row");
        Close(new[] { 0.0, 3, 3 }, coords[1], "unravel col");
        var perm = _engine.TensorRandperm<double>(10, seed: 7).ToArray().OrderBy(v => v).ToArray();
        Assert.Equal(Enumerable.Range(0, 10).Select(i => (double)i).ToArray(), perm);
        var ints = _engine.TensorRandint<double>(-3, 4, new[] { 1000 }, seed: 1).ToArray();
        Assert.True(ints.All(v => v >= -3 && v < 4 && v == Math.Floor(v)) && ints.Distinct().Count() == 7);
    }
}
