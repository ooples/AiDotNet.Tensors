using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>The public gradient checker (<see cref="GradCheck"/>): it passes correct gradients and catches wrong ones.</summary>
public class GradCheckTests
{
    private readonly CpuEngine _engine = new();
    private readonly ITestOutputHelper _out;

    public GradCheckTests(ITestOutputHelper output) => _out = output;

    private GradCheckOptions Options => new() { Engine = _engine };

    private static Tensor<double> Positive(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.4 + rng.NextDouble();
        return t;
    }

    [Fact]
    public void NumericalGradient_OfAScalarFunction_MatchesTheClosedForm()
    {
        // d/dx Σ log x = 1/x (the legacy NumericalGradient.ComputeForScalarFunction).
        var x = Positive(1, 6);
        var g = GradCheck.NumericalGradient(t => _engine.TensorSum(_engine.TensorLog(t)), x);
        for (int i = 0; i < x.Length; i++) Assert.Equal(1 / x[i], g[i], 8);
    }

    [Fact]
    public void NumericalVectorJacobianProduct_MatchesAnalyticalGradients()
    {
        var a = Positive(2, 2, 3);
        var b = Positive(3, 3, 2);
        var w = Positive(4, 2, 2);
        var numerical = GradCheck.NumericalGradients<double>(v => _engine.TensorMatMul(v[0], v[1]), new[] { a, b }, w);
        var analytical = GradCheck.AnalyticalGradients<double>(v => _engine.TensorMatMul(v[0], v[1]), new[] { a, b }, w, _engine);
        for (int i = 0; i < 2; i++)
        {
            var compare = GradCheck.Compare(analytical[i], numerical[i], 1e-7, 1e-6);
            Assert.True(compare.Passed, compare.ToString());
        }
        var single = GradCheck.NumericalVectorJacobianProduct(t => _engine.TensorExp(t), a, Positive(5, 2, 3));
        Assert.Equal(a.Length, single.Length);
    }

    [Fact]
    public void Check_PassesACorrectGradient_FloatAndDouble()
    {
        var x = Positive(6, 2, 3);
        Assert.True(GradCheck.Check(t => _engine.TensorMultiply(_engine.TensorExp(t), t), x, Options).Passed);
        var xf = new Tensor<float>(x.ToArray().Select(v => (float)v).ToArray(), new[] { 2, 3 });
        var resultF = GradCheck.Check(t => _engine.TensorMultiply(_engine.TensorExp(t), t), xf, Options);
        Assert.True(resultF.Passed, resultF.ToString());
        Assert.Equal(6, resultF.ElementsChecked);
    }

    [Fact]
    public void Check_CatchesAWrongGradient_AndReportsTheWorstElements()
    {
        // Fake quantization's straight-through gradient is 1 where the finite difference of the piecewise-constant
        // forward is 0: a deliberate disagreement the checker must report.
        var x = new Tensor<double>(new[] { 0.11, 0.37, -0.52, 0.81 }, new[] { 4 });
        var result = GradCheck.Check(t => _engine.TensorFakeQuantize(t, 4, 0.25), x, Options);
        _out.WriteLine(result.ToString());
        Assert.False(result.Passed);
        Assert.Equal(4, result.FailedElements);
        Assert.NotEmpty(result.WorstMismatches);
        var worst = result.WorstMismatches[0];
        Assert.Equal(1, worst.Order);
        Assert.Equal(0, worst.InputIndex);
        Assert.Equal(0.0, worst.Numerical, 6);
        Assert.Throws<InvalidOperationException>(() => result.ThrowIfFailed());
    }

    [Fact]
    public void Check_TreatsAnUnreachedInputAsZeroGradient()
    {
        var used = Positive(7, 3);
        var unused = Positive(8, 3);
        var result = GradCheck.CheckMany<double>(v => _engine.TensorExp(v[0]), new[] { used, unused }, Options);
        Assert.True(result.Passed, result.ToString());
        Assert.Equal(new[] { 1 }, result.InputsWithoutGradient.ToArray());
    }

    [Fact]
    public void GradGradCheck_PassesASmoothComposition()
    {
        // x³·y: second derivatives 6xy, 3x² mixed — needs the createGraph backward of multiply.
        var x = Positive(9, 4);
        var y = Positive(10, 4);
        var result = GradCheck.GradGradCheckMany<double>(
            v => _engine.TensorMultiply(_engine.TensorMultiply(_engine.TensorMultiply(v[0], v[0]), v[0]), v[1]), new[] { x, y }, Options);
        _out.WriteLine(result.ToString());
        Assert.True(result.Passed, result.ToString());
        Assert.True(result.SecondOrderChecked);
        Assert.Equal(16, result.ElementsChecked);
    }

    [Fact]
    public void Options_LimitInputsAndElements()
    {
        var x = Positive(11, 10);
        var y = Positive(12, 10);
        var result = GradCheck.CheckMany<double>(v => _engine.TensorMultiply(v[0], v[1]), new[] { x, y },
            new GradCheckOptions { Engine = _engine, InputsToCheck = new[] { 1 }, MaxElementsPerInput = 3, RandomOutputGradient = false });
        Assert.True(result.Passed, result.ToString());
        Assert.Equal(3, result.ElementsChecked);
    }
}
