using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Gradients for the variadic ops that genuinely lacked one: <c>TensorBlockDiag</c> and
/// <c>TensorCartesianProd</c>. Also covers <c>TensorNextAfter</c>'s pass-through gradient.
/// </summary>
/// <remarks>
/// <para>
/// Eleven variadic ops were reported by the gradcheck sweep as "no gradient for ANY of its 0 tensor
/// input(s)". That 0 was a HARNESS artifact: the sweep collected gradient targets with
/// <c>args.OfType&lt;Tensor&lt;double&gt;&gt;()</c>, which cannot see the elements of a
/// <c>Tensor&lt;T&gt;[]</c> parameter. Once the sweep flattens array arguments, 9 of the 11 turned out
/// to have working gradients all along; only these two were real.
/// </para>
/// <para>
/// Both are pure data-movement ops, so their gradients are exact and can be asserted against closed
/// forms rather than only finite differences.
/// </para>
/// </remarks>
public class VariadicOpGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public VariadicOpGradTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Mat(int rows, int cols, double start)
    {
        var t = new Tensor<double>([rows, cols]);
        for (int i = 0; i < t.Length; i++) t[i] = start + i;
        return t;
    }

    private static Tensor<double> Vec(params double[] v)
    {
        var t = new Tensor<double>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    /// <summary>
    /// Each block is copied verbatim, so under a plain sum loss every element of every input gets
    /// exactly 1 — and nothing leaks from the off-block zeros.
    /// </summary>
    [Fact]
    public void BlockDiag_SumLoss_GivesUnitGradientToEveryBlockElement()
    {
        var a = Mat(2, 3, 1);
        var b = Mat(1, 2, 10);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorBlockDiag([a, b]);
        Assert.Equal(new[] { 3, 5 }, outT.Shape.ToArray());

        var loss = _engine.ReduceSum(outT, null);
        var grads = tape.ComputeGradients(loss, [a, b]);

        foreach (var (name, t) in new[] { ("a", a), ("b", b) })
        {
            Assert.True(grads.TryGetValue(t, out var g) && g is not null, $"no gradient for {name}");
            for (int i = 0; i < t.Length; i++) Assert.Equal(1.0, g![i]);
        }
    }

    /// <summary>
    /// Weighting the output non-uniformly proves each block reads its OWN region of the gradient: a
    /// transposed or mis-offset block walk would pull weights from the wrong cells.
    /// </summary>
    [Fact]
    public void BlockDiag_WeightedLoss_ReadsTheCorrectDiagonalBlock()
    {
        var a = Mat(2, 2, 1);
        var b = Mat(2, 2, 5);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorBlockDiag([a, b]);   // 4x4
        var w = new Tensor<double>([4, 4]);
        for (int i = 0; i < 16; i++) w[i] = i;        // distinct weight per output cell
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [a, b]);

        // a occupies rows 0-1, cols 0-1 -> w cells 0,1,4,5
        Assert.Equal(0.0, grads[a]![0]);
        Assert.Equal(1.0, grads[a]![1]);
        Assert.Equal(4.0, grads[a]![2]);
        Assert.Equal(5.0, grads[a]![3]);
        // b occupies rows 2-3, cols 2-3 -> w cells 10,11,14,15
        Assert.Equal(10.0, grads[b]![0]);
        Assert.Equal(11.0, grads[b]![1]);
        Assert.Equal(14.0, grads[b]![2]);
        Assert.Equal(15.0, grads[b]![3]);
    }

    /// <summary>
    /// Every element of input k is reused across total/size_k output rows, so a sum loss gives it
    /// exactly that repeat count. This is the assertion that catches treating the backward as a
    /// reshape instead of a scatter-add.
    /// </summary>
    [Fact]
    public void CartesianProd_SumLoss_GradientEqualsTheRepeatCount()
    {
        var a = Vec(1, 2, 3);     // size 3
        var b = Vec(10, 20);      // size 2

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorCartesianProd([a, b]);
        Assert.Equal(new[] { 6, 2 }, outT.Shape.ToArray());

        var loss = _engine.ReduceSum(outT, null);
        var grads = tape.ComputeGradients(loss, [a, b]);

        // total = 6; a repeats 6/3 = 2 times per element, b repeats 6/2 = 3 times.
        for (int i = 0; i < a.Length; i++) Assert.Equal(2.0, grads[a]![i]);
        for (int i = 0; i < b.Length; i++) Assert.Equal(3.0, grads[b]![i]);
    }

    /// <summary>Three inputs of differing sizes, to confirm the multi-index walk generalises.</summary>
    [Fact]
    public void CartesianProd_ThreeInputs_GradientEqualsTheRepeatCount()
    {
        var a = Vec(1, 2);        // 2
        var b = Vec(3, 4, 5);     // 3
        var c = Vec(6, 7, 8, 9);  // 4

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(_engine.TensorCartesianProd([a, b, c]), null);
        var grads = tape.ComputeGradients(loss, [a, b, c]);

        int total = 2 * 3 * 4;
        for (int i = 0; i < a.Length; i++) Assert.Equal(total / 2.0, grads[a]![i]);
        for (int i = 0; i < b.Length; i++) Assert.Equal(total / 3.0, grads[b]![i]);
        for (int i = 0; i < c.Length; i++) Assert.Equal(total / 4.0, grads[c]![i]);
    }

    /// <summary>
    /// NextAfter is identity plus one ULP, so gradient passes through to a unchanged. b only selects
    /// the direction of the step — a piecewise-constant influence — so it receives exactly 0.
    /// </summary>
    [Fact]
    public void NextAfter_PassesGradientThroughToA_AndZeroToB()
    {
        var a = Vec(1.0, 2.0, -3.0, 0.5);
        var b = Vec(2.0, 1.0, 0.0, 0.5);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var outT = _engine.TensorNextAfter(a, b);
        var w = Vec(1.0, 2.0, 3.0, 4.0);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [a, b]);

        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(w[i], grads[a]![i]);
            Assert.Equal(0.0, grads[b]![i]);
        }
    }

    /// <summary>
    /// Confirms the pass-through choice agrees with what finite differences actually measure: at
    /// h=1e-6 the 1-ULP staircase is indistinguishable from the identity.
    /// </summary>
    [Fact]
    public void NextAfter_PassThrough_AgreesWithFiniteDifferences()
    {
        var a = Vec(1.0, 2.0, -3.0, 0.5);
        var b = Vec(5.0, 5.0, 5.0, 5.0);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var loss = _engine.ReduceSum(_engine.TensorNextAfter(a, b), null);
        var grads = tape.ComputeGradients(loss, [a]);

        const double h = 1e-6;
        for (int i = 0; i < a.Length; i++)
        {
            double orig = a[i];
            a[i] = orig + h; double lp = _engine.TensorSum(_engine.TensorNextAfter(a, b));
            a[i] = orig - h; double lm = _engine.TensorSum(_engine.TensorNextAfter(a, b));
            a[i] = orig;
            double numerical = (lp - lm) / (2 * h);
            _out.WriteLine($"nextafter d/da[{i}] analytical={grads[a]![i]:G10} numerical={numerical:G10}");
            Assert.True(Math.Abs(grads[a]![i] - numerical) < 1e-6,
                $"nextafter d/da[{i}]: analytical {grads[a]![i]:G10} vs numerical {numerical:G10}");
        }
    }
}
