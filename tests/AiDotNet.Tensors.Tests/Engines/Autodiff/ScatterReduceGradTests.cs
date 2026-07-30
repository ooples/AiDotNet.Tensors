using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <c>TensorScatterReduce</c> gradients across all five reduction modes and both
/// <c>includeSelf</c> settings.
/// </summary>
/// <remarks>
/// The op recorded nothing, so neither the destination nor the scattered source received a gradient.
/// The backward replays the forward's per-slot decisions — each slot's Mean divisor, which contributor
/// won an AMin/AMax slot, and whether the destination value survived an <c>includeSelf: false</c>
/// reset — rather than deriving them from the output, which is what keeps the two consistent.
/// </remarks>
public class ScatterReduceGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public ScatterReduceGradTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Vals(int[] shape, double start, double step = 1.0)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + i * step;
        return t;
    }

    /// <summary>indices [3,1] -> all three source rows land on destination row 0.</summary>
    private static Tensor<int> AllToRowZero()
    {
        var t = new Tensor<int>([3, 1]);
        for (int i = 0; i < 3; i++) t[i] = 0;
        return t;
    }

    private void CheckFiniteDifferences(
        ScatterReduceMode mode, bool includeSelf, string label)
    {
        var tensor = Vals([3, 1], 0.6, 0.7);
        var source = Vals([3, 1], 0.4, 0.5);
        var indices = AllToRowZero();

        Func<Tensor<double>> fwd = () =>
            _engine.TensorScatterReduce(tensor, 0, indices, source, mode, includeSelf);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        // Distinct output weights so misrouted gradient is detectable.
        var w = Vals([3, 1], 1.0);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(fwd(), w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        const double h = 1e-6;
        foreach (var (name, t) in new[] { ("tensor", tensor), ("source", source) })
        {
            Assert.True(grads.TryGetValue(t, out var g) && g is not null, $"{label}: no gradient for {name}");
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + h;
                double lp = _engine.TensorSum(_engine.TensorMultiply(fwd(), w));
                t[i] = orig - h;
                double lm = _engine.TensorSum(_engine.TensorMultiply(fwd(), w));
                t[i] = orig;
                double numerical = (lp - lm) / (2 * h);
                double denom = Math.Max(1.0, Math.Max(Math.Abs(g![i]), Math.Abs(numerical)));
                double rel = Math.Abs(g[i] - numerical) / denom;
                _out.WriteLine($"{label} d/d{name}[{i}] analytical={g[i]:G10} numerical={numerical:G10} rel={rel:E3}");
                Assert.True(rel < 1e-6,
                    $"{label}: d/d{name}[{i}] analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
    }

    [Theory]
    [InlineData(ScatterReduceMode.Sum, true)]
    [InlineData(ScatterReduceMode.Sum, false)]
    [InlineData(ScatterReduceMode.Mean, true)]
    [InlineData(ScatterReduceMode.Mean, false)]
    [InlineData(ScatterReduceMode.Prod, true)]
    [InlineData(ScatterReduceMode.Prod, false)]
    [InlineData(ScatterReduceMode.AMax, true)]
    [InlineData(ScatterReduceMode.AMin, true)]
    public void AllModes_GradientMatchesFiniteDifferences(ScatterReduceMode mode, bool includeSelf)
        => CheckFiniteDifferences(mode, includeSelf, $"{mode}/self={includeSelf}");

    /// <summary>
    /// CLOSED FORM for Sum: every contributor to a slot gets that slot's weight verbatim. With all 3
    /// source rows targeting row 0 and includeSelf, row 0's weight (1.0) reaches the destination row 0
    /// and all three source rows.
    /// </summary>
    [Fact]
    public void Sum_EveryContributorGetsTheSlotWeight()
    {
        var tensor = Vals([3, 1], 0.6, 0.7);
        var source = Vals([3, 1], 0.4, 0.5);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var w = Vals([3, 1], 1.0);      // weights 1, 2, 3
        var outT = _engine.TensorScatterReduce(tensor, 0, AllToRowZero(), source, ScatterReduceMode.Sum, true);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        // Destination row 0 participated in the reduction; rows 1 and 2 were untouched copies.
        Assert.Equal(1.0, grads[tensor]![0]);
        Assert.Equal(2.0, grads[tensor]![1]);
        Assert.Equal(3.0, grads[tensor]![2]);
        // All three source rows fed slot 0, so each gets weight 1.
        for (int i = 0; i < 3; i++) Assert.Equal(1.0, grads[source]![i]);
    }

    /// <summary>
    /// CLOSED FORM for Mean with includeSelf: slot 0 averaged 4 values (self + 3 sources), so each
    /// contributor receives weight/4. This is the assertion that catches a backward reusing the wrong
    /// divisor (e.g. 3, ignoring self).
    /// </summary>
    [Fact]
    public void Mean_ContributorsShareTheSlotWeightByCount()
    {
        var tensor = Vals([3, 1], 0.6, 0.7);
        var source = Vals([3, 1], 0.4, 0.5);

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var w = Vals([3, 1], 1.0);
        var outT = _engine.TensorScatterReduce(tensor, 0, AllToRowZero(), source, ScatterReduceMode.Mean, true);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        Assert.Equal(0.25, grads[tensor]![0]);      // 1.0 / 4 contributors
        for (int i = 0; i < 3; i++) Assert.Equal(0.25, grads[source]![i]);
    }

    /// <summary>
    /// CLOSED FORM for AMax: winner-takes-all. source rows are 0.4, 0.9, 1.4 and self is 0.6, so
    /// source row 2 (1.4) wins slot 0 and receives the whole weight while every other contributor to
    /// that slot receives exactly 0.
    /// </summary>
    [Fact]
    public void AMax_OnlyTheWinningContributorGetsGradient()
    {
        var tensor = Vals([3, 1], 0.6, 0.7);     // 0.6, 1.3, 2.0
        var source = Vals([3, 1], 0.4, 0.5);     // 0.4, 0.9, 1.4

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var w = Vals([3, 1], 1.0);
        var outT = _engine.TensorScatterReduce(tensor, 0, AllToRowZero(), source, ScatterReduceMode.AMax, true);
        Assert.Equal(1.4, outT[0]);              // max(0.6, 0.4, 0.9, 1.4)
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        Assert.Equal(0.0, grads[tensor]![0]);    // self lost slot 0
        Assert.Equal(0.0, grads[source]![0]);
        Assert.Equal(0.0, grads[source]![1]);
        Assert.Equal(1.0, grads[source]![2]);    // winner
    }

    /// <summary>
    /// A zero factor must not collapse a Prod slot's gradient. Deriving d/dfactor as out/factor divides
    /// by zero here; the non-zero-product plus zero-count formulation gives the correct finite value.
    /// </summary>
    [Fact]
    public void Prod_WithAZeroFactor_StaysFinite()
    {
        var tensor = Vals([3, 1], 0.6, 0.7);
        var source = new Tensor<double>([3, 1]);
        source[0] = 0.0;    // the single zero factor in slot 0
        source[1] = 0.9;
        source[2] = 1.4;

        using var tape = new GradientTape<double>();
        tape.BindEngineIfUnset(_engine);
        var w = Vals([3, 1], 1.0);
        var outT = _engine.TensorScatterReduce(tensor, 0, AllToRowZero(), source, ScatterReduceMode.Prod, true);
        Assert.Equal(0.0, outT[0]);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(outT, w), null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);

        // d/dsource[0] = product of the others = 0.6 * 0.9 * 1.4
        Assert.Equal(0.6 * 0.9 * 1.4, grads[source]![0], 12);
        // Every other factor's product-of-others includes the zero, so their gradients are 0.
        Assert.Equal(0.0, grads[source]![1]);
        Assert.Equal(0.0, grads[source]![2]);
        Assert.Equal(0.0, grads[tensor]![0]);
        foreach (var g in new[] { grads[source]![0], grads[source]![1], grads[tensor]![0] })
        {
            Assert.False(double.IsNaN(g));
            Assert.False(double.IsInfinity(g));
        }
    }
}
