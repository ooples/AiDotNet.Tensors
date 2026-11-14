using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Create-graph backwards at the edges the random-input sweeps do not reach: RG-LRU's clipped square-root derivative,
/// RWKV-4 keys spanning a range wider than exp can represent, and scatter_reduce("prod") with zero factors. In each
/// case the create-graph gradient must equal the fast kernel's (the same function differentiated both ways) and be
/// differentiable again.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class HigherOrderEdgeCaseTests : IDisposable
{
    private readonly IEngine _previousEngine;
    private readonly CpuEngine _engine = new();

    public HigherOrderEdgeCaseTests()
    {
        _previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _engine;
    }

    public void Dispose() => AiDotNetEngine.Current = _previousEngine;

    private static Tensor<double> Filled(int[] shape, Func<int, double> value)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = value(i);
        return t;
    }

    // The gradients of ⟨f(inputs), u⟩ with and without create-graph.
    private (Dictionary<Tensor<double>, Tensor<double>> Fast, Dictionary<Tensor<double>, Tensor<double>> Graph) BothWays(
        Func<Tensor<double>> f, Tensor<double>[] inputs, int seed)
    {
        Dictionary<Tensor<double>, Tensor<double>> Run(bool createGraph)
        {
            using var tape = new GradientTape<double>();
            var y = f();
            var rng = new Random(seed);
            var u = Filled((int[])y._shape.Clone(), _ => rng.NextDouble() * 2 - 1);
            return tape.ComputeGradients(_engine.ReduceSum(_engine.TensorMultiply(y, u), null), inputs, createGraph);
        }
        return (Run(false), Run(true));
    }

    private static void AssertSame(Dictionary<Tensor<double>, Tensor<double>> fast, Dictionary<Tensor<double>, Tensor<double>> graph,
        Tensor<double>[] inputs, double tolerance)
    {
        for (int i = 0; i < inputs.Length; i++)
        {
            Assert.True(fast.ContainsKey(inputs[i]) == graph.ContainsKey(inputs[i]), $"input {i}: gradient present in only one mode");
            if (!fast.ContainsKey(inputs[i])) continue;
            var a = fast[inputs[i]];
            var b = graph[inputs[i]];
            for (int e = 0; e < a.Length; e++)
            {
                Assert.False(double.IsNaN(b[e]) || double.IsInfinity(b[e]), $"input {i}[{e}]: create-graph gradient {b[e]}");
                Assert.True(Math.Abs(a[e] - b[e]) <= tolerance * (1 + Math.Abs(a[e])),
                    $"input {i}[{e}]: fast kernel {a[e]:G12} vs create-graph {b[e]:G12}");
            }
        }
    }

    [Fact]
    public void RgLru_CreateGraphGradient_UsesTheKernelsClippedSquareRootDerivative()
    {
        // a = r·sigmoid(-decay) ≈ 0.9999999: |da s| = a/sqrt(1 - a²) ≈ 2200, past the ±1000 clip.
        int b = 1, t = 3, d = 2;
        var value = Filled(new[] { b, t, d }, i => 0.4 + 0.1 * i);
        var recGate = Filled(new[] { b, t, d }, i => i % 2 == 0 ? 0.9999999 : 0.6);
        var inpGate = Filled(new[] { b, t, d }, i => 0.5 + 0.05 * i);
        var decay = Filled(new[] { d }, _ => -40.0);
        var inputs = new[] { value, recGate, inpGate, decay };
        var (fast, graph) = BothWays(() => _engine.RgLruScanForward(value, recGate, inpGate, decay), inputs, 3);
        AssertSame(fast, graph, inputs, 1e-9);

        // And differentiable again: the second derivative in the (smooth) value and input-gate directions.
        var check = GradCheck.GradGradCheckMany<double>(_ => _engine.RgLruScanForward(value, recGate, inpGate, decay), inputs,
            new GradCheckOptions { Engine = _engine, InputsToCheck = new[] { 0, 2 }, Epsilon = 1e-7, AbsoluteTolerance = 1e-4, RelativeTolerance = 1e-3 });
        Assert.True(check.Passed, check.ToString());
    }

    [Fact]
    public void Rwkv4_CreateGraphGradient_StaysFiniteForKeysSpanningMoreThanExpRange()
    {
        // Keys from -450 to +450: e^{±450} each fit a double, but their ratio (e^{900}) does not, which a single global
        // shift could not represent.
        int b = 1, t = 4, d = 2;
        var r = Filled(new[] { b, t, d }, i => 0.3 + 0.1 * i);
        var k = Filled(new[] { b, t, d }, i => i % 2 == 0 ? -450.0 + 300 * (i / 2) : 0.2 * i);
        var v = Filled(new[] { b, t, d }, i => 0.5 - 0.1 * i);
        var decay = Filled(new[] { d }, i => -0.5 + i);
        var first = Filled(new[] { d }, i => 0.3 - 0.2 * i);
        var inputs = new[] { r, k, v, decay, first };
        var (fast, graph) = BothWays(() => _engine.Rwkv4WkvForward(r, k, v, decay, first), inputs, 5);
        AssertSame(fast, graph, inputs, 1e-8);
    }

    [Fact]
    public void ScatterReduceProd_WithZeroFactors_IsDifferentiableTwice()
    {
        // Destination slot 0 holds a zero destination factor; slot 1 a single zero source factor; slot 2 none.
        var tensor = new Tensor<double>(new[] { 0.0, 0.7, 1.3, 0.9 }, new[] { 4 });
        var source = new Tensor<double>(new[] { 0.6, 0.0, 1.1, -0.8, 0.5, 1.2 }, new[] { 6 });
        var index = new Tensor<int>(new[] { 0, 1, 1, 2, 0, 2 }, new[] { 6 });
        Tensor<double> F(IReadOnlyList<Tensor<double>> x) => _engine.TensorScatterReduce(x[0], 0, index, x[1], ScatterReduceMode.Prod, true);

        var inputs = new[] { tensor, source };
        var (fast, graph) = BothWays(() => F(inputs), inputs, 7);
        AssertSame(fast, graph, inputs, 1e-12);

        // Products are polynomials, so central differences are exact up to rounding even at the zeros.
        var check = GradCheck.GradGradCheckMany<double>(F, inputs, new GradCheckOptions { Engine = _engine });
        Assert.True(check.Passed, check.ToString());
    }

    [Fact]
    public void ScatterReduceProd_WithTwoZeroSourcesInASlot_MatchesTorch()
    {
        // torch: each of two zero source factors in one slot gets gradient 0 (and the product of the others is 0).
        var tensor = new Tensor<double>(new[] { 0.5, 2.0 }, new[] { 2 });
        var source = new Tensor<double>(new[] { 0.0, 0.0, 3.0 }, new[] { 3 });
        var index = new Tensor<int>(new[] { 0, 0, 1 }, new[] { 3 });
        var inputs = new[] { tensor, source };
        var (fast, graph) = BothWays(() => _engine.TensorScatterReduce(tensor, 0, index, source, ScatterReduceMode.Prod, true), inputs, 9);
        AssertSame(fast, graph, inputs, 1e-12);
        Assert.Equal(0.0, graph[source][0]);
        Assert.Equal(0.0, graph[source][1]);
    }
}
