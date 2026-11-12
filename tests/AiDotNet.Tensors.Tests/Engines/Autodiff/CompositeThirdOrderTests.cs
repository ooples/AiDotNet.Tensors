using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Third derivatives through ops whose create-graph backward differentiates a composite of recorded ops
/// (FusedLinear and everything built on it: the RNN and GRU cells, GLU, ...), with every backward under createGraph
/// on one tape. Each level must actually HAVE a gradient: the composite used to be differentiated under a nested tape
/// whose entries the next create-graph backward never walked, so the second level came back empty — and a check that
/// treats a missing gradient as zero against finite differences of a function that is then also zero passes
/// vacuously.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CompositeThirdOrderTests : IDisposable
{
    private readonly IEngine _previous;
    private readonly CpuEngine _e = new();

    public CompositeThirdOrderTests()
    {
        _previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _e;
    }

    public void Dispose() => AiDotNetEngine.Current = _previous;

    private static Tensor<double> T(int[] shape, int seed, double scale = 1)
    {
        var r = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (0.35 + r.NextDouble() * 0.6) * scale;
        return t;
    }

    private Tensor<double> Contract(Dictionary<Tensor<double>, Tensor<double>> g, Tensor<double>[] xs, Tensor<double>[] ws, int level)
    {
        Tensor<double>? total = null;
        for (int i = 0; i < xs.Length; i++)
        {
            Assert.True(g.TryGetValue(xs[i], out var gi) && gi is not null, $"level {level}: no gradient for input {i}");
            var term = _e.ReduceSum(_e.TensorMultiply(gi!, ws[i]), null);
            total = total is null ? term : _e.TensorAdd(total, term);
        }
        return total!;
    }

    private void AssertThirdOrder(Func<Tensor<double>[], Tensor<double>> f, params Tensor<double>[] xs)
    {
        var u = T(f(xs)._shape, 1);
        var v = xs.Select((s, i) => T(s._shape, 100 + i)).ToArray();
        var w = xs.Select((s, i) => T(s._shape, 200 + i)).ToArray();

        // K(x) = Σ⟨∂h/∂x, w⟩, h = Σ⟨∂L/∂x, v⟩, L = ⟨f(x)², u⟩, and its gradient by a third create-graph backward.
        double K(bool differentiate, out Dictionary<Tensor<double>, Tensor<double>>? third)
        {
            third = null;
            using var tape = new GradientTape<double>();
            var y = f(xs);
            var loss = _e.ReduceSum(_e.TensorMultiply(_e.TensorMultiply(y, y), u), null);
            var h = Contract(tape.ComputeGradients(loss, xs, createGraph: true), xs, v, 1);
            var k = Contract(tape.ComputeGradients(h, xs, createGraph: true), xs, w, 2);
            if (differentiate) third = tape.ComputeGradients(k, xs, createGraph: true);
            return k[0];
        }

        K(true, out var analytic);
        for (int s = 0; s < xs.Length; s++)
        {
            Assert.True(analytic!.TryGetValue(xs[s], out var g) && g is not null, $"level 3: no gradient for input {s}");
            for (int e = 0; e < Math.Min(4, xs[s].Length); e++)
            {
                double original = xs[s][e];
                xs[s][e] = original + 1e-5;
                double hi = K(false, out _);
                xs[s][e] = original - 1e-5;
                double lo = K(false, out _);
                xs[s][e] = original;
                double numerical = (hi - lo) / 2e-5;
                Assert.True(Math.Abs(g![e] - numerical) <= 1e-6 + 1e-5 * Math.Abs(numerical),
                    $"input {s}[{e}]: third-order {g[e]:G10} vs finite differences {numerical:G10}");
            }
        }
    }

    [Fact]
    public void FusedLinear_ThirdDerivative() =>
        AssertThirdOrder(a => _e.FusedLinear(a[0], a[1], a[2], FusedActivationType.None), T(new[] { 2, 3 }, 5), T(new[] { 3, 4 }, 6), T(new[] { 4 }, 7));

    [Fact]
    public void FusedLinearTanh_ThirdDerivative() =>
        AssertThirdOrder(a => _e.FusedLinear(a[0], a[1], a[2], FusedActivationType.Tanh), T(new[] { 2, 3 }, 5), T(new[] { 3, 4 }, 6), T(new[] { 4 }, 7));

    [Fact]
    public void Glu_ThirdDerivative() => AssertThirdOrder(a => _e.GLU(a[0], -1), T(new[] { 2, 4 }, 3));

    [Theory]
    [InlineData(RnnCellType.RnnTanh)]
    [InlineData(RnnCellType.RnnRelu)]
    public void RnnCell_ThirdDerivative(RnnCellType cell) =>
        AssertThirdOrder(a => _e.TensorRnnCell(a[0], a[1], a[2], a[3], a[4], a[5], cell),
            T(new[] { 2, 3 }, 5), T(new[] { 2, 4 }, 6), T(new[] { 4, 3 }, 7), T(new[] { 4, 4 }, 8), T(new[] { 4 }, 9), T(new[] { 4 }, 10));

    [Fact]
    public void GruCell_ThirdDerivative() =>
        AssertThirdOrder(a => _e.TensorGruCell(a[0], a[1], a[2], a[3], a[4], a[5]),
            T(new[] { 2, 3 }, 5), T(new[] { 2, 4 }, 6), T(new[] { 12, 3 }, 7), T(new[] { 12, 4 }, 8), T(new[] { 12 }, 9), T(new[] { 12 }, 10));

    [Fact]
    public void Bilinear_ThirdDerivative() =>
        AssertThirdOrder(a => _e.TensorBilinear(a[0], a[1], a[2], a[3]),
            T(new[] { 2, 3 }, 5), T(new[] { 2, 2 }, 6), T(new[] { 4, 3, 2 }, 7), T(new[] { 4 }, 8));

    [Fact]
    public void HyperbolicLinear_ThirdDerivative() =>
        AssertThirdOrder(a => _e.TensorHyperbolicLinear(a[0], a[1], a[2], 1.0),
            T(new[] { 2, 3 }, 5, 0.2), T(new[] { 4, 3 }, 6), T(new[] { 4 }, 7, 0.1));
}
