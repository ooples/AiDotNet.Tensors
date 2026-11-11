using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// FusedLinearInto writes into a caller's buffer and is not sweepable (it returns nothing). Under a tape it must be
/// the same differentiable function as FusedLinear for every activation: the same values, first-order gradients and
/// create-graph second-order gradients (FusedLinear's are pinned against finite differences by the gradgrad sweep).
/// It used to record nothing, so a network built on it had no gradient, let alone a Hessian.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class FusedLinearIntoHigherOrderTests : IDisposable
{
    private readonly IEngine _previous;
    private readonly CpuEngine _engine = new();

    public FusedLinearIntoHigherOrderTests()
    {
        _previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _engine;
    }

    public void Dispose() => AiDotNetEngine.Current = _previous;

    public static IEnumerable<object[]> Activations()
        => Enum.GetValues(typeof(FusedActivationType)).Cast<FusedActivationType>()
            // Gumbel-Softmax samples fresh noise per call, so two calls cannot be compared value for value.
            .Where(a => a != FusedActivationType.GumbelSoftmax)
            .Select(a => new object[] { a });

    private static Tensor<double> Make(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 1.6 - 0.8;
        return t;
    }

    // value, first-order gradient, and the gradient of ⟨first-order gradient, v⟩ (a Hessian-vector product), for x.
    private (double[] Value, double[] Grad, double[] Hvp) Derivatives(Func<Tensor<double>, Tensor<double>> f, Tensor<double> x)
    {
        var v = Make(x._shape, 99);
        using var tape = new GradientTape<double>();
        var y = f(x);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(y, y), null);
        var g = tape.ComputeGradients(loss, new[] { x }, createGraph: true)[x];
        var h = _engine.ReduceSum(_engine.TensorMultiply(g, v), null);
        var hv = tape.ComputeGradients(h, new[] { x }).TryGetValue(x, out var second) ? second.ToArray() : new double[x.Length];
        return (y.ToArray(), g.ToArray(), hv);
    }

    [Theory]
    [MemberData(nameof(Activations))]
    public void FusedLinearInto_UnderATape_IsFusedLinear_ToSecondOrder(FusedActivationType activation)
    {
        var w = Make(new[] { 3, 4 }, 2);
        var b = Make(new[] { 4 }, 3);
        var x = Make(new[] { 2, 3 }, 1);

        var fused = Derivatives(input => _engine.FusedLinear(input, w, b, activation), x);
        var into = Derivatives(input =>
        {
            var destination = new Tensor<double>(new[] { 2, 4 });
            _engine.FusedLinearInto(destination, input, w, b, activation);
            return destination;
        }, x);

        void Same(double[] expected, double[] actual, string what)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12 * (1 + Math.Abs(expected[i])),
                    $"{activation} {what}[{i}]: FusedLinear {expected[i]:G17} vs FusedLinearInto {actual[i]:G17}");
        }
        Same(fused.Value, into.Value, "value");
        Same(fused.Grad, into.Grad, "gradient");
        Same(fused.Hvp, into.Hvp, "Hessian-vector product");
    }
}
