using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// An in-place op in the middle of a chain must pass its own derivative back to whatever produced the tensor it
/// overwrote: d/dx sum(op_(2x)) = 2 op'(2x). The tape keys gradients by tensor object, and the overwritten tensor is
/// the output of both the producer and the in-place op, so these pin that the two gradients stay apart.
/// </summary>
public class InPlaceAutogradChainTests
{
    private static readonly double[] Xs = { -1.5, -0.25, 0.5, 2.0 };

    private static Tensor<double> X() => new Tensor<double>((double[])Xs.Clone(), new[] { 4 });

    private static double[] Expected(System.Func<double, double> derivative)
    {
        var g = new double[Xs.Length];
        for (int i = 0; i < Xs.Length; i++) g[i] = 2.0 * derivative(2.0 * Xs[i]);
        return g;
    }

    private static double[] TapeGradient(System.Action<IEngine, Tensor<double>> inPlace)
    {
        IEngine engine = new CpuEngine();
        var x = X();
        using var tape = new GradientTape<double>();
        var y = engine.TensorMultiplyScalar(x, 2.0);
        inPlace(engine, y);
        var loss = engine.ReduceSum(y, null);
        return tape.ComputeGradients(loss, new[] { x })[x].ToArray();
    }

    private static double[] StreamingGradient(System.Action<IEngine, Tensor<double>> inPlace)
    {
        IEngine engine = new CpuEngine();
        var x = X();
        using var tape = new GradientTape<double>();
        var y = engine.TensorMultiplyScalar(x, 2.0);
        inPlace(engine, y);
        var loss = engine.ReduceSum(y, null);
        double[]? gx = null;
        tape.ComputeGradientsStreaming(loss, new[] { x }, (src, grad) => gx = grad.ToArray());
        Assert.NotNull(gx);
        return gx!;
    }

    private static void AssertClose(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
    }

    [Fact]
    public void ReLUInPlace_InAChain_ComputeGradients() =>
        AssertClose(Expected(v => v > 0 ? 1 : 0), TapeGradient((e, y) => e.ReLUInPlace(y)));

    [Fact]
    public void ReLUInPlace_InAChain_Streaming() =>
        AssertClose(Expected(v => v > 0 ? 1 : 0), StreamingGradient((e, y) => e.ReLUInPlace(y)));

    [Fact]
    public void GeneratedAbsInPlace_InAChain_ComputeGradients() =>
        AssertClose(Expected(v => v > 0 ? 1 : -1), TapeGradient((e, y) => e.TensorAbsInPlace(y)));

    [Fact]
    public void GeneratedAbsInPlace_InAChain_Streaming() =>
        AssertClose(Expected(v => v > 0 ? 1 : -1), StreamingGradient((e, y) => e.TensorAbsInPlace(y)));

    [Fact]
    public void AUseOfTheOldValue_KeepsItsGradient()
    {
        // z reads y before relu_ overwrites it: d/dx [sum((2x)^2) + sum(relu(2x))] = 8x + 2 [2x > 0]. torch raises a
        // version error here; the rebased history differentiates it.
        IEngine engine = new CpuEngine();
        var x = X();
        using var tape = new GradientTape<double>();
        var y = engine.TensorMultiplyScalar(x, 2.0);
        var z = engine.ReduceSum(engine.TensorMultiply(y, y), null);
        engine.ReLUInPlace(y);
        var loss = engine.TensorAdd(z, engine.ReduceSum(y, null));
        var g = tape.ComputeGradients(loss, new[] { x })[x].ToArray();
        for (int i = 0; i < Xs.Length; i++) Assert.Equal(8 * Xs[i] + (Xs[i] > 0 ? 2 : 0), g[i], 12);
    }

    [Fact]
    public void SelfAliasedAddInPlace_DifferentiatesAsTwiceTheValue()
    {
        IEngine engine = new CpuEngine();
        var x = X();
        using var tape = new GradientTape<double>();
        var y = engine.TensorMultiplyScalar(x, 2.0);
        engine.TensorAddInPlace(y, y);                 // y = 4x
        var g = tape.ComputeGradients(engine.ReduceSum(y, null), new[] { x })[x].ToArray();
        Assert.All(g, v => Assert.Equal(4.0, v, 12));
    }

    [Fact]
    public void FormerlyUnrecordedInPlaceOps_NowCarryTheirDerivative()
        // tanh_ used to record nothing, so the producer got the gradient as if tanh_ had not run.
        => AssertClose(Expected(v => 1 - Math.Tanh(v) * Math.Tanh(v)), TapeGradient((e, y) => e.TanhInPlace(y)));

    [Fact]
    public void TwoInPlaceOpsInARow_ComputeGradients() =>
        // tanh_(abs_(2x)): derivative 2 * sign(2x) * (1 - tanh(|2x|)^2)
        AssertClose(
            Expected(v => (v > 0 ? 1 : -1) * (1 - System.Math.Tanh(System.Math.Abs(v)) * System.Math.Tanh(System.Math.Abs(v)))),
            TapeGradient((e, y) => { e.TensorAbsInPlace(y); e.TanhInPlace(y); }));
}
