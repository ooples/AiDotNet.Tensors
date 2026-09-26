using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A GraphMode recording must hand its backward the same saved state the eager tape records.
/// These ops recorded their compiled node with a backward that indexes savedState but passed
/// none, so a compiled training step threw IndexOutOfRangeException on the first backward
/// (AiDotNet's EfficientConformer CTC head fell back to eager every step). TaylorSoftmax also
/// recorded the plain softmax backward. Each case compares the compiled gradient with eager.
/// </summary>
[Collection("CompilationGlobalState")]
public class GraphModeSavedStateTests
{
    private static Tensor<double> Filled(int[] shape, Func<int, double> value)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = value(i);
        return t;
    }

private static Tensor<double> EagerGradient(
        IEngine engine, int[] paramShape, Func<int, double> values, Func<IEngine, Tensor<double>, Tensor<double>> op)
    {
        var x = Filled(paramShape, values);
        using var tape = new GradientTape<double>();
        var y = op(engine, x);
        var w = Filled(y._shape, i => 0.5 + 0.25 * (i % 5));
        var loss = engine.ReduceSum(engine.TensorMultiply(y, w), null);
        return tape.ComputeGradients(loss, sources: new[] { x })[x];
    }

    private static void AssertSameGradient(Tensor<double> eager, Tensor<double>? compiled, string step)
    {
        Assert.NotNull(compiled);
        if (compiled is null) return;
        Assert.Equal(eager.Length, compiled.Length);
        for (int i = 0; i < eager.Length; i++)
            Assert.True(Math.Abs(eager[i] - compiled[i]) < 1e-9,
                $"{step}: grad[{i}] eager={eager[i]:R} compiled={compiled[i]:R}");
    }

    /// <summary>
    /// Replays the compiled plan twice: once at the traced values, then again after the parameter
    /// changes in place. Each replay's gradient must match a fresh eager gradient at the same
    /// values, so a backward that kept saved state from an earlier replay cannot pass.
    /// </summary>
    private static void AssertCompiledGradientMatchesEager(
        int[] paramShape, Func<int, double> init, Func<IEngine, Tensor<double>, Tensor<double>> op)
    {
        var engine = new CpuEngine();
        Func<int, double> changed = i => init(i) + 0.05 * ((i % 3) + 1);

        var xF = Filled(paramShape, init);
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var y = op(engine, xF);
            var w = Filled(y._shape, i => 0.5 + 0.25 * (i % 5));
            engine.ReduceSum(engine.TensorMultiply(y, w), null);
            plan = scope.CompileTraining(new[] { xF });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
            AssertSameGradient(EagerGradient(engine, paramShape, init, op), xF.Grad, "first replay");

            for (int i = 0; i < xF.Length; i++) xF[i] = changed(i);
            plan.Step();
            AssertSameGradient(EagerGradient(engine, paramShape, changed, op), xF.Grad, "replay after the parameter changed");
        }
    }

    [Fact]
    public void CtcLoss_CompiledBackward_MatchesEager()
    {
        var targets = new Tensor<int>(new[] { 1, 2, 3 }, new[] { 3 });
        AssertCompiledGradientMatchesEager(new[] { 5, 2, 4 }, i => Math.Log(0.25) + 0.01 * (i % 7),
            (e, x) => e.TensorCTCLoss(x, targets, new[] { 5, 5 }, new[] { 2, 1 }, 0));
    }

    [Fact]
    public void Sparsemax_CompiledBackward_MatchesEager()
        => AssertCompiledGradientMatchesEager(new[] { 3, 5 }, i => 0.3 * (i % 4) - 0.2 * (i % 3),
            (e, x) => e.Sparsemax(x, -1));

    [Fact]
    public void TaylorSoftmax_CompiledBackward_MatchesEager()
        => AssertCompiledGradientMatchesEager(new[] { 3, 5 }, i => 0.1 * (i % 6) - 0.2,
            (e, x) => e.TaylorSoftmax(x, 2, -1));

    [Fact]
    public void RepeatElements_CompiledBackward_MatchesEager()
        => AssertCompiledGradientMatchesEager(new[] { 3, 4 }, i => 0.1 * i,
            (e, x) => e.TensorRepeatElements(x, 2, 0));

    [Fact]
    public void ComplexMagnitude_CompiledBackward_MatchesEager()
        => AssertCompiledGradientMatchesEager(new[] { 6 }, i => 0.5 + 0.3 * i,
            (e, x) => e.TensorComplexMagnitude(x));

    [Fact]
    public void ScatterAdd_CompiledBackward_MatchesEager()
    {
        var destination = Filled(new[] { 5, 3 }, i => 0.01 * i);
        var indices = new Tensor<int>(new[] { 1, 3 }, new[] { 2 });
        AssertCompiledGradientMatchesEager(new[] { 2, 3 }, i => 0.2 * i,
            (e, x) => e.TensorScatterAdd(destination, indices, x, 0));
    }

    [Fact]
    public void ScatterSoftmax_CompiledBackward_MatchesEager()
    {
        var indices = new Tensor<int>(new[] { 0, 0, 1, 1, 1, 2 }, new[] { 6 });
        AssertCompiledGradientMatchesEager(new[] { 6 }, i => 0.3 * i - 0.5,
            (e, x) => e.ScatterSoftmax(x, indices, 0, 3));
    }

    [Fact]
    public void MaskedFillBitMask_CompiledBackward_MatchesEager()
    {
        var mask = new Tensor<Bit>(new[] { 3, 4 });
        for (int i = 0; i < mask.Length; i++) mask[i] = i % 3 == 0;
        AssertCompiledGradientMatchesEager(new[] { 3, 4 }, i => 0.1 * i,
            (e, x) => e.TensorMaskedFill(x, mask, 0.0));
    }

    [Fact]
    public void ReduceLogVariance_CompiledBackward_MatchesEager()
        => AssertCompiledGradientMatchesEager(new[] { 3, 5 }, i => 0.1 * i + 0.05 * (i % 3),
            (e, x) => e.ReduceLogVariance(x, new[] { 1 }, false, 1e-8));
    [Fact]
    public void TrilinearInterpolate_CompiledBackward_MatchesEager()
    {
        var positions = new Tensor<double>(new[] { 0.5, 1.25, 0.75, 1.5, 0.25, 1.75, 0.9, 0.9, 0.3 }, new[] { 3, 3 });
        AssertCompiledGradientMatchesEager(new[] { 3, 3, 3, 2 }, i => 0.1 * (i % 7) - 0.2,
            (e, x) => e.TensorTrilinearInterpolate(x, positions));
    }
}