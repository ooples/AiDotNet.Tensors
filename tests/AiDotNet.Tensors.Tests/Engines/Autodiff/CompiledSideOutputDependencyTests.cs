using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A compiled training step whose consumer reads only an op's integer side output (the argmax indices of a max-pool
/// or a reduce-max). Those indices are not a graph input, so nothing ordered the consumer after the producer's
/// refresh, and when nothing read the producer's pooled values dead-code elimination dropped the producer: every
/// replay used the trace's indices. MaxPool2DWithTensorIndices also computed its indices once from the lazy trace
/// and never refreshed them. Each case changes the input between steps so the argmax moves, and compares the plan's
/// loss and gradients with the tape's for that step.
/// </summary>
public class CompiledSideOutputDependencyTests
{
    private static void Fill(Tensor<double> t, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, rng.NextDouble() * 4 - 2);
    }

    private static void AssertSame(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * (1 + Math.Abs(expected[i])),
                $"{what}[{i}]: tape {expected[i]:R}, compiled {actual[i]:R}");
    }

    private static void AssertPlanFollowsTheTape(Func<IEngine, Tensor<double>> loss, Tensor<double> changing,
        Tensor<double>[] parameters, string what)
    {
        var engine = new CpuEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            Fill(changing, 100);
            ICompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                loss(engine);
                plan = scope.CompileTraining(parameters);
            }
            using (plan)
            {
                for (int step = 0; step < 4; step++)
                {
                    Fill(changing, 200 + step);
                    double expectedLoss;
                    var expected = new double[parameters.Length][];
                    using (var tape = new GradientTape<double>())
                    {
                        var l = loss(engine);
                        expectedLoss = l.GetFlat(0);
                        var grads = tape.ComputeGradients(l, parameters);
                        for (int i = 0; i < parameters.Length; i++)
                            expected[i] = grads.TryGetValue(parameters[i], out var g) ? g.ToArray() : new double[parameters[i].Length];
                    }
                    double actualLoss = plan.Step().GetFlat(0);
                    Assert.True(Math.Abs(expectedLoss - actualLoss) <= 1e-9 * (1 + Math.Abs(expectedLoss)),
                        $"{what}, step {step}: tape loss {expectedLoss:R}, compiled {actualLoss:R}");
                    for (int i = 0; i < parameters.Length; i++)
                        AssertSame(expected[i], plan.Gradients[i].ToArray(), $"{what}, step {step}, parameter {i}");
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    // Ldexp reads only the reduce's argmax as its exponent; the reduced values themselves are unused.
    [Fact]
    public void Ldexp_OnReduceMaxIndices_FollowsTheRefreshedArgmax()
    {
        var x = new Tensor<double>(new[] { 4, 5 });
        var y = new Tensor<double>(new[] { 0.5, -1.25, 2.0, 0.75 }, new[] { 4 });
        AssertPlanFollowsTheTape(e =>
        {
            e.ReduceMaxWithTensorIndices(x, new[] { 1 }, false, out var argmax);
            return e.ReduceSum(e.TensorLdexp(y, argmax), null, false);
        }, x, new[] { x, y }, "ldexp(y, argmax x)");
    }

    // ScatterAdd reads only the pool's indices; the pooled values are unused.
    [Fact]
    public void ScatterAdd_OnMaxPool3DIndices_FollowsTheRefreshedArgmax()
    {
        var x = new Tensor<double>(new[] { 1, 1, 4, 4, 4 });
        var source = new Tensor<double>(new[] { 1, 1, 2, 2, 2 });
        Fill(source, 7);
        // A position ramp, so the loss and the source's gradient depend on where each value lands.
        var ramp = new Tensor<double>(new[] { 64, 1 });
        for (int i = 0; i < 64; i++) ramp.SetFlat(i, i + 1);
        AssertPlanFollowsTheTape(e =>
        {
            // The indices are flat input positions (d * 16 + h * 4 + w), one per pooled cell; ScatterAdd flattens them.
            e.MaxPool3DWithTensorIndices(x, new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, out var indices);
            var scattered = e.ScatterAdd(e.Reshape(source, new[] { 8, 1 }), indices, 0, 64);
            return e.ReduceSum(e.TensorMultiply(scattered, ramp), null, false);
        }, x, new[] { x, source }, "scatterAdd(source, pool indices)");
    }

    // MaxUnpool reads the 2D pool's indices; the indices used to be computed once from the lazy trace.
    [Fact]
    public void MaxUnpool_OnMaxPool2DIndices_FollowsTheRefreshedArgmax()
    {
        var x = new Tensor<double>(new[] { 1, 2, 4, 4 });
        var w = new Tensor<double>(new[] { 1, 2, 4, 4 });
        Fill(w, 11);
        AssertPlanFollowsTheTape(e =>
        {
            var pooled = e.MaxPool2DWithTensorIndices(x, new[] { 2, 2 }, new[] { 2, 2 }, out var indices);
            var unpooled = e.TensorMaxUnpool(pooled, indices, new[] { 4, 4 });
            return e.ReduceSum(e.TensorMultiply(unpooled, w), null, false);
        }, x, new[] { x, w }, "maxUnpool(maxPool2D x)");
    }
}
