using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A compiled op's replay closure captured its argument arrays by reference. Editing a pad array after tracing
/// changed the padding of every replay while the recorded output shape stayed the trace's, and repointing an element
/// of a variadic op's operand array made replays read a tensor the graph never recorded as a dependency. A plan must
/// keep computing what was traced.
/// </summary>
public class CompiledArgumentSnapshotTests
{
    private static Tensor<double> Rnd(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertPlanMatches(Func<IEngine, Tensor<double>> traced, Action afterTrace,
        Tensor<double>[] parameters, string what)
    {
        var engine = new CpuEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            double expectedLoss;
            var expected = new double[parameters.Length][];
            using (var tape = new GradientTape<double>())
            {
                var loss = traced(engine);
                expectedLoss = loss.GetFlat(0);
                var grads = tape.ComputeGradients(loss, parameters);
                for (int i = 0; i < parameters.Length; i++)
                    expected[i] = grads.TryGetValue(parameters[i], out var g) ? g.ToArray() : new double[parameters[i].Length];
            }

            ICompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                traced(engine);
                plan = scope.CompileTraining(parameters);
            }
            afterTrace();
            using (plan)
            {
                for (int step = 0; step < 2; step++)
                {
                    double loss = plan.Step().GetFlat(0);
                    Assert.True(Math.Abs(expectedLoss - loss) <= 1e-12 * (1 + Math.Abs(expectedLoss)),
                        $"{what}, step {step}: traced loss {expectedLoss:R}, replayed {loss:R}");
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        var actual = plan.Gradients[i].ToArray();
                        for (int k = 0; k < actual.Length; k++)
                            Assert.True(Math.Abs(expected[i][k] - actual[k]) <= 1e-12 * (1 + Math.Abs(expected[i][k])),
                                $"{what}, step {step}, parameter {i}[{k}]: traced {expected[i][k]:R}, replayed {actual[k]:R}");
                    }
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    // [1, 1] and [2, 0] pad the last axis to the same width, so only the values reveal which one a replay used.
    [Fact]
    public void PadNd_PadArrayEditedAfterTracing_ReplaysTheTracedPadding()
    {
        var x = Rnd(new[] { 2, 3 }, 1);
        var w = Rnd(new[] { 2, 5 }, 2);
        var pad = new[] { 1, 1 };
        AssertPlanMatches(e => e.ReduceSum(e.TensorMultiply(e.PadNd(x, pad, PadMode.Reflect), w), null, false),
            () => { pad[0] = 2; pad[1] = 0; }, new[] { x, w }, "PadNd");
    }

    // The operand array's second element is repointed at another tensor of the same shape after tracing.
    [Fact]
    public void MultiDot_OperandArrayRepointedAfterTracing_ReplaysTheTracedOperands()
    {
        var a = Rnd(new[] { 2, 3 }, 3);
        var b = Rnd(new[] { 3, 2 }, 4);
        var other = Rnd(new[] { 3, 2 }, 5);
        var matrices = new[] { a, b };
        AssertPlanMatches(e => e.ReduceSum(e.TensorMultiDot(matrices), null, false),
            () => matrices[1] = other, new[] { a, b }, "MultiDot");
    }
}
