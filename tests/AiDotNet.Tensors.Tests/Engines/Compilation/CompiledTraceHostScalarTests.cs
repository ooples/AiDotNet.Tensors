using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// One-element tensors inside a traced forward: a series model de-normalizes its output by one-element statistics
/// (output * std + mean). The compiled plan's broadcast multiply and add must give the eager tape's loss and gradients,
/// step after step. (Host-scalar reductions of traced values fail closed in a training trace: HostScalarTrainingTraceTests.)
/// </summary>
[Collection("CompilationGlobalState")]
public class CompiledTraceHostScalarTests
{
    private static Tensor<T> Filled<T>(int[] shape, int seed, double scale)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var rng = new Random(seed);
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = ops.FromDouble((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static Tensor<T> Forward<T>(IEngine engine, Tensor<T> input, Tensor<T> w, Tensor<T> std, Tensor<T> mean, Tensor<T> target)
    {
        var h = engine.TensorMatMul(input, w);                                     // [4, 6]
        var denorm = engine.TensorAdd(engine.TensorMultiply(h, std), mean);         // [4, 6] * [1] + [1]
        var diff = engine.TensorSubtract(denorm, target);
        return engine.ReduceSum(engine.TensorMultiply(diff, diff), null);
    }

    private static void Check<T>()
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var engine = new CpuEngine();
        var input = Filled<T>(new[] { 4, 5 }, 1, 2);
        var target = Filled<T>(new[] { 4, 6 }, 2, 2);
        var std = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { ops.FromDouble(0.7) }));
        var mean = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { ops.FromDouble(0.3) }));

        var wEager = Filled<T>(new[] { 5, 6 }, 3, 1);
        using var tape = new GradientTape<T>();
        var eagerLoss = Forward(engine, input, wEager, std, mean, target);
        var expected = tape.ComputeGradients(eagerLoss, sources: new[] { wEager })[wEager].ToArray();
        double expectedLoss = ops.ToDouble(eagerLoss[0]);

        var w = Filled<T>(new[] { 5, 6 }, 3, 1);
        ICompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(new[] { w }))
        {
            Forward(engine, input, w, std, mean, target);
            plan = scope.CompileTraining(new[] { w });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 3; step++)
            {
                double loss = ops.ToDouble(plan.Step()[0]);
                Assert.True(Math.Abs(loss - expectedLoss) <= 1e-4 * (1 + Math.Abs(expectedLoss)),
                    $"{typeof(T).Name} step {step}: compiled loss {loss}, eager {expectedLoss}");
                var actual = w.Grad!.ToArray();
                for (int j = 0; j < expected.Length; j++)
                    Assert.True(Math.Abs(ops.ToDouble(expected[j]) - ops.ToDouble(actual[j])) <= 1e-4 * (1 + Math.Abs(ops.ToDouble(expected[j]))),
                        $"{typeof(T).Name} step {step} grad[{j}]: eager {expected[j]}, compiled {actual[j]}");
            }
        }
    }

    [Fact] public void Float_MatchesTheTape() => Check<float>();
    [Fact] public void Double_MatchesTheTape() => Check<double>();
}
