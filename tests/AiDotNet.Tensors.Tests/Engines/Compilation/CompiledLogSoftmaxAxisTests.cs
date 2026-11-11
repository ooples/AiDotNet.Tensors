using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression: the compiled plans' specialized LogSoftmax forward (a pinned per-row kernel) accepted every rank-2
/// LogSoftmax step and ran it over the LAST axis, ignoring the axis the node saved. A log-softmax (or log-softmin) over
/// axis 0 therefore replayed as one over axis 1 -- in inference and in training plans alike.
/// </summary>
public class CompiledLogSoftmaxAxisTests
{
    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        return new Tensor<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
            .Select(_ => (float)(rng.NextDouble() * 6 - 3)).ToArray(), shape);
    }

    private static void Close(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * (1 + Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, false)]
    [InlineData(0, true)]
    public void CompiledInference_ReplaysTheRecordedAxis_OnLiveInputs(int axis, bool softmin)
    {
        var engine = new CpuEngine();
        var x = Rand(1, 4, 9);
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.EnableInference())
        {
            var y = softmin ? engine.TensorLogSoftmin(x, axis) : engine.TensorLogSoftmax(x, axis);
            plan = scope.CompileInference(y, new[] { x });
        }
        using (plan)
        {
            for (int round = 0; round < 3; round++)
            {
                Rand(10 + round, 4, 9).AsSpan().CopyTo(x.AsWritableSpan());
                var fresh = new Tensor<float>(x.ToArray(), new[] { 4, 9 });
                var eager = softmin ? engine.TensorLogSoftmin(fresh, axis) : engine.TensorLogSoftmax(fresh, axis);
                Close(eager.ToArray(), plan.Execute().ToArray(), $"axis {axis} round {round}");
            }
        }
    }

    [Fact]
    public void CompiledTraining_LogSoftmaxOverAxis0_MatchesTheTape()
    {
        var engine = new CpuEngine();
        var p = Rand(2, 5, 7);
        var coef = Rand(3, 5, 7);
        Tensor<float> Forward() => engine.ReduceSum(engine.TensorMultiply(engine.TensorLogSoftmax(engine.TensorMultiply(p, p), 0), coef), null);
        float[] expectedLoss, expectedGrad;
        using (var tape = new GradientTape<float>())
        {
            var loss = Forward();
            expectedLoss = loss.ToArray();
            expectedGrad = tape.ComputeGradients(loss, new[] { p })[p].ToArray();
        }
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward();
            plan = scope.CompileTraining(new[] { p });
        }
        using (plan)
        {
            var loss = plan.Step();
            Close(expectedLoss, loss.ToArray(), "loss");
            Close(expectedGrad, plan.Gradients[0].ToArray(), "grad");
        }
    }
}
