using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The lazy node for MaxPool2D(input, int poolSize, int stride, int padding) saved only its pool size and stride but
/// named MaxPool2DBackward, which casts savedState[0] to the argmax array: a compiled training plan threw
/// InvalidCastException on its first step. Its gradient must now match the eager tape's, unpadded and padded.
/// </summary>
[Collection("CompiledTrainingPlanSerial")]
public class CompiledIntMaxPoolTrainingTests
{
    private static Tensor<float> Seq(int[] shape, int seed)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)Math.Sin(seed * 7 + i * 0.31) * 0.3f;
        return new Tensor<float>(a, shape);
    }

    private static Tensor<float> Forward(IEngine e, Tensor<float> x, Tensor<float> k, int padding)
    {
        var pooled = e.MaxPool2D(e.Conv2D(x, k, 1, 1), 2, 2, padding);
        return e.ReduceSum(e.TensorMultiply(pooled, pooled), null);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void CompiledGradient_MatchesTheEagerTape(int padding)
    {
        var engine = new CpuEngine();
        var x = Seq(new[] { 2, 3, 7, 7 }, 1);

        var kEager = Seq(new[] { 4, 3, 3, 3 }, 2);
        float[] expected;
        using (var tape = new GradientTape<float>())
        {
            var loss = Forward(engine, x, kEager, padding);
            expected = tape.ComputeGradients(loss, sources: new[] { kEager })[kEager].ToArray();
        }

        var k = Seq(new[] { 4, 3, 3, 3 }, 2);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward(engine, x, k, padding);
            plan = scope.CompileTraining(new[] { k });
        }
        using (plan)
        {
            plan.Step();
            var actual = plan.Gradients[0].ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])),
                    $"padding {padding}: dK[{i}] compiled {actual[i]}, eager {expected[i]}");
        }
    }
}
