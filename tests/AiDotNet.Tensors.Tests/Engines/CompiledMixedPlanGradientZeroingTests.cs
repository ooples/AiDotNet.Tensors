using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A compiled CPU training plan that mixes specialized and generic backward actions zeroes only the gradient
/// buffers something ACCUMULATES into (multi-consumer tensors and the inputs of generic steps) after its first step.
/// These run a CNN-shaped graph that has both kinds (Conv2D / ReLU / MaxPool / MatMul specialized; channel-bias add
/// and adaptive average pool generic) for several steps with fixed parameters: every step must reproduce the tape's
/// gradients, and step N must equal step 1 bit for bit -- a buffer that missed its zeroing would add onto the
/// previous step's gradient.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class CompiledMixedPlanGradientZeroingTests
{
    private static Tensor<float> Rnd(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static Tensor<float> Forward(IEngine engine, Tensor<float> x, Tensor<float> k1, Tensor<float> b1,
        Tensor<float> k2, Tensor<float> b2, Tensor<float> w, Tensor<float> coef, bool graph)
    {
        var h = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(x, k1, 1, 1, 1), b1));
        h = graph
            ? engine.MaxPool2DWithIndices(h, new[] { 2, 2 }, new[] { 2, 2 }, out _)
            : engine.MaxPool2DWithTensorIndices(h, new[] { 2, 2 }, new[] { 2, 2 }, out _);
        h = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(h, k2, 1, 1, 1), b2));
        h = engine.AdaptiveAvgPool2D(h, 2, 2);
        var logits = engine.TensorMatMul(engine.Reshape(h, new[] { x._shape[0], 6 * 4 }), w);
        return engine.ReduceSum(engine.TensorMultiply(logits, coef), null);
    }

    [Fact]
    public void EveryStepReproducesTheFirstAndTheTape()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = new CpuEngine();
            var x = Rnd(new[] { 3, 2, 10, 10 }, 1);
            var k1 = Rnd(new[] { 4, 2, 3, 3 }, 2, 0.5f);
            var b1 = Rnd(new[] { 4 }, 3, 0.1f);
            var k2 = Rnd(new[] { 6, 4, 3, 3 }, 4, 0.5f);
            var b2 = Rnd(new[] { 6 }, 5, 0.1f);
            var w = Rnd(new[] { 24, 5 }, 6, 0.5f);
            var coef = Rnd(new[] { 3, 5 }, 7);
            var parameters = new[] { k1, b1, k2, b2, w };

            float[][] tape;
            using (var t = new GradientTape<float>())
            {
                var loss = Forward(engine, x, k1, b1, k2, b2, w, coef, graph: false);
                var g = t.ComputeGradients(loss, parameters);
                tape = Array.ConvertAll(parameters, p => (float[])g[p].GetFlattenedData().Clone());
            }

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(engine, x, k1, b1, k2, b2, w, coef, graph: true);
                plan = scope.CompileTraining(parameters);
            }
            try
            {
                float[][]? first = null;
                for (int step = 0; step < 3; step++)
                {
                    plan.Step();
                    var now = Array.ConvertAll(plan.Gradients, g => g.AsSpan().ToArray());
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        for (int i = 0; i < tape[p].Length; i++)
                        {
                            float expected = tape[p][i], got = now[p][i];
                            Assert.True(Math.Abs(got - expected) <= 1e-4f * (1f + Math.Abs(expected)),
                                $"step {step} parameter {p} element {i}: tape {expected:R} plan {got:R}");
                            if (first is not null)
                                Assert.True(BitConverter.SingleToInt32Bits(first[p][i]) == BitConverter.SingleToInt32Bits(got),
                                    $"step {step} parameter {p} element {i} drifted from step 0: {first[p][i]:R} -> {got:R}");
                        }
                    }
                    first ??= now;
                }
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }
}
