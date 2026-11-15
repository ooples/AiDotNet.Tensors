using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// On a device engine the gradients that alias views point at (a residual add's output gradient, which both operands'
/// gradients view, and a channel-bias add's) are liveness-pooled with the other gradients, and their views re-pointed with
/// them. A residual network trained for several SGD steps, captured or not, must reach the CPU plan's parameters with
/// that pooling on and off, and pooling them must pool more gradients than leaving them out.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuAliasGradientPoolingTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static (float[] Parameters, int PooledGradients) Train(IEngine engine, int steps, bool captured)
    {
        // Residual blocks h = h + ReLU(ChannelBias(Conv(Tanh(h)))): the add's output gradient is an alias target (both
        // operands view it, the residual stream through a multi-consumer merge where the backward allows it), and so is
        // the channel-bias add's. Each block's gradients die before the next block's backward starts, so they can share.
        var x = Filled(new[] { 2, 4, 8, 8 }, 1, 2f);
        var parameters = new List<Tensor<float>>();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = x;
            for (int b = 0; b < 4; b++)
            {
                var k = Filled(new[] { 4, 4, 3, 3 }, 10 + b, 0.3f);
                var bias = Filled(new[] { 4 }, 20 + b, 0.5f);
                parameters.Add(k); parameters.Add(bias);
                var conv = engine.Conv2D(engine.Tanh(h), k, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
                h = engine.TensorAdd(h, engine.ReLU(engine.TensorChannelBiasAdd(conv, bias)));
            }
            engine.ReduceSum(engine.TensorMultiply(h, h), null);
            plan = scope.CompileTraining(parameters.ToArray());
        }
        using (plan)
        {
            int pooled = plan is CompiledTrainingPlan<float> compiled ? compiled.PooledGradientCount : -1;
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.001f);
            if (!captured && plan is CompiledTrainingPlan<float> uncaptured) uncaptured.DisableGraphStep();
            for (int s = 0; s < steps; s++) plan.Step();
            var all = new List<float>();
            foreach (var p in parameters) all.AddRange(p.ToArray());
            return (all.ToArray(), pooled);
        }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pooled_alias_target_gradients_train_like_the_cpu(bool captured)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        bool previous = CompiledTrainingPlan<float>.PoolAliasTargetGradients;
        using (gpu)
        {
            try
            {
                const int steps = 8;
                var expected = Train(new CpuEngine(), steps, captured: false).Parameters;
                CompiledTrainingPlan<float>.PoolAliasTargetGradients = false;
                var separate = Train(gpu, steps, captured);
                CompiledTrainingPlan<float>.PoolAliasTargetGradients = true;
                var pooled = Train(gpu, steps, captured);

                foreach (var (actual, name) in new[] { (separate.Parameters, "separate"), (pooled.Parameters, "pooled") })
                {
                    Assert.Equal(expected.Length, actual.Length);
                    for (int i = 0; i < expected.Length; i++)
                        Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-3f * (1f + Math.Abs(expected[i])),
                            $"{name} [{i}]: cpu {expected[i]}, gpu {actual[i]}");
                }
                Assert.True(pooled.PooledGradients > separate.PooledGradients,
                    $"pooling alias targets pooled {pooled.PooledGradients} gradients, leaving them out {separate.PooledGradients}");
            }
            finally
            {
                CompiledTrainingPlan<float>.PoolAliasTargetGradients = previous;
            }
        }
    }
}
