using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A GPU plan runs a convolution layer's Conv2D -> channel bias add (-> ReLU) as one fused convolution, so the raw
/// convolution output is never stored. Trained for several SGD steps it must reach the CPU plan's parameters, with and
/// without the ReLU and with the bias feeding a BatchNorm, and the GPU plan must have fused it (no separate bias-add or
/// ReLU backward). A stride-2 pair sharing its input (ResNet's downsampling) stays unfused for the shortcut-pair path.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuConvBiasFusionTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    public enum Tail { None, ReLU, BatchNorm }

    private static (float[] Parameters, IReadOnlyList<string> Backward) Train(IEngine engine, Tail tail, int steps)
    {
        var x = Filled(new[] { 3, 4, 9, 9 }, 1, 2f);
        var k = Filled(new[] { 5, 4, 3, 3 }, 2, 0.4f);
        var bias = Filled(new[] { 5 }, 3, 0.5f);
        var gamma = Filled(new[] { 5 }, 4, 1f);
        var beta = Filled(new[] { 5 }, 5, 1f);
        var parameters = tail == Tail.BatchNorm ? new[] { k, bias, gamma, beta } : new[] { k, bias };
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var y = engine.TensorChannelBiasAdd(engine.Conv2D(x, k, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 }), bias);
            if (tail == Tail.ReLU) y = engine.ReLU(y);
            if (tail == Tail.BatchNorm) y = engine.BatchNorm(y, gamma, beta, 1e-5, out _, out _);
            engine.ReduceSum(engine.TensorMultiply(y, y), null);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.002f);
            for (int s = 0; s < steps; s++) plan.Step();
            var backward = plan is CompiledTrainingPlan<float> compiled ? compiled.BackwardStepKinds : Array.Empty<string>();
            var all = new List<float>();
            foreach (var p in parameters) all.AddRange(p.ToArray());
            return (all.ToArray(), backward);
        }
    }

    private static DirectGpuTensorEngine? TryGpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            return null;
        }
        if (gpu.IsGpuAvailable) return gpu;
        gpu.Dispose();
        return null;
    }

    [SkippableTheory]
    [InlineData(Tail.None)]
    [InlineData(Tail.ReLU)]
    [InlineData(Tail.BatchNorm)]
    public void Fused_conv_bias_trains_like_the_cpu(Tail tail)
    {
        var gpu = TryGpu();
        if (gpu is null)
        {
            Skip.If(true, "No GPU device.");
            return;
        }
        using (gpu)
        {
            const int steps = 4;
            var expected = Train(new CpuEngine(), tail, steps).Parameters;
            var (actual, backward) = Train(gpu, tail, steps);
            Assert.Contains(backward, kind => kind == "generic:FusedConv2DBias");
            Assert.DoesNotContain(backward, kind => kind == "generic:TensorChannelBiasAdd" || kind == "generic:ReLU");
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-3f * (1f + Math.Abs(expected[i])),
                    $"[{i}]: cpu {expected[i]}, gpu {actual[i]} ({tail})");
        }
    }

    [SkippableFact]
    public void A_stride_two_pair_sharing_its_input_stays_unfused()
    {
        var gpu = TryGpu();
        if (gpu is null)
        {
            Skip.If(true, "No GPU device.");
            return;
        }
        using (gpu)
        {
            var x = Filled(new[] { 2, 4, 8, 8 }, 1, 2f);
            var main = Filled(new[] { 6, 4, 3, 3 }, 2, 0.4f);
            var shortcut = Filled(new[] { 6, 4, 1, 1 }, 3, 0.4f);
            var b1 = Filled(new[] { 6 }, 4, 0.5f);
            var b2 = Filled(new[] { 6 }, 5, 0.5f);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var y1 = gpu.TensorChannelBiasAdd(gpu.Conv2D(x, main, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 }), b1);
                var y2 = gpu.TensorChannelBiasAdd(gpu.Conv2D(x, shortcut, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 1, 1 }), b2);
                var s = gpu.TensorAdd(y1, y2);
                gpu.ReduceSum(gpu.TensorMultiply(s, s), null);
                plan = scope.CompileTraining(new[] { main, shortcut, b1, b2 });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.001f);
                plan.Step();
                var backward = ((CompiledTrainingPlan<float>)plan).BackwardStepKinds;
                Assert.DoesNotContain(backward, kind => kind == "generic:FusedConv2DBias");
            }
        }
    }
}
