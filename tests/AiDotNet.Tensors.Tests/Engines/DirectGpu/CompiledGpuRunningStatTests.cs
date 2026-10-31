using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A compiled training plan on a GPU engine updates a BatchNorm-style running statistic in place every step
/// (runningMean = 0.9 * runningMean + 0.1 * batchMean, as AiDotNet's BatchNorm layers record it). The statistic is a
/// host tensor the plan binds to a device buffer; the in-place updates run on that buffer. After several steps it must
/// hold every step's contribution, equal to the same plan on the CPU, whether or not the host read it in between. With a
/// ReLU after the BatchNorm the pair fuses (the fused node writes the batch statistics the update reads), and the
/// statistic must be the same.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuRunningStatTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static float[] Run(IEngine engine, int steps, bool readBetweenSteps, bool relu, out IReadOnlyList<string> backwardKinds)
    {
        var x = Filled(new[] { 4, 3, 8, 8 }, 1, 2f);
        var kernel = Filled(new[] { 6, 3, 3, 3 }, 2, 0.6f);
        var gamma = Filled(new[] { 6 }, 3, 1f);
        var beta = Filled(new[] { 6 }, 4, 1f);
        // Nonzero, as a running statistic is after its first step: every in-place update then changes it.
        var runningMean = Filled(new[] { 6 }, 5, 2f);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var conv = engine.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
            var y = engine.BatchNorm(conv, gamma, beta, 1e-5, out var mean, out _);
            engine.TensorMultiplyScalarInPlace(runningMean, 0.9f);
            engine.TensorAddInPlace(runningMean, engine.TensorMultiplyScalar(mean, 0.1f));
            if (relu) y = engine.ReLU(y);
            engine.ReduceSum(engine.TensorMultiply(y, y), null);
            plan = scope.CompileTraining(new[] { kernel, gamma, beta });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int s = 0; s < steps; s++)
            {
                plan.Step();
                if (readBetweenSteps) _ = runningMean.ToArray();
            }
            backwardKinds = plan is CompiledTrainingPlan<float> compiled ? compiled.BackwardStepKinds : Array.Empty<string>();
            return runningMean.ToArray();
        }
    }

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void RunningStatistic_AccumulatesEveryStep_LikeTheCpu(bool readBetweenSteps, bool relu)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        using (gpu)
        {
            const int steps = 4;
            var expected = Run(new CpuEngine(), steps, readBetweenSteps: false, relu, out _);
            var actual = Run(gpu, steps, readBetweenSteps, relu, out var gpuBackward);
            // The GPU plan runs the pair fused: one BatchNorm+ReLU backward and no separate ReLU backward.
            if (relu)
            {
                Assert.Contains(gpuBackward, k => k.Contains("BatchNormReLU"));
                Assert.DoesNotContain(gpuBackward, k => k == "generic:ReLU");
            }
            // With the learning rate at zero every step's batch mean is the same m, so after n steps the statistic is
            // 0.9^n * r0 + (1 - 0.9^n) * m: a lost or skipped update leaves a different value.
            Assert.Contains(expected, v => Math.Abs(v) > 1e-3f);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * (1f + Math.Abs(expected[i])),
                    $"[{i}]: cpu {expected[i]}, gpu {actual[i]} (readBetweenSteps={readBetweenSteps}, relu={relu})");
        }
    }
}
