// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="RunningStatistics.UpdateInPlace{T}"/>: running = keep * running + batchScale * batch, in place, over
/// several updates -- the first touches a host-backed statistic (the GPU engine scales it through the unary in-place
/// path, which binds it), later ones run as one resident AddScaled. Checked against a double reference on the CPU engine
/// and, where CUDA is available, the GPU engine, reading the statistic back from the host after each update.
/// </summary>
[Collection("DirectGpuSerial")]
public class RunningStatisticsTests
{
    private static void Check(IEngine engine)
    {
        const int n = 37;
        var rng = new Random(11);
        var running = new Tensor<float>(new[] { n });
        var expected = new double[n];
        for (int i = 0; i < n; i++) { running[i] = (float)rng.NextDouble(); expected[i] = running[i]; }
        const float keep = 0.9f, scale = 0.1f * 1.02f;
        for (int step = 0; step < 4; step++)
        {
            var batch = new Tensor<float>(new[] { n });
            for (int i = 0; i < n; i++) batch[i] = (float)(rng.NextDouble() * 4 - 2);
            RunningStatistics.UpdateInPlace(engine, running, batch, keep, scale);
            for (int i = 0; i < n; i++) expected[i] = keep * expected[i] + scale * (double)batch[i];
            var host = running.ToArray();
            for (int i = 0; i < n; i++)
                Assert.True(Math.Abs(host[i] - expected[i]) <= 1e-5 * (1 + Math.Abs(expected[i])),
                    $"step {step} element {i}: {host[i]} vs {expected[i]}");
        }
    }

    private static void CheckGraphModeReplay(IEngine engine)
    {
        const int n = 19;
        var rng = new Random(23);
        var running = new Tensor<float>(new[] { n });
        var param = new Tensor<float>(new[] { n });
        var expected = new double[n];
        for (int i = 0; i < n; i++)
        {
            running[i] = (float)rng.NextDouble();
            expected[i] = running[i];
            param[i] = (float)(rng.NextDouble() * 2.0 - 0.5);
        }
        const float keep = 0.9f, scale = 0.1f;

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var scaled = engine.TensorMultiplyScalar(param, 2.0f);
            RunningStatistics.UpdateInPlace(engine, running, scaled, keep, scale);
            engine.ReduceSum(engine.TensorMultiply(scaled, scaled), null);
            plan = scope.CompileTraining(new[] { param });
        }
        using (plan)
        {
            for (int step = 0; step < 3; step++)
            {
                var pSnap = param.ToArray();
                plan.Step();
                for (int i = 0; i < n; i++) expected[i] = keep * expected[i] + scale * (2.0 * pSnap[i]);
                var host = running.ToArray();
                for (int i = 0; i < n; i++)
                    Assert.True(Math.Abs(host[i] - expected[i]) <= 1e-5 * (1 + Math.Abs(expected[i])),
                        $"graph step {step} element {i}: {host[i]} vs {expected[i]}");
            }
        }
    }

    [Fact]
    public void Cpu_update_matches_the_ema()
    {
        var cpu = new CpuEngine();
        Check(cpu);
        CheckGraphModeReplay(cpu);
    }

    [SkippableFact]
    public void Gpu_update_matches_the_ema_from_first_touch_onward()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.IsAvailable, "No CUDA driver.");
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "No DirectGpu backend.");
        Check(engine);
        CheckGraphModeReplay(engine);
    }
}
