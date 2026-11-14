using System;
using AiDotNet.Tensors.Engines;
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

    [Fact]
    public void Cpu_update_matches_the_ema() => Check(new CpuEngine());

    [SkippableFact]
    public void Gpu_update_matches_the_ema_from_first_touch_onward()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.IsAvailable, "No CUDA driver.");
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "No DirectGpu backend.");
        Check(engine);
    }
}
