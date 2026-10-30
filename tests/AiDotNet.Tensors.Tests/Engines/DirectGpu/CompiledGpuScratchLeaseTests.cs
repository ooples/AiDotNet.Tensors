using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A compiled GPU plan leases each backward output's scratch for the action that writes it, instead of keeping a buffer
/// per output for the plan's life. A residual network trained for several SGD steps must reach the same parameters as
/// the CPU plan with leases on and off, and the leased plan must hold fewer scratch bytes.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuScratchLeaseTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static (float[] Parameters, long ScratchBytes) Train(IEngine engine, int steps)
    {
        // Identical residual blocks h += ReLU(BatchNorm(Conv(h))): each convolution's input gradient is the second
        // contribution to h's gradient (the add passes it the first), so it is written to scratch and added, never
        // redirected, and every block's is the same size. Then a concatenation, whose backward slices its gradient.
        var x = Filled(new[] { 4, 6, 8, 8 }, 1, 2f);
        var parameters = new System.Collections.Generic.List<Tensor<float>>();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = x;
            for (int b = 0; b < 4; b++)
            {
                var k = Filled(new[] { 6, 6, 3, 3 }, 10 + b, 0.2f);
                var gamma = Filled(new[] { 6 }, 20 + b, 1f);
                var beta = Filled(new[] { 6 }, 30 + b, 1f);
                parameters.Add(k); parameters.Add(gamma); parameters.Add(beta);
                var conv = engine.Conv2D(h, k, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
                h = engine.TensorAdd(h, engine.ReLU(engine.BatchNorm(conv, gamma, beta, 1e-5, out _, out _)));
            }
            var cat = engine.TensorConcatenate(new[] { h, x }, 1);
            engine.ReduceSum(engine.TensorMultiply(cat, cat), null);
            plan = scope.CompileTraining(parameters.ToArray());
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.001f);
            for (int s = 0; s < steps; s++) plan.Step();
            long bytes = plan is CompiledTrainingPlan<float> compiled ? compiled.ScratchBufferBytes : -1;
            var all = new System.Collections.Generic.List<float>();
            foreach (var p in parameters) all.AddRange(p.ToArray());
            return (all.ToArray(), bytes);
        }
    }

    [SkippableFact]
    public void Leased_backward_scratch_trains_like_the_cpu_and_holds_fewer_bytes()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        bool previous = CompiledTrainingPlan<float>.LeaseBackwardScratch;
        using (gpu)
        {
            try
            {
                const int steps = 4;
                var expected = Train(new CpuEngine(), steps).Parameters;
                CompiledTrainingPlan<float>.LeaseBackwardScratch = false;
                var kept = Train(gpu, steps);
                CompiledTrainingPlan<float>.LeaseBackwardScratch = true;
                var leased = Train(gpu, steps);

                foreach (var (actual, name) in new[] { (kept.Parameters, "kept"), (leased.Parameters, "leased") })
                {
                    Assert.Equal(expected.Length, actual.Length);
                    for (int i = 0; i < expected.Length; i++)
                        Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-3f * (1f + Math.Abs(expected[i])),
                            $"{name} [{i}]: cpu {expected[i]}, gpu {actual[i]}");
                }
                Assert.True(leased.ScratchBytes > 0 && leased.ScratchBytes < kept.ScratchBytes,
                    $"leased plan holds {leased.ScratchBytes} scratch bytes, kept plan {kept.ScratchBytes}");
            }
            finally
            {
                CompiledTrainingPlan<float>.LeaseBackwardScratch = previous;
            }
        }
    }
}
