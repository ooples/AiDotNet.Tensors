using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="CompiledTrainingPlan{T}.TryApplyGradients"/> runs only the plan's optimizer on gradients computed outside
/// it, for a step the plan cannot replay. A plan that takes one step's gradients from the eager tape must end where a
/// plan that ran every step itself ends: its Adam moments, step count and clipping carried the outside step. With the
/// gradients handed over as zeros instead, the plans must differ, so the comparison sees the handed-over gradients.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledPlanApplyGradientsTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static Tensor<float> Loss(IEngine engine, Tensor<float> x, Tensor<float> w1, Tensor<float> w2)
    {
        var hidden = engine.ReLU(engine.TensorMatMul(x, w1));
        var output = engine.TensorMatMul(hidden, w2);
        return engine.ReduceSum(engine.TensorMultiply(output, output), null);
    }

    // Three Adam steps with clipping. Step outsideAt's gradients come from the eager tape (zeroed when zeroOutside),
    // applied through TryApplyGradients; the plan runs every other step (all of them when outsideAt is -1).
    private static float[] Train(IEngine engine, int outsideAt, bool zeroOutside)
    {
        var x = Filled(new[] { 8, 16 }, 1, 2f);
        var w1 = Filled(new[] { 16, 32 }, 2, 0.5f);
        var w2 = Filled(new[] { 32, 4 }, 3, 0.5f);
        var parameters = new[] { w1, w2 };
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Loss(engine, x, w1, w2);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
            plan.SetMaxGradNorm(5.0);
            for (int s = 0; s < 3; s++)
            {
                if (s == outsideAt)
                {
                    Dictionary<Tensor<float>, Tensor<float>> gradients;
                    using (var tape = new GradientTape<float>())
                        gradients = tape.ComputeGradients(Loss(engine, x, w1, w2), parameters);
                    if (zeroOutside) gradients = new Dictionary<Tensor<float>, Tensor<float>>();
                    Assert.True(((CompiledTrainingPlan<float>)plan).TryApplyGradients(gradients));
                }
                else
                {
                    plan.Step();
                }
            }
            var all = new List<float>();
            foreach (var p in parameters) all.AddRange(p.ToArray());
            return all.ToArray();
        }
    }

    private static void AssertSameRun(IEngine engine, int outsideAt)
    {
        var fused = Train(engine, outsideAt: -1, zeroOutside: false);
        var outside = Train(engine, outsideAt, zeroOutside: false);
        var zeroed = Train(engine, outsideAt, zeroOutside: true);
        double worst = 0, zeroedWorst = 0;
        for (int i = 0; i < fused.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(fused[i] - outside[i]) / (1 + Math.Abs(fused[i])));
            zeroedWorst = Math.Max(zeroedWorst, Math.Abs(fused[i] - zeroed[i]) / (1 + Math.Abs(fused[i])));
        }
        Assert.True(worst <= 1e-4, $"outside step ended {worst:G3} from the all-fused run");
        Assert.True(zeroedWorst > 1e-3, $"zero gradients handed over ended only {zeroedWorst:G3} from the all-fused run");
    }

    // The first step, and a step after the plan has run on the device (its gradients then live in device buffers).
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void An_outside_step_on_the_cpu_continues_the_plans_optimizer(int outsideAt)
        => AssertSameRun(new CpuEngine(), outsideAt);

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    public void An_outside_step_on_the_gpu_continues_the_plans_optimizer(int outsideAt)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        using (gpu) AssertSameRun(gpu, outsideAt);
    }
}
