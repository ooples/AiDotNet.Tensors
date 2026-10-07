using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// On CUDA a compiled plan's fp32 Adam/AdamW update is device-stepped: the non-finite-gradient decision, the step
/// counter and the learning rate live in device memory, so a training step reads nothing back to the host (that read
/// used to block the host until the whole step had run, every step). These pin that the device-stepped update keeps
/// the host-stepped contract: a non-finite step is discarded and counted, the counters the host reports are exact
/// when read, and the learning-rate schedule is followed past the end of the device's learning-rate table.
/// </summary>
[Collection("DirectGpuSerial")]
public class DeviceSteppedOptimizerTests : IDisposable
{
    private const int Length = 16;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static bool TryCuda(out DirectGpuTensorEngine? engine)
    {
        engine = null;
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (!candidate.IsGpuAvailable || candidate.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)
            {
                candidate.Dispose();
                return false;
            }
            engine = candidate;
            return true;
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException)
        {
            return false;
        }
    }

    // loss = sum(w * x): the gradient with respect to w is exactly x, so a non-finite x is a non-finite gradient and a
    // constant x is a constant gradient, with no model behaviour in between.
    private static (ICompiledTrainingPlan<float> Plan, Tensor<float> Weights, Tensor<float> Input) Build(DirectGpuTensorEngine gpu)
    {
        var w = new Tensor<float>(new[] { Length });
        var x = new Tensor<float>(new[] { Length });
        for (int i = 0; i < Length; i++) { w[i] = 1f + 0.1f * i; x[i] = 1f; }
        w.Gpu();
        IEngine engine = gpu;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { w }))
        {
            engine.ReduceSum(engine.TensorMultiply(w, x), null);
            plan = scope.CompileTraining(new[] { w });
        }
        return (plan, w, x);
    }

    private static void Feed(Tensor<float> x, float value)
    {
        var span = x.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = value;
    }

    private static bool DeviceSteppingEngaged(ICompiledTrainingPlan<float> plan) =>
        typeof(CompiledTrainingPlan<float>)
            .GetField("_devOptState", BindingFlags.NonPublic | BindingFlags.Instance)!
            .GetValue(plan) is not null;

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void A_non_finite_step_is_discarded_on_the_device_and_counted_exactly(bool captureGraph)
    {
        Skip.IfNot(TryCuda(out var gpu) && gpu is not null, "No CUDA backend.");
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_DEVICE_STEP_OPTIMIZER") == "0",
            "Device-stepped optimizer is disabled for this process.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            var (plan, w, x) = Build(gpu);
            using (plan)
            {
                if (!captureGraph) ((CompiledTrainingPlan<float>)plan).DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);

                const int finiteSteps = 6;   // past the graph warm-up, so the captured case replays
                for (int s = 0; s < finiteSteps; s++) { Feed(x, 1f); plan.Step(); }
                Assert.True(DeviceSteppingEngaged(plan), "the device-stepped update never ran, so nothing here is tested");
                Assert.Equal(finiteSteps, ((CompiledTrainingPlan<float>)plan).OptimizerStep);
                Assert.False(plan.LastStepSkippedNonFiniteGradients);
                Assert.Equal(0, plan.NonFiniteStepsSkipped);

                var before = w.ToArray();
                Feed(x, float.NaN);
                plan.Step();
                Assert.True(plan.LastStepSkippedNonFiniteGradients, "a NaN gradient step was not reported as skipped");
                Assert.Equal(1, plan.NonFiniteStepsSkipped);
                Assert.Equal(finiteSteps, ((CompiledTrainingPlan<float>)plan).OptimizerStep);   // a discarded step does not advance the counter
                var afterBad = w.ToArray();
                for (int i = 0; i < Length; i++)
                    Assert.True(before[i] == afterBad[i], $"w[{i}] moved on a discarded step: {before[i]} -> {afterBad[i]}");

                Feed(x, 1f);
                plan.Step();
                Assert.False(plan.LastStepSkippedNonFiniteGradients);
                Assert.Equal(1, plan.NonFiniteStepsSkipped);
                Assert.Equal(finiteSteps + 1, ((CompiledTrainingPlan<float>)plan).OptimizerStep);
                var afterGood = w.ToArray();
                for (int i = 0; i < Length; i++)
                {
                    Assert.True(!float.IsNaN(afterGood[i]) && !float.IsInfinity(afterGood[i]), $"w[{i}] is not finite after recovering: {afterGood[i]}");
                    Assert.True(afterGood[i] < afterBad[i], $"w[{i}] did not move on the recovery step");
                }
            }
        }
    }

    /// <summary>
    /// The device keeps learning rates for a bounded number of future steps and the host refills the table (one read)
    /// before a step could run past it. A step-decay schedule whose drop lands after the first table makes a missed
    /// or misaligned refill visible: with a constant unit gradient, Adam moves each weight by exactly the step's
    /// learning rate (up to float rounding), so the final weights are the initial ones minus the schedule's sum.
    /// </summary>
    [SkippableFact]
    public void The_learning_rate_schedule_is_followed_past_the_device_table()
    {
        Skip.IfNot(TryCuda(out var gpu) && gpu is not null, "No CUDA backend.");
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_DEVICE_STEP_OPTIMIZER") == "0",
            "Device-stepped optimizer is disabled for this process.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            var (plan, w, x) = Build(gpu);
            using (plan)
            {
                const int dropAt = 4100, steps = 4300;
                var schedule = LrSchedule.Step(1e-3, stepSize: dropAt, gamma: 0.1);
                plan.ConfigureOptimizer(OptimizerType.Adam, schedule);
                var initial = w.ToArray();
                for (int s = 0; s < steps; s++) { Feed(x, 1f); plan.Step(); }
                Assert.True(DeviceSteppingEngaged(plan), "the device-stepped update never ran, so nothing here is tested");
                Assert.Equal(steps, ((CompiledTrainingPlan<float>)plan).OptimizerStep);

                double travelled = 0;
                for (int t = 1; t <= steps; t++) travelled += schedule.GetLr(t);
                var final = w.ToArray();
                for (int i = 0; i < Length; i++)
                {
                    double expected = initial[i] - travelled;
                    Assert.True(Math.Abs(final[i] - expected) <= 2e-3,
                        $"w[{i}] = {final[i]}, expected {expected:G6} (initial {initial[i]} minus the schedule's sum {travelled:G6})");
                }
            }
        }
    }
}
