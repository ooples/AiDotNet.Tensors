using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// <see cref="FusedOptimizerExtras.VelocityIncludesLearningRate"/>: SGD momentum whose velocity carries the learning
/// rate (v = mu*v + lr_t*g; p -= v) instead of the classical v = mu*v + g; p -= lr_t*v. The two agree at a constant rate
/// and differ under a schedule. The flag must reach the step (the plan keeps a hand-written clone of the extras, which
/// once dropped it, so the plan quietly ran the classical rule) and must survive a checkpoint.
/// </summary>
[Collection("DirectGpuSerial")]
public class MomentumLearningRateVelocityTests
{
    private const float Mu = 0.9f;
    private const int Steps = 8;
    private static readonly float[] Init = { 0.9f, -0.4f, 1.7f, -2.3f, 0.05f, 0.6f, -1.1f, 0.3f, 2.2f, -0.7f, 0.45f };

    private static LrSchedule Schedule() => LrSchedule.Exponential(0.1, 0.7);

    // f(w) = sum(w^2): the gradient is 2w at the parameters before the step. Step k (1-based) runs at GetLr(k).
    private static double[] Reference(bool lrInVelocity, float weightDecay = 0f)
    {
        var schedule = Schedule();
        var w = Array.ConvertAll(Init, x => (double)x);
        var v = new double[w.Length];
        for (int k = 1; k <= Steps; k++)
        {
            double lr = (float)schedule.GetLr(k);
            for (int i = 0; i < w.Length; i++)
            {
                double g = 2 * w[i] + weightDecay * w[i];
                if (lrInVelocity) { v[i] = Mu * v[i] + lr * g; w[i] -= v[i]; }
                else { v[i] = Mu * v[i] + g; w[i] -= lr * v[i]; }
            }
        }
        return w;
    }

    private static float[] Run(bool lrInVelocity, Func<ICompiledTrainingPlan<float>, FusedOptimizerCheckpoint?>? capture = null)
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>((float[])Init.Clone(), new[] { Init.Length });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(weight, weight), null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGDMomentum, Schedule(), Mu, 0f, 0f, 0f,
                new FusedOptimizerExtras { VelocityIncludesLearningRate = lrInVelocity });
            for (int k = 0; k < Steps; k++) plan.Step();
            capture?.Invoke(plan);
        }
        return weight.ToArray();
    }

    private static double MaxAbs(float[] actual, double[] expected)
    {
        double max = 0;
        for (int i = 0; i < actual.Length; i++) max = Math.Max(max, Math.Abs(actual[i] - expected[i]));
        return max;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Cpu_plan_runs_the_requested_rule_under_a_schedule(bool lrInVelocity)
    {
        double err = MaxAbs(Run(lrInVelocity), Reference(lrInVelocity));
        Assert.True(err < 1e-5, $"lrInVelocity={lrInVelocity}: plan vs reference max |diff| {err:E3}");
    }

    [Fact]
    public void The_two_rules_really_differ_under_a_schedule()
    {
        double diff = 0;
        var a = Reference(true);
        var b = Reference(false);
        for (int i = 0; i < a.Length; i++) diff = Math.Max(diff, Math.Abs(a[i] - b[i]));
        Assert.True(diff > 1e-2, $"the rules agreed to {diff:E3}: the test could not see the flag");
    }

    [Fact]
    public void The_flag_survives_a_checkpoint_round_trip()
    {
        FusedOptimizerCheckpoint? checkpoint = null;
        Run(true, plan => checkpoint = Assert.IsType<CompiledTrainingPlan<float>>(plan).CaptureFusedOptimizerCheckpoint());
        if (checkpoint is null)
            throw new InvalidOperationException("Expected compiled plan to capture a fused optimizer checkpoint.");
        Assert.True(checkpoint.Extras.VelocityIncludesLearningRate);

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            FusedOptimizerCheckpointSerializer.Write(writer, checkpoint);
        stream.Position = 0;
        using var reader = new BinaryReader(stream);
        var restored = FusedOptimizerCheckpointSerializer.Read(reader);
        Assert.True(restored is not null && restored.Extras.VelocityIncludesLearningRate);
    }

    [Fact]
    public void Nesterov_is_rejected()
    {
        Assert.Throws<ArgumentException>(() =>
            new FusedOptimizerExtras { VelocityIncludesLearningRate = true, Nesterov = true }.Validate());
    }

    [SkippableTheory]
    [InlineData(true, 0f)]
    [InlineData(false, 0f)]
    [InlineData(true, 0.05f)]
    [InlineData(false, 0.05f)]
    public void Gpu_plan_runs_the_requested_rule_under_a_schedule(bool lrInVelocity, float weightDecay)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend.");
        var prevEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var hostWeight = new Tensor<float>((float[])Init.Clone(), new[] { Init.Length });
            var weight = gpu.UploadToGpu(hostWeight, AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.Weight);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                gpu.ReduceSum(gpu.TensorMultiply(weight, weight), null);
                plan = scope.CompileTraining(new[] { weight });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGDMomentum, Schedule(), Mu, 0f, 0f, weightDecay,
                    new FusedOptimizerExtras { VelocityIncludesLearningRate = lrInVelocity });
                for (int k = 0; k < Steps; k++) plan.Step();
            }
            double err = MaxAbs(weight.ToArray(), Reference(lrInVelocity, weightDecay));
            Assert.True(err < 1e-4, $"GPU lrInVelocity={lrInVelocity}, wd={weightDecay}: plan vs reference max |diff| {err:E3}");
        }
        finally
        {
            AiDotNetEngine.Current = prevEngine;
        }
    }
}
