using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// <see cref="FusedOptimizerExtras.DecoupledWeightDecay"/> for the AMSGrad plan kernel. AiDotNet maps
/// AdamW(UseAMSGrad) onto OptimizerType.AMSGrad, whose weight decay was always Adam's L2 (g += wd·p): the
/// fused AdamW + AMSGrad run was L2-regularized AMSGrad, not AdamW. With the flag the plan must follow PyTorch
/// AdamW(amsgrad=True): p *= 1 - lr·wd, then the AMSGrad step on the undecayed gradient. Without it the old
/// coupled convention must be unchanged.
/// </summary>
[Collection("DirectGpuSerial")]
public class AMSGradDecoupledWeightDecayTests
{
    private const double Lr = 0.05, B1 = 0.9, B2 = 0.999, Eps = 1e-8, Wd = 0.2;
    private const int Steps = 6;

    // f(w) = sum(w²) so the gradient is 2w, evaluated before the step (as the plan's backward is).
    private static double[] Reference(double[] w0, bool decoupled)
    {
        var w = (double[])w0.Clone();
        var m = new double[w.Length];
        var v = new double[w.Length];
        var vMax = new double[w.Length];
        for (int t = 1; t <= Steps; t++)
        {
            double bc1 = 1 - Math.Pow(B1, t), bc2 = 1 - Math.Pow(B2, t);
            for (int i = 0; i < w.Length; i++)
            {
                double g = 2 * w[i];
                if (decoupled) w[i] *= 1 - Lr * Wd;
                else g += Wd * w[i];
                m[i] = B1 * m[i] + (1 - B1) * g;
                v[i] = B2 * v[i] + (1 - B2) * g * g;
                vMax[i] = Math.Max(vMax[i], v[i]);
                w[i] -= Lr * (m[i] / bc1) / (Math.Sqrt(vMax[i] / bc2) + Eps);
            }
        }
        return w;
    }

    private static readonly double[] Init = { 0.9, -0.4, 1.7, -2.3, 0.05, 0.6, -1.1, 0.3, 2.2, -0.7, 0.45 };

    private static double[] RunFloat(bool decoupled)
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new[] { Init.Length });
        for (int i = 0; i < Init.Length; i++) weight[i] = (float)Init[i];
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(weight, weight), null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.AMSGrad, (float)Lr, (float)B1, (float)B2, (float)Eps, (float)Wd,
                new FusedOptimizerExtras { DecoupledWeightDecay = decoupled });
            for (int s = 0; s < Steps; s++) plan.Step();
        }
        var result = new double[Init.Length];
        for (int i = 0; i < Init.Length; i++) result[i] = weight[i];
        return result;
    }

    private static double[] RunDouble(bool decoupled)
    {
        var engine = new CpuEngine();
        var weight = new Tensor<double>(new[] { Init.Length });
        for (int i = 0; i < Init.Length; i++) weight[i] = Init[i];
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(weight, weight), null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.AMSGrad, (float)Lr, (float)B1, (float)B2, (float)Eps, (float)Wd,
                new FusedOptimizerExtras { DecoupledWeightDecay = decoupled });
            for (int s = 0; s < Steps; s++) plan.Step();
        }
        var result = new double[Init.Length];
        for (int i = 0; i < Init.Length; i++) result[i] = weight[i];
        return result;
    }

    private static void AssertClose(double[] expected, double[] actual, double tol, string what)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"{what}: w[{i}] expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Float_plan_matches_the_reference_for_either_decay_convention(bool decoupled)
        => AssertClose(Reference(Init, decoupled), RunFloat(decoupled), 2e-5, decoupled ? "decoupled" : "coupled");

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Double_plan_matches_the_reference_for_either_decay_convention(bool decoupled)
        // The plan carries lr/betas/wd as float, so compare against a reference fed the same float-rounded values.
        => AssertClose(Reference(Init, decoupled), RunDouble(decoupled), 1e-6, decoupled ? "decoupled" : "coupled");

    [Fact]
    public void The_two_conventions_really_differ_so_the_flag_is_observable()
    {
        var a = RunFloat(true);
        var b = RunFloat(false);
        double maxAbs = 0;
        for (int i = 0; i < a.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
        Assert.True(maxAbs > 1e-3, $"decoupled and coupled AMSGrad agreed to {maxAbs} - the flag had no effect");
    }

    /// <summary>
    /// Same contract for LAMB's algorithm-selecting extras: the runtime-state clone the checkpoint is taken from used
    /// to drop them, so a clipped / uncorrected LAMB plan restored as plain LAMB.
    /// </summary>
    [Fact]
    public void Lamb_extras_reach_the_checkpoint()
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new[] { 1f, -2f, 3f }, new[] { 3 });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(weight, weight), null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.LAMB, 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f,
                new FusedOptimizerExtras { LambMaxTrustRatio = 10f, LambDisableBiasCorrection = true });
            plan.Step();
            var checkpoint = Assert.IsType<FusedOptimizerCheckpoint>(
                Assert.IsType<CompiledTrainingPlan<float>>(plan).CaptureFusedOptimizerCheckpoint());
            Assert.Equal(10f, checkpoint.Extras.LambMaxTrustRatio);
            Assert.True(checkpoint.Extras.LambDisableBiasCorrection);
        }
    }

    [Fact]
    public void The_flag_survives_a_checkpoint_round_trip()
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new[] { 1f, -2f, 3f }, new[] { 3 });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMultiply(weight, weight), null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.AMSGrad, 0.01f, 0.9f, 0.999f, 1e-8f, 0.1f,
                new FusedOptimizerExtras { DecoupledWeightDecay = true });
            plan.Step();
            var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            var checkpoint = Assert.IsType<FusedOptimizerCheckpoint>(compiled.CaptureFusedOptimizerCheckpoint());
            Assert.True(checkpoint.Extras.DecoupledWeightDecay);

            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
                FusedOptimizerCheckpointSerializer.Write(writer, checkpoint);
            stream.Position = 0;
            using var reader = new BinaryReader(stream);
            var restored = FusedOptimizerCheckpointSerializer.Read(reader);
            Assert.NotNull(restored);
            Assert.True(restored!.Extras.DecoupledWeightDecay);
        }
    }

    /// <summary>The GPU plan branch (pre-scale on device, then the AMSGrad kernel with no decay).</summary>
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void Gpu_plan_matches_the_reference_for_either_decay_convention(bool decoupled)
    {
        var prior = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception) { }
        Skip.IfNot(gpu is not null && gpu.IsGpuAvailable, "GPU backend did not resolve.");
        try
        {
            AiDotNetEngine.Current = gpu!;
            var weight = new Tensor<float>(new[] { Init.Length });
            for (int i = 0; i < Init.Length; i++) weight[i] = (float)Init[i];
            weight.Gpu();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { weight }))
            {
                gpu!.ReduceSum(gpu.TensorMultiply(weight, weight), null);
                plan = scope.CompileTraining(new[] { weight });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.AMSGrad, (float)Lr, (float)B1, (float)B2, (float)Eps, (float)Wd,
                    new FusedOptimizerExtras { DecoupledWeightDecay = decoupled });
                for (int s = 0; s < Steps; s++) plan.Step();
            }
            var actual = weight.ToArray();
            var result = new double[Init.Length];
            for (int i = 0; i < Init.Length; i++) result[i] = actual[i];
            AssertClose(Reference(Init, decoupled), result, 5e-5, (decoupled ? "decoupled" : "coupled") + " (GPU)");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }
}
