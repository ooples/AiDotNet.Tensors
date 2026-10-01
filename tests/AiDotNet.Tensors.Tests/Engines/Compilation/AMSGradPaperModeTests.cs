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
/// <see cref="FusedOptimizerExtras.AmsgradDisableBiasCorrection"/>: AMSGrad as published (Reddi, Kale &amp; Kumar,
/// "On the Convergence of Adam and Beyond", ICLR 2018, Algorithm 2), which has no bias correction:
/// vMax = max(vMax, v), w -= lr·m / (sqrt(vMax) + eps). The default stays PyTorch's amsgrad=True
/// (m/bc1 over sqrt(vMax/bc2) + eps). Both conventions must hold on every plan path, with either weight-decay
/// convention, and the flag must survive a checkpoint.
/// </summary>
[Collection("DirectGpuSerial")]
public class AMSGradPaperModeTests
{
    private const double Lr = 0.05, B1 = 0.9, B2 = 0.999, Eps = 1e-8, Wd = 0.2;
    private const int Steps = 6;

    private static readonly double[] Init = { 0.9, -0.4, 1.7, -2.3, 0.05, 0.6, -1.1, 0.3, 2.2, -0.7, 0.45 };

    // f(w) = sum(w²) so the gradient is 2w, evaluated before the step (as the plan's backward is).
    // ConfigureOptimizer takes lr/betas/eps/wd as float, so the reference runs on those float-rounded values. Without
    // bias correction nothing cancels the rounding: the first step is lr·(1-b1)/sqrt(1-b2), and 1-(float)0.999 is off
    // by 1.3e-5 relative.
    private static double[] Reference(bool paper, bool decoupled)
    {
        double Lr = (float)AMSGradPaperModeTests.Lr, B1 = (float)AMSGradPaperModeTests.B1;
        double B2 = (float)AMSGradPaperModeTests.B2, Eps = (float)AMSGradPaperModeTests.Eps, Wd = (float)AMSGradPaperModeTests.Wd;
        var w = (double[])Init.Clone();
        var m = new double[w.Length];
        var v = new double[w.Length];
        var vMax = new double[w.Length];
        for (int t = 1; t <= Steps; t++)
        {
            double bc1 = paper ? 1 : 1 - Math.Pow(B1, t);
            double bc2 = paper ? 1 : 1 - Math.Pow(B2, t);
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

    private static FusedOptimizerExtras Extras(bool paper, bool decoupled)
        => new FusedOptimizerExtras { AmsgradDisableBiasCorrection = paper, DecoupledWeightDecay = decoupled };

    private static double[] RunFloat(bool paper, bool decoupled)
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
                Extras(paper, decoupled));
            for (int s = 0; s < Steps; s++) plan.Step();
        }
        var result = new double[Init.Length];
        for (int i = 0; i < Init.Length; i++) result[i] = weight[i];
        return result;
    }

    private static double[] RunDouble(bool paper, bool decoupled)
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
                Extras(paper, decoupled));
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

    private static string Label(bool paper, bool decoupled)
        => (paper ? "paper" : "pytorch") + "/" + (decoupled ? "decoupled" : "coupled");

    [Theory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(false, false)]
    public void Float_plan_matches_the_reference(bool paper, bool decoupled)
        => AssertClose(Reference(paper, decoupled), RunFloat(paper, decoupled), 2e-5, Label(paper, decoupled));

    [Theory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(false, false)]
    public void Double_plan_matches_the_reference(bool paper, bool decoupled)
        => AssertClose(Reference(paper, decoupled), RunDouble(paper, decoupled), 1e-6, Label(paper, decoupled));

    [Fact]
    public void The_two_conventions_really_differ_so_the_flag_is_observable()
    {
        var paper = RunFloat(true, false);
        var pytorch = RunFloat(false, false);
        double maxAbs = 0;
        for (int i = 0; i < paper.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(paper[i] - pytorch[i]));
        Assert.True(maxAbs > 1e-3, $"paper and PyTorch AMSGrad agreed to {maxAbs} - the flag had no effect");
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
            plan.ConfigureOptimizer(OptimizerType.AMSGrad, 0.01f, 0.9f, 0.999f, 1e-8f, 0f, Extras(true, false));
            plan.Step();
            var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            var checkpoint = Assert.IsType<FusedOptimizerCheckpoint>(compiled.CaptureFusedOptimizerCheckpoint());
            Assert.True(checkpoint.Extras.AmsgradDisableBiasCorrection);

            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
                FusedOptimizerCheckpointSerializer.Write(writer, checkpoint);
            stream.Position = 0;
            using var reader = new BinaryReader(stream);
            var restored = FusedOptimizerCheckpointSerializer.Read(reader);
            Assert.NotNull(restored);
            Assert.True(restored is not null && restored.Extras.AmsgradDisableBiasCorrection);
        }
    }

    /// <summary>The GPU plan branch: the same AMSGrad kernel, fed the rescaled lr and eps.</summary>
    [SkippableTheory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, false)]
    public void Gpu_plan_matches_the_reference(bool paper, bool decoupled)
    {
        var prior = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        bool resolved = gpu is not null && gpu.IsGpuAvailable;
        // The POCL lane sets this so the AMSGrad device path cannot pass CI by skipping.
        if (!resolved && string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no GPU backend resolved.");
        Skip.IfNot(resolved, "GPU backend did not resolve.");
        if (gpu is null) return;
        try
        {
            AiDotNetEngine.Current = gpu;
            var weight = new Tensor<float>(new[] { Init.Length });
            for (int i = 0; i < Init.Length; i++) weight[i] = (float)Init[i];
            weight.Gpu();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { weight }))
            {
                gpu.ReduceSum(gpu.TensorMultiply(weight, weight), null);
                plan = scope.CompileTraining(new[] { weight });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.AMSGrad, (float)Lr, (float)B1, (float)B2, (float)Eps, (float)Wd,
                    Extras(paper, decoupled));
                for (int s = 0; s < Steps; s++) plan.Step();
            }
            var actual = weight.ToArray();
            var result = new double[Init.Length];
            for (int i = 0; i < Init.Length; i++) result[i] = actual[i];
            AssertClose(Reference(paper, decoupled), result, 5e-5, Label(paper, decoupled) + " (GPU)");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }
}
