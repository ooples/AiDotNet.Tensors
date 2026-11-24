using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A training BatchNorm feeding a LeakyReLU of positive slope (a GAN discriminator's conv -&gt; BN -&gt; LeakyReLU) compiles
/// on the GPU to one <c>BatchNormLeakyReLU</c> node: one kernel writes LeakyReLU(BN(x)) and one differentiates both. It
/// must train like the unfused plan, over channel slices the register kernel takes and ones the split kernels take, and
/// write the batch statistics the absorbed nodes did. A slope of zero or below stays unfused: the fused backward's mask
/// is the output's sign, which is its input's only for a positive slope.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledBatchNormLeakyReluStepTests : IDisposable
{
    private const int Channels = 8, Steps = 6;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledBatchNormLeakyReluStepTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static Tensor<float> Fill(int[] shape, float value)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = value;
        return t;
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public float[][] Statistics = Array.Empty<float[]>();
        public float[] RunningMean = Array.Empty<float>();
        public string[] StepNames = Array.Empty<string>();
    }

    private static Run Train(IEngine engine, bool onDevice, int batch, int size, float slope)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([batch, Channels, size, size], 1, 1f);
        var y = Rand([batch, Channels], 2, 1f);
        float wScale = 2f / MathF.Sqrt(Channels * 9);
        var w1 = Rand([Channels, Channels, 3, 3], 3, wScale);
        var g1 = Fill([Channels], 1.5f);
        var b1 = Fill([Channels], 0.1f);
        var w2 = Rand([Channels, Channels, 3, 3], 4, wScale);
        var g2 = Fill([Channels], 0.8f);
        var b2 = Fill([Channels], -0.2f);
        var parameters = new[] { w1, g1, b1, w2, g2, b2 };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        Tensor<float>[] statistics;
        var runningMean = new Tensor<float>([Channels]);
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.FusedConv2D(x, w1, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.LeakyReLU(engine.BatchNorm(h, g1, b1, 1e-5, out var mean1, out var variance1), slope);
            // AiDotNet's BatchNormalizationLayer running-mean update, reading the batch mean the fused node writes.
            engine.TensorMultiplyScalarInPlace(runningMean, 0.9f);
            engine.TensorAddInPlace(runningMean, engine.TensorMultiplyScalar(mean1, 0.1f));
            h = engine.FusedConv2D(h, w2, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.LeakyReLU(engine.BatchNorm(h, g2, b2, 1e-5, out var mean2, out var variance2), slope);
            statistics = new[] { mean1, variance1, mean2, variance2 };
            engine.TensorMSELoss(engine.ReduceMean(h, new[] { 2, 3 }, keepDims: false), y);
            plan = scope.CompileTraining(parameters);
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
            var run = new Run { Losses = new double[Steps] };
            for (int s = 0; s < Steps; s++)
            {
                Rand(x.Shape.ToArray(), 100 + s, 1f).AsSpan().CopyTo(x.AsWritableSpan());
                Rand(y.Shape.ToArray(), 200 + s, 1f).AsSpan().CopyTo(y.AsWritableSpan());
                run.Losses[s] = plan.Step().ToArray()[0];
            }
            run.FinalWeights = parameters.Select(p => p.ToArray()).ToArray();
            run.Statistics = statistics.Select(s => s.ToArray()).ToArray();
            run.RunningMean = runningMean.ToArray();
            if (plan is CompiledTrainingPlan<float> concrete)
            {
                var steps = (CompiledStep<float>[]?)typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(concrete);
                run.StepNames = steps?.Select(s => s.OpName).ToArray() ?? Array.Empty<string>();
            }
            return run;
        }
    }

    private static Run TrainUnfused(IEngine engine, bool onDevice, int batch, int size, float slope)
    {
        var pattern = FusionPatternRegistry.Patterns.First(p => p.Name == "BatchNormLeakyReLU");
        Assert.True(FusionPatternRegistry.Unregister(pattern.Name));
        try { return Train(engine, onDevice, batch, size, slope); }
        finally { FusionPatternRegistry.Register(pattern); }
    }

    private static void AssertTrainsLike(Run reference, Run fused, string label, double tolerance)
    {
        for (int s = 0; s < Steps; s++)
            Assert.True(Math.Abs(reference.Losses[s] - fused.Losses[s]) <= tolerance * Math.Max(1, Math.Abs(reference.Losses[s])),
                $"{label} step {s}: loss {fused.Losses[s]:G7} != unfused {reference.Losses[s]:G7}");
        AssertClose(reference.FinalWeights, fused.FinalWeights, $"{label} parameter", tolerance);
        AssertClose(reference.Statistics, fused.Statistics, $"{label} statistic", tolerance);
        AssertClose(new[] { reference.RunningMean }, new[] { fused.RunningMean }, $"{label} running mean", tolerance);
    }

    private static void AssertClose(float[][] expected, float[][] actual, string label, double tolerance)
    {
        for (int p = 0; p < expected.Length; p++)
        {
            double maxErr = 0;
            for (int i = 0; i < expected[p].Length; i++)
                maxErr = Math.Max(maxErr, Math.Abs(expected[p][i] - actual[p][i]) / Math.Max(1.0, Math.Abs(expected[p][i])));
            Assert.True(maxErr <= tolerance, $"{label} {p}: max rel error {maxErr:E2} after {Steps} steps");
        }
    }

    private static DirectGpuTensorEngine? TryGpu()
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) return candidate;
            candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        return null;
    }

    // batch, size: a channel slice of 4 x 64 = 256 (the register kernel) and 32 x 256 = 8192 + 64 x 256 (the CUDA split).
    [SkippableTheory]
    [InlineData(4, 8)]
    [InlineData(64, 16)]
    public void A_gpu_plan_fuses_batchnorm_leakyrelu_and_trains_like_the_unfused_plans(int batch, int size)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            var unfusedGpu = TrainUnfused(gpu, onDevice: true, batch, size, 0.2f);
            Assert.Contains("LeakyReLU", unfusedGpu.StepNames);
            Assert.DoesNotContain("BatchNormLeakyReLU", unfusedGpu.StepNames);
            var fused = Train(gpu, onDevice: true, batch, size, 0.2f);
            _output.WriteLine("steps: " + string.Join(", ", fused.StepNames));
            _output.WriteLine("losses unfused: " + string.Join(", ", unfusedGpu.Losses.Select(l => l.ToString("G7"))));
            _output.WriteLine("losses fused  : " + string.Join(", ", fused.Losses.Select(l => l.ToString("G7"))));
            Assert.Equal(2, fused.StepNames.Count(n => n == "BatchNormLeakyReLU"));
            Assert.DoesNotContain("BatchNorm", fused.StepNames);
            Assert.DoesNotContain("LeakyReLU", fused.StepNames);
            Assert.DoesNotContain("BatchNormBatchMean", fused.StepNames);
            AssertTrainsLike(unfusedGpu, fused, "gpu fused vs gpu unfused", 1e-4);
            var cpu = TrainUnfused(new CpuEngine(), onDevice: false, batch, size, 0.2f);
            AssertTrainsLike(cpu, fused, "gpu fused vs cpu unfused", 1e-3);
        }
    }

    [SkippableFact]
    public void A_slope_of_zero_or_below_stays_unfused_and_still_trains_like_the_cpu()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            var plan = Train(gpu, onDevice: true, 4, 8, -0.1f);
            Assert.DoesNotContain("BatchNormLeakyReLU", plan.StepNames);
            Assert.Contains("LeakyReLU", plan.StepNames);
            var cpu = Train(new CpuEngine(), onDevice: false, 4, 8, -0.1f);
            AssertTrainsLike(cpu, plan, "negative slope", 1e-3);
        }
    }
}
