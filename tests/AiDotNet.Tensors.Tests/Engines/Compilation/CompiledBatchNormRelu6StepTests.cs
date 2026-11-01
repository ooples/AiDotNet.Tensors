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
/// A training BatchNorm feeding a ReLU6 (every MobileNetV2 expansion and depthwise stage) compiles to one
/// <c>BatchNormReLU6</c> node whose backward takes its mask from the ReLU6 output. It trains like the unfused plan, and
/// the batch statistics it writes for the nodes it absorbed are not taken for caller-fed leaves: they were, and every
/// step's pre-pass downloaded, unbound and re-uploaded them (MobileNetV2 0.17 -> 0.21 s per epoch on OpenCL).
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledBatchNormRelu6StepTests : IDisposable
{
    private const int Batch = 4, Channels = 8, Size = 8, Steps = 6;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledBatchNormRelu6StepTests(ITestOutputHelper output) => _output = output;

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
        public Tensor<float>[] ExternalLeaves = Array.Empty<Tensor<float>>();
        public Tensor<float>[] Inputs = Array.Empty<Tensor<float>>();
    }

    private static Run Train(IEngine engine, bool onDevice)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Batch, Channels, Size, Size], 1, 1f);
        var y = Rand([Batch, Channels], 2, 1f);
        float wScale = 2f / MathF.Sqrt(Channels * 9);
        var w1 = Rand([Channels, Channels, 3, 3], 3, wScale);
        var g1 = Fill([Channels], 2f);
        var b1 = Fill([Channels], 0.5f);
        var w2 = Rand([Channels, Channels, 3, 3], 4, wScale);
        var g2 = Fill([Channels], 3f);
        var b2 = Fill([Channels], 1f);
        var parameters = new[] { w1, g1, b1, w2, g2, b2 };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        Tensor<float>[] statistics;
        var runningMean = new Tensor<float>([Channels]);
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            // Gammas of 2 and 3 with positive betas put a share of the normalized values above 6, so the cap's mask
            // is exercised from both ends.
            var h = engine.FusedConv2D(x, w1, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.TensorReLU6(engine.BatchNorm(h, g1, b1, 1e-5, out var mean1, out var variance1));
            // The running-mean update of AiDotNet's BatchNormalizationLayer, recorded in the graph: the caller's tensor
            // updated in place from the batch mean, which the fused node writes for the statistic node it absorbed.
            engine.TensorMultiplyScalarInPlace(runningMean, 0.9f);
            engine.TensorAddInPlace(runningMean, engine.TensorMultiplyScalar(mean1, 0.1f));
            h = engine.FusedConv2D(h, w2, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.TensorReLU6(engine.BatchNorm(h, g2, b2, 1e-5, out var mean2, out var variance2));
            statistics = new[] { mean1, variance1, mean2, variance2 };
            engine.TensorMSELoss(engine.ReduceMean(h, new[] { 2, 3 }, keepDims: false), y);
            plan = scope.CompileTraining(parameters);
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
            var run = new Run { Losses = new double[Steps], Inputs = new[] { x, y } };
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
                run.ExternalLeaves = (Tensor<float>[]?)typeof(CompiledTrainingPlan<float>)
                    .GetMethod("GraphExternalLeaves", BindingFlags.NonPublic | BindingFlags.Instance)?.Invoke(concrete, null)
                    ?? Array.Empty<Tensor<float>>();
            }
            return run;
        }
    }

    private static Run TrainUnfused()
    {
        var pattern = FusionPatternRegistry.Patterns.First(p => p.Name == "BatchNormReLU6");
        Assert.True(FusionPatternRegistry.Unregister(pattern.Name));
        try { return Train(new CpuEngine(), onDevice: false); }
        finally { FusionPatternRegistry.Register(pattern); }
    }

    private static void AssertTrainsLike(Run reference, Run fused, string label)
    {
        for (int s = 0; s < Steps; s++)
            Assert.True(Math.Abs(reference.Losses[s] - fused.Losses[s]) <= 1e-4 * Math.Max(1, Math.Abs(reference.Losses[s])),
                $"{label} step {s}: loss {fused.Losses[s]:G7} != unfused {reference.Losses[s]:G7}");
        AssertClose(reference.FinalWeights, fused.FinalWeights, $"{label} parameter");
        AssertClose(reference.Statistics, fused.Statistics, $"{label} statistic");
        AssertClose(new[] { reference.RunningMean }, new[] { fused.RunningMean }, $"{label} running mean");
    }

    private static void AssertClose(float[][] expected, float[][] actual, string label)
    {
        for (int p = 0; p < expected.Length; p++)
        {
            double maxErr = 0;
            for (int i = 0; i < expected[p].Length; i++)
                maxErr = Math.Max(maxErr, Math.Abs(expected[p][i] - actual[p][i]) / Math.Max(1.0, Math.Abs(expected[p][i])));
            Assert.True(maxErr <= 1e-4, $"{label} {p}: max rel error {maxErr:E2} after {Steps} steps");
        }
    }

    private void AssertFusedWithOnlyCallerLeaves(Run fused, string label)
    {
        _output.WriteLine($"{label} steps: " + string.Join(", ", fused.StepNames));
        Assert.Equal(2, fused.StepNames.Count(n => n == "BatchNormReLU6"));
        Assert.DoesNotContain("BatchNorm", fused.StepNames);
        Assert.DoesNotContain("ReLU6", fused.StepNames);
        Assert.DoesNotContain("BatchNormBatchMean", fused.StepNames);
        // The running-mean update reads the first BatchNorm's batch mean, which the fused node writes.
        Assert.Equal(fused.Inputs.Length, fused.ExternalLeaves.Length);
        foreach (var leaf in fused.ExternalLeaves)
            Assert.Contains(fused.Inputs, input => ReferenceEquals(input, leaf));
    }

    [Fact]
    public void A_cpu_plan_fuses_batchnorm_relu6_and_trains_like_the_unfused_plan()
    {
        var reference = TrainUnfused();
        Assert.Contains("ReLU6", reference.StepNames);
        Assert.DoesNotContain("BatchNormReLU6", reference.StepNames);
        var fused = Train(new CpuEngine(), onDevice: false);
        AssertFusedWithOnlyCallerLeaves(fused, "cpu");
        AssertTrainsLike(reference, fused, "cpu");
    }

    [SkippableFact]
    public void A_gpu_plan_fuses_batchnorm_relu6_and_trains_like_the_unfused_cpu_plan()
    {
        DirectGpuTensorEngine? gpu = null;
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) gpu = candidate; else candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            var reference = TrainUnfused();
            var fused = Train(gpu, onDevice: true);
            _output.WriteLine("losses unfused: " + string.Join(", ", reference.Losses.Select(l => l.ToString("G7"))));
            _output.WriteLine("losses gpu    : " + string.Join(", ", fused.Losses.Select(l => l.ToString("G7"))));
            AssertFusedWithOnlyCallerLeaves(fused, "gpu");
            AssertTrainsLike(reference, fused, "gpu");
        }
    }
}
