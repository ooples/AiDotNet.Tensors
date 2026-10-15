using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled training step with a training-mode BatchNorm must capture into a CUDA graph like any other. The plan
/// records each batch statistic as a node that copies it into its output; that copy went through host spans, a
/// synchronous download that aborted every capture (CUDA 900), so every network with a training-mode BatchNorm
/// (ResNet-18: 18.2 ms a step) silently fell back to the uncaptured step (13.6 ms once it captures).
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledTrainingBatchNormCaptureTests : IDisposable
{
    private const int Batch = 8, Channels = 3, Filters = 4, Size = 8, Outputs = 2, Steps = 10;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledTrainingBatchNormCaptureTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static (double[] Losses, float[][] Weights, bool Replayed, string? Failure) Train(DirectGpuTensorEngine gpu, bool capture)
    {
        var x = Rand([Batch, Channels, Size, Size], 1, 1f);
        var y = Rand([Batch, Outputs], 2, 1f);
        var conv = Rand([Filters, Channels, 3, 3], 3, 0.3f);
        var gamma = new Tensor<float>(Enumerable.Repeat(1f, Filters).ToArray(), [Filters]);
        var beta = new Tensor<float>([Filters]);
        var w = Rand([Filters, Outputs], 4, 0.3f);
        var b = new Tensor<float>([Outputs]);
        var parameters = new[] { conv, gamma, beta, w, b };
        foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = gpu.FusedConv2D(x, conv, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = gpu.ReLU(gpu.BatchNorm(h, gamma, beta, 1e-5, out _, out _));
            var pooled = gpu.Reshape(gpu.GlobalAvgPool2D(h), new[] { Batch, Filters });
            var pred = gpu.FusedLinear(pooled, w, b, FusedActivationType.None);
            var diff = gpu.TensorSubtract(pred, y);
            gpu.ReduceMean(gpu.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
            plan = scope.CompileTraining(parameters);
        }

        using (plan)
        {
            var concrete = (CompiledTrainingPlan<float>)plan;
            if (!capture) concrete.DisableGraphStep();
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05);
            var losses = new double[Steps];
            for (int s = 0; s < Steps; s++) losses[s] = plan.Step().ToArray()[0];
            var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
            return (losses, parameters.Select(p => p.ToArray()).ToArray(), exec != IntPtr.Zero, concrete.GraphCaptureFailureReason);
        }
    }

    [SkippableFact]
    public void A_step_with_a_training_mode_batch_norm_captures_and_trains_like_the_eager_step()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu!;
            Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0", "Graph capture is disabled for this process.");

            var eager = Train(gpu, capture: false);
            var captured = Train(gpu, capture: true);
            _output.WriteLine("eager    " + string.Join(" ", eager.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine("captured " + string.Join(" ", captured.Losses.Select(l => l.ToString("G6"))));

            Assert.True(eager.Losses.Skip(Steps / 2).Average() < eager.Losses.Take(Steps / 2).Average(), "the eager reference did not learn");
            Assert.False(eager.Replayed, "the eager reference captured a graph");
            Assert.True(captured.Replayed, $"the BatchNorm step did not capture: {captured.Failure ?? "no failure recorded"}");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(eager.Losses[s] - captured.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6}");
            for (int p = 0; p < eager.Weights.Length; p++)
                for (int i = 0; i < eager.Weights[p].Length; i++)
                    Assert.True(Math.Abs(eager.Weights[p][i] - captured.Weights[p][i]) <= 1e-5f,
                        $"param {p}[{i}]: captured {captured.Weights[p][i]} != eager {eager.Weights[p][i]}");
        }
    }
}
