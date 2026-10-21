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
/// A captured (whole-step CUDA graph) training step of a residual conv + BatchNorm block: it trains like the same plan on
/// the CPU engine, and its graph carries no dead zero-fills or first-write gradient copies. Before, every BatchNorm and
/// convolution zero-filled its pooled outputs, the gradients of non-differentiable leaves were re-zeroed every step, and
/// every gradient's first contribution was copied into its accumulator: a ResNet-18 step replayed 227 memset and 106
/// memcpy nodes (~1 ms of a 5.5 ms step).
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledConvBnGraphStepTests : IDisposable
{
    private const int Batch = 4, Channels = 8, Size = 8, Steps = 8;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledConvBnGraphStepTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
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
        public bool ReplayedAGraph;
        public string[] StepNames = Array.Empty<string>();
        public float[][] Statistics = Array.Empty<float[]>();
    }

    private static Run Train(IEngine engine, bool onDevice)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Batch, Channels, Size, Size], 1, 1f);
        var y = Rand([Batch, Channels], 2, 1f);
        float wScale = 1f / MathF.Sqrt(Channels * 9);
        var w1 = Rand([Channels, Channels, 3, 3], 3, wScale);
        var g1 = Fill([Channels], 1f);
        var b1 = new Tensor<float>([Channels]);
        var w2 = Rand([Channels, Channels, 3, 3], 4, wScale);
        var g2 = Fill([Channels], 1f);
        var b2 = new Tensor<float>([Channels]);
        var parameters = new[] { w1, g1, b1, w2, g2, b2 };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        Tensor<float>[] statistics;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.FusedConv2D(x, w1, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.ReLU(engine.BatchNorm(h, g1, b1, 1e-5, out var mean1, out var variance1));
            h = engine.FusedConv2D(h, w2, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.ReLU(engine.TensorAdd(engine.BatchNorm(h, g2, b2, 1e-5, out var mean2, out var variance2), x));
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
            // The batch mean and variance the BatchNorms hand out (on the GPU plan, written by the fused nodes in place
            // of the statistic nodes they absorbed), after the last step: what a running-statistics update outside the
            // graph would read.
            run.Statistics = statistics.Select(s => s.ToArray()).ToArray();
            if (plan is CompiledTrainingPlan<float> concrete)
            {
                run.ReplayedAGraph = (IntPtr)typeof(CompiledTrainingPlan<float>)
                    .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)! != IntPtr.Zero;
                var steps = (CompiledStep<float>[]?)typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete);
                run.StepNames = steps?.Select(s => s.OpName).ToArray() ?? Array.Empty<string>();
            }
            return run;
        }
    }

    // Two residual blocks: the first block's fused BatchNorm+add+ReLU output is read by the second block's convolution
    // and its residual add, so its gradient has two contributions.
    private static (double[] Losses, float[][] Weights) TrainTwoBlocks(IEngine engine)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Batch, Channels, Size, Size], 1, 1f);
        var y = Rand([Batch, Channels], 2, 1f);
        float wScale = 1f / MathF.Sqrt(Channels * 9);
        var ps = new System.Collections.Generic.List<Tensor<float>>();
        for (int i = 0; i < 4; i++) { ps.Add(Rand([Channels, Channels, 3, 3], 3 + i, wScale)); ps.Add(Fill([Channels], 1f)); ps.Add(new Tensor<float>([Channels])); }
        var parameters = ps.ToArray();
        foreach (var p in parameters) p.Gpu();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = x;
            for (int blk = 0; blk < 2; blk++)
            {
                var w1 = parameters[6 * blk]; var g1 = parameters[6 * blk + 1]; var b1 = parameters[6 * blk + 2];
                var w2 = parameters[6 * blk + 3]; var g2 = parameters[6 * blk + 4]; var b2 = parameters[6 * blk + 5];
                var t = engine.FusedConv2D(h, w1, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
                t = engine.ReLU(engine.BatchNorm(t, g1, b1, 1e-5, out _, out _));
                t = engine.FusedConv2D(t, w2, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
                h = engine.ReLU(engine.TensorAdd(engine.BatchNorm(t, g2, b2, 1e-5, out _, out _), h));
            }
            engine.TensorMSELoss(engine.ReduceMean(h, new[] { 2, 3 }, keepDims: false), y);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
            var losses = new double[Steps];
            for (int s = 0; s < Steps; s++)
            {
                Rand(x.Shape.ToArray(), 100 + s, 1f).AsSpan().CopyTo(x.AsWritableSpan());
                Rand(y.Shape.ToArray(), 200 + s, 1f).AsSpan().CopyTo(y.AsWritableSpan());
                losses[s] = plan.Step().ToArray()[0];
            }
            return (losses, parameters.Select(p => p.ToArray()).ToArray());
        }
    }

    /// <summary>
    /// A residual block's input gradient has two contributions; the plan parks the second (GradientDeferral) and the
    /// fused BatchNorm+ReLU backward sums it as it loads, instead of an add pass: training is bit for bit the same.
    /// </summary>
    [SkippableFact]
    public void Deferring_the_second_gradient_contribution_trains_bit_for_bit_like_adding_it()
    {
        DirectGpuTensorEngine? gpu = null;
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) gpu = candidate; else candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu)
        {
            Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "The dual-gradient kernel is CUDA-only.");
            bool prior = AiDotNet.Tensors.Engines.Autodiff.GradientDeferral.Enabled;
            try
            {
                AiDotNet.Tensors.Engines.Autodiff.GradientDeferral.Enabled = false;
                var added = TrainTwoBlocks(gpu);
                AiDotNet.Tensors.Engines.Autodiff.GradientDeferral.Enabled = true;
                long before = System.Threading.Interlocked.Read(ref AiDotNet.Tensors.Engines.Autodiff.GradientDeferral.ParkedCount);
                var deferred = TrainTwoBlocks(gpu);
                long parked = System.Threading.Interlocked.Read(ref AiDotNet.Tensors.Engines.Autodiff.GradientDeferral.ParkedCount) - before;
                Assert.True(parked > 0, "no gradient contribution was deferred");
                Assert.Equal(added.Losses, deferred.Losses);
                for (int p = 0; p < added.Weights.Length; p++) Assert.Equal(added.Weights[p], deferred.Weights[p]);
            }
            finally
            {
                AiDotNet.Tensors.Engines.Autodiff.GradientDeferral.Enabled = prior;
            }
        }
    }

    /// <summary>
    /// The fused BatchNorm+ReLU backward rebuilds the ReLU mask from the input (batchnorm_bwd_relumask) instead of reading
    /// the ReLU output: training must be bit for bit the same as reading it, with the rebuilt mask actually used.
    /// </summary>
    [SkippableFact]
    public void Rebuilding_the_relu_mask_from_the_input_trains_bit_for_bit_like_reading_it()
    {
        DirectGpuTensorEngine? gpu = null;
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) gpu = candidate; else candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu)
        {
            Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "The mask kernel is CUDA-only.");
            bool prior = AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_batchNormMaskFromInput;
            try
            {
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_batchNormMaskFromInput = false;
                var read = Train(gpu, onDevice: true);
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_batchNormMaskFromInput = true;
                long before = System.Threading.Interlocked.Read(
                    ref AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.BatchNormMaskFromInputLaunches);
                var rebuilt = Train(gpu, onDevice: true);
                long launches = System.Threading.Interlocked.Read(
                    ref AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.BatchNormMaskFromInputLaunches) - before;
                Assert.True(launches > 0, "the fused BatchNorm+ReLU backward never rebuilt the mask from the input");
                Assert.Equal(read.Losses, rebuilt.Losses);
                for (int p = 0; p < read.FinalWeights.Length; p++) Assert.Equal(read.FinalWeights[p], rebuilt.FinalWeights[p]);
            }
            finally
            {
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_batchNormMaskFromInput = prior;
            }
        }
    }

    [SkippableFact]
    public void A_captured_conv_batchnorm_step_trains_like_the_cpu_plan_without_dead_fills_or_copies()
    {
        DirectGpuTensorEngine? gpu = null;
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) gpu = candidate; else candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu)
        {
            Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");
            var cuda = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)gpu.GetBackend()!;

            var reference = Train(new CpuEngine(), onDevice: false);
            var captured = Train(gpu, onDevice: true);
            _output.WriteLine($"captured graph: {cuda.LastCaptureKernelNodeCount} kernel, {cuda.LastCaptureMemsetNodeCount} memset, " +
                              $"{cuda.LastCaptureMemcpyNodeCount} memcpy nodes");
            _output.WriteLine("losses cpu     : " + string.Join(", ", reference.Losses.Select(l => l.ToString("G7"))));
            _output.WriteLine("losses captured: " + string.Join(", ", captured.Losses.Select(l => l.ToString("G7"))));

            Assert.True(captured.ReplayedAGraph, "the plan is not replaying a graph after warm-up");
            // The GPU plan fuses the first BatchNorm with its ReLU (BatchNormReluFusionPattern) and the second with the
            // residual add and the ReLU after it (BatchNormAddReluFusionPattern); the CPU plan fuses neither.
            _output.WriteLine("captured steps: " + string.Join(", ", captured.StepNames));
            Assert.Equal(1, captured.StepNames.Count(n => n == "BatchNormReLU"));
            Assert.Equal(1, captured.StepNames.Count(n => n == "BatchNormAddReLU"));
            Assert.DoesNotContain("BatchNorm", captured.StepNames);
            Assert.DoesNotContain("BatchNormReLU", reference.StepNames);
            Assert.DoesNotContain("BatchNormAddReLU", reference.StepNames);
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(reference.Losses[s] - captured.Losses[s]) <= 1e-4 * Math.Max(1, Math.Abs(reference.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G7} != cpu {reference.Losses[s]:G7}");
            for (int p = 0; p < reference.FinalWeights.Length; p++)
            {
                double maxErr = 0;
                for (int i = 0; i < reference.FinalWeights[p].Length; i++)
                    maxErr = Math.Max(maxErr, Math.Abs(reference.FinalWeights[p][i] - captured.FinalWeights[p][i]));
                Assert.True(maxErr <= 1e-4, $"parameter {p}: max |captured - cpu| {maxErr:E2} after {Steps} steps");
            }
            // The batch statistics: mean1, variance1 (biased, the engine's contract), mean2, variance2, as the CPU plan's.
            string[] statisticNames = { "mean1", "variance1", "mean2", "variance2" };
            for (int s = 0; s < reference.Statistics.Length; s++)
            {
                double maxErr = 0;
                for (int i = 0; i < reference.Statistics[s].Length; i++)
                    maxErr = Math.Max(maxErr, Math.Abs(reference.Statistics[s][i] - captured.Statistics[s][i])
                                              / Math.Max(1.0, Math.Abs(reference.Statistics[s][i])));
                Assert.True(maxErr <= 1e-4, $"{statisticNames[s]}: max rel |captured - cpu| {maxErr:E2} after {Steps} steps");
            }
            // Measured on an RTX 3080: 2 memset nodes (the loss mean's output) and 1 memcpy (the loss seed). Before, this
            // graph carried 24 memset and 13 memcpy nodes; the residual add's gradient copies went with the fused add.
            Assert.True(cuda.LastCaptureMemsetNodeCount <= 2,
                $"{cuda.LastCaptureMemsetNodeCount} memset nodes in the captured step: a dead zero-fill came back");
            Assert.True(cuda.LastCaptureMemcpyNodeCount <= 1,
                $"{cuda.LastCaptureMemcpyNodeCount} memcpy nodes in the captured step: a first-write gradient copy came back");
        }
    }
}
