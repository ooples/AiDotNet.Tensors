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
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.FusedConv2D(x, w1, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.ReLU(engine.BatchNorm(h, g1, b1, 1e-5, out _, out _));
            h = engine.FusedConv2D(h, w2, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None);
            h = engine.ReLU(engine.TensorAdd(engine.BatchNorm(h, g2, b2, 1e-5, out _, out _), x));
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
            if (plan is CompiledTrainingPlan<float> concrete)
                run.ReplayedAGraph = (IntPtr)typeof(CompiledTrainingPlan<float>)
                    .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)! != IntPtr.Zero;
            return run;
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
            // Measured on an RTX 3080: 2 memset and 3 memcpy nodes (the loss mean's output, the loss seed, and the residual
            // add's gradient handed to both of its inputs). Before, this graph carried 24 memset and 13 memcpy nodes.
            Assert.True(cuda.LastCaptureMemsetNodeCount <= 2,
                $"{cuda.LastCaptureMemsetNodeCount} memset nodes in the captured step: a dead zero-fill came back");
            Assert.True(cuda.LastCaptureMemcpyNodeCount <= 3,
                $"{cuda.LastCaptureMemcpyNodeCount} memcpy nodes in the captured step: a first-write gradient copy came back");
        }
    }
}
