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
/// A compiled training step through a max-pool on the GPU captures and replays a whole-step CUDA graph, and trains like
/// the same plan on the CPU engine. The pool's compiled backward recovered the window winners as a host index array,
/// so every step downloaded them; inside the capture that read aborted it (CUDA 900) and the plan fell back to its eager
/// path for good (measured: the parity CNN's step 27.2 ms against 0.46 ms for PyTorch's CUDA graph).
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledMaxPoolGraphStepTests : IDisposable
{
    private const int Batch = 8, Channels = 4, Size = 8, Classes = 5, Steps = 8;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledMaxPoolGraphStepTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static (double[] Losses, float[][] Weights, bool Replayed) Train(IEngine engine, bool onDevice)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Batch, 1, Size, Size], 1, 1f);
        var y = Rand([Batch, Classes], 2, 1f);
        var w1 = Rand([Channels, 1, 3, 3], 3, 1f / 3f);
        var w2 = Rand([Channels * (Size / 2) * (Size / 2), Classes], 4, 0.2f);
        var b2 = new Tensor<float>([Classes]);
        var parameters = new[] { w1, w2, b2 };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.FusedConv2D(x, w1, null, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
            h = engine.MaxPool2D(h, 2);
            h = engine.Reshape(h, new[] { Batch, Channels * (Size / 2) * (Size / 2) });
            engine.TensorMSELoss(engine.FusedLinear(h, w2, b2, FusedActivationType.None), y);
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
            bool replayed = plan is CompiledTrainingPlan<float> concrete
                && (IntPtr)typeof(CompiledTrainingPlan<float>)
                    .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)! != IntPtr.Zero;
            return (losses, parameters.Select(p => p.ToArray()).ToArray(), replayed);
        }
    }

    [SkippableFact]
    public void A_compiled_step_through_a_max_pool_replays_a_captured_graph_and_trains_like_the_cpu_plan()
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

            var reference = Train(new CpuEngine(), onDevice: false);
            var captured = Train(gpu, onDevice: true);
            _output.WriteLine("losses cpu     : " + string.Join(", ", reference.Losses.Select(l => l.ToString("G7"))));
            _output.WriteLine("losses captured: " + string.Join(", ", captured.Losses.Select(l => l.ToString("G7"))));

            Assert.True(captured.Replayed, "the compiled step through the max-pool did not replay a captured graph");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(reference.Losses[s] - captured.Losses[s]) <= 1e-4 * Math.Max(1, Math.Abs(reference.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G7} != cpu {reference.Losses[s]:G7}");
            for (int p = 0; p < reference.Weights.Length; p++)
            {
                double maxErr = 0;
                for (int i = 0; i < reference.Weights[p].Length; i++)
                    maxErr = Math.Max(maxErr, Math.Abs(reference.Weights[p][i] - captured.Weights[p][i]));
                Assert.True(maxErr <= 1e-4, $"parameter {p}: max |captured - cpu| {maxErr:E2} after {Steps} steps");
            }
        }
    }
}
