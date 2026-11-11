using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// #1549: a compiled training step under FP16 autocast captures and replays a whole-step CUDA graph. The resident
/// in-place gradient accumulation was switched off under autocast, so an activation with two consumers (a residual)
/// accumulated its gradient on the host; that download aborted capture (CUDA 900) and the step ran eager: an FP16
/// transformer step took 13x its FP32 time. Gradients stay FP32 under autocast, so the resident add is correct there.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledAutocastGraphStepTests : IDisposable
{
    private const int Rows = 64, Width = 64, Steps = 8;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledAutocastGraphStepTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    // h = relu(x W1 + b1); out = h W2 + b2 + h (residual: h's gradient is accumulated from two consumers).
    private (double[] Losses, bool Replayed, string? Failure) Train(IEngine engine, bool onDevice, bool autocast)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Rows, Width], 1, 1f);
        var y = Rand([Rows, Width], 2, 1f);
        var w1 = Rand([Width, Width], 3, 0.1f);
        var b1 = new Tensor<float>([Width]);
        var w2 = Rand([Width, Width], 4, 0.1f);
        var b2 = new Tensor<float>([Width]);
        var parameters = new[] { w1, b1, w2, b2 };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        using var scope16 = autocast ? new AutocastScope(PrecisionMode.Float16) : null;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.FusedLinear(x, w1, b1, FusedActivationType.ReLU);
            var o = engine.TensorAdd(engine.FusedLinear(h, w2, b2, FusedActivationType.None), h);
            engine.TensorMSELoss(o, y);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
            var losses = new double[Steps];
            for (int s = 0; s < Steps; s++) losses[s] = plan.Step().ToArray()[0];
            var concrete = plan as CompiledTrainingPlan<float>;
            bool replayed = concrete is not null
                && (IntPtr)typeof(CompiledTrainingPlan<float>)
                    .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)! != IntPtr.Zero;
            return (losses, replayed, concrete?.GraphCaptureFailureReason);
        }
    }

    [SkippableFact]
    public void An_fp16_autocast_compiled_step_with_a_residual_replays_a_captured_graph()
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

            var reference = Train(new CpuEngine(), onDevice: false, autocast: false);
            var captured = Train(gpu, onDevice: true, autocast: true);
            _output.WriteLine("losses fp32 cpu      : " + string.Join(", ", reference.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine("losses fp16 captured : " + string.Join(", ", captured.Losses.Select(l => l.ToString("G6"))));

            Assert.True(captured.Replayed, "the FP16 autocast step did not replay a captured graph: " + captured.Failure);
            // FP16 operands round to 2^-11; the curves agree to that, not to FP32.
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(reference.Losses[s] - captured.Losses[s]) <= 2e-2 * Math.Max(1, Math.Abs(reference.Losses[s])),
                    $"step {s}: FP16 captured loss {captured.Losses[s]:G6} != FP32 cpu {reference.Losses[s]:G6}");
            Assert.True(captured.Losses[Steps - 1] < captured.Losses[0], "the FP16 step did not train");
        }
    }
}
