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

    // conv(1->8, 3x3) -> relu -> maxpool 2 -> conv(8->16, 3x3) -> relu -> flatten -> linear. Under FP16 autocast both
    // convolutions and both their gradients run on the 16-bit route inside the captured graph (#1549).
    private (double[] Losses, bool Replayed, string? Failure) TrainCnn(IEngine engine, bool onDevice, bool autocast)
    {
        const int batch = 8, size = 12, classes = 5;
        AiDotNetEngine.Current = engine;
        var x = Rand([batch, 1, size, size], 11, 1f);
        var y = Rand([batch, classes], 12, 1f);
        var w1 = Rand([8, 1, 3, 3], 13, 0.4f);
        var b1 = new Tensor<float>([8]);
        var w2 = Rand([16, 8, 3, 3], 14, 0.15f);
        var b2 = new Tensor<float>([16]);
        var w3 = Rand([16 * (size / 2) * (size / 2), classes], 15, 0.05f);
        var b3 = new Tensor<float>([classes]);
        var parameters = new[] { w1, b1, w2, b2, w3, b3 };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        using var scope16 = autocast ? new AutocastScope(PrecisionMode.Float16) : null;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.FusedConv2D(x, w1, b1, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
            h = engine.MaxPool2D(h, 2);
            h = engine.FusedConv2D(h, w2, b2, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
            h = engine.Reshape(h, new[] { batch, 16 * (size / 2) * (size / 2) });
            engine.TensorMSELoss(engine.FusedLinear(h, w3, b3, FusedActivationType.None), y);
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
    public void An_fp16_autocast_compiled_cnn_runs_its_convolutions_in_fp16_inside_the_captured_graph()
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
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.Gpu.INativeReducedPrecisionConv { NativeReducedPrecisionMath: true },
                "Needs cuDNN tensor-core convolution.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            var reference = TrainCnn(new CpuEngine(), onDevice: false, autocast: false);
            var before = gpu.AutocastConvDispatchCounts;
            var captured = TrainCnn(gpu, onDevice: true, autocast: true);
            var after = gpu.AutocastConvDispatchCounts;
            _output.WriteLine("losses fp32 cpu      : " + string.Join(", ", reference.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine("losses fp16 captured : " + string.Join(", ", captured.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine($"16-bit dispatches: forward {after.Forward - before.Forward}, dX {after.BackwardData - before.BackwardData}, dW {after.BackwardFilter - before.BackwardFilter}");

            Assert.True(captured.Replayed, "the FP16 autocast CNN step did not replay a captured graph: " + captured.Failure);
            // Issued while the step's body runs eagerly (the pre-pass and any eager step); the replays reuse them.
            Assert.True(after.Forward - before.Forward >= 2, "the convolutions' forward did not take the 16-bit route");
            Assert.True(after.BackwardFilter - before.BackwardFilter >= 2, "the weight gradients did not take the 16-bit route");
            Assert.True(after.BackwardData - before.BackwardData >= 1, "the second conv's input gradient did not take the 16-bit route");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(reference.Losses[s] - captured.Losses[s]) <= 2e-2 * Math.Max(1, Math.Abs(reference.Losses[s])),
                    $"step {s}: FP16 captured loss {captured.Losses[s]:G6} != FP32 cpu {reference.Losses[s]:G6}");
            Assert.True(captured.Losses[Steps - 1] < captured.Losses[0], "the FP16 CNN did not train");
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
