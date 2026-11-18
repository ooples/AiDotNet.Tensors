using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class CapturedGradientPublicationTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void Every_captured_step_publishes_current_gradients_after_the_previous_readback_was_consumed()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        using (gpu)
        {
            bool available = gpu?.IsGpuAvailable == true;
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
                Assert.True(available, "A working GPU is required by this validation lane.");
            Skip.If(!available, "No direct GPU backend available.");
            Skip.If(gpu!.GetBackend() is not CudaBackend, "Whole-step capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0", "Capture disabled for this process.");
            AiDotNetEngine.Current = gpu;
            var weight = new Tensor<float>(new[] { 0.4f, -0.3f, 0.2f, -0.5f, 0.6f, 0.1f }, [2, 3]).Gpu();
            // Fixed resident data isolate result publication from external-input refresh and closure rebinding.
            var input = new Tensor<float>(new[] { 0.5f, 0.8f, -0.7f, 0.9f, -0.4f, 0.6f }, [2, 3]).Gpu();
            var target = new Tensor<float>(new[] { -0.2f, 0.3f, 0.4f, -0.1f, 0.2f, -0.3f }, [2, 3]).Gpu();
            var xs = input.ToArray();
            var ys = target.ToArray();
            Tensor<float> predicted;
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { weight }))
            {
                predicted = gpu.TensorMultiply(weight, input);
                GradientRecording.RetainOutput(predicted);
                var residual = gpu.TensorSubtract(predicted, target);
                plan = scope.CompileTraining(new[] { weight }, gpu.ReduceSum(gpu.TensorMultiply(residual, residual), null));
                Assert.Null(scope.HostReadOp);
            }
            Tensor<float>? retiredGradient = null;
            float[]? expectedRetiredGradient = null;
            long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.01, weightDecay: 0);
                float[]? previous = null;
                for (int step = 0; step < 7; step++)
                {
                    var before = weight.ToArray();
                    var expectedOutput = before.Select((w, i) => w * xs[i]).ToArray();
                    var expectedGradient = before.Select((w, i) => 2 * xs[i] * (w * xs[i] - ys[i])).ToArray();
                    double expectedLoss = expectedOutput.Select((value, i) => (double)(value - ys[i]) * (value - ys[i])).Sum();
                    Assert.True(Math.Abs(plan.Step().ToArray()[0] - expectedLoss) < 2e-6, $"step {step} loss");
                    var actual = plan.Gradients[0].ToArray(); // Consume the one-shot host mirror every step.
                    Check(expectedOutput, predicted.ToArray(), step, "forward");
                    Check(expectedGradient, actual, step, "gradient");
                    Check(actual, plan.Gradients[0].ToArray(), step, "repeated gradient read");
                    if (previous is not null)
                        Assert.True(actual.Zip(previous, (a, b) => Math.Abs(a - b)).Any(delta => delta > 1e-4f),
                            $"step {step} repeated the previous step's gradients");
                    previous = actual;
                }
                Assert.Null(plan.GraphCaptureFailureReason);
                var graph = (IntPtr)typeof(CompiledTrainingPlan<float>)
                    .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;
                Assert.NotEqual(IntPtr.Zero, graph);
                Assert.True(CompiledTrainingPlan<float>.GraphReplayCount >= replaysBefore + 2,
                    "The test must read gradients after actual graph replays, not only warmup or capture.");

                var finalBefore = weight.ToArray();
                expectedRetiredGradient = finalBefore.Select((w, i) => 2 * xs[i] * (w * xs[i] - ys[i])).ToArray();
                plan.Step();
                retiredGradient = plan.Gradients[0]; // Retain the tensor, without consuming its final host mirror.
            }
            Check(expectedRetiredGradient!, retiredGradient!.ToArray(), 7, "gradient after plan retirement");
        }
    }

    private static void Check(float[] expected, float[] actual, int step, string value)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(expected[i] - actual[i]) < 2e-6f,
                $"step {step}, {value}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }
}
