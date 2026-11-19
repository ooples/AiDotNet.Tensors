using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A compiled plan's external leaf that the device rewrites before every step - a dropout mask redrawn on the device and
/// copied into the step's tensor - must keep the captured step graph. The copy used to land in a fresh device buffer:
/// the graph's baked pointer still read the buffer the capture pre-pass bound, so the plan saw its leaf rebound, retired
/// the graph and trained eagerly from then on (Informer and Autoformer on the 3090 never replayed). Each replay must also
/// read that step's mask, not the capture step's or a stale host copy, so the parameters must match a CPU plan fed the
/// same masks.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuDeviceRewrittenLeafTests
{
    private const int Rows = 16, Columns = 32, Steps = 10;

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static Tensor<float> Mask(int step)
    {
        var rng = new Random(100 + step);
        var t = new Tensor<float>(new[] { Rows, Columns });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() < 0.3 ? 0f : 1f / 0.7f;
        return t;
    }

    private static (float[] Parameters, string? Failure, long Replays) Train(IEngine engine)
    {
        var x = Filled(new[] { Rows, Columns }, 1, 2f);
        var w = Filled(new[] { Columns, Columns }, 2, 0.2f);
        var mask = Mask(0);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.TensorMatMul(engine.TensorMultiply(x, mask), w);
            engine.ReduceSum(engine.TensorMultiply(h, h), null);
            plan = scope.CompileTraining(new[] { w });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-3f);
            long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
            for (int s = 0; s < Steps; s++)
            {
                // The step's mask reaches the leaf through a device copy of a device result, as a GPU-drawn mask does.
                using var fresh = engine.TensorMultiplyScalar(Mask(s), 1f);
                engine.TensorCopy(fresh, mask);
                plan.Step();
            }
            var compiled = plan as CompiledTrainingPlan<float>;
            return (w.ToArray(), compiled?.GraphCaptureFailureReason,
                CompiledTrainingPlan<float>.GraphReplayCount - replaysBefore);
        }
    }

    [SkippableFact]
    public void A_leaf_the_device_rewrites_before_each_step_keeps_the_captured_graph_and_trains_like_the_cpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.IGpuGraphCapture { SupportsGraphCapture: true },
                "The backend does not capture step graphs.");
            var expected = Train(new CpuEngine()).Parameters;
            var (actual, failure, replays) = Train(gpu);

            Assert.Null(failure);
            // Warm-up steps and the capture step run uncaptured; every later step must replay the graph.
            Assert.True(replays >= Steps - 5, $"the plan replayed its captured graph {replays} times in {Steps} steps");
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-3f * (1f + Math.Abs(expected[i])),
                    $"w[{i}]: cpu {expected[i]}, gpu {actual[i]}");
        }
    }
}
