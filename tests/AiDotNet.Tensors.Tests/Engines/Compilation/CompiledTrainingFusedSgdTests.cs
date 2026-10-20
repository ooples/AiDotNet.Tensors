using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The ping-pong SGD step fused into the weight-gradient kernels (CudaBackend.FusedSgd): a captured conv step must
/// train bit for bit like the same step with every update left to sgd_pingpong_update, through each fused kernel (the
/// direct Winograd epilogue, the split Winograd finish, the split-K reduce), with the dW route pinned so the fused
/// kernels are the ones that run, and a non-finite step must still be undone for the fused tensors.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledTrainingFusedSgdTests : IDisposable
{
    private const int Steps = 10;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public void Dispose()
    {
        AiDotNetEngine.Current = _prior;
        CudaBackend.FusedSgdEnabled = true;
        TunedKernelPolicy.SetOverride(TunedKernelOp.Conv2DBackwardFilter, null);
    }

    private static bool TryGpu(out DirectGpuTensorEngine? engine)
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (!candidate.IsGpuAvailable) { candidate.Dispose(); engine = null; return false; }
            engine = candidate;
            return true;
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { engine = null; return false; }
    }

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public float[][] FinalGradients = Array.Empty<float[]>();
        public float[][] WeightsBeforeNonFiniteStep = Array.Empty<float[]>();
        public float[][] WeightsAfterNonFiniteStep = Array.Empty<float[]>();
        public int[] FusedTensors = new int[2];
        public bool ReplayedParityGraphs;
        public int NonFiniteStepsSkipped;
    }

    // network: 0 = two stride-1 3x3 convs (256 -> 256 on 8x8: the direct Winograd entry; 256 -> 32: the split one plus
    // its finish), 1 = one stride-2 3x3 conv (64 -> 64, 16x16 -> 8x8: an r8 tile grid small enough to split along K).
    private static Run Train(DirectGpuTensorEngine gpu, int network, bool fused, int nonFiniteStep)
    {
        CudaBackend.FusedSgdEnabled = fused;
        Tensor<float> x, y;
        Tensor<float>[] parameters;
        if (network == 0)
        {
            x = Rand([4, 256, 8, 8], 1, 1f);
            y = Rand([4, 32, 8, 8], 2, 1f);
            parameters = new[] { Rand([256, 256, 3, 3], 3, 0.03f), Rand([32, 256, 3, 3], 4, 0.03f) };
        }
        else
        {
            x = Rand([4, 64, 16, 16], 1, 1f);
            y = Rand([4, 64, 8, 8], 2, 1f);
            parameters = new[] { Rand([64, 64, 3, 3], 3, 0.05f) };
        }
        foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            Tensor<float> pred;
            if (network == 0)
            {
                var h = ((IEngine)gpu).ReLU(((IEngine)gpu).Conv2D(x, parameters[0], stride: 1, padding: 1));
                pred = ((IEngine)gpu).Conv2D(h, parameters[1], stride: 1, padding: 1);
            }
            else
            {
                pred = ((IEngine)gpu).Conv2D(x, parameters[0], stride: 2, padding: 1);
            }
            gpu.TensorMSELoss(pred, y);
            plan = scope.CompileTraining(parameters);
        }

        var run = new Run();
        using (plan)
        {
            var concrete = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.SGD, LrSchedule.Cosine(0.05, Steps, 1e-3), weightDecay: 1e-3);
            var losses = new double[Steps];
            for (int s = 0; s < Steps; s++)
            {
                // A different batch every step, through the host span, as the compiled training step feeds it.
                var xs = x.AsWritableSpan();
                for (int i = 0; i < xs.Length; i++) xs[i] = (float)Math.Sin(0.37 * i + s);
                if (s == nonFiniteStep)
                {
                    xs[0] = float.NaN;
                    run.WeightsBeforeNonFiniteStep = parameters.Select(p => p.ToArray()).ToArray();
                }
                x.IncrementVersion();
                losses[s] = plan.Step().ToArray()[0];
                if (s == nonFiniteStep) run.WeightsAfterNonFiniteStep = parameters.Select(p => p.ToArray()).ToArray();
            }
            run.Losses = losses;
            run.FinalWeights = parameters.Select(p => p.ToArray()).ToArray();
            run.FinalGradients = plan.Gradients.Select(g => g.ToArray()).ToArray();
            run.ReplayedParityGraphs = concrete.ReplaysParityGraphs;
            run.NonFiniteStepsSkipped = plan.NonFiniteStepsSkipped;
            var io = typeof(CompiledTrainingPlan<float>).GetField("_stepIo", BindingFlags.NonPublic | BindingFlags.Instance)!
                .GetValue(concrete);
            if (io?.GetType().GetProperty("FusedTensorCounts", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)
                ?.GetValue(io) is int[] counts)
                run.FusedTensors = (int[])counts.Clone();
        }
        return run;
    }

    [SkippableTheory]
    [InlineData(0, "nvrtc.conv2d_dw.winograd_fused", -1)]
    [InlineData(0, "nvrtc.conv2d_dw.winograd_fused", 6)]
    [InlineData(0, "nvrtc.conv2d_dw.winograd_fused", 7)]
    [InlineData(1, "nvrtc.conv2d_dw.r8_64x128", -1)]
    [InlineData(1, "nvrtc.conv2d_dw.r8_64x128", 7)]
    public void Fused_sgd_trains_bit_for_bit_like_the_separate_update(int network, string dwRoute, int nonFiniteStep)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is CudaBackend, "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0", "Graph capture is disabled for this process.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_SGD_PINGPONG") == "0", "The ping-pong update is disabled for this process.");
            TunedKernelPolicy.SetOverride(TunedKernelOp.Conv2DBackwardFilter, dwRoute);

            var separate = Train(gpu, network, fused: false, nonFiniteStep);
            var fusedRun = Train(gpu, network, fused: true, nonFiniteStep);
            Assert.True(separate.ReplayedParityGraphs && fusedRun.ReplayedParityGraphs, "the ping-pong graphs did not replay");
            Assert.Equal(new[] { 0, 0 }, separate.FusedTensors);
            // Every conv weight's update ran in its dW kernel, in both step graphs.
            int expected = network == 0 ? 2 : 1;
            Assert.Equal(new[] { expected, expected }, fusedRun.FusedTensors);

            int expectedSkips = nonFiniteStep >= 0 ? 1 : 0;
            Assert.Equal(expectedSkips, separate.NonFiniteStepsSkipped);
            Assert.Equal(expectedSkips, fusedRun.NonFiniteStepsSkipped);
            for (int s = 0; s < Steps; s++)
                if (s != nonFiniteStep)
                    Assert.Equal(separate.Losses[s], fusedRun.Losses[s]);
            for (int p = 0; p < separate.FinalWeights.Length; p++)
            {
                Assert.Equal(separate.FinalWeights[p], fusedRun.FinalWeights[p]);
                // The fused kernels still publish the gradient.
                Assert.Equal(separate.FinalGradients[p], fusedRun.FinalGradients[p]);
            }
            if (nonFiniteStep >= 0)
                for (int p = 0; p < fusedRun.WeightsBeforeNonFiniteStep.Length; p++)
                    Assert.Equal(fusedRun.WeightsBeforeNonFiniteStep[p], fusedRun.WeightsAfterNonFiniteStep[p]);
        }
    }
}
