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
/// Whole-step CUDA-graph capture is ON by default for float compiled training on CUDA: after
/// <c>GraphWarmupSteps</c> eager steps the plan captures one step and replays it from then on. A replay is only
/// correct if it trains EXACTLY like the eager step it replaces, so the oracle is the same plan with capture disabled:
/// same initial weights, same batch, same optimizer — the per-step losses and the final weights must agree, and the
/// captured plan must really be replaying a graph (otherwise a silently failing capture would pass vacuously).
/// </summary>
/// <remarks>
/// Four defects stacked on this path, each found with this oracle (a 1024-3x1024-10 MLP trained through AiDotNet's
/// NeuralNetwork.Train on the GPU stopped learning after three steps; with capture disabled it trained):
/// <list type="number">
/// <item>The MSE loss is ReduceMean(...).Reshape([1]); the reshape dropped the device buffer, so copying the loss out
/// went through the host — a CUDA 900 inside the capture, so capture always failed.</item>
/// <item>A failed capture left the gradient accumulators bound to device buffers; the eager fallback zeroed only the
/// host arrays, so its in-place adds piled up on the device and the optimizer read host arrays that never got them.</item>
/// <item>The optimizer retired the captured graph after EVERY update, even when all parameters were updated on the
/// device — so the plan cycled warm-up, capture, destroy, and hit defect 2 on every cycle.</item>
/// <item>A replay never re-armed the loss tensor's host download, so every loss reported after capture was the
/// capture step's.</item>
/// </list>
/// </remarks>
[Collection("DirectGpuSerial")]
public class CompiledTrainingGraphStepParityTests : IDisposable
{
    private const int Batch = 32, Inputs = 24, Hidden = 48, Outputs = 6, Steps = 10;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledTrainingGraphStepParityTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static bool TryGpu(out DirectGpuTensorEngine? engine)
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (!candidate.IsGpuAvailable) { candidate.Dispose(); engine = null; return false; }
            engine = candidate;
            return true;
        }
        catch (Exception) { engine = null; return false; }
    }

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    /// <summary>
    /// A different batch every step, written into the plan's persistent input and target through a host span - exactly
    /// how AiDotNet's compiled training step feeds each batch. A replayed graph must see it: with the target frozen at
    /// the capture step's batch, the model memorized that one batch (reported loss -> 0 while the real fit got worse).
    /// </summary>
    private static void FeedBatch(Tensor<float> x, Tensor<float> y, int step)
    {
        Rand([Batch, Inputs], 100 + step, 1f).AsSpan().CopyTo(x.AsWritableSpan());
        Rand([Batch, Outputs], 200 + step, 1f).AsSpan().CopyTo(y.AsWritableSpan());
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public bool ReplayedAGraph;
    }

    private static Run Train(DirectGpuTensorEngine gpu, bool capture, bool failCapture = false, bool composedMse = false)
    {
        var x = Rand([Batch, Inputs], 1, 1f);
        var y = Rand([Batch, Outputs], 2, 1f);
        var w1 = Rand([Inputs, Hidden], 3, 0.3f);
        var b1 = new Tensor<float>([Hidden]);
        var w2 = Rand([Hidden, Outputs], 4, 0.3f);
        var b2 = new Tensor<float>([Outputs]);
        var parameters = new[] { w1, b1, w2, b2 };
        // Device-owned parameters, as AiDotNet's compiled step makes them before compiling.
        foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = gpu.FusedLinear(x, w1, b1, FusedActivationType.ReLU);
            var pred = gpu.FusedLinear(h, w2, b2, FusedActivationType.None);
            if (composedMse)
            {
                // AiDotNet's MeanSquaredErrorLoss.ComputeTapeLoss: mean over all axes of (pred - y)^2.
                var diff = gpu.TensorSubtract(pred, y);
                gpu.ReduceMean(gpu.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
            }
            else
            {
                gpu.TensorMSELoss(pred, y);
            }
            plan = scope.CompileTraining(parameters);
        }

        using (plan)
        {
            var concrete = (CompiledTrainingPlan<float>)plan;
            if (!capture) concrete.DisableGraphStep();
            concrete.FailNextCaptureForTesting = failCapture;
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
            var losses = new double[Steps];
            for (int s = 0; s < Steps; s++)
            {
                FeedBatch(x, y, s);
                losses[s] = plan.Step().ToArray()[0];
            }
            var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
            return new Run
            {
                Losses = losses,
                FinalWeights = parameters.Select(p => p.ToArray()).ToArray(),
                ReplayedAGraph = exec != IntPtr.Zero,
            };
        }
    }

    [SkippableFact]
    public void A_captured_step_trains_exactly_like_the_eager_step()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            var eager = Train(gpu, capture: false);
            var captured = Train(gpu, capture: true);
            _output.WriteLine("eager    " + string.Join(" ", eager.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine("captured " + string.Join(" ", captured.Losses.Select(l => l.ToString("G6"))));

            // The eager plan must actually learn, or agreeing with it proves nothing.
            Assert.True(eager.Losses.Skip(Steps / 2).Average() < eager.Losses.Take(Steps / 2).Average(), "the eager reference did not learn");
            Assert.False(eager.ReplayedAGraph, "the eager reference captured a graph");
            // A capture that fails falls back to eager and would agree trivially.
            Assert.True(captured.ReplayedAGraph, "the captured plan is not replaying a graph after warm-up");

            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(eager.Losses[s] - captured.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6}");
            for (int p = 0; p < eager.FinalWeights.Length; p++)
                for (int i = 0; i < eager.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(eager.FinalWeights[p][i] - captured.FinalWeights[p][i]) <= 1e-5f,
                        $"param {p}[{i}]: captured {captured.FinalWeights[p][i]} != eager {eager.FinalWeights[p][i]}");
        }
    }

    /// <summary>
    /// A capture that fails (any capture-unsafe op) falls back to the eager step. It must then train exactly like a
    /// plan that never tried: the pre-pass it ran bound the gradient accumulators to device buffers, and leaving them
    /// bound made the eager step's in-place gradient adds land on never-zeroed device memory.
    /// </summary>
    [SkippableFact]
    public void A_failed_capture_trains_exactly_like_the_eager_step()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            var eager = Train(gpu, capture: false);
            var failed = Train(gpu, capture: true, failCapture: true);
            _output.WriteLine("eager  " + string.Join(" ", eager.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine("failed " + string.Join(" ", failed.Losses.Select(l => l.ToString("G6"))));

            Assert.False(failed.ReplayedAGraph, "the forced capture failure did not take effect");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(eager.Losses[s] - failed.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: loss after a failed capture {failed.Losses[s]:G6} != eager {eager.Losses[s]:G6}");
            for (int p = 0; p < eager.FinalWeights.Length; p++)
                for (int i = 0; i < eager.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(eager.FinalWeights[p][i] - failed.FinalWeights[p][i]) <= 1e-5f,
                        $"param {p}[{i}]: after a failed capture {failed.FinalWeights[p][i]} != eager {eager.FinalWeights[p][i]}");
        }
    }

    /// <summary>
    /// The same oracle with AiDotNet's MSE composition (ReduceMean over (pred - y)^2), which is what
    /// NeuralNetwork.Train records. Its ReduceMeanBackward has to stay on the device inside the capture (a synchronous
    /// Fill made it fall to the CPU there - CUDA 906), and its x*x backward exposed the eager step's stale gradient
    /// accumulation (see Every_step_computes_the_tape_gradient).
    /// </summary>
    [SkippableFact]
    public void A_captured_step_with_a_composed_mean_loss_trains_exactly_like_the_eager_step()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            var eager = Train(gpu, capture: false, composedMse: true);
            var captured = Train(gpu, capture: true, composedMse: true);
            _output.WriteLine("eager    " + string.Join(" ", eager.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine("captured " + string.Join(" ", captured.Losses.Select(l => l.ToString("G6"))));

            Assert.True(eager.Losses.Skip(Steps / 2).Average() < eager.Losses.Take(Steps / 2).Average(), "the eager reference did not learn");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(eager.Losses[s] - captured.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6}");
            for (int p = 0; p < eager.FinalWeights.Length; p++)
                for (int i = 0; i < eager.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(eager.FinalWeights[p][i] - captured.FinalWeights[p][i]) <= 1e-5f,
                        $"param {p}[{i}]: captured {captured.FinalWeights[p][i]} != eager {eager.FinalWeights[p][i]}");
            Assert.True(captured.ReplayedAGraph, "the captured plan is not replaying a graph after warm-up");
        }
    }

    /// <summary>
    /// Ground truth, independent of every GPU path: at each step the compiled plan's parameter gradients must equal a
    /// CPU-engine GradientTape's gradients at the same weights and batch. "Captured equals eager" is not enough on its
    /// own - the eager step used to accumulate each step's gradient onto the previous step's cached device copy, so
    /// both sides of that comparison were wrong together.
    /// </summary>
    [SkippableTheory]
    [InlineData(false, false, 0.0)]
    [InlineData(false, true, 0.0)]
    [InlineData(true, false, 0.0)]
    [InlineData(true, true, 0.0)]
    // L2 regularization (AiDotNet's optimizer default is 0.01): the plan must add strength * theta to each gradient.
    [InlineData(false, true, 0.05)]
    [InlineData(true, true, 0.05)]
    public void Every_step_computes_the_tape_gradient(bool capture, bool composedMse, double l2)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            var x = Rand([Batch, Inputs], 1, 1f);
            var y = Rand([Batch, Outputs], 2, 1f);
            var parameters = new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]), Rand([Hidden, Outputs], 4, 0.3f), new Tensor<float>([Outputs]) };
            foreach (var p in parameters) p.Gpu();

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                Loss(gpu, x, y, parameters, composedMse);
                plan = scope.CompileTraining(parameters);
            }
            using (plan)
            {
                var concrete = (CompiledTrainingPlan<float>)plan;
                if (!capture) concrete.DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
                plan.SetL2Regularization(l2);
                var gradients = (Tensor<float>[])typeof(CompiledTrainingPlan<float>)
                    .GetField("_gradients", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
                for (int s = 0; s < Steps; s++)
                {
                    FeedBatch(x, y, s);
                    var expected = CpuTapeGradients(x, y, parameters, composedMse);
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        var theta = parameters[p].ToArray();   // the pre-step weights the L2 term is taken at
                        for (int i = 0; i < expected[p].Length; i++) expected[p][i] += (float)l2 * theta[i];
                    }
                    plan.Step();
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        var actual = gradients[p].TryGetGpuBuffer() is { } buffer
                            ? gpu.GetBackend()!.DownloadBuffer(buffer)
                            : gradients[p].ToArray();
                        double diff = 0, norm = 0;
                        for (int i = 0; i < expected[p].Length; i++)
                        {
                            double d = actual[i] - expected[p][i];
                            diff += d * d; norm += (double)expected[p][i] * expected[p][i];
                        }
                        Assert.True(Math.Sqrt(diff) <= 1e-3 * Math.Sqrt(norm) + 1e-6,
                            $"step {s}, param {p}: |plan - tape| = {Math.Sqrt(diff):G4}, |tape| = {Math.Sqrt(norm):G4}");
                    }
                }
            }
        }
    }

    private static Tensor<float> Loss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w, bool composedMse)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var pred = e.FusedLinear(h, w[2], w[3], FusedActivationType.None);
        if (!composedMse) return e.TensorMSELoss(pred, y);
        var diff = e.TensorSubtract(pred, y);
        return e.ReduceMean(e.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
    }

    private static float[][] CpuTapeGradients(Tensor<float> x, Tensor<float> y, Tensor<float>[] parameters, bool composedMse)
    {
        var prior = AiDotNetEngine.Current;
        var cpu = new CpuEngine();
        try
        {
            AiDotNetEngine.Current = cpu;
            Tensor<float> Copy(Tensor<float> t) => new Tensor<float>(t.ToArray(), t.Shape.ToArray());
            var w = parameters.Select(Copy).ToArray();
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
            var loss = Loss(cpu, Copy(x), Copy(y), w, composedMse);
            var grads = tape.ComputeGradients(loss, w);
            return w.Select(p => grads[p].ToArray()).ToArray();
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}

