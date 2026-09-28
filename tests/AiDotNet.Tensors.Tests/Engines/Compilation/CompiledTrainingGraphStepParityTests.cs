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
        Rand(x.Shape.ToArray(), 100 + step, 1f).AsSpan().CopyTo(x.AsWritableSpan());
        Rand(y.Shape.ToArray(), 200 + step, 1f).AsSpan().CopyTo(y.AsWritableSpan());
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public bool ReplayedAGraph;
        public string[] StepStates = Array.Empty<string>();
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
            var states = new string[Steps];
            for (int s = 0; s < Steps; s++)
            {
                FeedBatch(x, y, s);
                // Reference loss from the exact host batch and pre-step weights: a mismatch message then says whether the
                // plan computed a wrong loss or reported one it never downloaded.
                double reference = CpuLoss(x, y, parameters, composedMse);
                var lossTensor = plan.Step();
                states[s] = $"resident={lossTensor.IsGpuResident} pending={lossTensor.HasPendingGpuData} cpuRef={reference:G6}";
                losses[s] = lossTensor.ToArray()[0];
            }
            var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
            return new Run
            {
                Losses = losses,
                FinalWeights = parameters.Select(p => p.ToArray()).ToArray(),
                ReplayedAGraph = exec != IntPtr.Zero,
                StepStates = states,
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
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6} (captured {captured.StepStates[s]}, eager {eager.StepStates[s]})");
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
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6} (captured {captured.StepStates[s]}, eager {eager.StepStates[s]})");
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

    /// <summary>
    /// The same ground truth on the CPU engine, which AiDotNet's compiled training step uses whenever no GPU is
    /// selected: every step's plan gradient (plus the L2 term) must equal a fresh GradientTape's on that step's batch.
    /// </summary>
    [Theory]
    [InlineData(false, 0.0)]
    [InlineData(true, 0.0)]
    [InlineData(true, 0.05)]
    public void Every_cpu_step_computes_the_tape_gradient(bool composedMse, double l2)
    {
        var prior = AiDotNetEngine.Current;
        var cpu = new CpuEngine();
        try
        {
            AiDotNetEngine.Current = cpu;
            var x = Rand([Batch, Inputs], 1, 1f);
            var y = Rand([Batch, Outputs], 2, 1f);
            var parameters = new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]), Rand([Hidden, Outputs], 4, 0.3f), new Tensor<float>([Outputs]) };

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                Loss(cpu, x, y, parameters, composedMse);
                plan = scope.CompileTraining(parameters);
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
                plan.SetL2Regularization(l2);
                var gradients = (Tensor<float>[])typeof(CompiledTrainingPlan<float>)
                    .GetField("_gradients", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;
                // Reference Adam (Kingma & Ba, PyTorch defaults) on the tape gradients: the plan's weights must follow it.
                var m = parameters.Select(t => new double[t.Length]).ToArray();
                var v = parameters.Select(t => new double[t.Length]).ToArray();
                for (int s = 0; s < Steps; s++)
                {
                    FeedBatch(x, y, s);
                    var expected = CpuTapeGradients(x, y, parameters, composedMse);
                    var expectedWeights = new double[parameters.Length][];
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        var theta = parameters[p].ToArray();
                        for (int i = 0; i < expected[p].Length; i++) expected[p][i] += (float)l2 * theta[i];
                        expectedWeights[p] = new double[theta.Length];
                        for (int i = 0; i < theta.Length; i++)
                        {
                            m[p][i] = 0.9 * m[p][i] + 0.1 * expected[p][i];
                            v[p][i] = 0.999 * v[p][i] + 0.001 * expected[p][i] * expected[p][i];
                            double mHat = m[p][i] / (1 - Math.Pow(0.9, s + 1)), vHat = v[p][i] / (1 - Math.Pow(0.999, s + 1));
                            expectedWeights[p][i] = theta[i] - 1e-2 * mHat / (Math.Sqrt(vHat) + 1e-8);
                        }
                    }
                    plan.Step();
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        var w = parameters[p].ToArray();
                        for (int i = 0; i < w.Length; i++)
                            Assert.True(Math.Abs(w[i] - expectedWeights[p][i]) <= 1e-4,
                                $"step {s}, param {p}[{i}]: plan weight {w[i]} != reference Adam {expectedWeights[p][i]}");
                        var actual = gradients[p].ToArray();
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
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    /// <summary>
    /// A training loop recompiles when the batch shape changes (the short last batch of an epoch). The new plan must
    /// continue the optimizer - moments, step count - or Adam restarts from step 1 every epoch. Reference: one Adam
    /// trajectory across both batch sizes.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_plan_for_a_new_batch_shape_continues_the_previous_plans_optimizer(bool onGpu)
    {
        var prior = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        if (onGpu) Skip.IfNot(TryGpu(out gpu) && gpu is not null, "GPU backend did not resolve.");
        IEngine engine = onGpu ? gpu! : new CpuEngine();
        try
        {
            AiDotNetEngine.Current = engine;
            var parameters = new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]), Rand([Hidden, Outputs], 4, 0.3f), new Tensor<float>([Outputs]) };
            if (onGpu) foreach (var p in parameters) p.Gpu();
            ICompiledTrainingPlan<float> Compile(Tensor<float> x, Tensor<float> y)
            {
                using var scope = GraphMode.EnableTraining(parameters);
                Loss(engine, x, y, parameters, composedMse: true);
                return scope.CompileTraining(parameters);
            }
            var m = parameters.Select(t => new double[t.Length]).ToArray();
            var v = parameters.Select(t => new double[t.Length]).ToArray();
            int t = 0;
            void StepAndCheck(ICompiledTrainingPlan<float> plan, Tensor<float> x, Tensor<float> y, int s)
            {
                FeedBatch(x, y, s);
                var g = CpuTapeGradients(x, y, parameters, composedMse: true);
                t++;
                var expected = new double[parameters.Length][];
                for (int p = 0; p < parameters.Length; p++)
                {
                    var theta = parameters[p].ToArray();
                    expected[p] = new double[theta.Length];
                    for (int i = 0; i < theta.Length; i++)
                    {
                        m[p][i] = 0.9 * m[p][i] + 0.1 * g[p][i];
                        v[p][i] = 0.999 * v[p][i] + 0.001 * g[p][i] * g[p][i];
                        expected[p][i] = theta[i] - 1e-2 * (m[p][i] / (1 - Math.Pow(0.9, t))) / (Math.Sqrt(v[p][i] / (1 - Math.Pow(0.999, t))) + 1e-8);
                    }
                }
                plan.Step();
                for (int p = 0; p < parameters.Length; p++)
                {
                    var w = parameters[p].ToArray();
                    for (int i = 0; i < w.Length; i++)
                        Assert.True(Math.Abs(w[i] - expected[p][i]) <= 1e-4,
                            $"step {t}, param {p}[{i}]: plan weight {w[i]} != reference Adam {expected[p][i]}");
                }
            }

            var xFull = Rand([Batch, Inputs], 1, 1f);
            var yFull = Rand([Batch, Outputs], 2, 1f);
            var xTail = Rand([Batch / 2 + 3, Inputs], 5, 1f);
            var yTail = Rand([Batch / 2 + 3, Outputs], 6, 1f);
            using var full = Compile(xFull, yFull);
            full.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
            for (int s = 0; s < 3; s++) StepAndCheck(full, xFull, yFull, s);

            using var tail = Compile(xTail, yTail);
            tail.ContinueOptimizerFrom(full);
            StepAndCheck(tail, xTail, yTail, 3);

            // And back: the full-batch plan continues from the tail plan's state, not its own stale one. Enough steps
            // for a GPU plan to warm up, capture and replay after the restore.
            full.ContinueOptimizerFrom(tail);
            for (int s = 4; s < 10; s++) StepAndCheck(full, xFull, yFull, s);
            if (onGpu && gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend
                && Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") != "0")
            {
                var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                    .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(full)!;
                Assert.True(exec != IntPtr.Zero, "the restored plan never captured a graph, so the replay path went untested");
            }

            // The plans share one moment store, so a further round trip copies nothing and keeps the captured graph
            // (it covers forward and backward only and never addresses the moments).
            var group = typeof(CompiledTrainingPlan<float>).GetField("_sharedMoments", BindingFlags.NonPublic | BindingFlags.Instance)!;
            Assert.NotNull(group.GetValue(full));
            Assert.Same(group.GetValue(full), group.GetValue(tail));
            var graphField = typeof(CompiledTrainingPlan<float>).GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
            var graphBefore = (IntPtr)graphField.GetValue(full)!;
            tail.ContinueOptimizerFrom(full);
            StepAndCheck(tail, xTail, yTail, 10);
            full.ContinueOptimizerFrom(tail);
            StepAndCheck(full, xFull, yFull, 11);
            Assert.Equal(graphBefore, (IntPtr)graphField.GetValue(full)!);

            // Disposing one member must not free the store the other still steps with.
            tail.ContinueOptimizerFrom(full);
            full.Dispose();
            StepAndCheck(tail, xTail, yTail, 12);
            StepAndCheck(tail, xTail, yTail, 13);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }

    /// <summary>
    /// A GC finalizer that releases a dead GPU result queues its device free; the next GPU op drains the queue. When
    /// that op ran inside the step capture, the free was issued on the capturing stream and RECORDED into the graph,
    /// so every launch freed memory the graph did not own. Measured only in the full test suite (a long process full
    /// of dead GPU results): step 4 - the first captured launch - reported a negative or zero mean-squared loss.
    /// Here the free is queued deterministically from inside the capture.
    /// </summary>
    [SkippableFact]
    public void A_finalizer_free_queued_during_capture_is_not_recorded_into_the_step_graph()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");
            var cuda = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)gpu.GetBackend()!;

            var eager = Train(gpu, capture: false);
            var clean = Train(gpu, capture: true);
            int cleanFreeNodes = cuda.LastCaptureFreeNodeCount;
            Assert.True(clean.ReplayedAGraph, "the reference plan is not replaying a graph after warm-up");
            int queued = 0;
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.TestHookInsideCapture = () =>
            {
                // A pre-capture allocation, released as its finalizer would: pointer claimed, free queued.
                var victim = cuda.AllocateBuffer(4096);
                var ptrField = victim.GetType().GetField("_devicePtr", BindingFlags.NonPublic | BindingFlags.Instance)!;
                var ctxField = victim.GetType().GetField("_context", BindingFlags.NonPublic | BindingFlags.Instance)!;
                var streamField = victim.GetType().GetField("_asyncFreeStream", BindingFlags.NonPublic | BindingFlags.Instance)!;
                var ptr = (IntPtr)ptrField.GetValue(victim)!;
                var ctx = (IntPtr)ctxField.GetValue(victim)!;
                var stream = (IntPtr)streamField.GetValue(victim)!;
                var gen = (long)victim.GetType().GetField("_contextGeneration", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(victim)!;
                ptrField.SetValue(victim, IntPtr.Zero);
                GC.SuppressFinalize(victim);
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.PendingFinalizerFrees.Enqueue((ptr, ctx, gen, stream));
                queued++;
            };
            Run captured;
            try { captured = Train(gpu, capture: true); }
            finally { AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.TestHookInsideCapture = null; }

            Assert.True(queued > 0, "the capture never opened, so nothing was tested");
            Assert.Equal(cleanFreeNodes, cuda.LastCaptureFreeNodeCount);
            Assert.True(captured.ReplayedAGraph, "the plan is not replaying a graph after warm-up");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(eager.Losses[s] - captured.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6} (captured {captured.StepStates[s]}, eager {eager.StepStates[s]})");
        }
    }

    /// <summary>
    /// The backend's compute stream is shared by every thread using the engine, and a stream capture records whatever
    /// is enqueued on the stream. When the step captured on that stream, another thread's op landed IN the step graph
    /// (replayed every step, never run for its own thread) or invalidated the capture. Measured in the full suite:
    /// captured losses of 0 and -176, and dozens of unrelated tests failing "cuEventQuery failed: Invalid value".
    /// Here a second thread runs an op on the same engine from inside the capture and waits for its result.
    /// </summary>
    [SkippableFact]
    public void Another_threads_op_during_a_step_capture_runs_for_that_thread_and_stays_out_of_the_graph()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");
            var cuda = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)gpu.GetBackend()!;

            var eager = Train(gpu, capture: false);
            var clean = Train(gpu, capture: true);
            int cleanKernels = cuda.LastCaptureKernelNodeCount;
            Assert.True(clean.ReplayedAGraph, "the reference plan is not replaying a graph after warm-up");

            var a = Rand([256], 11, 1f);
            var b = Rand([256], 12, 1f);
            float[]? foreignResult = null;
            Exception? foreignError = null;
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.TestHookInsideCapture = () =>
            {
                var other = new System.Threading.Thread(() =>
                {
                    try { foreignResult = gpu.TensorAdd(a, b).ToArray(); }
                    catch (Exception ex) { foreignError = ex; }
                });
                other.Start();
                other.Join();
            };
            Run captured;
            try { captured = Train(gpu, capture: true); }
            finally { AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.TestHookInsideCapture = null; }

            Assert.Null(foreignError);
            Assert.NotNull(foreignResult);
            var av = a.ToArray(); var bv = b.ToArray();
            for (int i = 0; i < av.Length; i++)
                Assert.True(Math.Abs(foreignResult![i] - (av[i] + bv[i])) <= 1e-5f, $"foreign op element {i} is wrong");
            Assert.Equal(cleanKernels, cuda.LastCaptureKernelNodeCount);
            Assert.True(captured.ReplayedAGraph, "the plan is not replaying a graph after warm-up");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(eager.Losses[s] - captured.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6} (captured {captured.StepStates[s]}, eager {eager.StepStates[s]})");
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

    /// <summary>
    /// A compiled plan can outlive the engine it captured its step graph on - AiDotNet caches the compiled training
    /// step per thread and disposes it on the next model's setup, after the previous model's engine is gone.
    /// Disposing it then threw "cuStreamSynchronize (graph destroy) failed: Invalid context" (measured in the
    /// AiDotNet model-family suite). The dead context already reclaimed the graph; disposal must just let go.
    /// </summary>
    [SkippableFact]
    public void A_plan_whose_engine_was_disposed_disposes_without_touching_the_dead_context()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
            "Whole-step graph capture is CUDA-only.");
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
            "Graph capture is disabled for this process.");
        AiDotNetEngine.Current = gpu;
        var x = Rand([Batch, Inputs], 1, 1f);
        var y = Rand([Batch, Outputs], 2, 1f);
        var parameters = new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]), Rand([Hidden, Outputs], 4, 0.3f), new Tensor<float>([Outputs]) };
        foreach (var p in parameters) p.Gpu();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            Loss(gpu, x, y, parameters, composedMse: false);
            plan = scope.CompileTraining(parameters);
        }
        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
        for (int s = 0; s < 6; s++) { FeedBatch(x, y, s); plan.Step(); }
        var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
            .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;
        Assert.True(exec != IntPtr.Zero, "no graph was captured, so the dead-context disposal went untested");

        AiDotNetEngine.Current = _prior;
        gpu.Dispose();
        plan.Dispose();
    }

    private static double CpuLoss(Tensor<float> x, Tensor<float> y, Tensor<float>[] parameters, bool composedMse)
    {
        var prior = AiDotNetEngine.Current;
        var cpu = new CpuEngine();
        try
        {
            AiDotNetEngine.Current = cpu;
            Tensor<float> Copy(Tensor<float> t) => new Tensor<float>(t.ToArray(), t.Shape.ToArray());
            return Loss(cpu, Copy(x), Copy(y), parameters.Select(Copy).ToArray(), composedMse).ToArray()[0];
        }
        finally { AiDotNetEngine.Current = prior; }
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

