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
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { engine = null; return false; }
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
        public float[][] LastGradients = Array.Empty<float[]>();
        public float[][] WeightsBeforeNonFiniteStep = Array.Empty<float[]>();
        public float[][] WeightsAfterNonFiniteStep = Array.Empty<float[]>();
        public bool SkipReportedAtNonFiniteStep;
        public bool SkipReportedAfterNonFiniteStep;
        public int NonFiniteStepsSkipped;
        public int OptimizerStep;
        public bool UpdateCapturedInGraph;   // the optimizer update ran inside the step graph (step I/O)
        public bool UpdateCapturedBeforeClip;   // ... at the step before clipOnFromStep
        public int CheckpointStep;   // the step count an optimizer checkpoint (export / save) records
        public bool ReplayedParityGraphs;   // two step graphs alternated (the ping-pong SGD update)
        public float[][] WeightsAfterDispose = Array.Empty<float[]>();
    }

    private static Run Train(DirectGpuTensorEngine gpu, bool capture, bool failCapture = false, bool composedMse = false, bool failInsideCapture = false, bool releasedInputBinding = false, bool retireGraphBeforeLastRead = false, bool droppedTargetBinding = false, int nonFiniteStep = -1, double maxGradNorm = 0, int clipOffFromStep = int.MaxValue, LrSchedule? schedule = null, OptimizerType optimizer = OptimizerType.Adam, int clipOnFromStep = int.MaxValue, bool pingPong = true)
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
            if (!pingPong) concrete.DisablePingPongForTesting();
            concrete.FailNextCaptureForTesting = failCapture;
            concrete.FailInsideNextCaptureForTesting = failInsideCapture;
            if (schedule is null) plan.ConfigureOptimizer(optimizer, learningRate: 1e-2f);
            else plan.ConfigureOptimizer(optimizer, schedule);
            plan.SetMaxGradNorm(maxGradNorm);
            var losses = new double[Steps];
            float[][] lastGradients = Array.Empty<float[]>();
            var states = new string[Steps];
            var run = new Run();
            for (int s = 0; s < Steps; s++)
            {
                FeedBatch(x, y, s);
                if (s == clipOffFromStep) plan.SetMaxGradNorm(0);
                if (s == clipOnFromStep)
                {
                    run.UpdateCapturedBeforeClip = typeof(CompiledTrainingPlan<float>)
                        .GetField("_capturedOptimizer", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete) is not null;
                    plan.SetMaxGradNorm(0.5);
                }
                if (s == nonFiniteStep)
                {
                    // One non-finite input makes every gradient non-finite: the whole step must be discarded.
                    x.AsWritableSpan()[0] = float.NaN;
                    run.WeightsBeforeNonFiniteStep = parameters.Select(p => p.ToArray()).ToArray();
                }
                if (releasedInputBinding && s == 3)
                {
                    // An earlier eager step can leave an input bound to a buffer that is no longer its to use: released
                    // (zero handle - the capture path's refresh threw, and the catch disabled the whole-step graph) or,
                    // as here, returned to the pool with its handle intact, where refreshing the batch into it would
                    // write into whichever tensor rents it next.
                    var cb = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)gpu.GetBackend()!;
                    var released = cb.AllocateBuffer(new float[y.Length]);
                    released.Dispose();
                    y._gpuBuffer = released;
                    y._gpuBackend = cb;
                }
                if (droppedTargetBinding && s == 6)
                {
                    // After capture and a few replays the target loses its device binding (anything that drops a
                    // tensor's device copy). The graph still reads the buffer it baked in; skipping the refresh because
                    // the tensor is unbound trained on the previous step's target.
                    y._gpuBuffer = null;
                    y._gpuBackend = null;
                    y._gpuBufferVersion = -1;
                }
                // Reference loss from the exact host batch and pre-step weights: a mismatch message then says whether the
                // plan computed a wrong loss or reported one it never downloaded.
                double reference = CpuLoss(x, y, parameters, composedMse);
                var lossTensor = plan.Step();
                if (retireGraphBeforeLastRead && s == Steps - 1)
                {
                    // Retire the replayed graph BEFORE anything reads this step's results, as ConfigureOptimizer,
                    // EnableFrozenWeightOptimizations and Dispose do: the rollback must land the device results on
                    // the host, not discard them.
                    concrete.DisableGraphStep();
                }
                states[s] = $"resident={lossTensor.IsGpuResident} pending={lossTensor.HasPendingGpuData} cpuRef={reference:G6}";
                losses[s] = lossTensor.ToArray()[0];
                if (s == Steps - 1) lastGradients = plan.Gradients.Select(g => g.ToArray()).ToArray();
                if (s == nonFiniteStep)
                {
                    run.WeightsAfterNonFiniteStep = parameters.Select(p => p.ToArray()).ToArray();
                    run.SkipReportedAtNonFiniteStep = plan.LastStepSkippedNonFiniteGradients;
                }
                if (nonFiniteStep >= 0 && s == nonFiniteStep + 1) run.SkipReportedAfterNonFiniteStep = plan.LastStepSkippedNonFiniteGradients;
            }
            var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
            run.Losses = losses;
            run.FinalWeights = parameters.Select(p => p.ToArray()).ToArray();
            run.ReplayedAGraph = exec != IntPtr.Zero || retireGraphBeforeLastRead;
            run.LastGradients = lastGradients;
            run.StepStates = states;
            // Read before anything else resolves the plan's step count: a checkpoint must resolve it itself.
            run.CheckpointStep = concrete.CaptureFusedOptimizerCheckpoint()?.OptimizerStep ?? -1;
            run.NonFiniteStepsSkipped = plan.NonFiniteStepsSkipped;
            run.OptimizerStep = concrete.OptimizerStep;
            run.UpdateCapturedInGraph = typeof(CompiledTrainingPlan<float>)
                .GetField("_capturedOptimizer", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete) is not null;
            run.ReplayedParityGraphs = concrete.ReplaysParityGraphs;
            plan.Dispose();
            run.WeightsAfterDispose = parameters.Select(p => p.ToArray()).ToArray();
            return run;
        }
    }

    /// <summary>
    /// The captured SGD update writes the new weights into a second buffer set and two step graphs alternate (one
    /// reads set A and writes B, the other the reverse), so one pass over the gradients both updates and takes their
    /// finiteness reduction. It must train bit for bit like the in-place update: the same arithmetic per element, a
    /// non-finite step undone on either parity, the current weights readable between steps, and back in the plan's own
    /// buffers once the graphs go (an odd replay count leaves them in the second set).
    /// </summary>
    [SkippableTheory]
    [InlineData(-1)]
    [InlineData(6)]
    [InlineData(7)]
    public void Ping_pong_sgd_trains_bit_for_bit_like_the_in_place_update(int nonFiniteStep)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_SGD_PINGPONG") == "0",
                "The ping-pong update is disabled for this process.");

            var schedule = LrSchedule.Cosine(0.1, Steps, 1e-3);
            var inPlace = Train(gpu, capture: true, schedule: schedule, optimizer: OptimizerType.SGD, nonFiniteStep: nonFiniteStep, pingPong: false);
            var pingPong = Train(gpu, capture: true, schedule: schedule, optimizer: OptimizerType.SGD, nonFiniteStep: nonFiniteStep);
            Assert.True(inPlace.UpdateCapturedInGraph && pingPong.UpdateCapturedInGraph, "the update was not captured into the graph");
            Assert.False(inPlace.ReplayedParityGraphs, "the in-place oracle alternated graphs");
            Assert.True(pingPong.ReplayedParityGraphs, "the ping-pong plan did not alternate two step graphs");
            Assert.Equal(inPlace.NonFiniteStepsSkipped, pingPong.NonFiniteStepsSkipped);
            Assert.Equal(inPlace.OptimizerStep, pingPong.OptimizerStep);
            for (int s = 0; s < Steps; s++)
                if (s != nonFiniteStep)
                    Assert.Equal(inPlace.Losses[s], pingPong.Losses[s]);
            if (nonFiniteStep >= 0)
                for (int p = 0; p < pingPong.WeightsBeforeNonFiniteStep.Length; p++)
                    Assert.Equal(pingPong.WeightsBeforeNonFiniteStep[p], pingPong.WeightsAfterNonFiniteStep[p]);
            for (int p = 0; p < inPlace.FinalWeights.Length; p++)
            {
                Assert.Equal(inPlace.FinalWeights[p], pingPong.FinalWeights[p]);
                Assert.Equal(pingPong.FinalWeights[p], pingPong.WeightsAfterDispose[p]);
            }
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
            Assert.True(captured.UpdateCapturedInGraph, "the replayed graph does not run the optimizer update (step I/O)");
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
    /// A non-finite gradient on a replayed step must discard that step entirely - no moment or weight written - and
    /// report it, exactly as an eager step does. On CUDA the decision is made on the device (the multi-tensor Adam
    /// kernel reads the gradients' sum of squares) and the host learns of it from an asynchronous readback, so the
    /// skip report and the step counter must still come out as if the host had checked before launching.
    /// <para>With gradient clipping on, the guard reuses the clip's sum of squares of the same gradients; clipping
    /// switched off at the non-finite step must not let the previous step's (finite) clip sum stand in for it.</para>
    /// </summary>
    [SkippableTheory]
    [InlineData(0.0, int.MaxValue)]
    [InlineData(0.5, int.MaxValue)]
    [InlineData(0.5, 6)]
    public void A_non_finite_gradient_discards_the_replayed_step_like_the_eager_step(double maxGradNorm, int clipOffFromStep)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            const int nonFiniteStep = 6;   // after warm-up and capture: a replay
            var eager = Train(gpu, capture: false, nonFiniteStep: nonFiniteStep, maxGradNorm: maxGradNorm, clipOffFromStep: clipOffFromStep);
            var captured = Train(gpu, capture: true, nonFiniteStep: nonFiniteStep, maxGradNorm: maxGradNorm, clipOffFromStep: clipOffFromStep);
            Assert.True(captured.ReplayedAGraph, "the captured plan is not replaying a graph after warm-up");
            if (maxGradNorm == 0)
                Assert.True(captured.UpdateCapturedInGraph, "the discard was not made by the graph's own update (step I/O)");

            foreach (var (name, run) in new[] { ("eager", eager), ("captured", captured) })
            {
                Assert.True(run.SkipReportedAtNonFiniteStep, $"{name}: the non-finite step was not reported as skipped");
                Assert.False(run.SkipReportedAfterNonFiniteStep, $"{name}: the finite step after it was reported as skipped");
                Assert.Equal(1, run.NonFiniteStepsSkipped);
                Assert.Equal(Steps - 1, run.OptimizerStep);
                for (int p = 0; p < run.WeightsBeforeNonFiniteStep.Length; p++)
                    Assert.Equal(run.WeightsBeforeNonFiniteStep[p], run.WeightsAfterNonFiniteStep[p]);
            }
            for (int s = 0; s < Steps; s++)
            {
                if (s == nonFiniteStep) continue;
                Assert.True(Math.Abs(eager.Losses[s] - captured.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6}");
            }
            for (int p = 0; p < eager.FinalWeights.Length; p++)
                for (int i = 0; i < eager.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(eager.FinalWeights[p][i] - captured.FinalWeights[p][i]) <= 1e-5f,
                        $"param {p}[{i}]: captured {captured.FinalWeights[p][i]} != eager {eager.FinalWeights[p][i]}");
        }
    }

    /// <summary>
    /// The optimizer update inside the step graph reads each step's learning rate (and Adam's step number) from the
    /// device, picked by the device's own count of applied steps out of a table the host writes ahead. A schedule that
    /// changes the rate every step, and a non-finite step that must not advance the count, are where a stale or
    /// off-by-one choice would show: the replayed plan must train exactly like the eager plan.
    /// </summary>
    [SkippableTheory]
    [InlineData(OptimizerType.Adam, -1)]
    [InlineData(OptimizerType.Adam, 6)]
    [InlineData(OptimizerType.AdamW, 7)]
    [InlineData(OptimizerType.SGD, 6)]
    public void A_scheduled_update_inside_the_graph_trains_like_the_eager_step(OptimizerType optimizer, int nonFiniteStep)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            var schedule = LrSchedule.Cosine(optimizer == OptimizerType.SGD ? 0.1 : 2e-2, Steps, 1e-3);
            var eager = Train(gpu, capture: false, schedule: schedule, optimizer: optimizer, nonFiniteStep: nonFiniteStep);
            var captured = Train(gpu, capture: true, schedule: schedule, optimizer: optimizer, nonFiniteStep: nonFiniteStep);
            _output.WriteLine("eager    " + string.Join(" ", eager.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine("captured " + string.Join(" ", captured.Losses.Select(l => l.ToString("G6"))));
            Assert.True(captured.ReplayedAGraph, "the captured plan is not replaying a graph after warm-up");
            Assert.True(captured.UpdateCapturedInGraph, "the replayed graph does not run the optimizer update (step I/O)");
            Assert.Equal(eager.OptimizerStep, captured.OptimizerStep);
            Assert.Equal(eager.CheckpointStep, captured.CheckpointStep);
            Assert.Equal(eager.NonFiniteStepsSkipped, captured.NonFiniteStepsSkipped);
            for (int s = 0; s < Steps; s++)
            {
                if (s == nonFiniteStep) continue;
                Assert.True(Math.Abs(eager.Losses[s] - captured.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: captured loss {captured.Losses[s]:G6} != eager {eager.Losses[s]:G6}");
            }
            for (int p = 0; p < eager.FinalWeights.Length; p++)
                for (int i = 0; i < eager.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(eager.FinalWeights[p][i] - captured.FinalWeights[p][i]) <= 1e-5f,
                        $"param {p}[{i}]: captured {captured.FinalWeights[p][i]} != eager {eager.FinalWeights[p][i]}");
        }
    }

    /// <summary>
    /// Gradient clipping switched on after the update was captured into the graph: the graph's update knows nothing of
    /// clipping, so the plan must retire it and clip as the eager step does.
    /// </summary>
    [SkippableFact]
    public void Clipping_switched_on_after_capture_is_applied_like_the_eager_step()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            var eager = Train(gpu, capture: false, clipOnFromStep: 6);
            var captured = Train(gpu, capture: true, clipOnFromStep: 6);
            Assert.True(captured.ReplayedAGraph, "the captured plan is not replaying a graph after warm-up");
            Assert.True(captured.UpdateCapturedBeforeClip, "the update was not captured into the graph before clipping was switched on");
            Assert.False(captured.UpdateCapturedInGraph, "a graph whose update skips clipping kept replaying after clipping was switched on");
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
    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void A_failed_capture_trains_exactly_like_the_eager_step(bool failInsideCapture, bool composedMse)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            // failInsideCapture: the capture aborts from INSIDE, after its forward and backward already issued and
            // allocated there (a host read of a resident value - how a TabDDPM fused step failed). Buffers the aborted
            // capture handed out were graph-owned addresses, and the eager fallback then faulted on them (CUDA 700).
            var eager = Train(gpu, capture: false, composedMse: composedMse);
            var failed = Train(gpu, capture: true, failCapture: !failInsideCapture, composedMse: composedMse,
                failInsideCapture: failInsideCapture);
            _output.WriteLine("eager  " + string.Join(" ", eager.Losses.Select(l => l.ToString("G6"))));
            _output.WriteLine("failed " + string.Join(" ", failed.Losses.Select(l => l.ToString("G6"))));
            for (int s = 0; s < Steps; s++) _output.WriteLine($"failed step {s}: {failed.StepStates[s]}");

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
    /// Retiring a graph that RAN (ConfigureOptimizer, frozen-weight specialization, Dispose, the host optimizer path)
    /// must keep that step's results. The rollback used to discard every pending device-to-host download - right for
    /// a failed capture's pre-pass, wrong here, where the device holds the only copy of the loss and gradients.
    /// </summary>
    [SkippableFact]
    public void Retiring_a_replayed_graph_keeps_the_last_steps_loss_and_gradients()
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
            var retired = Train(gpu, capture: true, retireGraphBeforeLastRead: true);
            int last = Steps - 1;
            Assert.True(Math.Abs(eager.Losses[last] - retired.Losses[last]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[last])),
                $"last loss after retiring the graph {retired.Losses[last]:G6} != eager {eager.Losses[last]:G6}");
            for (int p = 0; p < eager.LastGradients.Length; p++)
                for (int i = 0; i < eager.LastGradients[p].Length; i++)
                    Assert.True(Math.Abs(eager.LastGradients[p][i] - retired.LastGradients[p][i]) <= 1e-5f,
                        $"grad {p}[{i}] after retiring the graph {retired.LastGradients[p][i]} != eager {eager.LastGradients[p][i]}");
        }
    }

    /// <summary>
    /// L2 regularization must land where the optimizer reads each gradient. It used to be written on the device
    /// whenever the gradient and parameter merely HAD device buffers, while the optimizer reads the host gradient
    /// unless the device copy is current and the parameter is resident - with mixed residency the term was silently
    /// dropped for some parameters. Every residency pattern must now train identically.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void L2_regularization_trains_identically_whatever_each_parameters_residency(bool capture)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            // A smooth hidden activation. With ReLU, a pre-activation within float rounding of zero lands on different
            // sides of the kink in the GPU forward and the CPU oracle (measured: one hidden unit flipped at step 4 of
            // the all-device run, every per-parameter sum still agreeing to 1e-4) - noise the L2 check cannot tell
            // from a misplaced term.
            const FusedActivationType Smooth = FusedActivationType.Tanh;
            float[][] RunWith(bool[] resident)
            {
                var x = Rand([Batch, Inputs], 1, 1f);
                var y = Rand([Batch, Outputs], 2, 1f);
                var parameters = new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]), Rand([Hidden, Outputs], 4, 0.3f), new Tensor<float>([Outputs]) };
                for (int i = 0; i < parameters.Length; i++) if (resident[i]) parameters[i].Gpu();
                ICompiledTrainingPlan<float> plan;
                using (var scope = GraphMode.EnableTraining(parameters))
                {
                    Loss(gpu, x, y, parameters, composedMse: true, Smooth);
                    plan = scope.CompileTraining(parameters);
                }
                using (plan)
                {
                    if (!capture) ((CompiledTrainingPlan<float>)plan).DisableGraphStep();
                    plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
                    plan.SetL2Regularization(0.05);
                    for (int s = 0; s < Steps; s++) { FeedBatch(x, y, s); plan.Step(); }
                }
                return parameters.Select(t => t.ToArray()).ToArray();
            }

            // Oracle: CPU tape gradients + 0.05*theta, stepped with textbook Adam (the plan's defaults).
            float[][] Reference()
            {
                var x = Rand([Batch, Inputs], 1, 1f);
                var y = Rand([Batch, Outputs], 2, 1f);
                var w = new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]), Rand([Hidden, Outputs], 4, 0.3f), new Tensor<float>([Outputs]) };
                var m = w.Select(t => new double[t.Length]).ToArray();
                var v = w.Select(t => new double[t.Length]).ToArray();
                for (int s = 0; s < Steps; s++)
                {
                    FeedBatch(x, y, s);
                    var g = CpuTapeGradients(x, y, w, composedMse: true, Smooth);
                    int t = s + 1;
                    for (int p = 0; p < w.Length; p++)
                    {
                        var theta = w[p].ToArray();
                        for (int i = 0; i < theta.Length; i++)
                        {
                            double gi = g[p][i] + 0.05 * theta[i];
                            m[p][i] = 0.9 * m[p][i] + 0.1 * gi;
                            v[p][i] = 0.999 * v[p][i] + 0.001 * gi * gi;
                            double update = 1e-2 * (m[p][i] / (1 - Math.Pow(0.9, t))) / (Math.Sqrt(v[p][i] / (1 - Math.Pow(0.999, t))) + 1e-8);
                            w[p][i] = (float)(theta[i] - update);
                        }
                    }
                }
                return w.Select(t => t.ToArray()).ToArray();
            }

            var expected = Reference();
            foreach (var pattern in new[] { new[] { true, true, true, true }, new[] { true, false, true, false }, new[] { false, true, false, true }, new[] { false, false, false, false } })
            {
                var actual = RunWith(pattern);
                for (int p = 0; p < expected.Length; p++)
                    for (int i = 0; i < expected[p].Length; i++)
                        Assert.True(Math.Abs(expected[p][i] - actual[p][i]) <= 1e-4f,
                            $"residency [{string.Join(",", pattern)}] capture={capture}: param {p}[{i}] {actual[p][i]} != reference {expected[p][i]}");
            }
        }
    }

    /// <summary>
    /// The state the residency theory cannot reach through a public flow (the eager step detaches a stale device
    /// gradient before backward; a replay keeps gradient versions current): a gradient bound to a device copy that is
    /// NOT current while both tensors have buffers. The optimizer reads the host gradient then, so the term must land
    /// there - the old rule wrote it into the stale device copy and the step trained without L2.
    /// </summary>
    [SkippableFact]
    public void L2_lands_in_the_host_gradient_when_the_device_copy_is_stale()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            var x = Rand([Batch, Inputs], 1, 1f);
            var y = Rand([Batch, Outputs], 2, 1f);
            var parameters = new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]), Rand([Hidden, Outputs], 4, 0.3f), new Tensor<float>([Outputs]) };
            foreach (var t in parameters) t.Gpu();
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                Loss(gpu, x, y, parameters, composedMse: true);
                plan = scope.CompileTraining(parameters);
            }
            using (plan)
            {
                var compiled = (CompiledTrainingPlan<float>)plan;
                compiled.DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
                plan.SetL2Regularization(0.05);
                FeedBatch(x, y, 0);
                plan.Step();

                var backend = gpu.GetBackend()!;
                var before = new float[plan.Gradients.Length][];
                for (int p = 0; p < plan.Gradients.Length; p++)
                {
                    var g = plan.Gradients[p];
                    Assert.NotNull(parameters[p].TryGetGpuBuffer());
                    before[p] = g.ToArray();
                    g._gpuBuffer?.Dispose();
                    g._gpuBuffer = backend.AllocateBuffer(new float[g.Length]);
                    g._gpuBackend = backend;
                    g._gpuBufferVersion = -1;   // bound, but not the current value
                }
                typeof(CompiledTrainingPlan<float>)
                    .GetMethod("ApplyL2Regularization", BindingFlags.NonPublic | BindingFlags.Instance)!
                    .Invoke(compiled, null);

                for (int p = 0; p < plan.Gradients.Length; p++)
                {
                    var g = plan.Gradients[p];
                    var theta = parameters[p].ToArray();
                    var host = g.GetBackingArrayForCacheLookupUnsafe()!;
                    for (int i = 0; i < theta.Length; i++)
                        Assert.True(Math.Abs(before[p][i] + 0.05f * theta[i] - host[i]) <= 1e-6f,
                            $"grad {p}[{i}]: host {host[i]} != {before[p][i]} + 0.05*{theta[i]}");
                    Assert.NotEqual(g._gpuBufferVersion, g.GpuCacheVersion);   // the stale copy is not revived
                }
            }
        }
    }

    [SkippableFact]
    public void An_input_that_lost_its_captured_binding_does_not_replay_a_stale_batch()
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
            var dropped = Train(gpu, capture: true, droppedTargetBinding: true);
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(eager.Losses[s] - dropped.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: loss {dropped.Losses[s]:G6} != eager {eager.Losses[s]:G6}");
            for (int p = 0; p < eager.FinalWeights.Length; p++)
                for (int i = 0; i < eager.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(eager.FinalWeights[p][i] - dropped.FinalWeights[p][i]) <= 1e-5f,
                        $"param {p}[{i}]: {dropped.FinalWeights[p][i]} != eager {eager.FinalWeights[p][i]}");
        }
    }

    [SkippableFact]
    public void An_input_bound_to_a_released_buffer_neither_blocks_capture_nor_corrupts_training()
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
            var released = Train(gpu, capture: true, releasedInputBinding: true);
            Assert.True(released.ReplayedAGraph, "a released input binding kept the step from ever capturing");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(eager.Losses[s] - released.Losses[s]) <= 1e-5 * Math.Max(1, Math.Abs(eager.Losses[s])),
                    $"step {s}: loss {released.Losses[s]:G6} != eager {eager.Losses[s]:G6}");
            for (int p = 0; p < eager.FinalWeights.Length; p++)
                for (int i = 0; i < eager.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(eager.FinalWeights[p][i] - released.FinalWeights[p][i]) <= 1e-5f,
                        $"param {p}[{i}]: {released.FinalWeights[p][i]} != eager {eager.FinalWeights[p][i]}");
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
            if (capture)
            {
                Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                    "Whole-step graph capture is CUDA-only.");
                Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                    "Graph capture is disabled for this process.");
            }
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
                if (capture)
                {
                    // A capture that failed falls back to eager and would agree trivially.
                    var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                        .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
                    Assert.True(exec != IntPtr.Zero, "no graph was captured, so the capture rows compared eager with eager");
                }
            }
        }
    }

    /// <summary>
    /// The tape ground truth for gradients that receive two contributions per step: the hidden activation feeds two
    /// layers and the output weight is shared by a fused linear and a plain matmul. In a resident GPU step the backward
    /// GEMMs write straight into the gradient buffers, overwriting on a buffer's first contribution of the step and
    /// adding on the later one (GEMM beta 0 / 1); a step that overwrote where it should add, or added onto the previous
    /// step's stale gradient, would still agree with its own eager twin, so only the tape can catch it.
    /// <para>The residual network covers the plan's other shortcut: a residual branch's output is used only by the
    /// add, so its gradient buffer IS the add's output gradient (no copy), while the residual stream's gets a copy
    /// plus the branch's contribution. The wide log-softmax network covers the step's pooled scratch: a log-softmax
    /// over more classes than the fused kernel takes is composed from row reductions whose scratch buffers the step
    /// reuses every replay.</para>
    /// </summary>
    [SkippableTheory]
    [InlineData(false, TwoPathNetwork.SharedWeightAndFanOut)]
    [InlineData(true, TwoPathNetwork.SharedWeightAndFanOut)]
    [InlineData(false, TwoPathNetwork.Residual)]
    [InlineData(true, TwoPathNetwork.Residual)]
    [InlineData(false, TwoPathNetwork.WideLogSoftmax)]
    [InlineData(true, TwoPathNetwork.WideLogSoftmax)]
    [InlineData(false, TwoPathNetwork.ResidualLayerNorm)]
    [InlineData(true, TwoPathNetwork.ResidualLayerNorm)]
    [InlineData(false, TwoPathNetwork.SharedBias)]
    [InlineData(true, TwoPathNetwork.SharedBias)]
    [InlineData(false, TwoPathNetwork.SoftmaxCrossEntropy)]
    [InlineData(true, TwoPathNetwork.SoftmaxCrossEntropy)]
    [InlineData(false, TwoPathNetwork.StandaloneRelu)]
    [InlineData(true, TwoPathNetwork.StandaloneRelu)]
    [InlineData(false, TwoPathNetwork.Pooling)]
    [InlineData(true, TwoPathNetwork.Pooling)]
    [InlineData(false, TwoPathNetwork.PreNormResidualStream)]
    [InlineData(true, TwoPathNetwork.PreNormResidualStream)]
    [InlineData(false, TwoPathNetwork.ResidualReadAfterFanOut)]
    [InlineData(true, TwoPathNetwork.ResidualReadAfterFanOut)]
    [InlineData(false, TwoPathNetwork.MeanAroundKeptAxis)]
    [InlineData(true, TwoPathNetwork.MeanAroundKeptAxis)]
    [InlineData(false, TwoPathNetwork.MaxPool)]
    [InlineData(true, TwoPathNetwork.MaxPool)]
    public void Every_step_computes_the_tape_gradient_through_the_gpu_step_shortcuts(bool capture, TwoPathNetwork network)
    {
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>[], Tensor<float>> loss = network switch
        {
            TwoPathNetwork.Residual => ResidualLoss,
            TwoPathNetwork.WideLogSoftmax => WideLogSoftmaxLoss,
            TwoPathNetwork.ResidualLayerNorm => ResidualLayerNormLoss,
            TwoPathNetwork.SharedBias => SharedBiasLoss,
            TwoPathNetwork.SoftmaxCrossEntropy => SoftmaxCrossEntropyLoss,
            TwoPathNetwork.StandaloneRelu => StandaloneReluLoss,
            TwoPathNetwork.Pooling => PoolingLoss,
            TwoPathNetwork.PreNormResidualStream => PreNormResidualStreamLoss,
            TwoPathNetwork.ResidualReadAfterFanOut => ResidualReadAfterFanOutLoss,
            TwoPathNetwork.MeanAroundKeptAxis => MeanAroundKeptAxisLoss,
            TwoPathNetwork.MaxPool => MaxPoolLoss,
            _ => SharedLoss,
        };
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            if (capture)
            {
                Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                    "Whole-step graph capture is CUDA-only.");
                Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                    "Graph capture is disabled for this process.");
            }
            var x = Rand([Batch, Inputs], 1, 1f);
            var y = Rand([Batch, Outputs], 2, 1f);
            var parameters = network switch
            {
                TwoPathNetwork.Residual => new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]),
                    Rand([Hidden, Hidden], 6, 0.3f), Rand([Hidden], 7, 0.1f), Rand([Hidden, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f) },
                TwoPathNetwork.SharedBias => new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]),
                    Rand([Hidden, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f), Rand([Hidden, Outputs], 6, 0.3f) },
                TwoPathNetwork.ResidualLayerNorm => new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]),
                    Ones(Hidden, 8, 0.2f), Rand([Hidden], 9, 0.1f), Rand([Hidden, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f) },
                TwoPathNetwork.StandaloneRelu => new[] { Rand([Inputs, Hidden], 3, 0.3f), Rand([Hidden], 10, 0.1f),
                    Rand([Hidden, Hidden], 6, 0.3f), Rand([Hidden], 7, 0.1f), Rand([Hidden, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f),
                    Rand([Hidden, Outputs], 11, 0.3f) },
                TwoPathNetwork.MaxPool => new[] { Rand([Inputs, Hidden], 3, 0.3f), Rand([Hidden], 10, 0.1f),
                    Rand([Hidden / 4, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f) },
                TwoPathNetwork.MeanAroundKeptAxis => new[] { Rand([Inputs, Hidden], 3, 0.3f), Rand([Hidden], 10, 0.1f),
                    Rand([Hidden, Outputs], 4, 0.3f) },
                TwoPathNetwork.ResidualReadAfterFanOut => new[] { Rand([Inputs, Hidden], 3, 0.3f), Rand([Hidden], 10, 0.1f),
                    Rand([Hidden, Hidden], 6, 0.3f), Rand([Hidden], 7, 0.1f), Rand([Hidden, Hidden], 12, 0.3f), Rand([Hidden], 13, 0.1f),
                    Rand([Hidden, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f), Rand([Hidden, Outputs], 11, 0.3f) },
                TwoPathNetwork.PreNormResidualStream => new[] { Rand([Inputs, Hidden], 3, 0.3f), Rand([Hidden], 10, 0.1f),
                    Ones(Hidden, 8, 0.2f), Rand([Hidden], 9, 0.1f), Rand([Hidden, Hidden], 6, 0.3f), Rand([Hidden], 7, 0.1f),
                    Rand([Hidden, Hidden], 12, 0.3f), Rand([Hidden], 13, 0.1f), Rand([Hidden, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f) },
                TwoPathNetwork.Pooling => new[] { Rand([Inputs, Hidden], 3, 0.3f), Rand([Hidden], 10, 0.1f),
                    Rand([8, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f), Rand([6, Outputs], 11, 0.3f) },
                TwoPathNetwork.WideLogSoftmax => new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]),
                    Rand([Hidden, WideClasses], 6, 0.3f), Rand([WideClasses], 7, 0.1f), Rand([WideClasses, Outputs], 4, 0.1f), Rand([Outputs], 5, 0.1f) },
                _ => new[] { Rand([Inputs, Hidden], 3, 0.3f), new Tensor<float>([Hidden]), Rand([Hidden, Outputs], 4, 0.3f), Rand([Outputs], 5, 0.1f) },
            };
            foreach (var p in parameters) p.Gpu();

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                loss(gpu, x, y, parameters);
                plan = scope.CompileTraining(parameters);
            }
            using (plan)
            {
                var concrete = (CompiledTrainingPlan<float>)plan;
                if (!capture) concrete.DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
                var gradients = typeof(CompiledTrainingPlan<float>)
                    .GetField("_gradients", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(concrete) as Tensor<float>[]
                    ?? throw new InvalidOperationException("CompiledTrainingPlan._gradients was not found.");
                for (int s = 0; s < Steps; s++)
                {
                    FeedBatch(x, y, s);
                    var expected = CpuTapeGradients(x, y, parameters, loss);
                    plan.Step();
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        var actual = gradients[p].TryGetGpuBuffer() is { } buffer && gpu.GetBackend() is { } backend
                            ? backend.DownloadBuffer(buffer)
                            : gradients[p].ToArray();
                        double diff = 0, norm = 0;
                        for (int i = 0; i < expected[p].Length; i++)
                        {
                            double d = actual[i] - expected[p][i];
                            diff += d * d; norm += (double)expected[p][i] * expected[p][i];
                        }
                        Assert.True(norm > 0, $"step {s}, param {p}: the tape gradient is zero, so the comparison proves nothing");
                        Assert.True(Math.Sqrt(diff) <= 1e-3 * Math.Sqrt(norm) + 1e-6,
                            $"step {s}, param {p}: |plan - tape| = {Math.Sqrt(diff):G4}, |tape| = {Math.Sqrt(norm):G4}");
                    }
                }
                if (capture)
                {
                    var exec = typeof(CompiledTrainingPlan<float>)
                        .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(concrete);
                    Assert.True(exec is IntPtr handle && handle != IntPtr.Zero,
                        "no graph was captured, so the capture row compared eager with eager");
                }
            }
        }
    }

    /// <summary>MSE of <c>FusedLinear(h, w2, b2) + h · w2</c> with <c>h = ReLU(x · w0 + b0)</c>: h and w2 each get two
    /// gradient contributions per step.</summary>
    private static Tensor<float> SharedLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var pred = e.TensorAdd(e.FusedLinear(h, w[2], w[3], FusedActivationType.None), e.TensorMatMul(h, w[2]));
        return e.TensorMSELoss(pred, y);
    }

    /// <summary>MSE of a residual block's projection: <c>r = h + ReLU(h · w2 + b2)</c>, <c>pred = r · w4 + b4</c>, with
    /// <c>h = ReLU(x · w0 + b0)</c>.</summary>
    private static Tensor<float> ResidualLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var r = e.TensorAdd(h, e.FusedLinear(h, w[2], w[3], FusedActivationType.ReLU));
        return e.TensorMSELoss(e.FusedLinear(r, w[4], w[5], FusedActivationType.None), y);
    }

    // Wider than DirectGpuTensorEngine's fused log-softmax row limit (256), so the GPU step composes it from row
    // reductions whose scratch comes from the step's pool.
    private const int WideClasses = 300;

    /// <summary>MSE of a projection of <c>LogSoftmax(h · w2 + b2)</c> over <see cref="WideClasses"/> classes, with
    /// <c>h = ReLU(x · w0 + b0)</c>.</summary>
    private static Tensor<float> WideLogSoftmaxLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var logProbabilities = e.TensorLogSoftmax(e.FusedLinear(h, w[2], w[3], FusedActivationType.None), 1);
        return e.TensorMSELoss(e.FusedLinear(logProbabilities, w[4], w[5], FusedActivationType.None), y);
    }

    /// <summary>MSE of a projection of two residual blocks sharing one LayerNorm, <c>r = h + LN(h)</c> and
    /// <c>s = r + LN(r)</c>, with <c>h = ReLU(x · w0 + b0)</c>: each LayerNorm input gets its own and the residual's
    /// gradient, and the shared gamma and beta two contributions each (overwrite, then add).</summary>
    private static Tensor<float> ResidualLayerNormLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var r = e.TensorAdd(h, e.LayerNorm(h, w[2], w[3], 1e-5, out _, out _));
        var s = e.TensorAdd(r, e.LayerNorm(r, w[2], w[3], 1e-5, out _, out _));
        return e.TensorMSELoss(e.FusedLinear(s, w[4], w[5], FusedActivationType.None), y);
    }

    /// <summary>MSE of <c>(h · w2 + b) + (h · w4 + b)</c> with <c>h = ReLU(x · w0 + b0)</c>: the two output layers share
    /// one bias, whose gradient is summed into its accumulator twice (overwrite, then add).</summary>
    private static Tensor<float> SharedBiasLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var pred = e.TensorAdd(e.FusedLinear(h, w[2], w[3], FusedActivationType.None),
            e.FusedLinear(h, w[4], w[3], FusedActivationType.None));
        return e.TensorMSELoss(pred, y);
    }

    // Zeroes the target's first row: an ignored row, which must count in neither the loss nor its denominator.
    private static readonly Tensor<float> IgnoreFirstRow = BuildIgnoreFirstRowMask();

    private static Tensor<float> BuildIgnoreFirstRowMask()
    {
        var mask = new Tensor<float>([Batch, Outputs]);
        for (int i = Outputs; i < mask.Length; i++) mask[i] = 1f;
        return mask;
    }

    /// <summary>TensorSoftmaxCrossEntropy of a two-layer classifier's logits against the batch's targets (soft and
    /// partly negative values, as FeedBatch draws them) with the first row ignored.</summary>
    private static Tensor<float> SoftmaxCrossEntropyLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var logits = e.FusedLinear(h, w[2], w[3], FusedActivationType.None);
        return e.TensorSoftmaxCrossEntropy(logits, e.TensorMultiply(y, IgnoreFirstRow));
    }

    /// <summary>MSE of <c>ReLU(ReLU(p) · w2 + b2) · w4 + b4 + p · w6</c> with <c>p = x · w0 + b0</c>, the ReLUs separate
    /// ops: the outer ReLU's input gradient is its accumulator's first contribution of the step (written straight
    /// in), while p's already holds the later matmul's contribution when the inner ReLU's backward runs (added).</summary>
    private static Tensor<float> StandaloneReluLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var p = e.FusedLinear(x, w[0], w[1], FusedActivationType.None);
        var c = e.TensorReLU(e.FusedLinear(e.TensorReLU(p), w[2], w[3], FusedActivationType.None));
        var pred = e.TensorAdd(e.FusedLinear(c, w[4], w[5], FusedActivationType.None), e.TensorMatMul(p, w[6]));
        return e.TensorMSELoss(pred, y);
    }

    /// <summary>MSE of <c>mean_1(r) · w2 + b2 + sum_2(r) · w4</c> with <c>r = ReLU(x · w0 + b0)</c> viewed as [batch, 6, 8]:
    /// the trailing-axis sum's gradient is r's first contribution (the broadcast writes it straight into the
    /// accumulator) and the middle-axis mean's is added on top.</summary>
    private static Tensor<float> PoolingLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var r = e.Reshape(e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU), new[] { Batch, 6, 8 });
        var mean = e.ReduceMean(r, new[] { 1 }, keepDims: false);
        var sum = e.ReduceSum(r, new[] { 2 }, keepDims: false);
        var pred = e.TensorAdd(e.FusedLinear(mean, w[2], w[3], FusedActivationType.None),
            e.FusedLinear(sum, w[4], w[3], FusedActivationType.None));
        return e.TensorMSELoss(pred, y);
    }

    /// <summary>MSE of a projection of two pre-norm residual blocks, <c>r1 = h + LN(h) · w4 + b5</c> and
    /// <c>r2 = r1 + LN(r1) · w6 + b7</c> (one LayerNorm shared), with <c>h = ReLU(x · w0 + b1)</c>: the residual stream feeds
    /// each add and the next LayerNorm, as a Transformer's does. On the GPU the stream's gradients share the adds' output
    /// gradient buffers (AliasMultiConsumerPassThroughs), the LayerNorms adding into them after the branches read them.</summary>
    private static Tensor<float> PreNormResidualStreamLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var r1 = e.TensorAdd(h, e.FusedLinear(e.LayerNorm(h, w[2], w[3], 1e-5, out _, out _), w[4], w[5], FusedActivationType.None));
        var r2 = e.TensorAdd(r1, e.FusedLinear(e.LayerNorm(r1, w[2], w[3], 1e-5, out _, out _), w[6], w[7], FusedActivationType.None));
        return e.TensorMSELoss(e.FusedLinear(r2, w[8], w[9], FusedActivationType.None), y);
    }

    /// <summary>MSE of <c>r · w6 + b7 + g · w8 + b7</c> with <c>f = h · w2 + b3</c>, <c>g = h · w4 + b5</c> recorded after f, and
    /// <c>r = h + f</c>, for <c>h = ReLU(x · w0 + b1)</c>: f's backward reads r's gradient AFTER g's adds h's contribution, so
    /// h's gradient must not share r's buffer (it would hand f a gradient with g's contribution in it).</summary>
    private static Tensor<float> ResidualReadAfterFanOutLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var f = e.FusedLinear(h, w[2], w[3], FusedActivationType.None);
        var g = e.FusedLinear(h, w[4], w[5], FusedActivationType.None);
        var r = e.TensorAdd(h, f);
        var pred = e.TensorAdd(e.FusedLinear(r, w[6], w[7], FusedActivationType.None), e.FusedLinear(g, w[8], w[7], FusedActivationType.None));
        return e.TensorMSELoss(pred, y);
    }

    /// <summary>MSE of <c>h · w2 + mean_{0,2}(h viewed as [batch, 6, 8])</c> with <c>h = ReLU(x · w0 + b1)</c>: a mean over a
    /// leading and a trailing block around one kept axis, as a convolution's bias gradient sums dY over N, H and W, used
    /// as the projection's bias. In a resident step that reduction runs as sum_around_kept.</summary>
    private static Tensor<float> MeanAroundKeptAxisLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU);
        var bias = e.ReduceMean(e.Reshape(h, new[] { Batch, Outputs, Hidden / Outputs }), new[] { 0, 2 }, keepDims: false);
        return e.TensorMSELoss(e.FusedLinear(h, w[2], bias, FusedActivationType.None), y);
    }

    /// <summary>MSE of a projection of a 2x2 max pool, <c>pool(h viewed as [batch, 3, 4, 4]) · w2 + b3</c> with
    /// <c>h = ReLU(x · w0 + b1)</c>: the pool's input gradient is its accumulator's first contribution, which on the GPU the
    /// per-window backward writes straight in (every cell: the winner's gradient, every other cell 0).</summary>
    private static Tensor<float> MaxPoolLoss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w)
    {
        var h = e.Reshape(e.FusedLinear(x, w[0], w[1], FusedActivationType.ReLU), new[] { Batch, 3, 4, 4 });
        // The int-index overload, as AiDotNet's pooling layer calls it: a compiled plan records it without indices.
        var pooled = e.MaxPool2DWithIndices(h, new[] { 2, 2 }, new[] { 2, 2 }, out _);
        return e.TensorMSELoss(e.FusedLinear(e.Reshape(pooled, new[] { Batch, Hidden / 4 }), w[2], w[3], FusedActivationType.None), y);
    }

    // A LayerNorm gain near one: 1 + noise.
    private static Tensor<float> Ones(int length, int seed, float noise)
    {
        var t = Rand([length], seed, noise);
        for (int i = 0; i < length; i++) t[i] += 1f;
        return t;
    }

    /// <summary>Networks whose gradients the compiled GPU step forms through a shortcut the eager tape does not take.</summary>
    public enum TwoPathNetwork
    {
        /// <summary>An activation feeding two layers; a weight shared by a linear and a matmul.</summary>
        SharedWeightAndFanOut,
        /// <summary>A residual block: the stream gets the add's gradient plus the branch's.</summary>
        Residual,
        /// <summary>A log-softmax over more classes than the fused kernel takes (pooled composed path).</summary>
        WideLogSoftmax,
        /// <summary>Residual blocks sharing a LayerNorm: its gradients written straight into their accumulators.</summary>
        ResidualLayerNorm,
        /// <summary>Two output layers sharing one bias: its gradient's column sums added into the accumulator.</summary>
        SharedBias,
        /// <summary>A classifier trained with the fused softmax cross-entropy (soft, negative and ignored targets).</summary>
        SoftmaxCrossEntropy,
        /// <summary>Separate ReLU ops: one gradient written into its accumulator, one added to an earlier contribution.</summary>
        StandaloneRelu,
        /// <summary>A middle-axis mean and a trailing-axis sum of one activation: broadcast gradients, written then added.</summary>
        Pooling,
        /// <summary>Two pre-norm residual blocks: the residual stream feeds each add and the next LayerNorm.</summary>
        PreNormResidualStream,
        /// <summary>A residual add whose branch is read after another consumer of the stream added into it.</summary>
        ResidualReadAfterFanOut,
        /// <summary>A mean around one kept axis (a convolution's bias-gradient shape), used as a bias.</summary>
        MeanAroundKeptAxis,
        /// <summary>A non-overlapping 2x2 max pool whose input gradient is a first contribution.</summary>
        MaxPool,
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

    private static Tensor<float> Loss(IEngine e, Tensor<float> x, Tensor<float> y, Tensor<float>[] w, bool composedMse,
        FusedActivationType hidden = FusedActivationType.ReLU)
    {
        var h = e.FusedLinear(x, w[0], w[1], hidden);
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

    private static float[][] CpuTapeGradients(Tensor<float> x, Tensor<float> y, Tensor<float>[] parameters, bool composedMse,
        FusedActivationType hidden = FusedActivationType.ReLU)
    {
        var prior = AiDotNetEngine.Current;
        var cpu = new CpuEngine();
        try
        {
            AiDotNetEngine.Current = cpu;
            Tensor<float> Copy(Tensor<float> t) => new Tensor<float>(t.ToArray(), t.Shape.ToArray());
            var w = parameters.Select(Copy).ToArray();
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
            var loss = Loss(cpu, Copy(x), Copy(y), w, composedMse, hidden);
            var grads = tape.ComputeGradients(loss, w);
            return w.Select(p => grads[p].ToArray()).ToArray();
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    private static float[][] CpuTapeGradients(Tensor<float> x, Tensor<float> y, Tensor<float>[] parameters,
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>[], Tensor<float>> loss)
    {
        var prior = AiDotNetEngine.Current;
        var cpu = new CpuEngine();
        try
        {
            AiDotNetEngine.Current = cpu;
            Tensor<float> Copy(Tensor<float> t) => new Tensor<float>(t.ToArray(), t.Shape.ToArray());
            var w = parameters.Select(Copy).ToArray();
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
            var grads = tape.ComputeGradients(loss(cpu, Copy(x), Copy(y), w), w);
            return w.Select(p => grads[p].ToArray()).ToArray();
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}

