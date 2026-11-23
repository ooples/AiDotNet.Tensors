using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled training plan whose forward reads a non-trainable state tensor and then updates it in place (a MoE
/// router's selection bias nudged after every training forward) keeps that state on the device. Once the step runs as
/// a captured CUDA graph, every replay updates the device copy without running host code. A host read of the state
/// must still see the CURRENT device value after every replay, not the value the first host read downloaded: a
/// checkpoint written after an earlier one otherwise saved the earlier step's state.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CompiledGraphInPlaceStateHostReadTests
{
    private const float Rate = 0.125f;

    // Any GPU backend runs the plan device-resident; only CUDA captures and replays the step as a graph.
    private static DirectGpuTensorEngine RequireGpu()
    {
        var engine = new DirectGpuTensorEngine();
        if (engine.SupportsGpu && engine.IsGpuAvailable) return engine;
        engine.Dispose();
        Assert.False(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1",
            "The validation lane requires a working GPU backend.");
        Skip.If(true, "No DirectGpuTensorEngine available.");
        return null!;
    }

    private static bool CapturesGraphs(DirectGpuTensorEngine engine) => engine.QueryBackend is CudaBackend;

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    // A plan whose forward reads `state` and then nudges it in place by Rate * sign(colSum(x @ w)). With a zero
    // learning rate that sign is a constant of the step; it is returned in `sign`, computed on the host.
    private static CompiledTrainingPlan<float> BuildPlan(DirectGpuTensorEngine engine, Tensor<float> state, out float[] sign)
    {
        var x = Filled(new[] { 8, 16 }, 1, 2f);
        var w = Filled(new[] { 16, 4 }, 2, 1f);
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { w }))
        {
            var h = engine.TensorMatMul(x, w);                                   // [8, 4]
            var y = engine.TensorAdd(h, state);                                  // the forward reads the state
            var loss = engine.ReduceSum(engine.TensorMultiply(y, y), null, false);
            using (new NoGradScope<float>())
            {
                // Then nudges it in place by a constant of the step (the learning rate is 0, so h never changes).
                var colSum = engine.ReduceSum(h, new[] { 0 }, true);             // [1, 4]
                engine.TensorAddInPlace(state, engine.TensorMultiplyScalar(engine.TensorSign(colSum), Rate));
            }
            plan = (CompiledTrainingPlan<float>)scope.CompileTraining(new[] { w }, loss);
        }

        var xs = x.ToArray(); var ws = w.ToArray();
        sign = new float[4];
        for (int j = 0; j < 4; j++)
        {
            double s = 0;
            for (int r = 0; r < 8; r++)
                for (int k = 0; k < 16; k++) s += xs[r * 16 + k] * ws[k * 4 + j];
            sign[j] = Math.Sign(s);
        }
        plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
        return plan;
    }

    // The host read of `state` must equal `origin` nudged `steps` times.
    private static void AssertState(Tensor<float> state, float[] origin, int steps, float[] sign, string context)
    {
        var actual = state.ToArray();
        for (int j = 0; j < 4; j++)
        {
            float expected = origin[j] + steps * Rate * sign[j];
            Assert.True(Math.Abs(expected - actual[j]) <= 1e-5f * (1 + Math.Abs(expected)),
                $"{context} [{j}]: expected {expected:R}, host read {actual[j]:R}");
        }
    }

    [SkippableTheory]
    [InlineData(1)]   // a host read after every step
    [InlineData(3)]   // a host read every third step (a periodic checkpoint)
    public void InPlaceState_HostRead_FollowsEveryGraphReplay(int readEvery)
    {
        using var engine = RequireGpu();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var state = Filled(new[] { 1, 4 }, 3, 1f);
            var initial = state.ToArray();
            using var plan = BuildPlan(engine, state, out var sign);
            long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
            const int steps = 12;
            int checkedReads = 0;
            for (int step = 1; step <= steps; step++)
            {
                plan.Step();
                if (step % readEvery != 0) continue;
                AssertState(state, initial, step, sign,
                    $"step {step} (graph replays {CompiledTrainingPlan<float>.GraphReplayCount - replaysBefore}, " +
                    $"capture failure: {plan.GraphCaptureFailureReason ?? "none"})");
                checkedReads++;
            }
            Assert.True(checkedReads >= 4);
            // On CUDA the case under test is the captured replay: it must actually have run.
            if (CapturesGraphs(engine))
                Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > replaysBefore, plan.GraphCaptureFailureReason);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    /// <summary>
    /// A host write to the state between replays (a checkpoint restore) is what the next step continues from. The
    /// re-arm after that step must not let the pre-write device copy overwrite it, and later replays build on it.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]   // the write lands right after a step, before any host read
    [InlineData(true)]    // a host read first downloads the device copy, then the write replaces it
    public void InPlaceState_HostWriteBetweenReplays_IsWhatTheNextStepContinuesFrom(bool readBeforeWrite)
    {
        using var engine = RequireGpu();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var state = Filled(new[] { 1, 4 }, 3, 1f);
            var initial = state.ToArray();
            using var plan = BuildPlan(engine, state, out var sign);
            long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
            const int before = 6, after = 5;
            for (int step = 1; step <= before; step++) plan.Step();
            if (readBeforeWrite) AssertState(state, initial, before, sign, $"step {before}");

            var written = new[] { 3.5f, -2.25f, 0.75f, -1.5f };
            for (int j = 0; j < 4; j++) state[j] = written[j];
            AssertState(state, written, 0, sign, "right after the host write");

            for (int step = 1; step <= after; step++)
            {
                plan.Step();
                AssertState(state, written, step, sign,
                    $"step {step} after the write (capture failure: {plan.GraphCaptureFailureReason ?? "none"})");
            }
            if (CapturesGraphs(engine))
                Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > replaysBefore, plan.GraphCaptureFailureReason);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
}
