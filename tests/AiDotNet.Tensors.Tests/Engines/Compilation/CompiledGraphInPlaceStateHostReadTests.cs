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
    private static CudaBackend RequireCuda()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1"
                || Environment.GetEnvironmentVariable("AIDOTNET_EXPECT_GPU_BACKEND") == "CUDA";
            Assert.False(required, "The validation lane requires a working CUDA backend.");
            Skip.If(true, "No CUDA device available.");
        }
        return backend;
    }

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    [SkippableTheory]
    [InlineData(1)]   // a host read after every step
    [InlineData(3)]   // a host read every third step (a periodic checkpoint)
    public void InPlaceState_HostRead_FollowsEveryGraphReplay(int readEvery)
    {
        using var backendProof = RequireCuda();
        using var engine = new DirectGpuTensorEngine();
        Assert.IsType<CudaBackend>(engine.QueryBackend);
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var x = Filled(new[] { 8, 16 }, 1, 2f);
            var w = Filled(new[] { 16, 4 }, 2, 1f);
            var state = Filled(new[] { 1, 4 }, 3, 1f);
            var initial = state.ToArray();
            const float rate = 0.125f;
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
                    engine.TensorAddInPlace(state, engine.TensorMultiplyScalar(engine.TensorSign(colSum), rate));
                }
                plan = (CompiledTrainingPlan<float>)scope.CompileTraining(new[] { w }, loss);
            }

            // sign(colSum) of the fixed forward, computed on the host.
            var xs = x.ToArray(); var ws = w.ToArray();
            var sign = new float[4];
            for (int j = 0; j < 4; j++)
            {
                double s = 0;
                for (int r = 0; r < 8; r++)
                    for (int k = 0; k < 16; k++) s += xs[r * 16 + k] * ws[k * 4 + j];
                sign[j] = Math.Sign(s);
            }

            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
                const int steps = 12;
                int checkedReads = 0;
                for (int step = 1; step <= steps; step++)
                {
                    plan.Step();
                    if (step % readEvery != 0) continue;
                    var actual = state.ToArray();
                    for (int j = 0; j < 4; j++)
                    {
                        float expected = initial[j] + step * rate * sign[j];
                        Assert.True(Math.Abs(expected - actual[j]) <= 1e-5f * (1 + Math.Abs(expected)),
                            $"step {step} [{j}]: expected {expected:R}, host read {actual[j]:R} " +
                            $"(graph replays {CompiledTrainingPlan<float>.GraphReplayCount - replaysBefore}, " +
                            $"capture failure: {plan.GraphCaptureFailureReason ?? "none"})");
                    }
                    checkedReads++;
                }
                Assert.True(checkedReads >= 4);
                // The case under test is the captured replay: it must actually have run.
                Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > replaysBefore, plan.GraphCaptureFailureReason);
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
}
