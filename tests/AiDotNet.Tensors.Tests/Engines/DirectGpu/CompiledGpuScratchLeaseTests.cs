using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A compiled GPU plan leases each backward output's scratch for the action that writes it, instead of keeping a buffer
/// per output for the plan's life. A residual network trained for several SGD steps must reach the same parameters as
/// the CPU plan with leases on and off, and the leased plan must hold fewer scratch bytes.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuScratchLeaseTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static (float[] Parameters, long ScratchBytes) Train(IEngine engine, int steps, bool capture = false)
    {
        // Identical residual blocks h += ReLU(BatchNorm(Conv(h))): each convolution's input gradient is the second
        // contribution to h's gradient (the add passes it the first), so it is written to scratch and added, never
        // redirected, and every block's is the same size. Then a concatenation, whose backward slices its gradient.
        var x = Filled(new[] { 4, 6, 8, 8 }, 1, 2f);
        var parameters = new System.Collections.Generic.List<Tensor<float>>();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = x;
            for (int b = 0; b < 4; b++)
            {
                var k = Filled(new[] { 6, 6, 3, 3 }, 10 + b, 0.2f);
                var gamma = Filled(new[] { 6 }, 20 + b, 1f);
                var beta = Filled(new[] { 6 }, 30 + b, 1f);
                parameters.Add(k); parameters.Add(gamma); parameters.Add(beta);
                var conv = engine.Conv2D(h, k, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
                h = engine.TensorAdd(h, engine.ReLU(engine.BatchNorm(conv, gamma, beta, 1e-5, out _, out _)));
            }
            var cat = engine.TensorConcatenate(new[] { h, x }, 1);
            engine.ReduceSum(engine.TensorMultiply(cat, cat), null);
            plan = scope.CompileTraining(parameters.ToArray());
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.001f);
            // A captured step graph bakes in its scratch addresses, so a capturing plan keeps its slots (CUDA captures
            // by default): both arms run uncaptured, where leasing applies on every backend.
            if (!capture && plan is CompiledTrainingPlan<float> uncaptured) uncaptured.DisableGraphStep();
            for (int s = 0; s < steps; s++) plan.Step();
            long bytes = plan is CompiledTrainingPlan<float> compiled ? compiled.ScratchBufferBytes : -1;
            var all = new System.Collections.Generic.List<float>();
            foreach (var p in parameters) all.AddRange(p.ToArray());
            return (all.ToArray(), bytes);
        }
    }

    [SkippableFact]
    public void Leased_backward_scratch_trains_like_the_cpu_and_holds_fewer_bytes()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        bool previous = CompiledTrainingPlan<float>.LeaseBackwardScratch;
        using (gpu)
        {
            try
            {
                const int steps = 4;
                var expected = Train(new CpuEngine(), steps).Parameters;
                CompiledTrainingPlan<float>.LeaseBackwardScratch = false;
                var kept = Train(gpu, steps);
                CompiledTrainingPlan<float>.LeaseBackwardScratch = true;
                var leased = Train(gpu, steps);

                foreach (var (actual, name) in new[] { (kept.Parameters, "kept"), (leased.Parameters, "leased") })
                {
                    Assert.Equal(expected.Length, actual.Length);
                    for (int i = 0; i < expected.Length; i++)
                        Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-3f * (1f + Math.Abs(expected[i])),
                            $"{name} [{i}]: cpu {expected[i]}, gpu {actual[i]}");
                }
                Assert.True(leased.ScratchBytes > 0 && leased.ScratchBytes < kept.ScratchBytes,
                    $"leased plan holds {leased.ScratchBytes} scratch bytes, kept plan {kept.ScratchBytes}");
            }
            finally
            {
                CompiledTrainingPlan<float>.LeaseBackwardScratch = previous;
            }
        }
    }

    [SkippableFact]
    public void Leased_backward_scratch_under_a_captured_step_graph_trains_identically_and_holds_fewer_bytes()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        bool previous = CompiledTrainingPlan<float>.LeaseBackwardScratchUnderCapture;
        using (gpu)
        {
            try
            {
                // Enough steps for the plan to capture its step and replay the graph many times: a lease assignment the
                // capture baked in that a later run reassigned or freed would corrupt the replayed steps.
                const int steps = 16;
                CompiledTrainingPlan<float>.LeaseBackwardScratchUnderCapture = false;
                var kept = Train(gpu, steps, capture: true);
                long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
                CompiledTrainingPlan<float>.LeaseBackwardScratchUnderCapture = true;
                var leased = Train(gpu, steps, capture: true);
                Skip.If(CompiledTrainingPlan<float>.GraphReplayCount == replaysBefore,
                    "This backend did not capture the step, so there is no graph to bake in a lease.");

                Assert.Equal(kept.Parameters, leased.Parameters);
                Assert.True(leased.ScratchBytes > 0 && leased.ScratchBytes < kept.ScratchBytes,
                    $"leased plan holds {leased.ScratchBytes} scratch bytes, kept plan {kept.ScratchBytes}");
            }
            finally
            {
                CompiledTrainingPlan<float>.LeaseBackwardScratchUnderCapture = previous;
            }
        }
    }
    private static (float[] Parameters, long ScratchBytes) TrainDenseBlocks(IEngine engine, int steps, bool capture)
    {
        // Dense blocks h = cat(h, Conv3x3(ReLU(BatchNorm(h)))), then a 1x1 convolution average-pooled: every 3x3 output
        // is read only by its concatenation and the 1x1 output only by the pool, so none of them is kept for the backward.
        var x = Filled(new[] { 4, 8, 8, 8 }, 2, 2f);
        var parameters = new System.Collections.Generic.List<Tensor<float>>();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = x;
            int channels = 8;
            for (int b = 0; b < 4; b++)
            {
                var gamma = Filled(new[] { channels }, 40 + b, 1f);
                var beta = Filled(new[] { channels }, 50 + b, 1f);
                var k = Filled(new[] { 4, channels, 3, 3 }, 60 + b, 0.2f);
                parameters.Add(gamma); parameters.Add(beta); parameters.Add(k);
                var pre = engine.ReLU(engine.BatchNorm(h, gamma, beta, 1e-5, out _, out _));
                var grown = engine.Conv2D(pre, k, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
                h = engine.TensorConcatenate(new[] { h, grown }, 1);
                channels += 4;
            }
            var squeeze = Filled(new[] { 8, channels, 1, 1 }, 70, 0.2f);
            parameters.Add(squeeze);
            var reduced = engine.Conv2D(h, squeeze, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
            var pooled = engine.AvgPool2D(reduced, new[] { 2, 2 }, new[] { 2, 2 });
            engine.ReduceSum(engine.TensorMultiply(pooled, pooled), null);
            plan = scope.CompileTraining(parameters.ToArray());
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.001f);
            if (!capture && plan is CompiledTrainingPlan<float> uncaptured) uncaptured.DisableGraphStep();
            for (int s = 0; s < steps; s++) plan.Step();
            long bytes = plan is CompiledTrainingPlan<float> compiled ? compiled.ScratchBufferBytes : -1;
            var all = new System.Collections.Generic.List<float>();
            foreach (var p in parameters) all.AddRange(p.ToArray());
            return (all.ToArray(), bytes);
        }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Leased_forward_outputs_no_backward_reads_train_identically_and_hold_fewer_bytes(bool capture)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        bool previous = CompiledTrainingPlan<float>.LeaseForwardOutputs;
        using (gpu)
        {
            try
            {
                // Enough steps for a capturing backend to capture and replay many times (see the backward test above).
                const int steps = 16;
                var expected = TrainDenseBlocks(new CpuEngine(), steps, capture: false).Parameters;
                CompiledTrainingPlan<float>.LeaseForwardOutputs = false;
                var kept = TrainDenseBlocks(gpu, steps, capture);
                CompiledTrainingPlan<float>.LeaseForwardOutputs = true;
                var leased = TrainDenseBlocks(gpu, steps, capture);

                Assert.Equal(kept.Parameters, leased.Parameters);
                Assert.Equal(expected.Length, leased.Parameters.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - leased.Parameters[i]) <= 2e-3f * (1f + Math.Abs(expected[i])),
                        $"[{i}]: cpu {expected[i]}, gpu {leased.Parameters[i]}");
                Assert.True(leased.ScratchBytes > 0 && leased.ScratchBytes < kept.ScratchBytes,
                    $"leased plan holds {leased.ScratchBytes} scratch bytes, kept plan {kept.ScratchBytes}");
            }
            finally
            {
                CompiledTrainingPlan<float>.LeaseForwardOutputs = previous;
            }
        }
    }
}
