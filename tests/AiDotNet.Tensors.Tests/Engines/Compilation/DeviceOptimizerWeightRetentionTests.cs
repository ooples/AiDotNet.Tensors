using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled plan's device optimizer updates its parameters' persistent weight buffers in place, and the update graph
/// captured from it bakes their addresses. A training loop invalidates the engine's weight caches after every step
/// (AiDotNet's fused session does: InvalidateAllWeightCaches, then InvalidateResidentWeightBuffer for every weight with no
/// pending device data). Before the first update, a configured weight has no pending device data, so that invalidation
/// freed the buffer the optimizer held, and the first update -- the captured one, when every step before the capture
/// deferred its update or ran without an optimizer -- wrote freed memory (CUDA 700). The optimizer's weights must stay
/// bound through the invalidation (as torch's optimizer holds its parameters), and the run must match one without it.
/// </summary>
[Collection("DirectGpuSerial")]
public class DeviceOptimizerWeightRetentionTests
{
    private const int Rows = 4, In = 16, Hidden = 8, Out = 4, Steps = 12, Micro = 4;

    private static float[] Values(int count, int seed, float scale)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() - 0.5) * scale).ToArray();
    }

    private static float[] Inputs(int step) => Values(Rows * In, 300 + step, 2f);

    private static DirectGpuTensorEngine? CudaEngine()
    {
        bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1";
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Assert.False(required, "This lane requires GPU tests, and the GPU engine failed: " + ex.Message);
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return null;
        }
        if (!gpu.IsGpuAvailable || gpu.GetBackend() is not CudaBackend)
        {
            gpu.Dispose();
            Assert.False(required, "This lane requires GPU tests, and no CUDA device came up.");
            Skip.If(true, "No CUDA device (the captured update is CUDA's).");
            return null;
        }
        return gpu;
    }

    // The weights: two plain tensors, or two views of one flat parameter buffer.
    private static Tensor<float>[] Weights(bool flat)
    {
        var w1 = new Tensor<float>(new[] { In, Hidden }, new Vector<float>(Values(In * Hidden, 5, 0.4f)));
        var w2 = new Tensor<float>(new[] { Hidden, Out }, new Vector<float>(Values(Hidden * Out, 6, 0.4f)));
        if (!flat) return new[] { w1, w2 };
        var buffer = new ParameterBuffer<float>(new[] { new[] { In, Hidden }, new[] { Hidden, Out } });
        buffer.CopyFrom(new[] { w1, w2 });
        return buffer.CreateAllViews();
    }

    // What the training loop does after a step: drop every derived weight copy, then every weight's resident buffer
    // unless the device holds its current value (a pending download).
    private static void InvalidateLikeATrainingLoop(DirectGpuTensorEngine engine, Tensor<float>[] weights)
    {
        engine.InvalidateAllWeightCaches();
        foreach (var w in weights)
            if (!(w.HasPendingGpuData && w.TryGetGpuBuffer() is not null))
                engine.InvalidateResidentWeightBuffer(w);
    }

    private sealed class Result
    {
        public float[] Weights = Array.Empty<float>();
        public bool Split, HeldWhileConfigured, HeldAfterDispose = true;
        public long Deferred, UpdateLaunches;
        public int OptimizerStep;
    }

    // scenario "deferred": every step defers its update but each window's last (windows of Micro steps), so every
    // step before the capture (the 4th) is deferred and the first update is the captured one. scenario "late": no
    // accumulation, and the optimizer is configured only after the eager warm-up steps, so again the first update is
    // the captured one.
    private static Result Train(DirectGpuTensorEngine engine, bool flat, string scenario, bool invalidate)
    {
        var weights = Weights(flat);
        var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(Inputs(0)));
        var t = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(Values(Rows * Out, 2, 2f)));
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var d = engine.TensorSubtract(engine.TensorMatMul(engine.TensorMatMul(x, weights[0]), weights[1]), t);
            engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(d, d), null), 1f / Rows);
            plan = scope.CompileTraining(weights);
        }
        var result = new Result();
        using (plan)
        {
            var compiled = (CompiledTrainingPlan<float>)plan;
            void Configure()
            {
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
                plan.SetMaxGradNorm(1.0);
                result.HeldWhileConfigured = weights.All(engine.IsDeviceOptimizerWeight);
            }
            int step = 0;
            if (scenario == "deferred")
            {
                Configure();
                compiled.SetGradientExchange(exchange =>
                {
                    if (step % Micro < Micro - 1) exchange.DeferUpdate(1.0);
                    else exchange.CombineAccumulated(1.0 / Micro, 1.0 / Micro);
                });
            }
            for (step = 0; step < Steps; step++)
            {
                if (scenario == "late" && step == 3)
                {
                    Configure();
                    if (invalidate) InvalidateLikeATrainingLoop(engine, weights);
                }
                plan.SetInputs(new[] { new Tensor<float>(new[] { Rows, In }, new Vector<float>(Inputs(step))) });
                plan.Step();
                if (invalidate) InvalidateLikeATrainingLoop(engine, weights);
            }
            result.Weights = weights.SelectMany(w => w.ToArray()).ToArray();
            result.Split = compiled.HasSplitStepGraph;
            result.Deferred = compiled.DeferredUpdateCount;
            result.UpdateLaunches = compiled.UpdateGraphReplayCount;
            result.OptimizerStep = compiled.OptimizerStep;
        }
        result.HeldAfterDispose = weights.Any(engine.IsDeviceOptimizerWeight);
        return result;
    }

    [SkippableTheory]
    [InlineData("deferred", false)]
    [InlineData("deferred", true)]
    [InlineData("late", false)]
    [InlineData("late", true)]
    public void WeightCacheInvalidationBeforeTheFirstUpdate_KeepsTheOptimizersBuffers(string scenario, bool flat)
    {
        using var engine = CudaEngine();
        if (engine is null) return;
        var control = Train(engine, flat, scenario, invalidate: false);
        var invalidated = Train(engine, flat, scenario, invalidate: true);
        var initial = Weights(flat).SelectMany(w => w.ToArray()).ToArray();
        double moved = control.Weights.Zip(initial, (a, b) => Math.Abs(a - b)).Max();
        double diff = control.Weights.Zip(invalidated.Weights, (a, b) => Math.Abs(a - b)).Max();
        Console.WriteLine($"scenario={scenario} flat={flat} split={invalidated.Split} deferred={invalidated.Deferred} "
            + $"optimizerStep={invalidated.OptimizerStep} moved={moved:E3} diff={diff:E3} "
            + $"held={invalidated.HeldWhileConfigured}/{invalidated.HeldAfterDispose}");

        Assert.True(moved > 1e-3, "training must move the weights");
        // Plain weights take the device optimizer (a flat buffer's views keep the host update): it must hold them.
        if (!flat) Assert.True(invalidated.HeldWhileConfigured, "the configured device optimizer must hold its weights");
        int updates = scenario == "deferred" ? Steps / Micro : Steps - 3;
        Assert.Equal(updates, invalidated.OptimizerStep);
        Assert.Equal(scenario == "deferred" ? Steps - Steps / Micro : 0, invalidated.Deferred);
        Assert.Equal(control.Weights, invalidated.Weights);
        Assert.False(invalidated.HeldAfterDispose, "disposing the plan must end its hold on the weights");
    }

    /// <summary>
    /// A buffer freed under the configured optimizer anyway (here every weight cache is dropped, which a host must only
    /// do between runs) makes the update throw a clear error naming the parameter, not write freed memory.
    /// </summary>
    [SkippableFact]
    public void AnOptimizerWeightFreedUnderThePlan_FailsLoudly_NotWithAnIllegalAddress()
    {
        using var engine = CudaEngine();
        if (engine is null) return;
        var weights = Weights(flat: false);
        var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(Inputs(0)));
        var t = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(Values(Rows * Out, 2, 2f)));
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var d = engine.TensorSubtract(engine.TensorMatMul(engine.TensorMatMul(x, weights[0]), weights[1]), t);
            engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(d, d), null), 1f / Rows);
            plan = scope.CompileTraining(weights);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
            engine.DropAllWeightCaches();
            var error = Assert.Throws<InvalidOperationException>(() => plan.Step());
            Console.WriteLine(error.Message);
            Assert.Contains("was released before", error.Message);
        }
        // The device is still usable: nothing wrote freed memory.
        var probe = engine.TensorAdd(new Tensor<float>(new[] { 4 }, new Vector<float>(new float[] { 1, 2, 3, 4 })),
            new Tensor<float>(new[] { 4 }, new Vector<float>(new float[] { 1, 1, 1, 1 })));
        Assert.Equal(new float[] { 2, 3, 4, 5 }, probe.ToArray());
    }
}
