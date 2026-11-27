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
/// captured from it bakes their addresses. A training loop may invalidate the engine's weight caches after every step
/// (InvalidateAllWeightCaches, then InvalidateResidentWeightBuffer for every weight with no pending device data). Before
/// the first update, a configured weight has no pending device data, so that invalidation freed the buffer the optimizer
/// held, and the first update -- the captured one, when every step before the capture deferred its update or ran without
/// an optimizer -- wrote freed memory (CUDA 700). The optimizer's buffers must survive the invalidation (as torch's
/// optimizer holds its parameters), the run must match one without it, and host weights written after an invalidation
/// must reach the buffer the optimizer and any replayed graph read.
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

    // The float device loss against the double host loss: the gaps observed were at most 2.4e-7, at losses near 1.6.
    private static double LossTolerance(double want) => 2e-5 * Math.Max(1.0, Math.Abs(want));

    // (1/Rows) sum((x W1 W2 - t)^2) on the host, in double.
    private static double HostLoss(float[] xs, float[] w1, float[] w2, float[] target)
    {
        double sum = 0;
        for (int r = 0; r < Rows; r++)
        {
            var h = new double[Hidden];
            for (int j = 0; j < Hidden; j++)
                for (int i = 0; i < In; i++) h[j] += xs[r * In + i] * (double)w1[i * Hidden + j];
            for (int o = 0; o < Out; o++)
            {
                double y = 0;
                for (int j = 0; j < Hidden; j++) y += h[j] * w2[j * Out + o];
                double d = y - target[r * Out + o];
                sum += d * d;
            }
        }
        return sum / Rows;
    }

    private static ICompiledTrainingPlan<float> Compile(DirectGpuTensorEngine engine, Tensor<float>[] weights, float[] target)
    {
        var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(Inputs(0)));
        var t = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(target));
        using var scope = GraphMode.Enable();
        var d = engine.TensorSubtract(engine.TensorMatMul(engine.TensorMatMul(x, weights[0]), weights[1]), t);
        engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(d, d), null), 1f / Rows);
        return scope.CompileTraining(weights);
    }

    /// <summary>
    /// A caller that invalidates a held weight and then writes new host values (the order a host copy into a weight
    /// uses) must have those values reach the optimizer's buffer: the next step's loss is the loss at the new weights,
    /// on the eager steps and on the graph replays alike.
    /// </summary>
    [SkippableTheory]
    [InlineData("deferred", 2)]   // eager, before the first (captured) update
    [InlineData("deferred", 9)]   // the split graph replaying
    [InlineData("late", 4)]       // eager, the step after the late configure
    [InlineData("late", 9)]       // the step graph replaying
    public void AHostWriteAfterAnInvalidation_ReachesTheOptimizersBuffer(string scenario, int writeAt)
    {
        using var engine = CudaEngine();
        if (engine is null) return;
        var weights = Weights(flat: false);
        var target = Values(Rows * Out, 2, 2f);
        using var plan = Compile(engine, weights, target);
        var compiled = (CompiledTrainingPlan<float>)plan;
        int step = 0;
        if (scenario == "deferred")
        {
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
            compiled.SetGradientExchange(exchange =>
            {
                if (step % Micro < Micro - 1) exchange.DeferUpdate(1.0);
                else exchange.CombineAccumulated(1.0 / Micro, 1.0 / Micro);
            });
        }
        for (step = 0; step < Steps; step++)
        {
            if (scenario == "late" && step == 3) plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
            var xs = Inputs(step);
            double? expected = null;
            if (step == writeAt)
            {
                Assert.True(engine.IsDeviceOptimizerWeight(weights[0]), "the plan's device optimizer must hold its weights");
                // New weights unrelated to the current ones, large enough to move the loss well past the tolerance.
                // A rescale of the current weights (halving W1, negating W2) moved this model's loss by less than
                // the tolerance (7e-5 at step 9 of the late case), so a refresh that never happened passed.
                var oldW1 = weights[0].ToArray();
                var oldW2 = weights[1].ToArray();
                var w1 = Values(In * Hidden, 41, 1.2f);
                var w2 = Values(Hidden * Out, 42, 1.2f);
                engine.InvalidateResidentWeightBuffer(weights[0]);
                engine.InvalidateResidentWeightBuffer(weights[1]);
                for (int i = 0; i < w1.Length; i++) weights[0][i] = w1[i];
                for (int i = 0; i < w2.Length; i++) weights[1][i] = w2[i];
                expected = HostLoss(xs, w1, w2, target);
                double stale = HostLoss(xs, oldW1, oldW2, target);
                Assert.True(Math.Abs(expected.Value - stale) > 100 * LossTolerance(expected.Value),
                    $"the written weights must change the loss far beyond the tolerance (new {expected}, old {stale})");
            }
            plan.SetInputs(new[] { new Tensor<float>(new[] { Rows, In }, new Vector<float>(xs)) });
            long replaysBefore = compiled.PlanGraphReplayCountForTests;
            float loss = plan.Step().ToArray()[0];
            if (expected is { } want)
            {
                bool replayed = compiled.PlanGraphReplayCountForTests > replaysBefore;
                Console.WriteLine($"scenario={scenario} writeAt={writeAt} split={compiled.HasSplitStepGraph} replayed={replayed} loss={loss} host={want}");
                Assert.Equal(writeAt == 9, replayed);   // the case names a replay or an eager step: hold it to that
                Assert.True(Math.Abs(loss - want) <= LossTolerance(want),
                    $"the step after the host write must see the new weights: loss {loss}, host loss {want}");
            }
        }
    }

    /// <summary>
    /// The hold is balanced: a reconfigure replaces the plan's holds rather than adding to them, Dispose ends them, and a
    /// plan dropped without Dispose ends them when it is finalized.
    /// </summary>
    [SkippableFact]
    public void TheHoldIsReleasedOnReconfigureDisposeAndFinalization()
    {
        using var engine = CudaEngine();
        if (engine is null) return;
        Assert.Equal(0, engine.DeviceOptimizerWeightCountForTests);
        var plan = Compile(engine, Weights(flat: false), Values(Rows * Out, 2, 2f));
        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
        Assert.Equal(2, engine.DeviceOptimizerWeightCountForTests);
        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-3);
        Assert.Equal(2, engine.DeviceOptimizerWeightCountForTests);
        plan.Step();
        plan.Dispose();
        Assert.Equal(0, engine.DeviceOptimizerWeightCountForTests);

        DropUndisposedPlan(engine);
        for (int i = 0; i < 10 && engine.DeviceOptimizerWeightCountForTests != 0; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        Assert.Equal(0, engine.DeviceOptimizerWeightCountForTests);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void DropUndisposedPlan(DirectGpuTensorEngine engine)
    {
        var plan = Compile(engine, Weights(flat: false), Values(Rows * Out, 2, 2f));
        plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
        Assert.Equal(2, engine.DeviceOptimizerWeightCountForTests);
    }
}
