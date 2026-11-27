using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Gradient accumulation through the exchange (PyTorch DDP's <c>no_sync</c>): every micro-batch but the last of a
/// window defers its update into the plan's accumulator, and the last combines the accumulator with its own gradient
/// before the exchange, so the update runs once per window on the token-weighted mean gradient. The combined gradient
/// must be bitwise the eager accumulation <c>acc = g1*n1; acc += gm*nm; acc *= 1/N; acc += g_last*(n_last/N)</c>, with
/// each multiply and add rounded on its own; a deferred step must leave the weights, the optimizer step count and the
/// learning-rate schedule where they were, through the eager warm-up steps, the capture and the graph replays alike.
/// </summary>
[Collection("DirectGpuSerial")]
public class GradientExchangeAccumulationTests
{
    private const int Rows = 4, In = 16, Hidden = 8, Out = 4, Windows = 5;

    private static float[] Values(int count, int seed, float scale)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() - 0.5) * scale).ToArray();
    }

    // Micro-batch m of window w: its inputs, and its (made-up, unequal) valid-token count.
    private static float[] MicroInputs(int window, int micro) => Values(Rows * In, 100 + window * 8 + micro, 2f);

    private static long Tokens(int window, int micro) => 20 + (7 * window + 13 * micro) % 31;

    private static readonly float[] Target = Values(Rows * Out, 2, 2f);

    private static IEngine? ResolveEngine(string where, out bool cuda)
    {
        cuda = false;
        if (where == "cpu") return new CpuEngine();
        bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1";
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Assert.False(required, "This lane requires GPU tests, and the GPU engine failed: " + ex.Message);
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return null;
        }
        if (!gpu.IsGpuAvailable)
        {
            gpu.Dispose();
            Assert.False(required, "This lane requires GPU tests, and no DirectGpu device came up.");
            Skip.If(true, "No GPU device.");
            return null;
        }
        cuda = gpu.GetBackend() is CudaBackend;
        return gpu;
    }

    // The weights where they are current: the device buffer on a GPU engine, the host array on the CPU.
    private static float[] WeightsNow(IEngine engine, Tensor<float> w)
    {
        if (engine is DirectGpuTensorEngine gpu && w.TryGetGpuBuffer() is { } buffer)
        {
            var weights = new float[buffer.Size];
            TestRequire.NotNull(gpu.GetBackend(), "the engine's backend").DownloadBuffer(buffer, weights);
            return weights.Take(w.Length).ToArray();
        }
        return w.ToArray();
    }

    // The eager accumulation, in float, each operation rounded on its own (the order the eager accumulated step uses).
    private static float[] EagerCombined(IReadOnlyList<float[]> raw, IReadOnlyList<long> tokens)
    {
        long total = tokens.Sum();
        float inv = (float)(1.0 / total);
        float lastWeight = (float)((double)tokens[tokens.Count - 1] / total);
        var acc = new float[raw[0].Length];
        for (int m = 0; m < raw.Count - 1; m++)
        {
            float w = (float)tokens[m];
            for (int i = 0; i < acc.Length; i++)
            {
                float weighted = raw[m][i] * w;
                acc[i] = m == 0 ? weighted : acc[i] + weighted;
            }
        }
        var last = raw[raw.Count - 1];
        for (int i = 0; i < acc.Length; i++)
        {
            float scaled = acc[i] * inv;
            float scaledLast = last[i] * lastWeight;
            acc[i] = scaled + scaledLast;
        }
        return acc;
    }

    private sealed class Run
    {
        public readonly List<List<float[]>> Raw = new();          // [window][micro] the step's own gradient
        public readonly List<float[]> Combined = new();           // [window] after CombineAccumulated
        public readonly List<List<float[]>> WeightsAtStep = new(); // [window][micro] weights the step's backward read
        public readonly List<float> Losses = new(), ExchangedLosses = new();
        public float[] FinalWeights = Array.Empty<float>();
        public int OptimizerStep, Skipped;
        public long Exchanges, Deferred, UpdateLaunches;
        public bool Split;
        public bool EpilogueCaptured;
    }

    // mode "accumulate": defer every micro-batch but the last of each window, then combine. mode "replace": one step per
    // window (on the window's last micro-batch) whose exchange writes the given gradients instead (the reference).
    private static Run Train(IEngine engine, int Micro, bool accumulate, IReadOnlyList<float[]>? replacement = null)
    {
        var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(MicroInputs(0, 0)));
        var t = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(Target));
        var w1 = new Tensor<float>(new[] { In, Hidden }, new Vector<float>(Values(In * Hidden, 5, 0.4f)));
        var w2 = new Tensor<float>(new[] { Hidden, Out }, new Vector<float>(Values(Hidden * Out, 6, 0.4f)));
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var d = engine.TensorSubtract(engine.TensorMatMul(engine.TensorMatMul(x, w1), w2), t);
            engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(d, d), null), 1f / Rows);
            plan = scope.CompileTraining(new[] { w1, w2 });
        }
        var run = new Run();
        using (plan)
        {
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
            plan.SetMaxGradNorm(1.0);
            int window = 0, micro = 0;
            compiled.SetGradientExchange(exchange =>
            {
                Assert.Null(exchange.DeferUnavailableReason);
                run.WeightsAtStep[window].Add(WeightsNow(engine, w1).Concat(WeightsNow(engine, w2)).ToArray());
                run.ExchangedLosses.Add(exchange.ReadLoss());
                var g = new float[exchange.TotalLength];
                exchange.Read(g);
                if (!accumulate)
                {
                    exchange.Write(replacement![window]);
                    return;
                }
                run.Raw[window].Add(g);
                Assert.Equal(micro, exchange.AccumulatedSteps);
                if (micro < Micro - 1)
                {
                    exchange.DeferUpdate(Tokens(window, micro));
                    return;
                }
                long total = Enumerable.Range(0, Micro).Sum(m => Tokens(window, m));
                exchange.CombineAccumulated(1.0 / total, (double)Tokens(window, micro) / total);
                Assert.Equal(0, exchange.AccumulatedSteps);
                var combined = new float[exchange.TotalLength];
                exchange.Read(combined);
                run.Combined.Add(combined);
            });
            for (window = 0; window < Windows; window++)
            {
                run.Raw.Add(new List<float[]>());
                run.WeightsAtStep.Add(new List<float[]>());
                for (micro = accumulate ? 0 : Micro - 1; micro < Micro; micro++)
                {
                    plan.SetInputs(new[] { new Tensor<float>(new[] { Rows, In }, new Vector<float>(MicroInputs(window, micro))) });
                    run.Losses.Add(plan.Step().ToArray()[0]);
                }
            }
            run.FinalWeights = w1.ToArray().Concat(w2.ToArray()).ToArray();
            run.OptimizerStep = compiled.OptimizerStep;
            run.Skipped = compiled.NonFiniteStepsSkipped;
            run.Exchanges = compiled.GradientExchangeCount;
            run.Deferred = compiled.DeferredUpdateCount;
            run.UpdateLaunches = compiled.UpdateGraphReplayCount;
            run.Split = compiled.HasSplitStepGraph;
            run.EpilogueCaptured = compiled.EpilogueGraphHandleForTests != IntPtr.Zero;
        }
        return run;
    }

    // d/dW1, d/dW2 of (1/rows) sum((x W1 W2 - t)^2) over the batch's first rows, in double on the host.
    private static float[] HostGradient(float[] xs, float[] weights, int rows = Rows)
    {
        var w1 = weights.Take(In * Hidden).ToArray();
        var w2 = weights.Skip(In * Hidden).ToArray();
        var g1 = new double[In * Hidden];
        var g2 = new double[Hidden * Out];
        for (int r = 0; r < rows; r++)
        {
            var h = new double[Hidden];
            for (int j = 0; j < Hidden; j++)
                for (int i = 0; i < In; i++) h[j] += xs[r * In + i] * (double)w1[i * Hidden + j];
            var dy = new double[Out];
            for (int o = 0; o < Out; o++)
            {
                double y = 0;
                for (int j = 0; j < Hidden; j++) y += h[j] * w2[j * Out + o];
                dy[o] = 2.0 / rows * (y - Target[r * Out + o]);
            }
            for (int j = 0; j < Hidden; j++)
            {
                double dh = 0;
                for (int o = 0; o < Out; o++)
                {
                    g2[j * Out + o] += h[j] * dy[o];
                    dh += w2[j * Out + o] * dy[o];
                }
                for (int i = 0; i < In; i++) g1[i * Hidden + j] += xs[r * In + i] * dh;
            }
        }
        return g1.Concat(g2).Select(v => (float)v).ToArray();
    }

    // micro 4: the capture (the 4th step) is a window's last, so every window's update runs as the update graph. micro 3:
    // the capture is a deferred step (the first of window 1), and window 0 ran before it, eagerly.
    [SkippableTheory]
    [InlineData("cpu", 4)]
    [InlineData("device", 4)]
    [InlineData("cpu", 3)]
    [InlineData("device", 3)]
    public void DeferredMicroBatches_CombineBitwiseLikeTheEagerAccumulation_AndUpdateOncePerWindow(string where, int Micro)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out bool cuda), "the engine");
        using (engine as IDisposable)
        {
            var acc = Train(engine, Micro, accumulate: true);
            Console.WriteLine($"cuda={cuda} split={acc.Split} exchanges={acc.Exchanges} deferred={acc.Deferred} "
                + $"updateLaunches={acc.UpdateLaunches} optimizerStep={acc.OptimizerStep}");

            Assert.Equal(Windows * Micro, acc.Exchanges);
            Assert.Equal(Windows * (Micro - 1), acc.Deferred);
            // One applied step per window; a deferred step advances neither the count nor the schedule.
            Assert.Equal(Windows, acc.OptimizerStep);
            Assert.Equal(0, acc.Skipped);   // nor does it count as a discarded (non-finite) step
            // CUDA captures on the 4th step: every window whose last step comes at or after it updates as the update graph.
            Assert.Equal(cuda, acc.Split);
            Assert.Equal(cuda, acc.EpilogueCaptured);
            Assert.Equal(cuda ? Enumerable.Range(0, Windows).Count(w => (w + 1) * Micro >= 4) : 0, acc.UpdateLaunches);
            // Every step published its own loss, deferred or not.
            Assert.Equal(acc.Losses, acc.ExchangedLosses);

            for (int w = 0; w < Windows; w++)
            {
                // The deferred steps moved no weight: every micro-batch of a window read the window's first weights.
                for (int m = 1; m < Micro; m++) Assert.Equal(acc.WeightsAtStep[w][0], acc.WeightsAtStep[w][m]);
                if (w > 0) Assert.NotEqual(acc.WeightsAtStep[w - 1][0], acc.WeightsAtStep[w][0]);
                // Each raw gradient is its own micro-batch's at those weights.
                for (int m = 0; m < Micro; m++)
                {
                    var expected = HostGradient(MicroInputs(w, m), acc.WeightsAtStep[w][m]);
                    double diff = expected.Zip(acc.Raw[w][m], (a, b) => Math.Abs(a - b)).Max();
                    Assert.True(diff < 1e-4, $"window {w} micro {m}: the step's gradient is off by {diff}");
                }
                // Bitwise: the combined gradient is the eager accumulation of the raw gradients.
                var tokens = Enumerable.Range(0, Micro).Select(m => Tokens(w, m)).ToArray();
                var eager = EagerCombined(acc.Raw[w], tokens);
                for (int i = 0; i < eager.Length; i++)
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(eager[i]) == TestHelpers.MathCompat.SingleToInt32Bits(acc.Combined[w][i]),
                        $"window {w} element {i}: combined {acc.Combined[w][i]:R}, eager accumulation {eager[i]:R}");
            }

            // The update ran on the combined gradient: a plan that takes one step per window with the same gradients
            // written at its exchange follows the same weights (to the rounding of its eager warm-up updates), and one
            // given only the last micro-batch's gradient does not.
            var reference = Train(engine, Micro, accumulate: false, acc.Combined);
            var lastOnly = Train(engine, Micro, accumulate: false, acc.Raw.Select(r => r[Micro - 1]).ToList());
            double refDiff = acc.FinalWeights.Zip(reference.FinalWeights, (a, b) => Math.Abs(a - b)).Max();
            double lastDiff = acc.FinalWeights.Zip(lastOnly.FinalWeights, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"referenceDiff={refDiff:G4} lastOnlyDiff={lastDiff:G4}");
            Assert.Equal(Windows, reference.OptimizerStep);
            Assert.True(refDiff < 1e-5, $"the accumulated plan's weights left the reference by {refDiff}");
            Assert.True(lastDiff > 1e-4, $"the control matched ({lastDiff}): the test cannot tell");
        }
    }

    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void Deferral_IsRefusedUnderLossScaling_AndCombineNeedsAccumulatedSteps(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        {
            var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(MicroInputs(0, 0)));
            var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Values(In * Out, 5, 0.4f)));
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(engine.TensorMatMul(x, w),
                    engine.TensorMatMul(x, w)), null), 1f / Rows);
                plan = scope.CompileTraining(new[] { w });
            }
            using (plan)
            {
                var compiled = (CompiledTrainingPlan<float>)plan;
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
                Exception? combineFailure = null, deferFailure = null;
                string? reason = "unset";
                compiled.SetGradientExchange(exchange =>
                {
                    combineFailure = Record.Exception(() => exchange.CombineAccumulated(1.0, 1.0));
                    reason = exchange.DeferUnavailableReason;
                    deferFailure = Record.Exception(() => exchange.DeferUpdate(1.0));
                });
                ((ICompiledTrainingPlanLossScaling)plan).ConfigureLossScaling(new LossScalingOptions { InitialScale = 8f });
                plan.Step();
                Assert.IsType<InvalidOperationException>(combineFailure);
                Assert.Contains("loss scaling", reason);
                Assert.Contains("loss scaling", Assert.IsType<InvalidOperationException>(deferFailure).Message);
                Assert.Equal(0, compiled.DeferredUpdateCount);
                Assert.Equal(1, compiled.OptimizerStep);
            }
        }
    }

    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void ACallbackThatThrowsAfterDeferring_DropsTheWindow(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        {
            var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(MicroInputs(0, 0)));
            var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Values(In * Out, 5, 0.4f)));
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(engine.TensorMatMul(x, w),
                    engine.TensorMatMul(x, w)), null), 1f / Rows);
                plan = scope.CompileTraining(new[] { w });
            }
            using (plan)
            {
                var compiled = (CompiledTrainingPlan<float>)plan;
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
                int step = 0;
                var held = new List<int>();
                compiled.SetGradientExchange(exchange =>
                {
                    held.Add(exchange.AccumulatedSteps);
                    exchange.DeferUpdate(1.0);
                    // A collective after the local deferral fails on the third step.
                    if (++step == 3) throw new TimeoutException("peer gone");
                });
                plan.Step();
                plan.Step();
                Assert.Throws<TimeoutException>(() => plan.Step());
                Assert.Equal(0, compiled.AccumulatedGradientSteps);   // the window went with the failed step
                plan.Step();
                Assert.Equal(new[] { 0, 1, 2, 0 }, held);
                Assert.Equal(1, compiled.AccumulatedGradientSteps);
            }
        }
    }

    // The tail micro-batch's rows (a batch shape of its own, so a plan of its own), and the full micro-batches before it.
    private const int TailRows = 2, TailMicro = 2;

    private static ICompiledTrainingPlan<float> CompileBatch(IEngine engine, int rows, Tensor<float> w1, Tensor<float> w2,
        bool firstOnly = false)
    {
        var x = new Tensor<float>(new[] { rows, In }, new Vector<float>(MicroInputs(0, 0).Take(rows * In).ToArray()));
        var t = new Tensor<float>(new[] { rows, Out }, new Vector<float>(Target.Take(rows * Out).ToArray()));
        using var scope = GraphMode.Enable();
        var d = engine.TensorSubtract(engine.TensorMatMul(engine.TensorMatMul(x, w1), w2), t);
        engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(d, d), null), 1f / rows);
        return scope.CompileTraining(firstOnly ? new[] { w1 } : new[] { w1, w2 });
    }

    // Each window runs TailMicro full micro-batches through one plan and an uneven tail (TailRows rows) through another,
    // as a training loop with a plan per batch shape does. With one accumulator handed to both plans the tail combines
    // the full micro-batches' gradients, per parameter, bitwise like the eager accumulation. reattach mimics a loop
    // that moves the exchange to the plan it switches to (SetGradientExchange(null) on the one it leaves, which must not
    // drop the shared window); otherwise both stay attached and their graphs capture.
    [SkippableTheory]
    [InlineData("cpu", false)]
    [InlineData("device", false)]
    [InlineData("device", true)]
    public void AnUnevenTailMicroBatch_OnItsOwnPlan_CombinesTheWindowPerParameter(string where, bool reattach)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out bool cuda), "the engine");
        using (engine as IDisposable)
        {
            var shared = TrainWithTail(engine, shareAccumulator: true, reattach, out var run);
            Console.WriteLine($"cuda={cuda} fullSplit={run.FullSplit} tailSplit={run.TailSplit} "
                + $"tailOptimizerStep={run.TailOptimizerStep}");
            Assert.Equal(Windows, run.TailOptimizerStep);   // one applied step per window, on the tail's plan
            if (cuda && !reattach) Assert.True(run.FullSplit && run.TailSplit, "both plans should step as graphs");
            for (int w = 0; w < Windows; w++)
            {
                for (int m = 0; m <= TailMicro; m++)
                {
                    Assert.Equal(run.WeightsAt[w][0], run.WeightsAt[w][m]);   // the deferred steps moved no weight
                    int rows = m < TailMicro ? Rows : TailRows;
                    var expected = HostGradient(MicroInputs(w, m).Take(rows * In).ToArray(), run.WeightsAt[w][m], rows);
                    double diff = expected.Zip(run.Raw[w][m], (a, b) => Math.Abs(a - b)).Max();
                    Assert.True(diff < 1e-4, $"window {w} micro {m}: the step's gradient is off by {diff}");
                }
                var tokens = Enumerable.Range(0, TailMicro + 1).Select(m => Tokens(w, m)).ToArray();
                var eager = EagerCombined(run.Raw[w], tokens);
                for (int i = 0; i < eager.Length; i++)
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(eager[i]) == TestHelpers.MathCompat.SingleToInt32Bits(run.Combined[w][i]),
                        $"window {w} element {i}: combined {run.Combined[w][i]:R}, eager accumulation {eager[i]:R}");
            }
            // The update ran on the combined gradient (a one-step-per-window plan given it follows the same weights), and
            // with each plan's own accumulator (the tail's plan then sees no accumulated steps) the full micro-batches
            // are lost.
            var reference = Train(engine, TailMicro + 1, accumulate: false, run.Combined);
            double refDiff = shared.Zip(reference.FinalWeights, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"referenceDiff={refDiff:G4}");
            Assert.True(refDiff < 1e-5, $"the accumulated plans' weights left the reference by {refDiff}");
            if (reattach)
            {
                // Moving the exchange off a plan whose own accumulator holds the window is refused, not silent.
                var refused = Assert.Throws<InvalidOperationException>(
                    () => TrainWithTail(engine, shareAccumulator: false, reattach, out _));
                Assert.Contains("accumulation window was dropped", refused.Message);
            }
            else
            {
                var own = TrainWithTail(engine, shareAccumulator: false, reattach, out _);
                double ownDiff = shared.Zip(own, (a, b) => Math.Abs(a - b)).Max();
                Console.WriteLine($"ownAccumulatorDiff={ownDiff:G4}");
                Assert.True(ownDiff > 1e-4, $"the per-plan control matched ({ownDiff}): the test cannot tell");
            }
        }
    }

    private sealed class TailRun
    {
        public readonly List<List<float[]>> Raw = new(), WeightsAt = new();
        public readonly List<float[]> Combined = new();
        public int TailOptimizerStep;
        public bool FullSplit, TailSplit;
    }

    private static float[] TrainWithTail(IEngine engine, bool shareAccumulator, bool reattach, out TailRun run)
    {
        var w1 = new Tensor<float>(new[] { In, Hidden }, new Vector<float>(Values(In * Hidden, 5, 0.4f)));
        var w2 = new Tensor<float>(new[] { Hidden, Out }, new Vector<float>(Values(Hidden * Out, 6, 0.4f)));
        var r = run = new TailRun();
        using var accumulation = new GradientAccumulation<float>();
        using var full = CompileBatch(engine, Rows, w1, w2);
        using var tail = CompileBatch(engine, TailRows, w1, w2);
        full.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
        full.SetMaxGradNorm(1.0);
        tail.SetMaxGradNorm(1.0);
        int window = 0, micro = 0;
        Action<GradientExchange<float>> callback = exchange =>
        {
            r.WeightsAt[window].Add(WeightsNow(engine, w1).Concat(WeightsNow(engine, w2)).ToArray());
            var g = new float[exchange.TotalLength];
            exchange.Read(g);
            r.Raw[window].Add(g);
            if (micro < TailMicro)
            {
                exchange.DeferUpdate(Tokens(window, micro));
                return;
            }
            long total = Enumerable.Range(0, TailMicro + 1).Sum(m => Tokens(window, m));
            // As a distributed loop does: combine when the exchange reports accumulated steps.
            if (exchange.AccumulatedSteps > 0)
                exchange.CombineAccumulated(1.0 / total, (double)Tokens(window, micro) / total);
            var c = new float[exchange.TotalLength];
            exchange.Read(c);
            r.Combined.Add(c);
        };
        var share = shareAccumulator ? accumulation : null;
        ((CompiledTrainingPlan<float>)full).SetGradientExchange(callback, share);
        ((CompiledTrainingPlan<float>)tail).SetGradientExchange(callback, share);
        ICompiledTrainingPlan<float>? current = null;
        for (window = 0; window < Windows; window++)
        {
            r.Raw.Add(new List<float[]>());
            r.WeightsAt.Add(new List<float[]>());
            for (micro = 0; micro <= TailMicro; micro++)
            {
                bool isTail = micro == TailMicro;
                int rows = isTail ? TailRows : Rows;
                var plan = isTail ? tail : full;
                if (current is not null && !ReferenceEquals(current, plan))
                {
                    plan.ContinueOptimizerFrom(current);
                    if (reattach)
                    {
                        ((CompiledTrainingPlan<float>)current).SetGradientExchange(null);
                        ((CompiledTrainingPlan<float>)plan).SetGradientExchange(callback, share);
                    }
                }
                current = plan;
                plan.SetInputs(new[] { new Tensor<float>(new[] { rows, In },
                    new Vector<float>(MicroInputs(window, micro).Take(rows * In).ToArray())) });
                plan.Step();
                if (shareAccumulator) Assert.Equal(isTail ? 0 : micro + 1, accumulation.Steps);
            }
        }
        r.TailOptimizerStep = ((CompiledTrainingPlan<float>)tail).OptimizerStep;
        r.FullSplit = ((CompiledTrainingPlan<float>)full).HasSplitStepGraph;
        r.TailSplit = ((CompiledTrainingPlan<float>)tail).HasSplitStepGraph;
        return w1.ToArray().Concat(w2.ToArray()).ToArray();
    }

    // A parameter the window accumulated but the step that ends it has no gradient for: that step cannot update it, so
    // the combine refuses and drops the window instead of updating the others alone.
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void ATailPlanWithoutAnAccumulatedParameter_RefusesTheCombine(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        {
            var w1 = new Tensor<float>(new[] { In, Hidden }, new Vector<float>(Values(In * Hidden, 5, 0.4f)));
            var w2 = new Tensor<float>(new[] { Hidden, Out }, new Vector<float>(Values(Hidden * Out, 6, 0.4f)));
            using var accumulation = new GradientAccumulation<float>();
            using var full = CompileBatch(engine, Rows, w1, w2);
            using var tail = CompileBatch(engine, TailRows, w1, w2, firstOnly: true);
            full.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
            Exception? failure = null;
            ((CompiledTrainingPlan<float>)full).SetGradientExchange(e => e.DeferUpdate(1.0), accumulation);
            ((CompiledTrainingPlan<float>)tail).SetGradientExchange(
                e => failure = Record.Exception(() => e.CombineAccumulated(0.5, 0.5)), accumulation);
            full.Step();
            Assert.Equal(2, accumulation.HeldParameterCountForTests);
            tail.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);   // it trains w1 alone
            var before = WeightsNow(engine, w1);
            Assert.ThrowsAny<Exception>(() => tail.Step());
            Assert.Contains("has no gradient in the step that ends it",
                Assert.IsType<InvalidOperationException>(failure).Message);
            Assert.Equal(0, accumulation.Steps);
            Assert.Equal(0, accumulation.HeldParameterCountForTests);
            Assert.Equal(before, WeightsNow(engine, w1));   // no update ran
        }
    }

    private static ICompiledTrainingPlan<float> LinearPlan(IEngine engine) => CompileSmall(engine, out _);

    private static ICompiledTrainingPlan<float> CompileSmall(IEngine engine, out Tensor<float> w)
    {
        var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(MicroInputs(0, 0)));
        w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Values(In * Out, 5, 0.4f)));
        using var scope = GraphMode.Enable();
        engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(engine.TensorMatMul(x, w),
            engine.TensorMatMul(x, w)), null), 1f / Rows);
        return scope.CompileTraining(new[] { w });
    }

    // A window opened at one loss scale and combined at another would be unscaled as a whole by the new factor.
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void LossScaling_CannotChangeWhileDeferredStepsArePending(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        using (var plan = LinearPlan(engine))
        {
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
            compiled.SetGradientExchange(exchange => exchange.DeferUpdate(1.0));
            plan.Step();
            Assert.Equal(1, compiled.AccumulatedGradientSteps);

            var failure = Assert.Throws<InvalidOperationException>(() =>
                ((ICompiledTrainingPlanLossScaling)plan).ConfigureLossScaling(new LossScalingOptions { InitialScale = 8f }));
            Assert.Contains("pending", failure.Message);
            Assert.Equal(1, compiled.AccumulatedGradientSteps);   // the refusal changed nothing
        }
    }

    // The running callback could refill the accumulator the replacement clears, for the new exchange to combine.
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void TheExchange_CannotBeReplacedFromInsideItsOwnCallback(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        using (var plan = LinearPlan(engine))
        {
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
            Exception? replaceFailure = null;
            compiled.SetGradientExchange(exchange =>
            {
                replaceFailure = Record.Exception(() => compiled.SetGradientExchange(_ => { }));
                exchange.DeferUpdate(1.0);
            });
            plan.Step();
            Assert.IsType<InvalidOperationException>(replaceFailure);
            Assert.Equal(1, compiled.AccumulatedGradientSteps);   // the original exchange kept its window

            // Between steps it is replaced; the plan's own pending window is dropped, and loudly.
            var dropped = Assert.Throws<InvalidOperationException>(() => compiled.SetGradientExchange(null));
            Assert.Contains("accumulation window was dropped", dropped.Message);
            Assert.Equal(0, compiled.AccumulatedGradientSteps);
        }
    }

    // Schedule-Free writes its evaluation point into the parameters before the forward and only its update restores
    // them, so a deferred step would leave the evaluation point in the parameters.
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void Deferral_IsRefusedForAnOptimizerThatTransformsTheParametersBeforeTheForward(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        using (var plan = LinearPlan(engine))
        {
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.ScheduleFreeSGD, learningRate: 1e-2);
            string? reason = null;
            Exception? deferFailure = null;
            compiled.SetGradientExchange(exchange =>
            {
                reason = exchange.DeferUnavailableReason;
                deferFailure = Record.Exception(() => exchange.DeferUpdate(1.0));
            });
            plan.Step();
            Assert.Contains("Schedule-Free", reason);
            Assert.IsType<InvalidOperationException>(deferFailure);
            Assert.Equal(0, compiled.DeferredUpdateCount);
            Assert.Equal(1, compiled.OptimizerStep);
        }
    }

    // The exchange's contract, each broken once: a step that neither defers nor combines with steps pending, a combine
    // after the reduced gradients were written back, and replacing the exchange while the plan's own accumulator holds
    // a window. Each is refused before any update, and the window is dropped.
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void AnExchangeThatBreaksTheWindow_IsRefusedBeforeTheUpdate(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        {
            using var plan = CompileSmall(engine, out var w);
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
            string mode = "defer";
            Exception? inner = null;
            compiled.SetGradientExchange(exchange =>
            {
                if (mode == "defer") { exchange.DeferUpdate(1.0); return; }
                if (mode == "neither") return;
                // "write-then-combine" / "write-then-defer"
                var g = new float[exchange.TotalLength];
                exchange.Read(g);
                exchange.Write(g);
                inner = mode == "write-then-defer"
                    ? Record.Exception(() => exchange.DeferUpdate(1.0))
                    : Record.Exception(() => exchange.CombineAccumulated(0.5, 0.5));
                if (inner is not null) throw inner;
            });
            var before = w.ToArray();

            plan.Step();
            mode = "neither";
            var neither = Assert.Throws<InvalidOperationException>(() => plan.Step());
            Assert.Contains("neither deferred", neither.Message);
            Assert.Equal(0, compiled.AccumulatedGradientSteps);

            mode = "defer";
            plan.Step();
            mode = "write-then-combine";
            Assert.ThrowsAny<Exception>(() => plan.Step());
            Assert.Contains("combined before the exchange writes", Assert.IsType<InvalidOperationException>(inner).Message);

            mode = "write-then-defer";
            Assert.ThrowsAny<Exception>(() => plan.Step());
            Assert.Contains("cannot defer its update", Assert.IsType<InvalidOperationException>(inner).Message);

            mode = "defer";
            plan.Step();
            // The refused combine threw before touching the accumulator, so the window kept its step (a retry may combine).
            Assert.Equal(2, compiled.AccumulatedGradientSteps);
            var replaced = Assert.Throws<InvalidOperationException>(() => compiled.SetGradientExchange(e => e.DeferUpdate(1.0)));
            Assert.Contains("accumulation window was dropped", replaced.Message);
            Assert.Equal(0, compiled.AccumulatedGradientSteps);

            Assert.Equal(0, compiled.OptimizerStep);   // nothing above updated
            Assert.Equal(before, w.ToArray());
        }
    }

    // A deferred step runs no update, so it has no finiteness verdict of its own: LastStepSkippedNonFiniteGradients keeps
    // the last update's, on the eager steps and the captured replays (CUDA captures on the 4th step) alike. Windows of 2
    // until step 7, whose update is poisoned (skipped); then one long window of deferred steps, read only at its end
    // (a read after every step would resolve the verdict each time and hide a stale slot), optionally after the captured
    // graph was dropped and captured again (its replay numbering restarts).
    [SkippableTheory]
    [InlineData("cpu", false, false)]
    [InlineData("device", false, false)]
    [InlineData("device", true, false)]
    [InlineData("device", false, true)]
    public void ADeferredStep_KeepsTheLastUpdatesFinitenessVerdict(string where, bool readEveryStep, bool recapture)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out bool cuda), "the engine");
        using (engine as IDisposable)
        {
            using var plan = CompileSmall(engine, out _);
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
            plan.SetMaxGradNorm(1.0);
            // Long enough that, after a recapture, the new step I/O launches more replays than the old one's last update index
            // before the read: a stale index then names a replay of the new step I/O.
            const int Poisoned = 7, LongWindowEnd = 17;   // steps 8..16 defer, 17 combines
            int step = 0;
            Action<GradientExchange<float>> callback = exchange =>
            {
                bool defer = step < Poisoned + 1 ? step % 2 == 0 : step < LongWindowEnd;
                if (defer) { exchange.DeferUpdate(1.0); return; }
                exchange.CombineAccumulated(0.5, 0.5);
                if (step != Poisoned) return;
                var g = new float[exchange.TotalLength];
                exchange.Read(g);
                g[0] = float.NaN;
                exchange.Write(g);
            };
            compiled.SetGradientExchange(callback);
            var verdicts = new List<bool>();
            for (step = 0; step <= LongWindowEnd; step++)
            {
                if (recapture && step == Poisoned + 1)
                    compiled.SetGradientExchange(callback);   // at a window boundary: drops the captured graphs
                plan.Step();
                if (readEveryStep) verdicts.Add(compiled.LastStepSkippedNonFiniteGradients);
                else if (step == LongWindowEnd - 1)
                    Assert.True(compiled.LastStepSkippedNonFiniteGradients,
                        "after the deferred steps, the verdict must still be the poisoned update's");
            }
            Console.WriteLine($"cuda={cuda} split={compiled.HasSplitStepGraph} recapture={recapture} verdicts={string.Join(",", verdicts)}");
            Assert.False(compiled.LastStepSkippedNonFiniteGradients);   // the long window's update was finite
            if (readEveryStep)
                Assert.Equal(Enumerable.Range(0, LongWindowEnd + 1).Select(s => s >= Poisoned && s < LongWindowEnd).ToList(),
                    verdicts);
            Assert.Equal(cuda, compiled.HasSplitStepGraph);
            Assert.Equal(1, compiled.NonFiniteStepsSkipped);
            Assert.Equal(4, compiled.OptimizerStep);   // updates at 1, 3, 5 and 17 (7 skipped)
        }
    }

    // Plans sharing one accumulator step one at a time: a step while another holds it is refused before its exchange
    // touches the accumulator (and its update), a disposed accumulator refuses every step, and neither an accumulator in
    // use nor one without an exchange is accepted.
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void ASharedAccumulator_RefusesConcurrentAndDisposedUse(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        {
            using var plan = CompileSmall(engine, out var w);
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
            var accumulation = new GradientAccumulation<float>();
            compiled.SetGradientExchange(e => e.DeferUpdate(1.0), accumulation);
            plan.Step();
            Assert.Equal(1, accumulation.Steps);
            accumulation.EnterStep();   // another plan's step is using it
            var busy = Assert.Throws<InvalidOperationException>(() => plan.Step());
            Assert.Contains("in use by another step", busy.Message);
            Assert.Equal(1, accumulation.Steps);
            var disposeBusy = Assert.Throws<InvalidOperationException>(() => accumulation.Dispose());
            Assert.Contains("while a step is using it", disposeBusy.Message);
            accumulation.ExitStep();
            Assert.Throws<ArgumentException>(() => compiled.SetGradientExchange(null, accumulation));
            plan.Step();
            Assert.Equal(2, accumulation.Steps);
            accumulation.Dispose();
            Assert.Throws<ObjectDisposedException>(() => plan.Step());
            Assert.Equal(0, compiled.OptimizerStep);
        }
    }

    // A shared accumulator forgets, at a window's end, the parameters no step of that window used: a loop that builds
    // new plans (new parameters) does not keep the old ones' slots alive.
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void ASharedAccumulator_DropsTheSlotsOfParametersAWindowDidNotUse(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        {
            using var accumulation = new GradientAccumulation<float>();
            int step = 0;
            Action<GradientExchange<float>> callback = exchange =>
            {
                if (step % 2 == 0) exchange.DeferUpdate(1.0);
                else exchange.CombineAccumulated(0.5, 0.5);
            };
            using (var first = CompileSmall(engine, out _))
            {
                first.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
                ((CompiledTrainingPlan<float>)first).SetGradientExchange(callback, accumulation);
                for (step = 0; step < 2; step++) first.Step();
            }
            Assert.Equal(1, accumulation.SlotCountForTests);
            using var second = CompileSmall(engine, out _);
            var compiled = (CompiledTrainingPlan<float>)second;
            second.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
            compiled.SetGradientExchange(callback, accumulation);
            step = 0;
            second.Step();
            Assert.Equal(2, accumulation.SlotCountForTests);   // the old plan's slot stays until the window ends
            step = 1;
            second.Step();
            Assert.Equal(1, accumulation.SlotCountForTests);
            Assert.Equal(1, compiled.OptimizerStep);
        }
    }

    // Gradients another plan deferred into a shared accumulator are not combined by a loss-scaled plan: its update
    // would unscale them by its own scale.
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void ALossScaledPlan_RefusesToCombineASharedWindow(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out _), "the engine");
        using (engine as IDisposable)
        {
            using var accumulation = new GradientAccumulation<float>();
            using var deferring = CompileSmall(engine, out _);
            deferring.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
            ((CompiledTrainingPlan<float>)deferring).SetGradientExchange(e => e.DeferUpdate(1.0), accumulation);
            deferring.Step();
            Assert.Equal(1, accumulation.Steps);
            using var scaled = CompileSmall(engine, out _);
            var compiled = (CompiledTrainingPlan<float>)scaled;
            scaled.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-2);
            ((ICompiledTrainingPlanLossScaling)scaled).ConfigureLossScaling(new LossScalingOptions { InitialScale = 8f });
            Exception? combineFailure = null;
            compiled.SetGradientExchange(e => combineFailure = Record.Exception(() => e.CombineAccumulated(0.5, 0.5)),
                accumulation);
            Assert.Throws<InvalidOperationException>(() => scaled.Step());   // neither deferred nor combined: refused
            var refusal = Assert.IsType<InvalidOperationException>(combineFailure);
            Assert.Contains("cannot combine", refusal.Message);
            Assert.Contains("loss scaling", refusal.Message);
            Assert.Equal(0, compiled.OptimizerStep);
        }
    }
}
