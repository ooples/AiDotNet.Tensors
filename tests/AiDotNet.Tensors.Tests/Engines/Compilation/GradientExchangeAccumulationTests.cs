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

    // d/dW1, d/dW2 of (1/Rows) sum((x W1 W2 - t)^2), in double on the host.
    private static float[] HostGradient(float[] xs, float[] weights)
    {
        var w1 = weights.Take(In * Hidden).ToArray();
        var w2 = weights.Skip(In * Hidden).ToArray();
        var g1 = new double[In * Hidden];
        var g2 = new double[Hidden * Out];
        for (int r = 0; r < Rows; r++)
        {
            var h = new double[Hidden];
            for (int j = 0; j < Hidden; j++)
                for (int i = 0; i < In; i++) h[j] += xs[r * In + i] * (double)w1[i * Hidden + j];
            var dy = new double[Out];
            for (int o = 0; o < Out; o++)
            {
                double y = 0;
                for (int j = 0; j < Hidden; j++) y += h[j] * w2[j * Out + o];
                dy[o] = 2.0 / Rows * (y - Target[r * Out + o]);
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
                    Assert.True(BitConverter.SingleToInt32Bits(eager[i]) == BitConverter.SingleToInt32Bits(acc.Combined[w][i]),
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
}
