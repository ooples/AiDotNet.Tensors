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
/// A compiled step with a gradient exchange (PyTorch DDP's all-reduce point) runs as two CUDA graphs, the backward and
/// the update (clip and fused optimizer), with the exchange between their launches. The test plays one rank of two:
/// the exchange averages this rank's half-batch gradient with the other half's (computed at the same weights), which
/// is exactly the full-batch mean gradient, so the trajectory must match a plan trained on the whole batch.
/// </summary>
[Collection("DirectGpuSerial")]
public class GradientExchangeSplitStepTests
{
    private const int Batch = 8, In = 16, Out = 4, Steps = 12;

    private static float[] Values(int count, int seed, float scale)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() - 0.5) * scale).ToArray();
    }

    private static Tensor<float> Rows(float[] all, int width, int first, int count)
        => new Tensor<float>(new[] { count, width }, new Vector<float>(all.Skip(first * width).Take(count * width).ToArray()));

    private sealed class Result
    {
        public List<float> Losses = new();
        public List<float> ExchangedLosses = new();
        public float[] Weights = Array.Empty<float>();
        public long Exchanges, UpdateGraphLaunches;
        public bool Split;
    }

    // loss = (1/rows) * sum((x W - t)^2), trained with fused Adam and a global-norm clip.
    private static Result Train(IEngine engine, float[] xs, float[] ts, int firstRow, int rows,
        Func<float[], float[]>? peerGradient)
    {
        var x = Rows(xs, In, firstRow, rows);
        var t = Rows(ts, Out, firstRow, rows);
        var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Values(In * Out, 5, 0.4f)));
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var d = engine.TensorSubtract(engine.TensorMatMul(x, w), t);
            engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(d, d), null), 1f / rows);
            plan = scope.CompileTraining(new[] { w });
        }
        var result = new Result();
        using (plan)
        {
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
            plan.SetMaxGradNorm(1.0);
            if (peerGradient is not null)
            {
                compiled.SetGradientExchange(exchange =>
                {
                    Assert.Equal(In * Out, exchange.TotalLength);
                    Assert.Same(w, Assert.Single(exchange.Parameters));
                    // A host plan's gradients are host arrays; a device plan keeps its gradient on the device.
                    if (engine is DirectGpuTensorEngine) Assert.Equal(1, exchange.DeviceGradientCount);
                    else Assert.Equal(0, exchange.DeviceGradientCount);
                    var local = new float[exchange.TotalLength];
                    exchange.Read(local);
                    result.ExchangedLosses.Add(exchange.ReadLoss());
                    // The weights this step's backward read (the update has not run yet).
                    var peer = peerGradient(WeightsNow(engine, w));
                    exchange.Write(local.Zip(peer, (a, b) => (a + b) / 2f).ToArray());
                });
            }
            for (int s = 0; s < Steps; s++) result.Losses.Add(plan.Step().ToArray()[0]);
            result.Weights = w.ToArray();
            result.Exchanges = compiled.GradientExchangeCount;
            result.UpdateGraphLaunches = compiled.UpdateGraphReplayCount;
            result.Split = compiled.HasSplitStepGraph;
        }
        return result;
    }

    // The weights where they are current: the device buffer on a GPU engine, the host array on the CPU.
    private static float[] WeightsNow(IEngine engine, Tensor<float> w)
    {
        if (engine is DirectGpuTensorEngine gpu && w.TryGetGpuBuffer() is { } buffer)
        {
            var weights = new float[buffer.Size];
            TestRequire.NotNull(gpu.GetBackend(), "the engine's backend").DownloadBuffer(buffer, weights);
            return weights.Take(In * Out).ToArray();
        }
        return w.ToArray();
    }

    // "cpu": the host engine (the exchange reads and writes host gradients); "device": the machine's DirectGpu engine,
    // whatever its backend (the POCL lane runs this on OpenCL). Null engine: skipped. cuda: the split-graph path exists.
    private static IEngine? ResolveEngine(string where, out bool cuda)
    {
        cuda = false;
        if (where == "cpu") return new CpuEngine();
        // A lane that requires GPU tests (the POCL shard) fails when the device did not come up, instead of skipping.
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

    // d/dW (1/rows) sum((xW - t)^2) = (2/rows) x^T (xW - t), in double on the host.
    private static float[] HostGradient(float[] xs, float[] ts, int firstRow, int rows, float[] w)
    {
        var g = new double[In * Out];
        for (int r = firstRow; r < firstRow + rows; r++)
            for (int o = 0; o < Out; o++)
            {
                double y = 0;
                for (int i = 0; i < In; i++) y += xs[r * In + i] * (double)w[i * Out + o];
                double residual = y - ts[r * Out + o];
                for (int i = 0; i < In; i++) g[i * Out + o] += 2.0 / rows * xs[r * In + i] * residual;
            }
        return g.Select(v => (float)v).ToArray();
    }

    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void HalfBatchWithAnAveragingExchange_MatchesTheFullBatch_AndTheUpdateGraphReplaysEveryStep(string where)
    {
        var gpu = TestRequire.NotNull(ResolveEngine(where, out bool cuda), "the engine");
        using (gpu as IDisposable)
        {
            var xs = Values(Batch * In, 1, 2f);
            var ts = Values(Batch * Out, 2, 2f);

            var full = Train(gpu, xs, ts, 0, Batch, peerGradient: null);
            var rank0 = Train(gpu, xs, ts, 0, Batch / 2, w => HostGradient(xs, ts, Batch / 2, Batch / 2, w));

            // Control: the same half batch without the peer's half (an exchange that keeps the local gradient) does not
            // reach the full-batch weights, so the match above is the exchange's doing.
            var alone = Train(gpu, xs, ts, 0, Batch / 2, w => HostGradient(xs, ts, 0, Batch / 2, w));
            double aloneDiff = full.Weights.Zip(alone.Weights, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"exchanges={rank0.Exchanges} updateGraphLaunches={rank0.UpdateGraphLaunches} split={rank0.Split} "
                + $"maxDiff={full.Weights.Zip(rank0.Weights, (a, b) => Math.Abs(a - b)).Max():G4} controlDiff={aloneDiff:G4}");
            Assert.True(aloneDiff > 1e-3, $"the control matched the full batch ({aloneDiff}): the test cannot tell");
            Assert.Equal(Steps, rank0.Exchanges);   // every step, warm-up and replay alike
            // CUDA captures the step: 3 eager warm-up steps, the capture on the 4th, and from then on every step launches
            // both graphs. Elsewhere every step runs eagerly with the exchange between the backward and the update.
            Assert.Equal(cuda, rank0.Split);
            Assert.Equal(cuda ? Steps - 3 : 0, rank0.UpdateGraphLaunches);
            // The loss the exchange reads (to weight and reduce it across ranks) is the loss the step returns.
            Assert.Equal(rank0.Losses, rank0.ExchangedLosses);
            double maxDiff = full.Weights.Zip(rank0.Weights, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(maxDiff < 1e-4, $"half batch + exchange diverged from the full batch by {maxDiff}");
            Assert.True(full.Weights.Zip(Values(In * Out, 5, 0.4f), (a, b) => Math.Abs(a - b)).Max() > 1e-2,
                "training must move the weights");
        }
    }
    /// <summary>
    /// Loss scaling under an exchange (DDP + mixed precision): a peer that overflowed makes the reduced gradient
    /// non-finite on every rank, and every rank must skip that step and back its scale off. The overflow check therefore
    /// runs on the exchanged gradient, in the update graph; checked on the local gradient before the exchange (the
    /// backward graph), this rank would see finite values, take the step on the non-finite sum, and leave the ranks'
    /// scales and weights apart. RMSprop's update is not captured (the graph holds the backward only and the update runs
    /// after each replay): the scaler then runs uncaptured after the exchange, with the same outcome.
    /// </summary>
    [SkippableTheory]
    [InlineData(OptimizerType.SGD, true, "device")]
    [InlineData(OptimizerType.RMSprop, false, "device")]
    [InlineData(OptimizerType.SGD, false, "cpu")]
    [InlineData(OptimizerType.RMSprop, false, "cpu")]
    public void APeersOverflowAtTheExchange_SkipsTheStepAndBacksOffTheScale_LikeALocalOverflow(OptimizerType optimizer,
        bool capturedUpdateOnCuda, string where)
    {
        var gpu = TestRequire.NotNull(ResolveEngine(where, out bool cuda), "the engine");
        bool capturedUpdate = capturedUpdateOnCuda && cuda;
        using (gpu as IDisposable)
        {
            const int overflowStep = 6, steps = 9;   // past the 3 warm-up steps and the capture: a graph replay
            var xs = Values(Batch * In, 1, 2f);
            var ts = Values(Batch * Out, 2, 2f);
            var x = Rows(xs, In, 0, Batch);
            var t = Rows(ts, Out, 0, Batch);
            var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Values(In * Out, 5, 0.4f)));
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var d = gpu.TensorSubtract(gpu.TensorMatMul(x, w), t);
                gpu.TensorMultiplyScalar(gpu.ReduceSum(gpu.TensorMultiply(d, d), null), 1f / Batch);
                plan = scope.CompileTraining(new[] { w });
            }
            var weightsAtExchange = new List<float[]>();
            int step = 0;
            using (plan)
            {
                var compiled = (CompiledTrainingPlan<float>)plan;
                plan.ConfigureOptimizer(optimizer, learningRate: 1e-2);
                var scaled = (ICompiledTrainingPlanLossScaling)plan;
                scaled.ConfigureLossScaling(new LossScalingOptions { InitialScale = 1024f, GrowthInterval = 1000 });
                compiled.SetGradientExchange(exchange =>
                {
                    weightsAtExchange.Add(WeightsNow(gpu, w));
                    if (++step != overflowStep) return;
                    // The peer's half overflowed: the sum this rank receives is infinite in one element.
                    var g = new float[exchange.TotalLength];
                    exchange.Read(g);
                    g[3] = float.PositiveInfinity;
                    exchange.Write(g);
                });
                for (int s = 0; s < steps; s++) plan.Step();
                var state = Assert.NotNull(scaled.ReadLossScaleState());
                Console.WriteLine($"scale={state.Scale} overflowSteps={state.OverflowSteps} steps={state.Steps} "
                    + $"updateGraphLaunches={compiled.UpdateGraphReplayCount} split={compiled.HasSplitStepGraph}");
                Assert.Equal(capturedUpdate, compiled.HasSplitStepGraph);
                Assert.Equal(capturedUpdate ? steps - 3 : 0, compiled.UpdateGraphReplayCount);
                // The overflowed step changed nothing, the steps around it trained, and the scale backed off once.
                Assert.Equal(weightsAtExchange[overflowStep - 1], weightsAtExchange[overflowStep]);
                Assert.NotEqual(weightsAtExchange[overflowStep - 2], weightsAtExchange[overflowStep - 1]);
                Assert.NotEqual(weightsAtExchange[overflowStep], weightsAtExchange[overflowStep + 1]);
                Assert.All(w.ToArray(), v => Assert.True(!float.IsNaN(v) && !float.IsInfinity(v), "a non-finite update reached the weights"));
                Assert.Equal(1, state.OverflowSteps);
                Assert.Equal(512f, state.Scale);
            }
        }
    }

    /// <summary>
    /// The page-locked exchange path (every gradient and the loss downloaded in one batch, written back stream-ordered
    /// before the update) trains bitwise like the per-buffer Read/Write path, on a two-parameter plan (so the
    /// gradients' offsets in the staging matter) and through the captured update graph. The exchange rewrites the
    /// gradient (halves it and adds a fixed pattern), so a write that did not reach the update would show.
    /// </summary>
    [SkippableTheory]
    [InlineData("cpu")]
    [InlineData("device")]
    public void ThePageLockedExchange_TrainsBitwiseLikeThePerBufferExchange(string where)
    {
        var engine = TestRequire.NotNull(ResolveEngine(where, out bool cuda), "the engine");
        using (engine as IDisposable)
        {
            const int hidden = 8;
            var xs = Values(Batch * In, 1, 2f);
            var ts = Values(Batch * Out, 2, 2f);
            // mode 0: per-buffer Read/Write; 1: the plan's page-locked staging; 2: caller-owned host memory registered in
            // place, written back in two ranges (as a 2-rank shared-memory exchange writes each partition from its owner).
            (List<float> Losses, List<float> ExchangedLosses, float[] Weights, int Staged) Run(int mode)
            {
                bool staged = mode == 1;
                var x = Rows(xs, In, 0, Batch);
                var t = Rows(ts, Out, 0, Batch);
                var w1 = new Tensor<float>(new[] { In, hidden }, new Vector<float>(Values(In * hidden, 5, 0.4f)));
                var w2 = new Tensor<float>(new[] { hidden, Out }, new Vector<float>(Values(hidden * Out, 6, 0.4f)));
                ICompiledTrainingPlan<float> plan;
                using (var scope = GraphMode.Enable())
                {
                    var d = engine.TensorSubtract(engine.TensorMatMul(engine.TensorMatMul(x, w1), w2), t);
                    engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(d, d), null), 1f / Batch);
                    plan = scope.CompileTraining(new[] { w1, w2 });
                }
                var losses = new List<float>();
                var exchanged = new List<float>();
                int stagedSteps = 0;
                using (plan)
                {
                    var compiled = (CompiledTrainingPlan<float>)plan;
                    plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2);
                    plan.SetMaxGradNorm(1.0);
                    IntPtr external = IntPtr.Zero;
                    AiDotNet.Tensors.Engines.DirectGpu.IGradientHostStaging? registered = null;
                    compiled.SetGradientExchange(exchange =>
                    {
                        Assert.Equal(In * hidden + hidden * Out, exchange.TotalLength);
                        if (mode == 2)
                        {
                            int total = exchange.TotalLength;
                            if (external == IntPtr.Zero)
                            {
                                // Page-aligned, a page-multiple long, with the gradients at a nonzero offset.
                                external = System.Runtime.InteropServices.Marshal.AllocHGlobal(3 * 4096);
                                long aligned = ((long)external + 4095) / 4096 * 4096;
                                registered = exchange.TryRegisterHostStaging((IntPtr)aligned, 2 * 1024, out string? why);
                                if (registered is null && cuda) Assert.Fail("registration failed on CUDA: " + why);
                            }
                            if (registered is not null)
                            {
                                const int at = 100;
                                exchange.DownloadInto(registered, at);
                                exchanged.Add(exchange.ReadLoss());
                                stagedSteps++;
                                var h = registered.Host;
                                for (int i = 0; i < total; i++) h[at + i] = h[at + i] * 0.5f + (i % 7) * 1e-3f;
                                int half = total / 2 + 3;
                                exchange.UploadRangeFrom(registered, at, half, total - half);
                                exchange.UploadRangeFrom(registered, at, 0, half);
                                registered.WaitForTransfers();
                                return;
                            }
                        }
                        if (staged && exchange.TryReadStaged(out Span<float> g, out float loss))
                        {
                            stagedSteps++;
                            exchanged.Add(loss);
                            for (int i = 0; i < g.Length; i++) g[i] = g[i] * 0.5f + (i % 7) * 1e-3f;
                            exchange.WriteStaged();
                            return;
                        }
                        var local = new float[exchange.TotalLength];
                        exchange.Read(local);
                        exchanged.Add(exchange.ReadLoss());
                        for (int i = 0; i < local.Length; i++) local[i] = local[i] * 0.5f + (i % 7) * 1e-3f;
                        exchange.Write(local);
                    });
                    for (int s = 0; s < Steps; s++) losses.Add(plan.Step().ToArray()[0]);
                    Assert.Equal(cuda, compiled.HasSplitStepGraph);
                    registered?.Dispose();
                    if (external != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(external);
                }
                return (losses, exchanged, w1.ToArray().Concat(w2.ToArray()).ToArray(), stagedSteps);
            }

            var perBuffer = Run(0);
            var pageLocked = Run(1);
            var external = Run(2);
            Console.WriteLine($"cuda={cuda} stagedSteps={pageLocked.Staged} "
                + $"maxDiff={perBuffer.Weights.Zip(pageLocked.Weights, (a, b) => Math.Abs(a - b)).Max():G4}");
            // CUDA has the page-locked staging (every step, warm-up and replay); other backends and the host fall back.
            Assert.Equal(cuda ? Steps : 0, pageLocked.Staged);
            Assert.Equal(perBuffer.Losses, pageLocked.Losses);
            Assert.Equal(perBuffer.ExchangedLosses, pageLocked.ExchangedLosses);
            Assert.Equal(pageLocked.Losses, pageLocked.ExchangedLosses);
            Assert.Equal(perBuffer.Weights, pageLocked.Weights);
            // Registered caller memory, written back in two ranges: bitwise too (CUDA registers; elsewhere it falls back).
            Assert.Equal(cuda ? Steps : 0, external.Staged);
            Assert.Equal(perBuffer.Losses, external.Losses);
            Assert.Equal(perBuffer.ExchangedLosses, external.ExchangedLosses);
            Assert.Equal(perBuffer.Weights, external.Weights);
            Assert.True(perBuffer.Weights.Zip(Values(In * 8, 5, 0.4f), (a, b) => Math.Abs(a - b)).Max() > 1e-2,
                "training must move the weights");
        }
    }

    // A loss scale state to resume from with no usable scale is refused, not silently replaced by the default.
    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NaN)]
    public void ConfigureLossScaling_RejectsAResumeStateWithoutAFinitePositiveScale(float scale)
    {
        var engine = new CpuEngine();
        var x = Rows(Values(Batch * In, 1, 2f), In, 0, Batch);
        var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Values(In * Out, 5, 0.4f)));
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.ReduceSum(engine.TensorMatMul(x, w), null);
            plan = scope.CompileTraining(new[] { w });
        }
        using (plan)
        {
            var scaled = (ICompiledTrainingPlanLossScaling)plan;
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                scaled.ConfigureLossScaling(new LossScalingOptions(), new LossScaleState(scale, 0, 0, 0)));
        }
    }
}
