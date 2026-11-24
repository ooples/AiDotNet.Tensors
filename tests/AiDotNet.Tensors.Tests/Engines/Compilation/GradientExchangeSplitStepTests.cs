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
        var rng = new Random(seed);
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
    private static Result Train(DirectGpuTensorEngine engine, float[] xs, float[] ts, int firstRow, int rows,
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
        var backend = engine.GetBackend()!;
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
                    var local = new float[exchange.TotalLength];
                    exchange.Read(local);
                    result.ExchangedLosses.Add(exchange.ReadLoss());
                    // The weights this step's backward read (the update has not run yet).
                    var weightBuffer = w.TryGetGpuBuffer() ?? throw new InvalidOperationException("w is not on the device");
                    var weights = new float[weightBuffer.Size];
                    backend.DownloadBuffer(weightBuffer, weights);
                    var peer = peerGradient(weights.Take(In * Out).ToArray());
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

    [SkippableFact]
    public void HalfBatchWithAnAveragingExchange_MatchesTheFullBatch_AndTheUpdateGraphReplaysEveryStep()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "No CUDA device.");
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
            Assert.True(rank0.Split, "the step did not capture as a backward graph and an update graph");
            // The plan runs 3 eager warm-up steps, captures on the 4th, and from then on every step launches both graphs.
            Assert.Equal(Steps - 3, rank0.UpdateGraphLaunches);
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
    [InlineData(OptimizerType.SGD, true)]
    [InlineData(OptimizerType.RMSprop, false)]
    public void APeersOverflowAtTheExchange_SkipsTheStepAndBacksOffTheScale_LikeALocalOverflow(OptimizerType optimizer, bool capturedUpdate)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "No CUDA device.");
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
            var backend = gpu.GetBackend()!;
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
                    var buffer = w.TryGetGpuBuffer() ?? throw new InvalidOperationException("w is not on the device");
                    var weights = new float[buffer.Size];
                    backend.DownloadBuffer(buffer, weights);
                    weightsAtExchange.Add(weights.Take(In * Out).ToArray());
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
