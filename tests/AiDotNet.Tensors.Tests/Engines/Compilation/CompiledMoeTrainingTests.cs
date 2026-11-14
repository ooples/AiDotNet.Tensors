using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The fused MoE op (<see cref="MixtureOfExperts.SwiGluExperts"/>) inside a compiled training plan, with the
/// auxiliary-loss-free selection bias nudged in place from each step's loads (DeepSeek-V3 style, as HreLm's router
/// does). The oracle is the eager tape loop: per-step losses, final weights and the final bias must match it.
/// </summary>
/// <remarks>
/// The op had no graph node, so a training trace ran it once and froze its output and loads into the plan: every
/// replay returned the capture batch's MoE output, its inputs got no gradient, and the bias took the capture step's
/// nudge every step. HreLm's moe-g model trained that way for 40 steps ended 0.067 nats behind its eager run, and at a
/// near-zero learning rate its validation loss after step 2 already differed from eager.
/// </remarks>
[Collection("DirectGpuSerial")]
public class CompiledMoeTrainingTests : IDisposable
{
    private const int Sequences = 2, SeqLen = 16, N = Sequences * SeqLen, D = 8, G = 8, F = 8, K = 2, Steps = 8;
    private const float Lr = 0.05f, BiasRate = 0.01f;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledMoeTrainingTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static void Feed(Tensor<float> x, Tensor<float> y, int step)
    {
        Rand(x.Shape.ToArray(), 100 + step, 1f).AsSpan().CopyTo(x.AsWritableSpan());
        x.IncrementVersion();
        Rand(y.Shape.ToArray(), 200 + step, 1f).AsSpan().CopyTo(y.AsWritableSpan());
        y.IncrementVersion();
    }

    private sealed class Model
    {
        public readonly Tensor<float> X = new([N, D]), Y = new([N, D]);
        public readonly Tensor<float> Router = Rand([D, G], 1, 0.8f), W1 = Rand([G, D, F], 2, 0.5f),
            W3 = Rand([G, D, F], 3, 0.5f), W2 = Rand([G, F, D], 4, 0.5f);
        public readonly Tensor<float> Bias = new([1, G]);
        public Tensor<float>[] Parameters => new[] { Router, W1, W3, W2 };
        private readonly bool _readMidTrace;

        public Model(IEngine engine, bool readMidTrace = false, bool hostBias = false)
        {
            _readMidTrace = readMidTrace;
            if (engine is DirectGpuTensorEngine)
                foreach (var t in hostBias ? Parameters : Parameters.Concat(new[] { Bias })) t.Gpu();
        }

        /// <summary>The training forward: route, run the experts, nudge the bias toward balance, return the loss.</summary>
        public Tensor<float> Loss(IEngine e)
        {
            var scores = e.Sigmoid(e.TensorMatMul(X, Router));
            var y = MixtureOfExperts.SwiGluExperts(e, X, scores, Bias, W1, W3, W2, K, Sequences, out var loads);
            using (new NoGradScope<float>())
            {
                var total = e.Reshape(e.ReduceSum(loads, [0], true), [1, G]);
                var below = e.TensorAddScalar(e.TensorNegate(total), (float)N * K / G);
                e.TensorAddInPlace(Bias, e.TensorMultiplyScalar(e.TensorSign(below), BiasRate));
            }
            // An op that reads its input eagerly in a trace (HreLm's next RMSNorm on the GPU) realizes the MoE node
            // there, after the nudge was recorded; the node must still read the bias from before the nudge.
            if (_readMidTrace) _ = y.ToArray();
            return e.TensorMSELoss(y, Y);
        }
    }

    private static (double[] Losses, float[][] Weights, float[] Bias) TrainEager(IEngine e)
    {
        var m = new Model(e);
        var losses = new double[Steps];
        for (int s = 0; s < Steps; s++)
        {
            Feed(m.X, m.Y, s);
            using var tape = new GradientTape<float>();
            var loss = m.Loss(e);
            var grads = tape.ComputeGradients(loss, m.Parameters);
            losses[s] = loss.ToArray()[0];
            foreach (var p in m.Parameters)
            {
                var updated = p.ToArray().Zip(grads[p].ToArray(), (w, g) => w - Lr * g).ToArray();
                updated.AsSpan().CopyTo(p.AsWritableSpan());
                p.IncrementVersion();
            }
        }
        return (losses, m.Parameters.Select(p => p.ToArray()).ToArray(), m.Bias.ToArray());
    }

    private static (double[] Losses, float[][] Weights, float[] Bias, bool Replayed) TrainCompiled(IEngine e, bool capture,
        bool readMidTrace = false, bool hopThreads = false, bool hostBias = false)
    {
        var m = new Model(e, readMidTrace, hostBias);
        ICompiledTrainingPlan<float> plan;
        Feed(m.X, m.Y, 0);
        using (var scope = GraphMode.EnableTraining(m.Parameters))
        {
            m.Loss(e);
            plan = scope.CompileTraining(m.Parameters);
        }
        using (plan)
        {
            var concrete = (CompiledTrainingPlan<float>)plan;
            if (!capture) concrete.DisableGraphStep();
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: Lr);
            var losses = new double[Steps];
            for (int s = 0; s < Steps; s++)
            {
                if (s > 0) Feed(m.X, m.Y, s);
                if (!hopThreads) { losses[s] = plan.Step().ToArray()[0]; continue; }
                // An async training loop resumes on another pool thread after an await: each step here runs on a new
                // thread, one after another.
                double stepLoss = double.NaN;
                Exception? failure = null;
                var worker = new System.Threading.Thread(() =>
                {
                    try { stepLoss = plan.Step().ToArray()[0]; } catch (Exception ex) { failure = ex; }
                });
                worker.Start();
                worker.Join();
                if (failure is not null) throw new InvalidOperationException("step " + s + " failed", failure);
                losses[s] = stepLoss;
            }
            var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
            return (losses, m.Parameters.Select(p => p.ToArray()).ToArray(), m.Bias.ToArray(), exec != IntPtr.Zero);
        }
    }

    private void AssertSameTraining(string label, (double[] Losses, float[][] Weights, float[] Bias) expected,
        (double[] Losses, float[][] Weights, float[] Bias) actual, double tolerance)
    {
        _output.WriteLine($"{label} expected " + string.Join(" ", expected.Losses.Select(l => l.ToString("G6"))));
        _output.WriteLine($"{label} actual   " + string.Join(" ", actual.Losses.Select(l => l.ToString("G6"))));
        for (int s = 0; s < Steps; s++)
            Assert.True(Math.Abs(expected.Losses[s] - actual.Losses[s]) <= tolerance * Math.Max(1, Math.Abs(expected.Losses[s])),
                $"{label} step {s}: loss {actual.Losses[s]:G7} != {expected.Losses[s]:G7}");
        for (int p = 0; p < expected.Weights.Length; p++)
            for (int i = 0; i < expected.Weights[p].Length; i++)
                Assert.True(Math.Abs(expected.Weights[p][i] - actual.Weights[p][i]) <= tolerance,
                    $"{label} param {p}[{i}]: {actual.Weights[p][i]} != {expected.Weights[p][i]}");
        for (int i = 0; i < G; i++)
            Assert.True(Math.Abs(expected.Bias[i] - actual.Bias[i]) <= 1e-6f,
                $"{label} bias[{i}]: {actual.Bias[i]} != {expected.Bias[i]}");
    }

    // Positive controls: the batches differ, so frozen MoE outputs would change the losses, and the bias moved in
    // more than one direction, so a repeated capture-step nudge would end somewhere else.
    private static void AssertNotVacuous((double[] Losses, float[][] Weights, float[] Bias) eager)
    {
        Assert.True(eager.Losses.Distinct().Count() == Steps, "the eager losses repeat; the comparison would be vacuous");
        Assert.Contains(eager.Bias, b => Math.Abs(Math.Abs(b) - Steps * BiasRate) > 1e-6f);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Cpu_CompiledMoePlan_TrainsLikeTheEagerTape(bool readMidTrace)
    {
        var cpu = new CpuEngine();
        AiDotNetEngine.Current = cpu;
        var eager = TrainEager(cpu);
        AssertNotVacuous(eager);
        var compiled = TrainCompiled(cpu, capture: false, readMidTrace);
        AssertSameTraining("cpu", eager, (compiled.Losses, compiled.Weights, compiled.Bias), 1e-5);
    }

    [SkippableTheory]
    [InlineData(false, false, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    [InlineData(true, false, true)]
    public void Gpu_CompiledMoePlan_TrainsLikeTheEagerTape(bool capture, bool hopThreads, bool hostBias)
    {
        DirectGpuTensorEngine? gpu = null;
        try
        {
            gpu = new DirectGpuTensorEngine();
            Skip.IfNot(gpu.IsGpuAvailable, "GPU backend did not resolve.");
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "GPU backend did not resolve.");
        }
        using (gpu)
        {
            AiDotNetEngine.Current = gpu!;
            if (capture)
            {
                Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Whole-step graph capture is CUDA-only.");
                Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0", "Graph capture is disabled for this process.");
            }
            var eager = TrainEager(gpu!);
            AssertNotVacuous(eager);
            // A host-only bias (HreLm's router keeps its bias on the host): the in-place nudge took the host path and the
            // MoE re-uploaded the bias inside the capture, so the plan never captured.
            var compiled = TrainCompiled(gpu!, capture, hopThreads: hopThreads, hostBias: hostBias);
            AssertSameTraining(capture ? (hopThreads ? "gpu-captured-hopping" : "gpu-captured") : "gpu", eager, (compiled.Losses, compiled.Weights, compiled.Bias), 1e-4);
            // A thread change between steps is not a second trainer: it used to turn capture off for the engine's life.
            if (capture) Assert.True(compiled.Replayed, "the captured plan is not replaying a graph after warm-up");
        }
    }
}
