using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A linear head (no activation) whose only consumer is the mean squared error - every parity network's head and loss -
/// compiles on the GPU engine to one <c>FusedLinearMseLoss</c> node (one launch forward, one backward for dx, dw and db)
/// and trains like the CPU plan, which keeps the composed ops. A hidden layer below the head checks the head's input
/// gradient reaches it; a learnable target takes the composed backward (the kernel gives no target gradient). A head
/// past the kernels' shapes is not fused: it keeps the FusedLinear and FusedMseLoss nodes and still trains like the CPU.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledFusedLinearMseLossTests : IDisposable
{
    private const int Steps = 12;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledFusedLinearMseLossTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    // Targets from a fixed teacher, y = x t, so every shape has something to learn while x changes each step.
    private static void Teach(Tensor<float> x, Tensor<float> teacher, Tensor<float> y)
    {
        int m = x.Shape[0], k = x.Shape[1], n = teacher.Shape[1];
        var xs = x.ToArray();
        var ts = teacher.ToArray();
        var ys = new float[m * n];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                float s = 0;
                for (int i = 0; i < k; i++) s += xs[r * k + i] * ts[i * n + c];
                ys[r * n + c] = s;
            }
        ys.AsSpan().CopyTo(y.AsWritableSpan());
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public string[] StepNames = Array.Empty<string>();
    }

    private static Run Train(IEngine engine, bool onDevice, int batch, int inputs, int outputs, bool hidden, bool bias,
        bool learnableTarget, bool keepDims)
    {
        AiDotNetEngine.Current = engine;
        int headInputs = hidden ? 24 : inputs;
        // Inputs scaled by sqrt(16 / K) past 16 features, so one learning rate trains every shape.
        float xScale = MathF.Min(1f, MathF.Sqrt(16f / inputs));
        var x = Rand([batch, inputs], 1, xScale);
        var teacher = Rand([inputs, outputs], 7, MathF.Sqrt(3f / inputs));
        var y = new Tensor<float>([batch, outputs]);
        Teach(x, teacher, y);
        var w1 = Rand([inputs, headInputs], 4, 0.3f);
        var b1 = Rand([headInputs], 5, 0.1f);
        var w = Rand([headInputs, outputs], 3, 0.3f);
        var b = Rand([outputs], 6, 0.1f);
        var list = new System.Collections.Generic.List<Tensor<float>>();
        if (hidden) { list.Add(w1); list.Add(b1); }
        list.Add(w);
        if (bias) list.Add(b);
        if (learnableTarget) list.Add(y);
        var parameters = list.ToArray();
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = hidden ? engine.FusedLinear(x, w1, b1, FusedActivationType.ReLU) : x;
            var pred = engine.FusedLinear(h, w, bias ? b : null, FusedActivationType.None);
            engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(pred, y)), new[] { 0, 1 }, keepDims);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.3f);
            var run = new Run { Losses = new double[Steps] };
            for (int s = 0; s < Steps; s++)
            {
                Rand(x.Shape.ToArray(), 100 + s, xScale).AsSpan().CopyTo(x.AsWritableSpan());
                if (!learnableTarget) Teach(x, teacher, y);
                run.Losses[s] = plan.Step().ToArray()[0];
            }
            run.FinalWeights = parameters.Select(p => p.ToArray()).ToArray();
            var steps = (CompiledStep<float>[]?)typeof(CompiledTrainingPlan<float>)
                .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan);
            run.StepNames = steps?.Select(st => st.OpName).ToArray() ?? Array.Empty<string>();
            return run;
        }
    }

    [SkippableTheory]
    // The parity LSTM's head ([32, 64] x [64, 10]), with and without a hidden layer below it, without a bias, keepDims.
    [InlineData(32, 64, 10, false, true, false, false)]
    [InlineData(32, 64, 10, true, true, false, false)]
    [InlineData(16, 12, 5, true, false, false, true)]
    // Ragged sizes (K not a multiple of 4, odd M and N), each register variant (K <= 32, 64, 128), 16 outputs.
    [InlineData(7, 13, 3, true, true, false, false)]
    [InlineData(9, 33, 7, false, true, false, false)]
    [InlineData(24, 96, 16, false, true, false, false)]
    // 17 outputs, past the kernels' 16: not fused.
    [InlineData(8, 20, 17, false, true, false, false)]
    // A learnable target: the forward fuses, the backward composes (the target's gradient).
    [InlineData(16, 12, 5, false, true, true, false)]
    // 2048 inputs, past the kernels' shared memory: not fused.
    [InlineData(16, 2048, 8, false, true, false, false)]
    public void A_linear_head_into_mse_fuses_on_the_gpu_and_trains_like_the_cpu_plan(
        int batch, int inputs, int outputs, bool hidden, bool bias, bool learnableTarget, bool keepDims)
    {
        DirectGpuTensorEngine? gpu = null;
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) gpu = candidate; else candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu)
        {
            var reference = Train(new CpuEngine(), onDevice: false, batch, inputs, outputs, hidden, bias, learnableTarget, keepDims);
            var fused = Train(gpu!, onDevice: true, batch, inputs, outputs, hidden, bias, learnableTarget, keepDims);
            _output.WriteLine("steps    : " + string.Join(", ", fused.StepNames));
            _output.WriteLine("cpu      : " + string.Join(" ", reference.Losses.Select(l => l.ToString("G7"))));
            _output.WriteLine("gpu fused: " + string.Join(" ", fused.Losses.Select(l => l.ToString("G7"))));

            bool fits = outputs <= 16 && inputs <= 128;
            Assert.DoesNotContain("ReduceMean", fused.StepNames);
            Assert.DoesNotContain("FusedLinearMseLoss", reference.StepNames);   // the CPU plan keeps the composed ops
            if (fits)
            {
                Assert.Contains("FusedLinearMseLoss", fused.StepNames);
                Assert.DoesNotContain("FusedMseLoss", fused.StepNames);
                Assert.Equal(hidden ? 1 : 0, fused.StepNames.Count(n => n == "FusedLinear"));   // the head is in the fused node
            }
            else
            {
                Assert.DoesNotContain("FusedLinearMseLoss", fused.StepNames);
                Assert.Contains("FusedMseLoss", fused.StepNames);
                Assert.Equal(hidden ? 2 : 1, fused.StepNames.Count(n => n == "FusedLinear"));
            }
            Assert.True(reference.Losses.Skip(Steps / 2).Average() < reference.Losses.Take(Steps / 2).Average() || learnableTarget,
                "the CPU reference did not learn");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(reference.Losses[s] - fused.Losses[s]) <= 1e-4 * Math.Max(1, Math.Abs(reference.Losses[s])),
                    $"step {s}: fused loss {fused.Losses[s]:G7} != cpu {reference.Losses[s]:G7}");
            for (int p = 0; p < reference.FinalWeights.Length; p++)
                for (int i = 0; i < reference.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(reference.FinalWeights[p][i] - fused.FinalWeights[p][i]) <= 1e-4,
                        $"parameter {p}[{i}]: fused {fused.FinalWeights[p][i]} != cpu {reference.FinalWeights[p][i]}");
        }
    }
}
