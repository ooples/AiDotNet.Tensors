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
/// A mean squared error written as <c>ReduceMean(Square(Subtract(p, y)))</c> - the parity harness's loss and AiDotNet's
/// MeanSquaredErrorLoss - compiles on the GPU engine to one <c>FusedMseLoss</c> node (one launch forward, one backward)
/// and trains exactly like the CPU plan, which keeps the composed ops. A target that is itself trained (a learnable
/// target) must still receive its gradient: the fused backward computes it only when it is required.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledFusedMseLossTests : IDisposable
{
    private const int Batch = 16, Inputs = 12, Outputs = 5, Steps = 8;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledFusedMseLossTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public string[] StepNames = Array.Empty<string>();
    }

    private static Run Train(IEngine engine, bool onDevice, bool learnableTarget, bool keepDims)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Batch, Inputs], 1, 1f);
        var y = Rand([Batch, Outputs], 2, 1f);
        var w = Rand([Inputs, Outputs], 3, 0.3f);
        var b = new Tensor<float>([Outputs]);
        var parameters = learnableTarget ? new[] { w, b, y } : new[] { w, b };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var pred = engine.FusedLinear(x, w, b, FusedActivationType.None);
            engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(pred, y)), new[] { 0, 1 }, keepDims);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1f);
            var run = new Run { Losses = new double[Steps] };
            for (int s = 0; s < Steps; s++)
            {
                Rand(x.Shape.ToArray(), 100 + s, 1f).AsSpan().CopyTo(x.AsWritableSpan());
                if (!learnableTarget) Rand(y.Shape.ToArray(), 200 + s, 1f).AsSpan().CopyTo(y.AsWritableSpan());
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
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public void A_composed_mse_fuses_on_the_gpu_and_trains_like_the_cpu_plan(bool learnableTarget, bool keepDims)
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
            var reference = Train(new CpuEngine(), onDevice: false, learnableTarget, keepDims);
            var fused = Train(gpu!, onDevice: true, learnableTarget, keepDims);
            _output.WriteLine("steps    : " + string.Join(", ", fused.StepNames));
            _output.WriteLine("cpu      : " + string.Join(" ", reference.Losses.Select(l => l.ToString("G7"))));
            _output.WriteLine("gpu fused: " + string.Join(" ", fused.Losses.Select(l => l.ToString("G7"))));

            Assert.Contains("FusedMseLoss", fused.StepNames);
            Assert.DoesNotContain("ReduceMean", fused.StepNames);
            Assert.DoesNotContain("FusedMseLoss", reference.StepNames);   // the CPU plan keeps the composed ops
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
