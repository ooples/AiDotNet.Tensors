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
/// An encoder layer's attention written as torch writes it - one in-projection, its output split into query, key and
/// value column blocks (Narrow + Reshape), then MultiHeadAttentionCore - compiles on the GPU engine to one
/// <c>MultiHeadAttentionPacked</c> node that reads the blocks in place and writes their gradients straight into the
/// projection's gradient, and trains like the CPU plan, which keeps the composed ops. Blocks that do not exactly cover
/// the projection keep the composed form.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledPackedAttentionTests : IDisposable
{
    private const int Batch = 2, Seq = 24, Model = 32, Heads = 2, Steps = 6;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledPackedAttentionTests(ITestOutputHelper output) => _output = output;

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

    // blocks: how many model-wide column blocks the in-projection has; starts: the query, key and value blocks.
    private static Run Train(IEngine engine, bool onDevice, int blocks, int[] starts)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([Batch * Seq, Model], 1, 1f);
        var y = Rand([Batch * Seq, Model], 2, 1f);
        var inProj = Rand([Model, blocks * Model], 3, 0.3f);
        var inBias = Rand([blocks * Model], 4, 0.1f);
        var outProj = Rand([Model, Model], 5, 0.3f);
        var outBias = new Tensor<float>([Model]);
        var parameters = new[] { inProj, inBias, outProj, outBias };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var qkv = engine.FusedLinear(x, inProj, inBias, FusedActivationType.None);
            Tensor<float> Part(int i) => engine.Reshape(engine.TensorNarrow(qkv, 1, starts[i] * Model, Model), new[] { Batch, Seq, Model });
            var context = engine.MultiHeadAttentionCore(Part(0), Part(1), Part(2), Heads);
            var pred = engine.FusedLinear(engine.Reshape(context, new[] { Batch * Seq, Model }), outProj, outBias, FusedActivationType.None);
            engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(pred, y)), new[] { 0, 1 }, false);
            plan = scope.CompileTraining(parameters);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.2f);
            var run = new Run { Losses = new double[Steps] };
            for (int s = 0; s < Steps; s++)
                run.Losses[s] = plan.Step().ToArray()[0];
            run.FinalWeights = parameters.Select(p => p.ToArray()).ToArray();
            var steps = (CompiledStep<float>[]?)typeof(CompiledTrainingPlan<float>)
                .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan);
            run.StepNames = steps?.Select(st => st.OpName).ToArray() ?? Array.Empty<string>();
            return run;
        }
    }

    [SkippableTheory]
    [InlineData(3, 0, 1, 2, true)]    // torch's order
    [InlineData(3, 2, 0, 1, true)]    // any order of the three blocks
    [InlineData(4, 0, 1, 3, false)]   // a fourth block nothing reads: not covered, kept composed
    public void A_packed_qkv_attention_fuses_on_the_gpu_and_trains_like_the_cpu_plan(int blocks, int q, int k, int v, bool fuses)
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
            var starts = new[] { q, k, v };
            var reference = Train(new CpuEngine(), onDevice: false, blocks, starts);
            var device = Train(gpu!, onDevice: true, blocks, starts);
            _output.WriteLine("steps: " + string.Join(", ", device.StepNames));
            _output.WriteLine("cpu  : " + string.Join(" ", reference.Losses.Select(l => l.ToString("G7"))));
            _output.WriteLine("gpu  : " + string.Join(" ", device.Losses.Select(l => l.ToString("G7"))));

            if (fuses)
            {
                Assert.Contains("MultiHeadAttentionPacked", device.StepNames);
                Assert.DoesNotContain("Narrow", device.StepNames);
                Assert.DoesNotContain("TensorSlice", device.StepNames);
                Assert.DoesNotContain("MultiHeadAttentionCore", device.StepNames);
            }
            else
            {
                Assert.DoesNotContain("MultiHeadAttentionPacked", device.StepNames);
            }
            Assert.DoesNotContain("MultiHeadAttentionPacked", reference.StepNames);
            Assert.True(reference.Losses.Skip(Steps / 2).Average() < reference.Losses.Take(Steps / 2).Average(),
                "the CPU reference did not learn");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(reference.Losses[s] - device.Losses[s]) <= 1e-4 * Math.Max(1, Math.Abs(reference.Losses[s])),
                    $"step {s}: gpu loss {device.Losses[s]:G7} != cpu {reference.Losses[s]:G7}");
            for (int p = 0; p < reference.FinalWeights.Length; p++)
                for (int i = 0; i < reference.FinalWeights[p].Length; i++)
                    Assert.True(Math.Abs(reference.FinalWeights[p][i] - device.FinalWeights[p][i]) <= 1e-4,
                        $"parameter {p}[{i}]: gpu {device.FinalWeights[p][i]} != cpu {reference.FinalWeights[p][i]}");
        }
    }
}
