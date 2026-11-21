using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A convolution whose output runs through per-channel steps -- its bias, an inference BatchNorm's broadcast multiply and
/// add, a LeakyReLU or ReLU -- compiles on the GPU to one <c>ConvChannelEpilogue</c> node: the steps run as one pass each
/// way. It must train like the unfused plans (GPU and CPU) with the affine parameters trainable or frozen (a GAN generator
/// step's discriminator), for a convolution and a transposed convolution.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledConvChannelEpilogueStepTests : IDisposable
{
    private const int Channels = 8, Steps = 5;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    public CompiledConvChannelEpilogueStepTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed, float scale, float offset = 0f)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale) + offset;
        return t;
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public string[] StepNames = Array.Empty<string>();
    }

    // chain: 0 bias + LeakyReLU (a discriminator's first layer); 1 bias + scale + shift + LeakyReLU (its inference-BN
    // layers), the scale and shift trainable; 2 the same with them frozen; 3 scale + shift + ReLU, no bias.
    private static Tensor<float> Epilogue(IEngine engine, Tensor<float> h, Tensor<float> bias, Tensor<float> scale,
        Tensor<float> shift, int chain)
    {
        var host = (CpuEngine)engine;
        if (chain != 3) h = engine.TensorChannelBiasAdd(h, bias);
        if (chain >= 1)
        {
            h = host.TensorBroadcastMultiply(h, scale);
            h = host.TensorBroadcastAdd(h, shift);
        }
        return chain == 3 ? engine.ReLU(h) : engine.LeakyReLU(h, 0.2f);
    }

    private static Run Train(IEngine engine, bool onDevice, int batch, int size, int chain, bool transposed)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([batch, Channels, size, size], 1, 1f);
        var y = Rand([batch, Channels], 2, 1f);
        float wScale = 2f / MathF.Sqrt(Channels * 16);
        var w1 = Rand([Channels, Channels, 4, 4], 3, wScale);
        var c1 = Rand([Channels], 5, 0.3f);
        var s1 = Rand([1, Channels, 1, 1], 7, 0.5f, 1f);
        var t1 = Rand([1, Channels, 1, 1], 8, 0.3f);
        var w2 = Rand([Channels, Channels, 4, 4], 4, wScale);
        var c2 = Rand([Channels], 6, 0.3f);
        var s2 = Rand([1, Channels, 1, 1], 9, 0.5f, 1f);
        var t2 = Rand([1, Channels, 1, 1], 10, 0.3f);
        var parameters = chain == 2 ? new[] { w1, c1, w2, c2 } : new[] { w1, c1, s1, t1, w2, c2, s2, t2 };
        if (onDevice)
            foreach (var p in new[] { w1, c1, s1, t1, w2, c2, s2, t2 }) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.Conv2D(x, w1, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });
            h = Epilogue(engine, h, c1, s1, t1, chain);
            h = transposed
                ? engine.ConvTranspose2D(h, w2, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 0, 0 })
                : engine.Conv2D(h, w2, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
            h = Epilogue(engine, h, c2, s2, t2, chain);
            engine.TensorMSELoss(engine.ReduceMean(h, new[] { 2, 3 }, keepDims: false), y);
            plan = scope.CompileTraining(parameters);
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
            var run = new Run { Losses = new double[Steps] };
            for (int s = 0; s < Steps; s++)
            {
                Rand(x.Shape.ToArray(), 100 + s, 1f).AsSpan().CopyTo(x.AsWritableSpan());
                Rand(y.Shape.ToArray(), 200 + s, 1f).AsSpan().CopyTo(y.AsWritableSpan());
                run.Losses[s] = plan.Step().ToArray()[0];
            }
            run.FinalWeights = new[] { w1, c1, s1, t1, w2, c2, s2, t2 }.Select(p => p.ToArray()).ToArray();
            if (plan is CompiledTrainingPlan<float> concrete)
            {
                var steps = (CompiledStep<float>[]?)typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(concrete);
                run.StepNames = steps?.Select(s => s.OpName).ToArray() ?? Array.Empty<string>();
            }
            return run;
        }
    }

    private static Run TrainUnfused(IEngine engine, bool onDevice, int batch, int size, int chain, bool transposed)
    {
        // Suspend keeps the pattern's slot ahead of the conv + bias fusion when it comes back.
        using var suspended = FusionPatternRegistry.Suspend("ConvChannelEpilogue");
        Assert.NotNull(suspended);
        return Train(engine, onDevice, batch, size, chain, transposed);
    }

    private static void AssertTrainsLike(Run reference, Run fused, string label, double tolerance)
    {
        for (int s = 0; s < Steps; s++)
            Assert.True(Math.Abs(reference.Losses[s] - fused.Losses[s]) <= tolerance * Math.Max(1, Math.Abs(reference.Losses[s])),
                $"{label} step {s}: loss {fused.Losses[s]:G7} != unfused {reference.Losses[s]:G7}");
        for (int p = 0; p < reference.FinalWeights.Length; p++)
        {
            double maxErr = 0;
            for (int i = 0; i < reference.FinalWeights[p].Length; i++)
                maxErr = Math.Max(maxErr, Math.Abs(reference.FinalWeights[p][i] - fused.FinalWeights[p][i])
                                          / Math.Max(1.0, Math.Abs(reference.FinalWeights[p][i])));
            Assert.True(maxErr <= tolerance, $"{label} parameter {p}: max rel error {maxErr:E2} after {Steps} steps");
        }
    }

    private static DirectGpuTensorEngine? TryGpu()
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable) return candidate;
            candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        return null;
    }

    [SkippableTheory]
    [InlineData(4, 8, 0, false)]
    [InlineData(4, 8, 1, false)]
    [InlineData(4, 8, 2, false)]
    [InlineData(4, 8, 3, false)]
    [InlineData(4, 8, 1, true)]
    [InlineData(32, 32, 0, false)]
    [InlineData(32, 32, 2, true)]
    public void A_gpu_plan_runs_the_conv_epilogue_in_one_pass_and_trains_like_the_unfused_plans(
        int batch, int size, int chain, bool transposed)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            var unfusedGpu = TrainUnfused(gpu, onDevice: true, batch, size, chain, transposed);
            Assert.DoesNotContain("ConvChannelEpilogue", unfusedGpu.StepNames);
            var fused = Train(gpu, onDevice: true, batch, size, chain, transposed);
            _output.WriteLine("steps unfused: " + string.Join(", ", unfusedGpu.StepNames));
            _output.WriteLine("steps fused  : " + string.Join(", ", fused.StepNames));
            _output.WriteLine("losses unfused: " + string.Join(", ", unfusedGpu.Losses.Select(l => l.ToString("G9"))));
            _output.WriteLine("losses fused  : " + string.Join(", ", fused.Losses.Select(l => l.ToString("G9"))));
            Assert.Equal(2, fused.StepNames.Count(n => n == "ConvChannelEpilogue"));
            Assert.DoesNotContain("LeakyReLU", fused.StepNames);
            Assert.DoesNotContain("TensorBroadcastMultiply", fused.StepNames);
            AssertTrainsLike(unfusedGpu, fused, "gpu fused vs gpu unfused", 1e-5);
            var cpu = TrainUnfused(new CpuEngine(), onDevice: false, batch, size, chain, transposed);
            AssertTrainsLike(cpu, fused, "gpu fused vs cpu unfused", 1e-3);
        }
    }
}
