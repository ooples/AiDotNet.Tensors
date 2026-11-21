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
/// A convolution or transposed convolution whose channel bias feeds a training BatchNorm + activation (a DCGAN's
/// conv/convT -&gt; bias -&gt; BN -&gt; LeakyReLU/ReLU, a ResNet's conv -&gt; bias -&gt; BN -&gt; ReLU6) compiles on the GPU to
/// one <c>ConvBiasBatchNorm</c> node: the bias cancels in the BatchNorm, so the node skips the bias pass and its gradient
/// reduction. It must train like the unfused plans (GPU and CPU), write the batch means with the bias (the running-mean
/// update reads them), and give the bias its exact zero gradient.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledConvBiasBatchNormStepTests : IDisposable
{
    private const int Channels = 8, Steps = 6;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _output;

    private readonly bool _priorChannelsLast = AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_channelsLastAtFp32;

    // Channels-last is off for FP32 plans by default (measured slower); these tests exercise it where cuDNN takes it.
    public CompiledConvBiasBatchNormStepTests(ITestOutputHelper output)
    {
        _output = output;
        AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_channelsLastAtFp32 = true;
    }

    public void Dispose()
    {
        AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_channelsLastAtFp32 = _priorChannelsLast;
        AiDotNetEngine.Current = _prior;
    }

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static Tensor<float> Fill(int[] shape, float value)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = value;
        return t;
    }

    private sealed class Run
    {
        public double[] Losses = Array.Empty<double>();
        public float[][] FinalWeights = Array.Empty<float[]>();
        public float[][] Statistics = Array.Empty<float[]>();
        public float[] RunningMean = Array.Empty<float>();
        public float[][] Biases = Array.Empty<float[]>();
        public string[] StepNames = Array.Empty<string>();
    }

    // activation: 0 ReLU, 1 LeakyReLU(0.2), 2 ReLU6.
    private static Tensor<float> Activate(IEngine engine, Tensor<float> h, int activation) => activation switch
    {
        0 => engine.ReLU(h),
        1 => engine.LeakyReLU(h, 0.2f),
        _ => engine.TensorReLU6(h),
    };

    private static Run Train(IEngine engine, bool onDevice, int batch, int size, int activation)
    {
        AiDotNetEngine.Current = engine;
        var x = Rand([batch, Channels, size, size], 1, 1f);
        var y = Rand([batch, Channels], 2, 1f);
        float wScale = 2f / MathF.Sqrt(Channels * 16);
        // A strided 4x4 convolution (a discriminator's), then a 4x4/2 transposed convolution back up (a generator's).
        var w1 = Rand([Channels, Channels, 4, 4], 3, wScale);
        var c1 = Rand([Channels], 5, 0.5f);
        var g1 = Fill([Channels], 1.5f);
        var b1 = Fill([Channels], 0.1f);
        var w2 = Rand([Channels, Channels, 4, 4], 4, wScale);
        var c2 = Rand([Channels], 6, 0.5f);
        var g2 = Fill([Channels], 0.8f);
        var b2 = Fill([Channels], -0.2f);
        var parameters = new[] { w1, c1, g1, b1, w2, c2, g2, b2 };
        if (onDevice) foreach (var p in parameters) p.Gpu();

        ICompiledTrainingPlan<float> plan;
        Tensor<float>[] statistics;
        var runningMean = new Tensor<float>([Channels]);
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.TensorChannelBiasAdd(engine.Conv2D(x, w1, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 }), c1);
            h = Activate(engine, engine.BatchNorm(h, g1, b1, 1e-5, out var mean1, out var variance1), activation);
            // AiDotNet's BatchNormalizationLayer running-mean update, reading the batch mean the fused node writes.
            engine.TensorMultiplyScalarInPlace(runningMean, 0.9f);
            engine.TensorAddInPlace(runningMean, engine.TensorMultiplyScalar(mean1, 0.1f));
            h = engine.TensorChannelBiasAdd(
                engine.ConvTranspose2D(h, w2, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 0, 0 }), c2);
            h = Activate(engine, engine.BatchNorm(h, g2, b2, 1e-5, out var mean2, out var variance2), activation);
            statistics = new[] { mean1, variance1, mean2, variance2 };
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
            run.FinalWeights = parameters.Select(p => p.ToArray()).ToArray();
            run.Biases = new[] { c1.ToArray(), c2.ToArray() };
            run.Statistics = statistics.Select(s => s.ToArray()).ToArray();
            run.RunningMean = runningMean.ToArray();
            if (plan is CompiledTrainingPlan<float> concrete)
            {
                var steps = (CompiledStep<float>[]?)typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(concrete);
                run.StepNames = steps?.Select(s => s.OpName).ToArray() ?? Array.Empty<string>();
            }
            return run;
        }
    }

    private static Run TrainUnfused(IEngine engine, bool onDevice, int batch, int size, int activation)
    {
        // Suspend, not Unregister + Register: re-registering would move the pattern behind the conv + bias and BatchNorm
        // fusions it must run before.
        using var suspended = FusionPatternRegistry.Suspend("ConvBiasBatchNorm");
        Assert.NotNull(suspended);
        return Train(engine, onDevice, batch, size, activation);
    }

    private static void AssertTrainsLike(Run reference, Run fused, string label, double tolerance)
    {
        for (int s = 0; s < Steps; s++)
            Assert.True(Math.Abs(reference.Losses[s] - fused.Losses[s]) <= tolerance * Math.Max(1, Math.Abs(reference.Losses[s])),
                $"{label} step {s}: loss {fused.Losses[s]:G7} != unfused {reference.Losses[s]:G7}");
        AssertClose(reference.FinalWeights, fused.FinalWeights, $"{label} parameter", tolerance);
        AssertClose(reference.Statistics, fused.Statistics, $"{label} statistic", tolerance);
        AssertClose(new[] { reference.RunningMean }, new[] { fused.RunningMean }, $"{label} running mean", tolerance);
    }

    private static void AssertClose(float[][] expected, float[][] actual, string label, double tolerance)
    {
        for (int p = 0; p < expected.Length; p++)
        {
            double maxErr = 0;
            for (int i = 0; i < expected[p].Length; i++)
                maxErr = Math.Max(maxErr, Math.Abs(expected[p][i] - actual[p][i]) / Math.Max(1.0, Math.Abs(expected[p][i])));
            Assert.True(maxErr <= tolerance, $"{label} {p}: max rel error {maxErr:E2} after {Steps} steps");
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

    // batch, size: small channel slices (the register BatchNorm kernels) and large ones (the split kernels).
    [SkippableTheory]
    [InlineData(4, 8, 0)]
    [InlineData(4, 8, 1)]
    [InlineData(4, 8, 2)]
    [InlineData(64, 32, 1)]
    [InlineData(64, 32, 0)]
    public void A_gpu_plan_folds_the_conv_bias_into_the_batchnorm_and_trains_like_the_unfused_plans(
        int batch, int size, int activation)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            var unfusedGpu = TrainUnfused(gpu, onDevice: true, batch, size, activation);
            Assert.DoesNotContain("ConvBiasBatchNorm", unfusedGpu.StepNames);
            long activationsBefore = ChannelsLastLayoutPass.ChannelsLastActivations;
            long convolutionsBefore = AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.ChannelsLastConvolutions;
            var fused = Train(gpu, onDevice: true, batch, size, activation);
            _output.WriteLine("steps unfused: " + string.Join(", ", unfusedGpu.StepNames));
            _output.WriteLine("steps fused  : " + string.Join(", ", fused.StepNames));
            _output.WriteLine("losses unfused: " + string.Join(", ", unfusedGpu.Losses.Select(l => l.ToString("G7"))));
            _output.WriteLine("losses fused  : " + string.Join(", ", fused.Losses.Select(l => l.ToString("G7"))));
            Assert.Equal(2, fused.StepNames.Count(n => n == "ConvBiasBatchNorm"));
            Assert.DoesNotContain("BatchNorm", fused.StepNames);
            Assert.DoesNotContain("TensorChannelBiasAdd", fused.StepNames);
            Assert.DoesNotContain("BatchNormBatchMean", fused.StepNames);
            AssertTrainsLike(unfusedGpu, fused, "gpu fused vs gpu unfused", 1e-4);
            // Where the backend prefers channels-last (cuDNN tensor cores), the first layer's output -- read only by the
            // second convolution -- is channels-last: the layout must engage and train like the all-NCHW fused plan.
            // ReLU6 is not among the channels-last BatchNorm's activations: that plan stays NCHW.
            if (gpu.PrefersChannelsLast && activation == 2)
                Assert.Equal(activationsBefore, ChannelsLastLayoutPass.ChannelsLastActivations);
            if (gpu.PrefersChannelsLast && activation != 2)
            {
                Assert.True(ChannelsLastLayoutPass.ChannelsLastActivations > activationsBefore,
                    "the plan kept no activation channels-last");
                Assert.True(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.ChannelsLastConvolutions > convolutionsBefore,
                    "no channels-last convolution ran");
                bool prior = ChannelsLastLayoutPass.s_enabled;
                ChannelsLastLayoutPass.s_enabled = false;
                try
                {
                    var nchw = Train(gpu, onDevice: true, batch, size, activation);
                    AssertTrainsLike(nchw, fused, "gpu channels-last vs gpu NCHW", 1e-4);
                }
                finally
                {
                    ChannelsLastLayoutPass.s_enabled = prior;
                }
            }
            var cpu = TrainUnfused(new CpuEngine(), onDevice: false, batch, size, activation);
            AssertTrainsLike(cpu, fused, "gpu fused vs cpu unfused", 1e-3);
            // The exact gradient of a bias a training BatchNorm cancels is zero: SGD leaves it where it started.
            Assert.Equal(Rand([Channels], 5, 0.5f).ToArray(), fused.Biases[0]);
            Assert.Equal(Rand([Channels], 6, 0.5f).ToArray(), fused.Biases[1]);
        }
    }
}
