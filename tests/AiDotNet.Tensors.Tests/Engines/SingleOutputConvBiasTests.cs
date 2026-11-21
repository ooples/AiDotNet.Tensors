using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("EngineCurrentGlobalState")]
public sealed class SingleOutputConvBiasTests
{
#if NET7_0_OR_GREATER
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void MultiChannelWaveformsMatchBiasSeededConvolutionAcrossBatches(int dilation)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;
        const int batches = 2, channels = 16, filters = 8, width = 96, taps = 5;
        var random = new Random(479);
        float[] Values(int length) => Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() - .5)).ToArray();
        var signal = Values(batches * channels * width);
        var weights = Values(filters * channels * taps);
        var biases = Values(filters);
        using var input = new Tensor<float>(signal, new[] { batches, channels, 1, width });
        using var kernel = new Tensor<float>(weights, new[] { filters, channels, 1, taps });
        using var bias = new Tensor<float>(biases, new[] { filters });
        using var output = new Tensor<float>(new[] { batches, filters, 1, width });
        int padding = dilation * (taps / 2);
        bool previous = CpuEngine.SingleOutputConvBiasEnabled;
        try
        {
            CpuEngine.SingleOutputConvBiasEnabled = true;
            new CpuEngine().Conv2DBiasActivationInto(output, input, kernel, bias, false,
                new[] { 1, 1 }, new[] { 0, padding }, new[] { 1, dilation });
        }
        finally { CpuEngine.SingleOutputConvBiasEnabled = previous; }

        var actual = output.ToArray();
        for (int batch = 0; batch < batches; batch++)
            for (int filter = 0; filter < filters; filter++)
                for (int t = 0; t < width; t++)
                {
                    float sum = biases[filter];
                    for (int group = 0; group < channels; group += 8)
                        for (int tap = 0; tap < taps; tap++)
                            for (int ch = group; ch < group + 8; ch++)
                            {
                                int sample = t + tap * dilation - padding;
                                if ((uint)sample < (uint)width)
                                    sum = MathF.FusedMultiplyAdd(signal[(batch * channels + ch) * width + sample],
                                        weights[(filter * channels + ch) * taps + tap], sum);
                            }
                    Assert.Equal(sum, actual[(batch * filters + filter) * width + t]);
                }
    }
#endif

    [Theory]
    [InlineData(256)]
    [InlineData(273)]
    public void OffsetDestinationsMatchIndependentDoubleConvolutionAndLiveParameters(int width)
    {
        var engine = new CpuEngine();
        var inputStorage = Enumerable.Repeat(99f, 8 * width + 2).ToArray();
        var weightStorage = Enumerable.Repeat(98f, 8 * 7 + 2).ToArray();
        var outputStorage = Enumerable.Repeat(97f, width + 2).ToArray();
        using var input = Tensor<float>.FromMemory(inputStorage.AsMemory(1, 8 * width), new[] { 1, 8, 1, width });
        using var kernel = Tensor<float>.FromMemory(weightStorage.AsMemory(1, 56), new[] { 1, 8, 1, 7 });
        using var output = Tensor<float>.FromMemory(outputStorage.AsMemory(1, width), new[] { 1, 1, 1, width });
        using var bias = new Tensor<float>(new[] { .067f }, new[] { 1 });
        for (int i = 0; i < input.Length; i++) input[i] = .03f * (float)Math.Sin(i * .071);
        for (int i = 0; i < kernel.Length; i++) kernel[i] = .02f * (float)Math.Cos(i * .13);
        for (int replay = 0; replay < 2; replay++)
        {
            bool handled = engine.TrySingleOutputConvBiasInto(output, input, kernel, bias, new[] { 1, 1 }, new[] { 0, 3 }, new[] { 1, 1 });
#if NET5_0_OR_GREATER
            if (!Avx2.IsSupported || !Fma.IsSupported) { Assert.False(handled); return; }
#else
            Assert.False(handled); return;
#endif
            Assert.True(handled);
            for (int t = 0; t < width; t++)
            {
                double expected = bias[0];
                for (int channel = 0; channel < 8; channel++) for (int tap = 0; tap < 7; tap++)
                {
                    int from = t + tap - 3;
                    if ((uint)from < (uint)width) expected += (double)input[channel * width + from] * kernel[channel * 7 + tap];
                }
                Assert.InRange(Math.Abs(output[t] - expected), 0, 2e-6);
            }
            Assert.Equal(99f, inputStorage[0]); Assert.Equal(99f, inputStorage[^1]);
            Assert.Equal(98f, weightStorage[0]); Assert.Equal(98f, weightStorage[^1]);
            Assert.Equal(97f, outputStorage[0]); Assert.Equal(97f, outputStorage[^1]);
            kernel[0] += .125f; bias[0] -= .03125f;
        }
        using var overlap = Tensor<float>.FromMemory(inputStorage.AsMemory(1, width), new[] { 1, 1, 1, width });
        float[] snapshot = input.ToArray();
        Assert.False(engine.TrySingleOutputConvBiasInto(overlap, input, kernel, bias, new[] { 1, 1 }, new[] { 0, 3 }, new[] { 1, 1 }));
        Assert.Equal(snapshot, input.ToArray());
    }

    [Fact]
    public void CompiledPairPreservesIndependentVjpAndRespectsLaterObservers()
    {
        const int width = 273, channels = 8, taps = 3;
        string? prior = Environment.GetEnvironmentVariable("AIDOTNET_CPU_FUSED_OUTPUT_BIAS");
        string? priorEpilogue = Environment.GetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION");
        bool previousFused = CpuEngine.SingleOutputConvBiasEnabled;
        var previousEngine = AiDotNetEngine.Current;
        var previousObserver = CompiledTrainingPlan<float>.ForwardStepObserver;
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CPU_FUSED_OUTPUT_BIAS", "1");
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION", "1");
            CpuEngine.SingleOutputConvBiasEnabled = true;
            AiDotNetEngine.Current = new CpuEngine();
            var engine = AiDotNetEngine.Current;
            using var input = new Tensor<float>(Enumerable.Range(0, channels * width).Select(i => .03f * (float)Math.Sin(i * .07)).ToArray(), new[] { 1, channels, 1, width });
            using var kernel = new Tensor<float>(Enumerable.Range(0, channels * taps).Select(i => .02f * (float)Math.Cos(i * .13)).ToArray(), new[] { 1, channels, 1, taps });
            using var bias = new Tensor<float>(new[] { .067f }, new[] { 1 });
            using var seed = new Tensor<float>(Enumerable.Range(0, width).Select(i => .1f * (float)Math.Sin(i * .17)).ToArray(), new[] { 1, 1, 1, width });
            var parameters = new[] { input, kernel, bias };
            CompiledTrainingPlan<float> plan;
            using (var graph = GraphMode.EnableTraining(parameters))
            {
                var conv = engine.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 });
                var output = engine.TensorChannelBiasAdd(conv, bias);
                var loss = engine.ReduceSum(engine.TensorMultiply(output, seed));
                plan = graph.CompileTraining(parameters, loss);
            }
            using (plan)
            {
                for (int replay = 0; replay < 2; replay++)
                {
                    int before = CpuEngine.SingleOutputConvBiasRuns;
                    plan.Step();
#if NET5_0_OR_GREATER
                    Assert.Equal(before + (Avx2.IsSupported && Fma.IsSupported ? 1 : 0), CpuEngine.SingleOutputConvBiasRuns);
#endif
                    var gradients = plan.Gradients;
                    double expectedBias = seed.ToArray().Sum(value => (double)value);
                    Assert.InRange(Math.Abs(gradients[2][0] - expectedBias), 0, 2e-6);
                    for (int k = 0; k < kernel.Length; k++)
                    {
                        int channel = k / taps, tap = k % taps;
                        double expected = 0;
                        for (int t = 0; t < width; t++)
                        {
                            int from = t + tap - 1;
                            if ((uint)from < (uint)width) expected += (double)seed[t] * input[channel * width + from];
                        }
                        Assert.InRange(Math.Abs(gradients[1][k] - expected), 0, 2e-6);
                    }
                    kernel[0] += .125f; bias[0] -= .03125f;
                }
                int observed = 0, counter = CpuEngine.SingleOutputConvBiasRuns;
                CompiledTrainingPlan<float>.ForwardStepObserver = (_, _, _) => observed++;
                plan.Step();
                Assert.True(observed > 0);
                Assert.Equal(counter, CpuEngine.SingleOutputConvBiasRuns);
            }
        }
        finally
        {
            CompiledTrainingPlan<float>.ForwardStepObserver = previousObserver;
            AiDotNetEngine.Current = previousEngine;
            CpuEngine.SingleOutputConvBiasEnabled = previousFused;
            Environment.SetEnvironmentVariable("AIDOTNET_CPU_FUSED_OUTPUT_BIAS", prior);
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION", priorEpilogue);
        }
    }
}
