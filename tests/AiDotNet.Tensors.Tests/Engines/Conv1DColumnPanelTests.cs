using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class Conv1DColumnPanelTests
{
    [Theory]
    [InlineData(1, 32, 7, 2201, 17, 1, 8, 1)]
    [InlineData(2, 32, 7, 4501, 17, 2, 16, 2)]
    [InlineData(1, 64, 5, 2203, 11, 1, 15, 3)]
    public void WaveformPanelsMatchFloat64ReferenceIncludingTailsAndPadding(
        int batch, int channels, int outputs, int width, int taps, int stride, int padding, int dilation)
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { batch, channels, 1, width });
        var weights = new Tensor<float>(new[] { outputs, channels, 1, taps });
        var x = input.AsWritableSpan();
        var w = weights.AsWritableSpan();
        for (int i = 0; i < x.Length; i++) x[i] = (float)(.2 * Math.Sin(.11 * i));
        for (int i = 0; i < w.Length; i++) w[i] = (float)(.05 * Math.Cos(.13 * i));
        int outputWidth = (width + 2 * padding - dilation * (taps - 1) - 1) / stride + 1;
        int before = CpuEngine.Conv1DColumnPanelRunsOnThisThread;
#if NET5_0_OR_GREATER
        // Keep the specialized direct kernel out of this comparison so it exercises the new bounded panel route.
        using var suppressDirect = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.SuppressOnThisThread();
#endif
        var actual = engine.Conv2D(input, weights, new[] { 1, stride }, new[] { 0, padding }, new[] { 1, dilation }).ToArray();
        Assert.Equal(1, CpuEngine.Conv1DColumnPanelRunsOnThisThread - before);
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outputs; oc++)
                for (int t = 0; t < outputWidth; t++)
                {
                    double expected = 0;
                    for (int ic = 0; ic < channels; ic++)
                        for (int tap = 0; tap < taps; tap++)
                        {
                            int source = t * stride + tap * dilation - padding;
                            if ((uint)source < (uint)width)
                                expected += (double)x[(b * channels + ic) * width + source] * w[(oc * channels + ic) * taps + tap];
                        }
                    Assert.InRange(Math.Abs(actual[(b * outputs + oc) * outputWidth + t] - expected), 0,
                        2e-5 * Math.Max(1, Math.Abs(expected)));
                }
    }

    [Fact]
    public void ColumnPanelsAreIndependentOfWorkerCount()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 2, 32, 1, 4501 });
        var weights = new Tensor<float>(new[] { 7, 32, 1, 17 });
        input.Fill(.2f);
        weights.Fill(.03f);
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
#if NET5_0_OR_GREATER
            using var suppressDirect = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.SuppressOnThisThread();
#endif
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var single = engine.Conv2D(input, weights, new[] { 1, 2 }, new[] { 0, 16 }, new[] { 1, 2 }).ToArray();
            CpuParallelSettings.MaxDegreeOfParallelism = 4;
            var parallel = engine.Conv2D(input, weights, new[] { 1, 2 }, new[] { 0, 16 }, new[] { 1, 2 }).ToArray();
            Assert.Equal(single, parallel);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
    }

    [Theory]
    [InlineData(1, 7, 64, 2201, 17, 8, 1)]
    [InlineData(2, 7, 64, 2203, 11, 15, 3)]
    public void InputGradientMatchesFloat64ScatterAndAccumulatesIntoOffsetStorage(
        int batch, int channels, int outputs, int width, int taps, int padding, int dilation)
    {
        var engine = new CpuEngine();
        int outputWidth = width + 2 * padding - dilation * (taps - 1);
        var gradient = new Tensor<float>(new[] { batch, outputs, 1, outputWidth });
        var weights = new Tensor<float>(new[] { outputs, channels, 1, taps });
        var g = gradient.AsWritableSpan();
        var w = weights.AsWritableSpan();
        for (int i = 0; i < g.Length; i++) g[i] = (float)(.2 * Math.Sin(.11 * i));
        for (int i = 0; i < w.Length; i++) w[i] = (float)(.05 * Math.Cos(.13 * i));
        var expected = new double[batch * channels * width];
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < outputs; oc++)
                for (int t = 0; t < outputWidth; t++)
                    for (int ic = 0; ic < channels; ic++)
                        for (int tap = 0; tap < taps; tap++)
                        {
                            int source = t + tap * dilation - padding;
                            if ((uint)source < (uint)width)
                                expected[(b * channels + ic) * width + source] +=
                                    (double)g[(b * outputs + oc) * outputWidth + t] * w[(oc * channels + ic) * taps + tap];
                        }
#if NET5_0_OR_GREATER
        using var suppressDirect = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.SuppressOnThisThread();
#endif
        // A leading sentinel makes the written tensor a real nonzero-offset arena-like view.
        var storage = new Tensor<float>(new[] { expected.Length + 1 });
        storage.Fill(.25f);
        var destination = storage.Slice(0, 1, expected.Length + 1).Reshape(new[] { batch, channels, 1, width });
        Assert.Equal(1, destination.StorageOffset);
        int before = CpuEngine.Conv1DColumnPanelRunsOnThisThread;
        engine.Conv2DBackwardInputInto(destination, gradient, weights, destination.Shape.ToArray(),
            new[] { 1, 1 }, new[] { 0, padding }, new[] { 1, dilation }, accumulate: false);
        Assert.Equal(1, CpuEngine.Conv1DColumnPanelRunsOnThisThread - before);
        Assert.Equal(.25f, storage[0]);
        var actual = destination.ToArray();
        for (int i = 0; i < actual.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 3e-5 * Math.Max(1, Math.Abs(expected[i])));
        engine.Conv2DBackwardInputInto(destination, gradient, weights, destination.Shape.ToArray(),
            new[] { 1, 1 }, new[] { 0, padding }, new[] { 1, dilation }, accumulate: true);
        actual = destination.ToArray();
        for (int i = 0; i < actual.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - 2 * expected[i]), 0, 6e-5 * Math.Max(1, Math.Abs(expected[i])));
        Assert.Equal(.25f, storage[0]);
    }

    [Fact]
    public void ForwardPanelsReadAndWriteNonzeroStorageOffsets()
    {
        const int batch = 2, channels = 32, outputs = 7, width = 2201, taps = 17, padding = 8;
        var engine = new CpuEngine();
        var inputStorage = new Tensor<float>(new[] { batch * channels * width + 2 });
        var weightStorage = new Tensor<float>(new[] { outputs * channels * taps + 2 });
        var outputStorage = new Tensor<float>(new[] { batch * outputs * width + 2 });
        inputStorage.Fill(-10);
        weightStorage.Fill(-10);
        outputStorage.Fill(-10);
        var input = inputStorage.Slice(0, 1, inputStorage.Length - 1).Reshape(new[] { batch, channels, 1, width });
        var weights = weightStorage.Slice(0, 1, weightStorage.Length - 1).Reshape(new[] { outputs, channels, 1, taps });
        var output = outputStorage.Slice(0, 1, outputStorage.Length - 1).Reshape(new[] { batch, outputs, 1, width });
        input.Fill(.2f);
        weights.Fill(.03f);
        Assert.Equal(1, input.StorageOffset);
        Assert.Equal(1, weights.StorageOffset);
        Assert.Equal(1, output.StorageOffset);
#if NET5_0_OR_GREATER
        using var suppressDirect = AiDotNet.Tensors.Engines.Simd.DirectConvAvx2.SuppressOnThisThread();
#endif
        int before = CpuEngine.Conv1DColumnPanelRunsOnThisThread;
        engine.Conv2DInto(output, input, weights, new[] { 1, 1 }, new[] { 0, padding }, new[] { 1, 1 });
        Assert.Equal(1, CpuEngine.Conv1DColumnPanelRunsOnThisThread - before);
        var actual = output.ToArray();
        for (int i = 0; i < actual.Length; i++)
        {
            int t = i % width;
            int validTaps = Math.Min(taps, width + padding - t) - Math.Max(0, padding - t);
            double expected = validTaps * channels * .2f * .03f;
            Assert.InRange(Math.Abs(actual[i] - expected), 0, 2e-5 * Math.Max(1, Math.Abs(expected)));
        }
        Assert.Equal(-10f, outputStorage[0]);
        Assert.Equal(-10f, outputStorage[outputStorage.Length - 1]);
    }
}
