using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class Conv1DChannelBlockTests
{
    [Theory]
    [InlineData(1, 80, 32, 33, 7, 1, 3, 1)]
    [InlineData(7, 17, 37, 19, 5, 3, 2, 2)]
    [InlineData(2, 32, 277, 13, 3, 1, 1, 1)]
    [InlineData(7, 1024, 13, 257, 5, 1, 2, 1)]
    [InlineData(3, 17, 3, 5, 9, 2, 17, 2)]
    [InlineData(5, 19, 67, 23, 3, 2, 4, 3)]
    public void Panels_match_independent_double_convolution_with_channel_tail_batch_and_row_tiles(
        int batch, int channels, int time, int outputs, int taps, int stride, int padding, int dilation)
    {
        var engine = new CpuEngine();
        int outTime = (time + 2 * padding - dilation * (taps - 1) - 1) / stride + 1;
        float[] source = new float[batch * channels * time], weights = new float[outputs * channels * taps];
        for (int i = 0; i < source.Length; i++) source[i] = (float)(.2 * Math.Sin(i * .037));
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(.03 * Math.Cos(i * .019));
        using var input = new Tensor<float>(source, new[] { batch, channels, 1, time });
        using var kernel = new Tensor<float>(weights, new[] { outputs, channels, 1, taps });
        int before = CpuEngine.Conv1DChannelBlockRunsOnThisThread;
        float[] result = engine.Conv2D(input, kernel, new[] { 1, stride }, new[] { 0, padding }, new[] { 1, dilation }).ToArray();
        Assert.Equal(before + 1, CpuEngine.Conv1DChannelBlockRunsOnThisThread);
        Assert.Equal(batch * outputs * outTime, result.Length);
        for (int n = 0; n < batch; n++) for (int oc = 0; oc < outputs; oc++) for (int t = 0; t < outTime; t++)
        {
            double expected = 0;
            for (int ic = 0; ic < channels; ic++) for (int k = 0; k < taps; k++)
            {
                int location = t * stride - padding + k * dilation;
                if (location >= 0 && location < time)
                    expected += (double)source[(n * channels + ic) * time + location] * weights[(oc * channels + ic) * taps + k];
            }
            Assert.InRange(Math.Abs(result[(n * outputs + oc) * outTime + t] - expected), 0, 1e-5 * Math.Max(1, Math.Abs(expected)));
        }
        Assert.Equal(source, input.ToArray());
        Assert.Equal(weights, kernel.ToArray());
    }

    [Fact]
    public void Offset_views_preserve_sentinels_and_parameter_changes_are_visible()
    {
        var engine = new CpuEngine();
        using var inputStorage = new Tensor<float>(new[] { 2 * 17 * 41 + 2 });
        using var kernelStorage = new Tensor<float>(new[] { 19 * 17 * 3 + 2 });
        using var outputStorage = new Tensor<float>(new[] { 2 * 19 * 41 + 2 });
        inputStorage.Fill(-17f); kernelStorage.Fill(-19f); outputStorage.Fill(-23f);
        using var input = inputStorage.Slice(0, 1, inputStorage.Length - 1).Reshape(new[] { 2, 17, 1, 41 });
        using var kernel = kernelStorage.Slice(0, 1, kernelStorage.Length - 1).Reshape(new[] { 19, 17, 1, 3 });
        using var output = outputStorage.Slice(0, 1, outputStorage.Length - 1).Reshape(new[] { 2, 19, 1, 41 });
        input.Fill(.25f); kernel.Fill(.125f);
        int before = CpuEngine.Conv1DChannelBlockRunsOnThisThread;
        engine.Conv2DInto(output, input, kernel, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 });
        Assert.Equal(before + 1, CpuEngine.Conv1DChannelBlockRunsOnThisThread);
        var first = output.ToArray();
        kernel.Fill(.25f);
        engine.Conv2DInto(output, input, kernel, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 });
        var second = output.ToArray();
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i] * 2f, second[i]);
        Assert.Equal(-17f, inputStorage[0]); Assert.Equal(-17f, inputStorage[inputStorage.Length - 1]);
        Assert.Equal(-19f, kernelStorage[0]); Assert.Equal(-19f, kernelStorage[kernelStorage.Length - 1]);
        Assert.Equal(-23f, outputStorage[0]); Assert.Equal(-23f, outputStorage[outputStorage.Length - 1]);
    }
}
