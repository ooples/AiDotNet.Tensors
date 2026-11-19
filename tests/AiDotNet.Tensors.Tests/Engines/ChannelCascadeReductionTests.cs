using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class ChannelCascadeReductionTests
{
    [Theory]
    [InlineData(1, 1, 1, false)]
    [InlineData(1, 3, 8193, false)]
    [InlineData(2, 5, 259, true)]
    [InlineData(1, 2, 65539, true)]
    public void Offset_channel_reductions_match_double_sums_and_keep_ownership(int batch, int channels, int spatial, bool keepDims)
    {
        var engine = new CpuEngine();
        int length = batch * channels * spatial;
        using var storage = new Tensor<float>(new[] { length + 2 });
        storage.Fill(-37f);
        using var input = storage.Slice(0, 1, length + 1).Reshape(new[] { batch, channels, 1, spatial });
        float[] values = new float[length];
        for (int i = 0; i < length; i++) input[i] = values[i] = (float)(.2 * Math.Sin(i * .037));
        int before = CpuEngine.ChannelCascadeReductionRunsOnThisThread;
        using var result = engine.ReduceSum(input, new[] { 0, -2, -1 }, keepDims);
        Assert.Equal(before + 1, CpuEngine.ChannelCascadeReductionRunsOnThisThread);
        Assert.Equal(keepDims ? new[] { 1, channels, 1, 1 } : new[] { channels }, result.Shape.ToArray());
        float[] actual = result.ToArray();
        for (int channel = 0; channel < channels; channel++)
        {
            double expected = 0;
            for (int n = 0; n < batch; n++) for (int s = 0; s < spatial; s++) expected += values[(n * channels + channel) * spatial + s];
            Assert.InRange(Math.Abs(actual[channel] - expected), 0, 3e-6 * Math.Max(1, Math.Abs(expected)));
        }
        Assert.Equal(values, input.ToArray());
        Assert.Equal(-37f, storage[0]); Assert.Equal(-37f, storage[storage.Length - 1]);
        input.Fill(19f);
        Assert.Equal(actual, result.ToArray());
    }

    [Fact]
    public void Cascade_preserves_small_terms_in_a_long_cancelling_sum()
    {
        var engine = new CpuEngine();
        float[] values = new float[8194]; values[0] = 1f; values[values.Length - 1] = -1f;
        for (int i = 1; i < values.Length - 1; i++) values[i] = 1e-5f;
        using var input = new Tensor<float>(values, new[] { 1, 1, 1, values.Length });
        using var output = engine.ReduceSum(input, new[] { 0, 2, 3 });
        double expected = (double)1e-5f * (values.Length - 2);
        Assert.InRange(Math.Abs(output.ToArray()[0] - expected), 0, 5e-7);
    }

    [Fact]
    public void Channel_reduction_backward_broadcasts_to_every_input_element()
    {
        var engine = new CpuEngine();
        using var input = new Tensor<float>(new[] { 2, 3, 1, 17 }); input.Fill(.25f);
        using var tape = new GradientTape<float>();
        var reduced = engine.ReduceSum(input, new[] { 0, 2, 3 }, keepDims: true);
        var loss = engine.ReduceSum(reduced);
        var gradients = tape.ComputeGradients(loss, new[] { input });
        Assert.True(gradients.TryGetValue(input, out var gradient));
        Assert.Equal(input.Shape.ToArray(), gradient.Shape.ToArray());
        foreach (float value in gradient.ToArray()) Assert.Equal(1f, value);
    }
}
