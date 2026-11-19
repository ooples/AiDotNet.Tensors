using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class ChannelBiasResidualBackendTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Cuda_preserves_each_rounding_boundary_and_all_inputs_in_one_launch(bool writeActivation)
    {
        using var backend = new CudaBackend();
        Assert.True(backend.IsAvailable, "This validation requires actual CUDA.");
        const int batch = 2, channels = 3, spatial = 259, count = batch * channels * spatial;
        const float alpha = .1f, sentinel = 123.75f;
        float[] pattern = { 0f, BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)), 1f, -1f,
            float.Epsilon, -float.Epsilon, 1.17549435E-38f, -1.17549435E-38f,
            float.NaN, float.PositiveInfinity, float.NegativeInfinity, .75f, -.5f,
            BitConverter.Int32BitsToSingle(0x3f800001), 16777216f, -16777216f };
        float[] values = Enumerable.Range(0, count).Select(i => pattern[i % pattern.Length]).ToArray();
        float[] residuals = Enumerable.Range(0, count).Select(i => pattern[(i + 3) % pattern.Length]).ToArray();
        float[] biases = { -.75f, 0f, .5f };
        var padded = Enumerable.Repeat(sentinel, count + 7).ToArray();
        using var input = backend.AllocateBuffer(values);
        using var bias = backend.AllocateBuffer(biases);
        using var residual = backend.AllocateBuffer(residuals);
        using var biased = backend.AllocateBuffer(padded);
        using var expectedSum = backend.AllocateBuffer(padded);
        using var expectedActivated = backend.AllocateBuffer(padded);
        using var actualSum = backend.AllocateBuffer(padded);
        using var actualActivated = backend.AllocateBuffer(padded);
        backend.Copy(input, biased, count);
        backend.Conv2DBiasAdd(biased, bias, batch, channels, spatial);
        backend.Add(biased, residual, expectedSum, count);
        backend.LeakyRelu(expectedSum, expectedActivated, alpha, count);
        var fused = Assert.IsAssignableFrom<IChannelBiasResidualBackend>(backend);
        long launches = GpuLaunchProbe.Count, readbacks = GpuLaunchProbe.ReadbackBytes;
        Assert.True(fused.TryChannelBiasResidual(input, bias, residual, actualSum, actualActivated,
            batch, channels, spatial, alpha, writeActivation));
        Assert.Equal(launches + 1, GpuLaunchProbe.Count);
        Assert.Equal(readbacks, GpuLaunchProbe.ReadbackBytes);
        Same(backend.DownloadBuffer(expectedSum), backend.DownloadBuffer(actualSum));
        Same(writeActivation ? backend.DownloadBuffer(expectedActivated) : padded, backend.DownloadBuffer(actualActivated));
        Same(values, backend.DownloadBuffer(input));
        Same(biases, backend.DownloadBuffer(bias));
        Same(residuals, backend.DownloadBuffer(residual));
    }

    [Fact]
    public void Cuda_rejects_output_aliases_and_undersized_buffers_before_launch_or_write()
    {
        using var backend = new CudaBackend();
        Assert.True(backend.IsAvailable);
        using var input = backend.AllocateBuffer(new[] { 1f, -2f, 3f, -4f });
        using var bias = backend.AllocateBuffer(new[] { .5f });
        using var residual = backend.AllocateBuffer(new[] { -1f, 2f, -3f, 4f });
        using var output = backend.AllocateBuffer(new[] { 91f, 92f, 93f, 94f });
        using var activated = backend.AllocateBuffer(new[] { 81f, 82f, 83f, 84f });
        using var shortBuffer = backend.AllocateBuffer(new[] { 71f });
        var fused = Assert.IsAssignableFrom<IChannelBiasResidualBackend>(backend);
        long before = GpuLaunchProbe.Count;
        Assert.False(fused.TryChannelBiasResidual(input, bias, residual, input, activated, 1, 1, 4, .1f, true));
        Assert.False(fused.TryChannelBiasResidual(input, bias, residual, residual, activated, 1, 1, 4, .1f, true));
        Assert.False(fused.TryChannelBiasResidual(input, bias, residual, output, residual, 1, 1, 4, .1f, true));
        Assert.False(fused.TryChannelBiasResidual(input, bias, residual, output, output, 1, 1, 4, .1f, true));
        Assert.False(fused.TryChannelBiasResidual(input, bias, shortBuffer, output, activated, 1, 1, 4, .1f, true));
        Assert.False(fused.TryChannelBiasResidual(input, bias, residual, output, shortBuffer, 1, 1, 4, .1f, true));
        Assert.False(fused.TryChannelBiasResidual(input, bias, residual, output, activated, 1, 1, 4, float.NaN, true));
        Assert.Equal(before, GpuLaunchProbe.Count);
        Same(new[] { 1f, -2f, 3f, -4f }, backend.DownloadBuffer(input));
        Same(new[] { -1f, 2f, -3f, 4f }, backend.DownloadBuffer(residual));
        Same(new[] { 91f, 92f, 93f, 94f }, backend.DownloadBuffer(output));
        Same(new[] { 81f, 82f, 83f, 84f }, backend.DownloadBuffer(activated));
    }

    [Fact]
    public void Closing_engine_does_not_borrow_or_launch()
    {
        using var engine = new DirectGpuTensorEngine();
        engine.StrictGpuLeases = true;
        engine.Dispose();
        using var input = new Tensor<float>(new[] { 1, 2, 1, 3 });
        using var bias = new Tensor<float>(new[] { 2 });
        using var residual = new Tensor<float>(new[] { 1, 2, 1, 3 });
        long before = GpuLaunchProbe.Count;
        Assert.False(engine.TryChannelBiasResidualInference(input, bias, residual, .1f, true, out var output, out var activated));
        Assert.Null(output);
        Assert.Null(activated);
        Assert.Equal(before, GpuLaunchProbe.Count);
    }

    private static void Same(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int index = 0; index < expected.Length; ++index)
            if (float.IsNaN(expected[index])) Assert.True(float.IsNaN(actual[index]), $"Expected NaN at {index}.");
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected[index]), BitConverter.SingleToInt32Bits(actual[index]));
    }
}
