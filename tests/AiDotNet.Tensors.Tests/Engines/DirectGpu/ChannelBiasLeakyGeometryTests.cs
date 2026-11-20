using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class ChannelBiasLeakyGeometryTests
{
    private sealed class Buffer : IGpuBuffer
    {
        internal Buffer(int size, int handle, int bytesPerElement = 4) { Size = size; Handle = new IntPtr(handle); SizeInBytes = (long)size * bytesPerElement; }
        public int Size { get; }
        public long SizeInBytes { get; }
        public IntPtr Handle { get; }
        public void Dispose() { }
    }

    [Theory]
    [InlineData(0, 3, 5, .1f)]
    [InlineData(2, 0, 5, .1f)]
    [InlineData(2, 3, 0, .1f)]
    [InlineData(int.MaxValue, int.MaxValue, int.MaxValue, .1f)]
    [InlineData(2, 3, 5, 0f)]
    [InlineData(2, 3, 5, -1f)]
    [InlineData(2, 3, 5, float.NaN)]
    [InlineData(2, 3, 5, float.PositiveInfinity)]
    public void Invalid_geometry_or_slope_is_rejected(int batch, int channels, int spatial, float alpha)
        => Assert.False(ChannelBiasLeakyGeometry.TryCount(new Buffer(30, 1), new Buffer(3, 2), new Buffer(30, 3),
            batch, channels, spatial, alpha, out _));

    [Fact]
    public void Inputs_are_read_only_and_only_disjoint_fp32_destinations_are_accepted()
    {
        using var input = new Buffer(30, 1);
        using var bias = new Buffer(3, 2);
        using var output = new Buffer(37, 3);
        Assert.True(ChannelBiasLeakyGeometry.TryCount(input, bias, output, 2, 3, 5, .1f, out int count));
        Assert.Equal(30, count);
        Assert.False(ChannelBiasLeakyGeometry.TryCount(input, bias, input, 2, 3, 5, .1f, out _));
        Assert.False(ChannelBiasLeakyGeometry.TryCount(input, bias, new Buffer(30, 2), 2, 3, 5, .1f, out _));
        Assert.False(ChannelBiasLeakyGeometry.TryCount(new Buffer(30, 1, 2), bias, output, 2, 3, 5, .1f, out _));
        Assert.False(ChannelBiasLeakyGeometry.TryCount(input, new Buffer(3, 2, 2), output, 2, 3, 5, .1f, out _));
        Assert.False(ChannelBiasLeakyGeometry.TryCount(input, bias, new Buffer(30, 3, 2), 2, 3, 5, .1f, out _));
    }

    [Theory]
    [InlineData(typeof(CudaBackend))]
    [InlineData(typeof(HipBackend))]
    [InlineData(typeof(MetalBackend))]
    [InlineData(typeof(OpenClBackend))]
    [InlineData(typeof(VulkanBackend))]
#if NET7_0_OR_GREATER
    [InlineData(typeof(AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend))]
#endif
    public void All_six_backend_classes_implement_the_explicit_semantic(Type backend)
        => Assert.True(typeof(IChannelBiasLeakyBackend).IsAssignableFrom(backend));
}
