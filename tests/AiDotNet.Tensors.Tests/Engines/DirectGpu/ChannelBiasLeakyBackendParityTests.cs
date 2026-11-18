using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class ChannelBiasLeakyBackendParityTests
{
    private static void Verify(IDirectGpuBackend backend)
    {
        const int batch = 2, channels = 3, spatial = 259, count = batch * channels * spatial;
        const float alpha = .1f, sentinel = 123.75f;
        float[] pattern = { 0f, BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)), 1f, -1f, float.Epsilon, -float.Epsilon,
            1.17549435E-38f, -1.17549435E-38f, float.NaN, float.PositiveInfinity, float.NegativeInfinity,
            .75f, -.5f, BitConverter.Int32BitsToSingle(0x3f800001) };
        var values = Enumerable.Range(0, count).Select(i => pattern[i % pattern.Length]).ToArray();
        var biases = new[] { -.75f, 0f, .5f };
        var padded = Enumerable.Repeat(sentinel, count + 7).ToArray();
        using var input = backend.AllocateBuffer(values);
        using var bias = backend.AllocateBuffer(biases);
        using var sum = backend.AllocateBuffer(padded);
        using var baseline = backend.AllocateBuffer(padded);
        using var fusedOutput = backend.AllocateBuffer(padded);
        backend.Copy(input, sum, count);
        backend.Conv2DBiasAdd(sum, bias, batch, channels, spatial);
        backend.LeakyRelu(sum, baseline, alpha, count);
        var fused = Assert.IsAssignableFrom<IChannelBiasLeakyBackend>(backend);
        long launches = GpuLaunchProbe.Count, readbacks = GpuLaunchProbe.ReadbackBytes;
        Assert.True(fused.TryChannelBiasLeaky(input, bias, fusedOutput, batch, channels, spatial, alpha));
        Assert.Equal(launches + 1, GpuLaunchProbe.Count);
        Assert.Equal(readbacks, GpuLaunchProbe.ReadbackBytes);
        AssertSameValues(backend.DownloadBuffer(baseline), backend.DownloadBuffer(fusedOutput));
        AssertSameValues(values, backend.DownloadBuffer(input));
        AssertSameValues(biases, backend.DownloadBuffer(bias));
        long beforeRejected = GpuLaunchProbe.Count;
        Assert.False(fused.TryChannelBiasLeaky(input, bias, input, batch, channels, spatial, alpha));
        Assert.Equal(beforeRejected, GpuLaunchProbe.Count);
        AssertSameValues(values, backend.DownloadBuffer(input));
    }

    private static void AssertSameValues(float[] expected, float[] actual)
    {
        Assert.True(actual.Length >= expected.Length);
        for (int index = 0; index < expected.Length; ++index)
            if (float.IsNaN(expected[index])) Assert.True(float.IsNaN(actual[index]), $"Expected NaN at {index}.");
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected[index]), BitConverter.SingleToInt32Bits(actual[index]));
    }

    [SkippableFact]
    public void Cuda_matches_separate_bias_then_leaky_in_one_launch()
    { using var backend = new CudaBackend(); Skip.IfNot(backend.IsAvailable, "CUDA unavailable."); Verify(backend); }
    [SkippableFact]
    public void Hip_matches_separate_bias_then_leaky_in_one_launch()
    { using var backend = new HipBackend(); Skip.IfNot(backend.IsAvailable, "HIP unavailable."); Verify(backend); }
    [SkippableFact]
    public void OpenCl_matches_separate_bias_then_leaky_in_one_launch()
    { using var backend = new OpenClBackend(); Skip.IfNot(backend.IsAvailable, "OpenCL unavailable."); Verify(backend); }
    [SkippableFact]
    public void Metal_matches_separate_bias_then_leaky_in_one_launch()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal unavailable on this OS.");
        using var backend = new MetalBackend(); Skip.IfNot(backend.IsAvailable, "Metal unavailable."); Verify(backend);
    }
    [SkippableFact]
    public void Vulkan_matches_separate_bias_then_leaky_in_one_launch()
    { var backend = VulkanBackend.Instance; Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan unavailable."); Verify(backend); }
#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_matches_separate_bias_then_leaky_in_one_launch()
    {
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend();
        Skip.IfNot(backend.IsAvailable, "WebGPU unavailable."); Verify(backend);
    }
#endif
}
