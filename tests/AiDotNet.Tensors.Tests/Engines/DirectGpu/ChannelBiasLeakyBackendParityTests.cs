using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class ChannelBiasLeakyBackendParityTests
{
    [Fact]
    public void Closing_engine_does_not_borrow_a_backend_or_launch()
    {
        using var engine = new DirectGpuTensorEngine();
        engine.StrictGpuLeases = true;
        engine.Dispose();
        using var input = new Tensor<float>(new[] { 1, 2, 1, 3 });
        using var bias = new Tensor<float>(new[] { 2 });
        long before = GpuLaunchProbe.Count;
        Assert.False(engine.TryChannelBiasLeakyInference(input, bias, .1f, out var output));
        Assert.Null(output);
        Assert.Equal(before, GpuLaunchProbe.Count);
    }

    private static void Verify(IDirectGpuBackend backend, float[]? channelBiases = null)
    {
        const int batch = 2, channels = 3, spatial = 259, count = batch * channels * spatial;
        const float alpha = .1f, sentinel = 123.75f;
        float[] pattern = { 0f, MathCompat.Int32BitsToSingle(unchecked((int)0x80000000)), 1f, -1f, float.Epsilon, -float.Epsilon,
            1.17549435E-38f, -1.17549435E-38f, float.NaN, float.PositiveInfinity, float.NegativeInfinity,
            .75f, -.5f, MathCompat.Int32BitsToSingle(0x3f800001), MathCompat.Int32BitsToSingle(0x007fffff),
            -MathCompat.Int32BitsToSingle(0x00800001), MathCompat.Int32BitsToSingle(0x0c800000),
            MathCompat.Int32BitsToSingle(0x0d000000) };
        var values = Enumerable.Range(0, count).Select(i => pattern[i % pattern.Length]).ToArray();
        var biases = channelBiases ?? new[] { -.75f, 0f, .5f };
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
            else Assert.Equal(MathCompat.SingleToInt32Bits(expected[index]), MathCompat.SingleToInt32Bits(actual[index]));
    }

    [SkippableFact]
    public void Cuda_matches_separate_bias_then_leaky_in_one_launch()
    { using var backend = new CudaBackend(); Skip.IfNot(backend.IsAvailable, "CUDA unavailable."); Verify(backend); }
    [SkippableFact]
    public void Hip_matches_separate_bias_then_leaky_in_one_launch()
    { using var backend = new HipBackend(); Skip.IfNot(backend.IsAvailable, "HIP unavailable."); Verify(backend); }
    [SkippableFact]
    public void OpenCl_matches_separate_bias_then_leaky_in_one_launch()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL unavailable.");
        Verify(backend);
        float largestSubnormal = MathCompat.Int32BitsToSingle(0x007fffff);
        Verify(backend, new[] { float.Epsilon, -largestSubnormal, largestSubnormal });
    }
    [SkippableFact]
    public void Metal_matches_separate_bias_then_leaky_in_one_launch()
    {
#if NET5_0_OR_GREATER
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal unavailable on this OS.");
#else
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX), "Metal unavailable on this OS.");
#endif
        using var backend = new MetalBackend(); Skip.IfNot(backend.IsAvailable, "Metal unavailable."); Verify(backend);
    }
    [SkippableFact]
    public void Vulkan_matches_separate_bias_then_leaky_in_one_launch()
    { var backend = VulkanBackend.Instance; Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan unavailable."); Verify(backend); }
#if NET7_0_OR_GREATER
    [SkippableFact]
    public async Task WebGpu_matches_separate_bias_then_leaky_in_one_launch()
    {
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend();
        Skip.IfNot(await backend.InitializeAsync(), "WebGPU unavailable."); Verify(backend);
    }
#endif
}
