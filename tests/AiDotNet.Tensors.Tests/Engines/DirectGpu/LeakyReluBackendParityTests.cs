#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class LeakyReluBackendParityTests
{
    private static void Check(IDirectGpuBackend backend, float alpha)
    {
        const int count = 301; // Includes an incomplete kernel tile.
        var x = new float[count];
        var g = new float[count];
        for (int i = 0; i < count; i++)
        {
            x[i] = (i % 13 - 6) * .125f;
            g[i] = (i % 7 - 3) * .25f;
        }
        x[0] = 0f;
        x[1] = -0f;
        using var input = backend.AllocateBuffer(x);
        using var upstream = backend.AllocateBuffer(g);
        using var output = backend.AllocateBuffer(count);
        using var gradient = backend.AllocateBuffer(count);
        backend.LeakyRelu(input, output, alpha, count);
        backend.LeakyReluBackward(upstream, input, gradient, alpha, count);
        var actual = backend.DownloadBuffer(output);
        var dx = backend.DownloadBuffer(gradient);
        for (int i = 0; i < count; i++)
        {
            // CPU LeakyReLU and HiFi's prior slope*x+(1-slope)*relu(x) choose the negative slope at both signed zeros.
            double expected = x[i] > 0 ? x[i] : alpha * x[i];
            double derivative = g[i] * (x[i] > 0 ? 1 : alpha);
            Assert.True(Math.Abs(actual[i] - expected) <= 2e-6, $"output[{i}]: {actual[i]} != {expected}");
            Assert.True(Math.Abs(dx[i] - derivative) <= 2e-6, $"gradient[{i}], x={x[i]}: {dx[i]} != {derivative}");
        }
    }

    [SkippableTheory]
    [InlineData(.1f)] [InlineData(.01f)]
    public void Cuda_forward_and_backward_preserve_channel_activation_semantics(float alpha)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA unavailable.");
        Check(backend, alpha);
    }

    [SkippableTheory]
    [InlineData(.1f)] [InlineData(.01f)]
    public void Hip_forward_and_backward_preserve_channel_activation_semantics(float alpha)
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP unavailable.");
        Check(backend, alpha);
    }

    [SkippableTheory]
    [InlineData(.1f)] [InlineData(.01f)]
    public void OpenCl_forward_and_backward_preserve_channel_activation_semantics(float alpha)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL unavailable.");
        Check(backend, alpha);
    }

    [SkippableTheory]
    [InlineData(.1f)] [InlineData(.01f)]
    public void Metal_forward_and_backward_preserve_channel_activation_semantics(float alpha)
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal unavailable on this platform.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal unavailable.");
        Check(backend, alpha);
    }

    [SkippableTheory]
    [InlineData(.1f)] [InlineData(.01f)]
    public void Vulkan_forward_and_backward_preserve_channel_activation_semantics(float alpha)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler unavailable.");
        Check(backend, alpha);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [InlineData(.1f)] [InlineData(.01f)]
    public void WebGpu_forward_and_backward_preserve_channel_activation_semantics(float alpha)
        => Check(DawnHostedWebGpu.BackendOrSkip(), alpha);
#endif
}
#endif
