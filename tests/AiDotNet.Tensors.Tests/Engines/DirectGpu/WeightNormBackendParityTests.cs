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
public sealed class WeightNormBackendParityTests
{
    private static void Check(IDirectGpuBackend backend, bool accumulate)
    {
        const int rows = 3, columns = 513;
        const float epsilon = 1e-24f;
        var native = Assert.IsAssignableFrom<IGpuBatchExecution>(backend);
        var v = new float[rows * columns]; var h = new float[v.Length];
        var g = new[] { .75f, -1.25f, 0f };
        var oldV = new float[v.Length]; var oldG = new[] { -.25f, .5f, 1.25f };
        for (int i = 0; i < v.Length; i++)
        {
            int row = i / columns;
            v[i] = row == 1 ? 0 : (i % 11 - 5) * (row == 2 ? 1e-13f : .125f);
            h[i] = (i % 7 - 3) * .0625f;
            oldV[i] = (i % 5 - 2) * .03125f;
        }
        using var direction = backend.AllocateBuffer(v);
        using var gain = backend.AllocateBuffer(g);
        using var upstream = backend.AllocateBuffer(h);
        using var output = backend.AllocateBuffer(v.Length);
        using var stats = backend.AllocateBuffer(rows * 2);
        using var dv = backend.AllocateBuffer(oldV);
        using var dg = backend.AllocateBuffer(oldG);
        native.WeightNorm(direction, gain, output, rows, columns, epsilon);
        native.WeightNormStats(upstream, direction, stats, rows, columns, epsilon);
        native.WeightNormDirectionGradient(upstream, direction, gain, stats, dv, rows, columns, accumulate);
        native.WeightNormGainGradient(stats, dg, rows, accumulate);
        var actualW = backend.DownloadBuffer(output);
        var actualV = backend.DownloadBuffer(dv);
        var actualG = backend.DownloadBuffer(dg);
        for (int row = 0; row < rows; row++)
        {
            double square = 0, dot = 0;
            for (int j = 0; j < columns; j++)
            {
                int i = row * columns + j;
                square += (double)v[i] * v[i]; dot += (double)h[i] * v[i];
            }
            double inv = 1 / Math.Sqrt(square + epsilon);
            CheckValue(actualG[row], dot * inv + (accumulate ? oldG[row] : 0), $"dGain[{row}]");
            for (int j = 0; j < columns; j++)
            {
                int i = row * columns + j;
                CheckValue(actualW[i], v[i] * (g[row] * inv), $"output[{i}]");
                double expected = (g[row] * inv) * (h[i] - v[i] * (dot * inv * inv)) + (accumulate ? oldV[i] : 0);
                CheckValue(actualV[i], expected, $"dDirection[{i}]");
            }
        }
    }

    private static void CheckValue(float actual, double expected, string label)
        => Assert.True(float.IsFinite(actual) && Math.Abs(actual - expected) <= 5e-5 * (1 + Math.Abs(expected)),
            $"{label}: {actual:G9} != {expected:G17}");

    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Cuda_weight_norm_has_native_forward_and_vjp(bool accumulate)
    {
        using var backend = new CudaBackend(); Skip.IfNot(backend.IsAvailable, "CUDA unavailable.");
        Check(backend, accumulate);
    }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Hip_weight_norm_has_native_forward_and_vjp(bool accumulate)
    {
        using var backend = new HipBackend(); Skip.IfNot(backend.IsAvailable, "HIP unavailable.");
        Check(backend, accumulate);
    }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void OpenCl_weight_norm_has_native_forward_and_vjp(bool accumulate)
    {
        using var backend = new OpenClBackend(); Skip.IfNot(backend.IsAvailable, "OpenCL unavailable.");
        Check(backend, accumulate);
    }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Metal_weight_norm_has_native_forward_and_vjp(bool accumulate)
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal unavailable on this platform.");
        using var backend = new MetalBackend(); Skip.IfNot(backend.IsAvailable, "Metal unavailable.");
        Check(backend, accumulate);
    }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Vulkan_weight_norm_has_native_forward_and_vjp(bool accumulate)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler unavailable.");
        Check(backend, accumulate);
    }
#if NET7_0_OR_GREATER
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void WebGpu_weight_norm_has_native_forward_and_vjp(bool accumulate)
        => Check(DawnHostedWebGpu.BackendOrSkip(), accumulate);
#endif
}
#endif
