#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every backend's <see cref="IMultiTensorKernels"/> port must produce PyTorch's clip_grad_norm_ result: the scale
/// min(1, maxNorm / (‖g‖ + 1e-6)) over all tensors together, applied to every element; 1 when the norm is within
/// the bound or not finite. Tensor sizes straddle the 256-element chunk and work-group boundaries.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MultiTensorBackendParityTests
{
    private static readonly int[] Sizes = { 1, 255, 256, 257, 1000 };

    private static float[][] MakeTensors(bool poisonWithNaN)
    {
        var data = new float[Sizes.Length][];
        for (int t = 0; t < Sizes.Length; t++)
        {
            data[t] = new float[Sizes[t]];
            for (int i = 0; i < Sizes[t]; i++) data[t][i] = (((t * 131 + i * 17) % 29) - 14) * 0.0625f;
        }
        if (poisonWithNaN) data[3][100] = float.NaN;
        return data;
    }

    private static float ClipAndCheck(IDirectGpuBackend backend, float maxNorm, bool poisonWithNaN)
    {
        var multi = (IMultiTensorKernels)backend;
        var data = MakeTensors(poisonWithNaN);
        double sumSq = 0;
        foreach (var tensor in data)
            foreach (float v in tensor) sumSq += (double)v * v;
        double norm = Math.Sqrt(sumSq);
        float expectedScale = double.IsFinite(norm) ? (float)Math.Min(1.0, maxNorm / (norm + 1e-6)) : 1f;

        var buffers = new List<IGpuBuffer>();
        try
        {
            foreach (var tensor in data) buffers.Add(backend.AllocateBuffer(tensor));
            using var sum = backend.AllocateBuffer(new float[2]);
            using var scale = backend.AllocateBuffer(new float[1]);
            // Twice: the second call must start from zero, not add to the first.
            multi.MultiTensorSumOfSquares(buffers, Sizes, sum);
            multi.MultiTensorSumOfSquares(buffers, Sizes, sum);
            multi.ClipScaleFromSumOfSquares(sum, maxNorm, scale);
            float actualScale = backend.DownloadBuffer(scale)[0];
            Assert.True(Math.Abs(actualScale - expectedScale) <= 1e-5f * Math.Max(1f, expectedScale),
                $"scale: expected {expectedScale}, got {actualScale}");

            multi.MultiTensorScaleByDeviceScalar(buffers, Sizes, scale);
            for (int t = 0; t < data.Length; t++)
            {
                var scaled = backend.DownloadBuffer(buffers[t]);
                for (int i = 0; i < Sizes[t]; i++)
                {
                    float expected = data[t][i] * actualScale;
                    if (float.IsNaN(expected)) { Assert.True(float.IsNaN(scaled[i]), $"tensor {t}[{i}] should stay NaN"); continue; }
                    Assert.Equal(expected, scaled[i]);
                }
            }
            return actualScale;
        }
        finally
        {
            foreach (var buffer in buffers) buffer.Dispose();
        }
    }

    private static void Check(IDirectGpuBackend backend)
    {
        Assert.True(ClipAndCheck(backend, maxNorm: 1f, poisonWithNaN: false) < 1f, "maxNorm 1 must clip");
        Assert.Equal(1f, ClipAndCheck(backend, maxNorm: 1e6f, poisonWithNaN: false));
        Assert.Equal(1f, ClipAndCheck(backend, maxNorm: 1f, poisonWithNaN: true));
    }

    [SkippableFact]
    public void OpenCl_MultiTensorClip_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_MultiTensorClip_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable,
            "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_MultiTensorClip_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif