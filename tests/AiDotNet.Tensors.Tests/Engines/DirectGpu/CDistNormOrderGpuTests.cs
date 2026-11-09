using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The cdist and pdist kernels computed (sum |diff|^p)^(1/p) for every p: infinite at p = 0 and 1 at p = +inf. They
/// must agree with the CPU engine, whose p = 0 counts differing coordinates and p = +inf takes the largest difference.
/// </summary>
[Collection("DirectGpuSerial")]
public class CDistNormOrderGpuTests
{
    [SkippableTheory]
    [InlineData(0.0)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(3.0)]
    [InlineData(1.0)]
    public void Kernels_match_the_cpu_for_every_norm_order(double p)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        var cpu = new CpuEngine();
        // Integer-valued inputs, with repeats, so p = 0 has equal coordinates to skip and the norms are exact.
        var x1 = new Tensor<float>(new[] { 4, 5 }, new Vector<float>(Enumerable.Range(0, 20).Select(i => (float)(i * 7 % 5)).ToArray()));
        var x2 = new Tensor<float>(new[] { 3, 5 }, new Vector<float>(Enumerable.Range(0, 15).Select(i => (float)(i * 3 % 4)).ToArray()));
        bool saved = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        try
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            GpuLaunchProbe.Reset();
            var cd = gpu.TensorCDist(x1, x2, p).ToArray();
            var pd = gpu.TensorPDist(x1, p).ToArray();
            Assert.True(GpuLaunchProbe.Count > 0, "No GPU work was launched.");
            AssertClose(cpu.TensorCDist(x1, x2, p).ToArray(), cd);
            AssertClose(cpu.TensorPDist(x1, p).ToArray(), pd);
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = saved;
        }
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * (1 + Math.Abs(expected[i])),
                $"element {i}: cpu {expected[i]} gpu {actual[i]}");
    }
}
