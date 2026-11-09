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

    [SkippableTheory]
    [InlineData(2.0)]
    [InlineData(0.0)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(1.5)]
    public void Batched_and_broadcast_cdist_runs_on_the_device(double p)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        var cpu = new CpuEngine();
        // [2, 3, 4, 5] x [3, 2, 5]: the second operand broadcasts over the leading batch dim.
        var x1 = new Tensor<float>(new[] { 2, 3, 4, 5 }, new Vector<float>(Enumerable.Range(0, 120).Select(i => (float)(i * 7 % 5)).ToArray()));
        var x2 = new Tensor<float>(new[] { 3, 2, 5 }, new Vector<float>(Enumerable.Range(0, 30).Select(i => (float)(i * 3 % 4)).ToArray()));
        bool saved = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        try
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            GpuLaunchProbe.Reset();
            var gpuResult = gpu.TensorCDist(x1, x2, p);
            Assert.True(GpuLaunchProbe.Count > 0, "No GPU work was launched.");
            Assert.True(GpuLaunchProbe.Readbacks == 0, "The batched cdist read its result back to the host.");
            var expected = cpu.TensorCDist(x1, x2, p);
            Assert.Equal(expected.Shape.ToArray(), gpuResult.Shape.ToArray());
            AssertClose(expected.ToArray(), gpuResult.ToArray());
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = saved;
        }
    }

    [SkippableFact]
    public void Max_norm_kernels_propagate_nan_like_the_cpu()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");

        var cpu = new CpuEngine();
        // A NaN coordinate after a larger one: a NaN-dropping max returned 5 here.
        var x1 = new Tensor<float>(new[] { 6f, float.NaN, 0f, 1f, 2f, 3f }, new[] { 2, 3 });
        var x2 = new Tensor<float>(new[] { 1f, 1f, 1f }, new[] { 1, 3 });
        var cd = gpu.TensorCDist(x1, x2, double.PositiveInfinity).ToArray();
        var pd = gpu.TensorPDist(x1, double.PositiveInfinity).ToArray();
        Assert.True(float.IsNaN(cd[0]), $"cdist gave {cd[0]} for a row holding NaN.");
        Assert.True(float.IsNaN(pd[0]), $"pdist gave {pd[0]} for a pair holding NaN.");
        Assert.Equal(cpu.TensorCDist(x1, x2, double.PositiveInfinity).ToArray(), cd);
        Assert.Equal(cpu.TensorPDist(x1, double.PositiveInfinity).ToArray(), pd);
    }

    [SkippableFact]
    public void Direct_backend_calls_take_an_infinite_order_as_the_max_norm()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        var backend = gpu.PlacementBackend ?? throw new InvalidOperationException("The available engine has no backend.");

        // Differences (2, 1): the max-norm is 2; passing +inf straight to the kernel used to give 1.
        using var a = backend.AllocateBuffer(new[] { 3f, 2f });
        using var b = backend.AllocateBuffer(new[] { 1f, 1f });
        using var o = backend.AllocateBuffer(1);
        backend.CDist(a, b, o, 1, 1, 2, float.PositiveInfinity);
        Assert.Equal(2f, backend.DownloadBuffer(o)[0]);
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * (1 + Math.Abs(expected[i])),
                $"element {i}: cpu {expected[i]} gpu {actual[i]}");
    }
}
