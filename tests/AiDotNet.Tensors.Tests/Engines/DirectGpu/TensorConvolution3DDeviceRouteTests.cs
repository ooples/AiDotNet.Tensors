using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// TensorConvolution's 3-D path called the non-virtual array overload of Conv3D, so even on a device engine it ran on
/// the host. A uniform geometry must reach the device Conv3D (launches, no recorded fallback) and match the CPU; a
/// per-axis geometry stays on the host and records its fallback.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class TensorConvolution3DDeviceRouteTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public TensorConvolution3DDeviceRouteTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Seq(int[] shape, float scale)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)Math.Sin(i * 0.37) * scale;
        return new Tensor<float>(data, shape);
    }

    [SkippableFact]
    public void UniformGeometry_RunsOnTheDevice_AndMatchesTheCpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine;
        Skip.If(gpu is null, "No GPU engine.");
        var x = Seq(new[] { 1, 2, 5, 5, 5 }, 1f);
        var w = Seq(new[] { 3, 2, 3, 3, 3 }, 0.2f);
        var expected = new CpuEngine().TensorConvolution(x, w, null, new[] { 1 }, new[] { 1 }, new[] { 1 }).ToArray();

        GpuLaunchProbe.Reset();
        var actual = gpu.TensorConvolution(x, w, null, new[] { 1 }, new[] { 1 }, new[] { 1 }).ToArray();

        Assert.True(GpuLaunchProbe.Count > 0, "no device kernel launched");
        // Entries read "<count>x <op>: <reason>".
        Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, f => f.Contains("x TensorConvolution:", StringComparison.Ordinal));
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])), $"[{i}] cpu {expected[i]} gpu {actual[i]}");
    }

    [SkippableFact]
    public void PerAxisGeometry_RecordsItsHostFallback()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine;
        Skip.If(gpu is null, "No GPU engine.");
        var x = Seq(new[] { 1, 2, 5, 5, 5 }, 1f);
        var w = Seq(new[] { 3, 2, 3, 3, 3 }, 0.2f);
        var expected = new CpuEngine().TensorConvolution(x, w, null, new[] { 1, 2, 1 }, new[] { 1 }, new[] { 1 }).ToArray();

        GpuLaunchProbe.Reset();
        var actual = gpu.TensorConvolution(x, w, null, new[] { 1, 2, 1 }, new[] { 1 }, new[] { 1 }).ToArray();

        Assert.Contains(GpuLaunchProbe.Fallbacks, f => f.Contains("x TensorConvolution: guard or route declined", StringComparison.Ordinal));
        Assert.Equal(expected, actual);
    }
}
