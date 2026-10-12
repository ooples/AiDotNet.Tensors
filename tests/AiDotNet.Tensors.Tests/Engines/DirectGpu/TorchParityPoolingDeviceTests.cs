using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1112 tier 1: adaptive average / max pooling over 1 and 3 axes and Lp pooling run on the device (a gather of each
/// window's taps and a device reduction). Shapes include non-divisible adaptive sizes (overlapping, unequal windows),
/// inputs include negatives and a NaN, and Lp runs at an integer and a fractional power. Each case must match the CPU
/// element by element (NaN only where the CPU is NaN), launch device work, and record no host fallback.
/// </summary>
public abstract class TorchParityPoolingDeviceTestsBase
{
    private readonly IDirectGpuTensorEngineFixture _fixture;
    protected TorchParityPoolingDeviceTestsBase(IDirectGpuTensorEngineFixture fixture) => _fixture = fixture;

    private static Tensor<float> Input(int salt, bool withNaN, params int[] shape)
    {
        var rng = new Random(salt);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() - 0.5) * 6);
        if (withNaN) t[t.Length / 3] = float.NaN;
        return t;
    }

    private static void AssertMatches(string op, Tensor<float> cpu, Tensor<float> gpu)
    {
        Assert.Equal(cpu._shape, gpu._shape);
        for (int i = 0; i < cpu.Length; i++)
        {
            float c = cpu[i], g = gpu[i];
            if (float.IsNaN(c) || float.IsNaN(g)) { Assert.True(float.IsNaN(c) && float.IsNaN(g), $"{op}[{i}]: cpu {c} gpu {g}"); continue; }
            Assert.True(Math.Abs(c - g) <= 1e-5 * Math.Max(1.0, Math.Abs(c)), $"{op}[{i}]: cpu {c:R} gpu {g:R}");
        }
    }

    public static TheoryData<string> Cases() => new()
    {
        "avg1d", "avg1dNaN", "avg3d", "max1d", "max3d", "lp2", "lp3neg", "lp1_5", "lp3dStride",
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Pooling_MatchesCpu_OnDevice(string op)
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        var x1 = Input(1, op.EndsWith("NaN"), 3, 5, 37);
        var x3 = Input(2, false, 2, 3, 7, 9, 11);
        Func<IEngine, Tensor<float>> f = op switch
        {
            "avg1d" or "avg1dNaN" => e => e.TensorAdaptiveAvgPool1D(x1, 10),
            "avg3d" => e => e.TensorAdaptiveAvgPool3D(x3, new[] { 3, 4, 5 }),
            "max1d" => e => e.TensorAdaptiveMaxPool1D(x1, 10),
            "max3d" => e => e.TensorAdaptiveMaxPool3D(x3, new[] { 3, 4, 5 }),
            "lp2" => e => e.TensorLpPool(x1, 2.0, new[] { 4 }),
            "lp3neg" => e => e.TensorLpPool(x1, 3.0, new[] { 5 }, new[] { 3 }),
            "lp1_5" => e => e.TensorLpPool(x1, 1.5, new[] { 3 }),
            _ => e => e.TensorLpPool(x3, 2.0, new[] { 2, 3, 2 }, new[] { 2, 2, 3 }),
        };
        string probe = op.StartsWith("avg") ? (op.Contains("3d") ? "TensorAdaptiveAvgPool3D" : "TensorAdaptiveAvgPool1D")
            : op.StartsWith("max") ? (op.Contains("3d") ? "TensorAdaptiveMaxPool3D" : "TensorAdaptiveMaxPool1D")
            : "TensorLpPool";

        var expected = f(cpu);
        long launches = GpuLaunchProbe.Count;
        long fallbacks = TorchParityElementwiseDeviceTests.FallbackCount(probe);
        var actual = f(gpu);
        _ = actual[0];
        Assert.True(GpuLaunchProbe.Count > launches, $"{op}: no device work launched");
        Assert.Equal(fallbacks, TorchParityElementwiseDeviceTests.FallbackCount(probe));
        AssertMatches(op, expected, actual);
    }
}

/// <summary>The cases above on the machine's default DirectGpu backend.</summary>
[Collection("VulkanGlobalState")]
public class TorchParityPoolingDeviceTests : TorchParityPoolingDeviceTestsBase, IClassFixture<DirectGpuTensorEngineTestFixture>
{
    public TorchParityPoolingDeviceTests(DirectGpuTensorEngineTestFixture fixture) : base(fixture) { }
}

/// <summary>The cases above on the Vulkan backend (#1127).</summary>
[Collection("VulkanGlobalState")]
public class TorchParityPoolingVulkanDeviceTests : TorchParityPoolingDeviceTestsBase, IClassFixture<VulkanTensorEngineTestFixture>
{
    public TorchParityPoolingVulkanDeviceTests(VulkanTensorEngineTestFixture fixture) : base(fixture) { }
}
