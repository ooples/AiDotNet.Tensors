// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The CUDA column-concat copy launched one block row per tensor row, so a concat with more than 65535 rows (the
/// grid-Y limit) was an invalid-value launch and the op silently fell back to the CPU (TensorCartesianProd hit it).
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class TallStridedCopyTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public TallStridedCopyTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void ColumnConcat_TallerThanGridLimit_RunsOnDeviceAndMatchesCpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        IEngine cpu = new CpuEngine();
        const int rows = 70_001;   // > 65535, and not a multiple of it
        var a = new Tensor<float>(new[] { rows, 2 });
        var b = new Tensor<float>(new[] { rows, 3 });
        for (int i = 0; i < a.Length; i++) a[i] = i * 0.5f;
        for (int i = 0; i < b.Length; i++) b[i] = -i * 0.25f;

        GpuLaunchProbe.Reset();
        var got = gpu.TensorConcatenate(new[] { a, b }, 1);
        var fallbacks = GpuLaunchProbe.Fallbacks;
        var want = cpu.TensorConcatenate(new[] { a, b }, 1);

        Assert.Empty(fallbacks);
        Assert.Equal(want.Shape.ToArray(), got.Shape.ToArray());
        var g = got.ToArray();
        var w = want.ToArray();
        for (int i = 0; i < w.Length; i++)
            Assert.True(g[i] == w[i], $"element {i}: gpu {g[i]} != cpu {w[i]}");
    }
}
