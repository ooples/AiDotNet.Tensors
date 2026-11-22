// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// TensorTopK over short rows (a router's top-k of a few experts) runs as one launch on backends with the small-row
/// kernel. Values and indices must equal the CPU engine's, including ties (lower index first), for every k it serves.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class SmallTopKRowsGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public SmallTopKRowsGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(37, 8, 1)]
    [InlineData(37, 8, 2)]
    [InlineData(5, 8, 8)]
    [InlineData(300, 64, 6)]
    [InlineData(3, 1000, 16)]
    public void Small_row_top_k_matches_the_cpu_engine(int rows, int cols, int k)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.ISmallTopKRowsBackend, "Backend has no small-row TopK kernel.");
        var rng = new Random(rows * 31 + cols * 7 + k);
        var x = new Tensor<float>([rows, cols]);
        // Values on a coarse grid so most rows hold ties.
        for (int i = 0; i < x.Length; i++) x[i] = rng.Next(-4, 5) * 0.25f;

        var cpuTop = new CpuEngine().TensorTopK(x, k, 1, out Tensor<int> cpuIdx);
        int before = gpu.TopKSmallRowsCompleted;
        var gpuTop = ((IEngine)gpu).TensorTopK(x, k, 1, out Tensor<int> gpuIdx);   // the explicit IEngine override, not the CPU base
        Assert.Equal(before + 1, gpu.TopKSmallRowsCompleted);

        Assert.Equal(cpuTop.Shape.ToArray(), gpuTop.Shape.ToArray());
        Assert.Equal(cpuIdx.ToArray(), gpuIdx.ToArray());
        Assert.Equal(cpuTop.ToArray(), gpuTop.ToArray());
    }
}
