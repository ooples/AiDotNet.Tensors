// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Linq;
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

    // Under the op lease: a backend lookup outside one fails the strict-lease run (AIDOTNET_GPU_LEASE_STRICT=1).
    private static bool HasSmallTopK(DirectGpuTensorEngine gpu)
    {
        using (gpu.EnterGpuOp())
            return gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.ISmallTopKRowsBackend;
    }

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
        Skip.IfNot(HasSmallTopK(gpu), "Backend has no small-row TopK kernel.");
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

    // NaN and signed zeros: the kernel must order them as the CPU engine does (a NaN's rank, -0 against +0 and ties),
    // bit for bit in the values and exactly in the indices.
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    public void Small_row_top_k_orders_nan_and_signed_zeros_as_the_cpu_engine(int k)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(HasSmallTopK(gpu), "Backend has no small-row TopK kernel.");
        float nan = float.NaN, nz = -0f;
        float[][] rows =
        {
            new[] { 0f, nz, 0f, nz, 1f, -1f, nz, 0f },
            new[] { nz, nz, 0f, 0f, nz, 0f, nz, 0f },
            new[] { nan, 1f, nz, 0f, -1f, 2f, nan, -2f },
            new[] { 1f, nan, nan, 1f, nz, nan, 0f, -1f },
            new[] { nan, nan, nan, nan, nan, nan, nan, nan },
            new[] { float.NegativeInfinity, nan, float.PositiveInfinity, nz, 0f, nan, float.NegativeInfinity, 3f },
        };
        int cols = rows[0].Length;
        var x = new Tensor<float>([rows.Length, cols]);
        for (int r = 0; r < rows.Length; r++)
            for (int c = 0; c < cols; c++) x[r * cols + c] = rows[r][c];

        var cpuTop = new CpuEngine().TensorTopK(x, k, 1, out Tensor<int> cpuIdx);
        int before = gpu.TopKSmallRowsCompleted;
        var gpuTop = ((IEngine)gpu).TensorTopK(x, k, 1, out Tensor<int> gpuIdx);
        Assert.Equal(before + 1, gpu.TopKSmallRowsCompleted);

        Assert.Equal(cpuIdx.ToArray(), gpuIdx.ToArray());
        Assert.Equal(cpuTop.ToArray().Select(TestHelpers.MathCompat.SingleToInt32Bits).ToArray(),
            gpuTop.ToArray().Select(TestHelpers.MathCompat.SingleToInt32Bits).ToArray());
    }
}
