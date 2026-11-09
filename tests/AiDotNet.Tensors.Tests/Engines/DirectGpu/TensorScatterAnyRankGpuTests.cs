// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The GPU TensorScatter ran only [batch, columns] along axis 1 and fell back to the CPU kernel whenever a gradient tape
/// was active, so every training step paid a download and an upload. It now runs at any rank and axis and records its
/// own tape entry; values and both gradients must match the CPU engine.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class TensorScatterAnyRankGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public TensorScatterAnyRankGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Rank3_scatter_under_a_tape_matches_cpu(int axis)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        int[] shape = { 3, 5, 4 };
        const int count = 2;
        var idxShape = (int[])shape.Clone();
        idxShape[axis] = count;
        int outer = shape.Take(axis).Aggregate(1, (a, b) => a * b), inner = shape.Skip(axis + 1).Aggregate(1, (a, b) => a * b);
        var idx = new int[outer * count * inner];
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < count; j++)
                for (int k = 0; k < inner; k++)
                    idx[(o * count + j) * inner + k] = (j * 2 + o + k) % shape[axis]; // distinct within a fibre
        var indices = new Tensor<int>(idx, idxShape);

        var (cpuY, cpuGd, cpuGs) = Run(new CpuEngine(), shape, idxShape, indices, axis);
        bool previous = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var (gpuY, gpuGd, gpuGs) = Run(_fixture.Engine!, shape, idxShape, indices, axis);
            Assert.Equal(cpuY, gpuY);
            AssertClose(cpuGd, gpuGd);
            AssertClose(cpuGs, gpuGs);
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = previous;
        }
    }

    [SkippableFact]
    public void Gpu_index_copy_add_and_set_slice_reject_shapes_the_cpu_rejects()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var input = new Tensor<float>(new[] { 2, 3, 4 });
        var idx = new Tensor<int>(new[] { 0, 2 }, new[] { 2 });
        var sameLengthWrongShape = new Tensor<float>(new[] { 2, 4, 2 });
        Assert.Throws<ArgumentException>(() => gpu.Scatter(input, idx, sameLengthWrongShape, 1));
        Assert.Throws<ArgumentException>(() => gpu.ScatterAdd(input, idx, sameLengthWrongShape, 1));
        Assert.Throws<ArgumentException>(() => gpu.TensorSetSliceAxis(input, new Tensor<float>(new[] { 2, 3, 4 }), 1, 0));
        gpu.TensorSetSliceAxis(input, new Tensor<float>(new[] { 2, 4 }), 1, 0);
    }

    private static (float[] Y, float[] GradDestination, float[] GradSource) Run(
        IEngine engine, int[] shape, int[] idxShape, Tensor<int> indices, int axis)
    {
        int n = shape.Aggregate(1, (a, b) => a * b), m = idxShape.Aggregate(1, (a, b) => a * b);
        var dst = new Tensor<float>(Enumerable.Range(0, n).Select(i => i * 0.25f).ToArray(), shape);
        var src = new Tensor<float>(Enumerable.Range(0, m).Select(i => 100f + i).ToArray(), idxShape);
        var w = new Tensor<float>(Enumerable.Range(0, n).Select(i => 1f + (i % 7) * 0.5f).ToArray(), shape);
        using var tape = new GradientTape<float>();
        var y = engine.TensorScatter(dst, indices, src, axis);
        var loss = engine.ReduceSum(engine.TensorMultiply(y, w), null, keepDims: false);
        Dictionary<Tensor<float>, Tensor<float>> g = tape.ComputeGradients(loss, new[] { dst, src });
        return (y.ToArray(), g[dst].ToArray(), g[src].ToArray());
    }

    private static void AssertClose(float[] e, float[] a)
    {
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= 1e-5f * (1 + Math.Abs(e[i])), $"element {i}: gpu {a[i]} vs cpu {e[i]}");
    }
}
