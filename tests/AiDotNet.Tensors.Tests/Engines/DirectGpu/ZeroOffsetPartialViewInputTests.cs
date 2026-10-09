using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A contiguous view that starts at offset 0 but covers only part of its storage (the first row of a matrix) shares
/// its storage's backing array, and every buffer lookup keyed on that array returns the WHOLE storage. A GPU op on
/// such a view must read the view's own elements and return the view's length.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ZeroOffsetPartialViewInputTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ZeroOffsetPartialViewInputTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void FirstRowView_IsReadAsItsOwnElements()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine;
        Skip.If(gpu is null, "No GPU engine.");
        var data = new float[12];
        for (int i = 0; i < data.Length; i++) data[i] = i + 1;
        var full = new Tensor<float>(data, new[] { 3, 4 });
        var row = full.Slice(0);
        Assert.Equal(4, row.Length);

        var negated = gpu.TensorNegate(row).ToArray();

        Assert.Equal(new[] { -1f, -2f, -3f, -4f }, negated);
    }

    [SkippableFact]
    public void FirstRowView_OfADeviceResidentMatrix_IsReadAsItsOwnElements()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine;
        Skip.If(gpu is null, "No GPU engine.");
        var data = new float[12];
        for (int i = 0; i < data.Length; i++) data[i] = i + 1;
        var full = gpu.TensorAdd(new Tensor<float>(data, new[] { 3, 4 }), new Tensor<float>(new float[12], new[] { 3, 4 }));
        var row = full.Slice(0);

        var negated = gpu.TensorNegate(row).ToArray();

        Assert.Equal(new[] { -1f, -2f, -3f, -4f }, negated);
    }
}
