using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class DynamicShapeTests
{
    [Fact]
    public void FixedShape_ResolvesDirectly()
    {
        var shape = DynamicShape.FromFixed(new[] { 1, 3, 64, 64 });
        var resolved = shape.Resolve(new Dictionary<string, int>());
        Assert.Equal(new[] { 1, 3, 64, 64 }, resolved);
        Assert.False(shape.HasSymbolicDims);
    }

    [Fact]
    public void SymbolicBatch_ResolvesWithBinding()
    {
        var shape = new DynamicShape(
            DynamicDim.Symbolic("batch"),
            DynamicDim.Fixed(256),
            DynamicDim.Fixed(64),
            DynamicDim.Fixed(64));

        Assert.True(shape.HasSymbolicDims);

        var resolved = shape.Resolve(new Dictionary<string, int> { ["batch"] = 4 });
        Assert.Equal(new[] { 4, 256, 64, 64 }, resolved);
    }

    [Fact]
    public void MultipleSymbolicDims_AllResolve()
    {
        var shape = new DynamicShape(
            DynamicDim.Symbolic("batch"),
            DynamicDim.Symbolic("seq_len"),
            DynamicDim.Fixed(512));

        var bindings = new Dictionary<string, int> { ["batch"] = 8, ["seq_len"] = 128 };
        var resolved = shape.Resolve(bindings);

        Assert.Equal(new[] { 8, 128, 512 }, resolved);
    }

    [Fact]
    public void UnboundSymbolic_Throws()
    {
        var shape = new DynamicShape(
            DynamicDim.Symbolic("batch"),
            DynamicDim.Fixed(256));

        Assert.Throws<InvalidOperationException>(() =>
            shape.Resolve(new Dictionary<string, int>())); // "batch" not bound
    }

    [Fact]
    public void TryResolve_ReturnNullWhenUnbound()
    {
        var shape = new DynamicShape(
            DynamicDim.Symbolic("batch"),
            DynamicDim.Fixed(256));

        var result = shape.TryResolve(new Dictionary<string, int>());
        Assert.Null(result);
    }

    [Fact]
    public void TryResolve_ReturnsArrayWhenBound()
    {
        var shape = new DynamicShape(
            DynamicDim.Symbolic("batch"),
            DynamicDim.Fixed(256));

        var result = shape.TryResolve(new Dictionary<string, int> { ["batch"] = 2 });
        Assert.NotNull(result);
        Assert.Equal(new[] { 2, 256 }, result);
    }

    [Fact]
    public void ComputeSize_CorrectProduct()
    {
        var shape = new DynamicShape(
            DynamicDim.Symbolic("n"),
            DynamicDim.Fixed(64),
            DynamicDim.Fixed(64));

        int size = shape.ComputeSize(new Dictionary<string, int> { ["n"] = 4 });
        Assert.Equal(4 * 64 * 64, size);
    }

    [Fact]
    public void GetSymbolicNames_ReturnsUniqueNames()
    {
        var shape = new DynamicShape(
            DynamicDim.Symbolic("batch"),
            DynamicDim.Fixed(3),
            DynamicDim.Symbolic("height"),
            DynamicDim.Symbolic("height")); // duplicate

        var names = shape.GetSymbolicNames();
        Assert.Equal(2, names.Count);
        Assert.Contains("batch", names);
        Assert.Contains("height", names);
    }

    [Fact]
    public void ToString_ShowsDimensions()
    {
        var shape = new DynamicShape(
            DynamicDim.Symbolic("N"),
            DynamicDim.Fixed(256));

        Assert.Equal("[N, 256]", shape.ToString());
    }

    [Fact]
    public void DifferentBatchSizes_SameCompiledGraph()
    {
        // This simulates the key use case: compile once, specialize for different batch sizes
        var shape = new DynamicShape(
            DynamicDim.Symbolic("batch"),
            DynamicDim.Fixed(256),
            DynamicDim.Fixed(32),
            DynamicDim.Fixed(32));

        var batch1 = shape.Resolve(new Dictionary<string, int> { ["batch"] = 1 });
        var batch4 = shape.Resolve(new Dictionary<string, int> { ["batch"] = 4 });
        var batch16 = shape.Resolve(new Dictionary<string, int> { ["batch"] = 16 });

        Assert.Equal(1 * 256 * 32 * 32, batch1[0] * batch1[1] * batch1[2] * batch1[3]);
        Assert.Equal(4 * 256 * 32 * 32, batch4[0] * batch4[1] * batch4[2] * batch4[3]);
        Assert.Equal(16 * 256 * 32 * 32, batch16[0] * batch16[1] * batch16[2] * batch16[3]);
    }
}
