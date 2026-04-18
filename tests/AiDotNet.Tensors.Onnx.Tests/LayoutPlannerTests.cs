using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// C8 tests: LayoutPlanner's per-op policy. These lock in the policy so a
/// future cb=16 (AVX-512) flip doesn't silently change which ops get
/// packed. Kept explicit per op rather than table-driven so a regression
/// points at the exact op it broke.
/// </summary>
public class LayoutPlannerTests
{
    [Theory]
    [InlineData("Conv", 16, true)]
    [InlineData("Conv", 15, false)]                // not divisible by cBlock
    [InlineData("BatchNormalization", 32, true)]
    [InlineData("Relu", 8, true)]
    [InlineData("MaxPool", 8, true)]
    [InlineData("AveragePool", 16, true)]
    [InlineData("Gemm", 8, false)]                 // layout-breaking
    [InlineData("Softmax", 8, false)]
    [InlineData("UnknownOp", 8, false)]            // conservative default
    public void PrefersNchwc_MatchesPolicy(string opType, int C, bool expected)
    {
        var shape = new[] { 1, C, 14, 14 };
        Assert.Equal(expected, LayoutPlanner.PrefersNchwc(opType, shape));
    }

    [Theory]
    [InlineData("Gemm", true)]
    [InlineData("MatMul", true)]
    [InlineData("Flatten", true)]
    [InlineData("Transpose", true)]
    [InlineData("GlobalAveragePool", true)]
    [InlineData("Softmax", true)]
    [InlineData("Conv", false)]
    [InlineData("Relu", false)]
    public void RequiresNchw_MatchesPolicy(string opType, bool expected)
    {
        Assert.Equal(expected, LayoutPlanner.RequiresNchw(opType));
    }

    [Fact]
    public void PropagateLayout_PackedThroughPointwise_StaysPacked()
    {
        var outShape = new[] { 1, 16, 14, 14 };
        var r = LayoutPlanner.PropagateLayout("Relu", TensorLayout.Nchwc8, outShape);
        Assert.Equal(TensorLayout.Nchwc8, r);
    }

    [Fact]
    public void PropagateLayout_PackedThroughGemm_FallsToNchw()
    {
        var outShape = new[] { 1, 1000 };
        var r = LayoutPlanner.PropagateLayout("Gemm", TensorLayout.Nchwc8, outShape);
        Assert.Equal(TensorLayout.Nchw, r);
    }

    [Fact]
    public void PropagateLayout_PackedThroughGlobalPool_FallsToNchw()
    {
        // GlobalAveragePool collapses spatial → output is NCHW by policy.
        var outShape = new[] { 1, 16, 1, 1 };
        var r = LayoutPlanner.PropagateLayout("GlobalAveragePool", TensorLayout.Nchwc8, outShape);
        Assert.Equal(TensorLayout.Nchw, r);
    }

    [Fact]
    public void PropagateLayout_NchwInput_NeverSpontaneouslyPacks()
    {
        var outShape = new[] { 1, 16, 14, 14 };
        // Even for a PrefersNchwc op, if input is NCHW the planner stays
        // NCHW — the caller is responsible for inserting the reorder.
        var r = LayoutPlanner.PropagateLayout("Relu", TensorLayout.Nchw, outShape);
        Assert.Equal(TensorLayout.Nchw, r);
    }

    [Fact]
    public void WorthReordering_ThresholdIsTwo()
    {
        Assert.False(LayoutPlanner.WorthReordering(0));
        Assert.False(LayoutPlanner.WorthReordering(1));
        Assert.True(LayoutPlanner.WorthReordering(2));
        Assert.True(LayoutPlanner.WorthReordering(5));
    }
}
