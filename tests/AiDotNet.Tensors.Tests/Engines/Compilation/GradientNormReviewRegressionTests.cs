using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class GradientNormReviewRegressionTests
{
    [Fact]
    public void InvalidPolicy_IsRejectedBeforeEmptyOrCpuFallback()
    {
        var invalid = (NonFiniteGradientPolicy)99;
        Assert.Throws<ArgumentOutOfRangeException>(() => GradientClipping.ClipGradNorm(Array.Empty<Tensor<float>>(), 1f, invalid));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuOptimizer.TryClipGradNorm(Array.Empty<Tensor<float>>(), 1.0, invalid));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuOptimizer.TryClipGradNorm(Array.Empty<Tensor<float>>(), double.MaxValue, invalid));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(4096, 1)]
    [InlineData(4097, 2)]
    [InlineData(int.MaxValue, 524288)]
    public void ChunkCount_DoesNotOverflowAtIntMax(int length, int expected)
        => Assert.Equal(expected, DeterministicGradientNorm.ChunkCount(length));

    [Fact]
    public void InfiniteBound_DoesNotClipANormAboveFloatRange()
    {
        var gradient = new Tensor<float>(new[] { 4 });
        gradient[0] = float.MaxValue;
        gradient[1] = float.MaxValue;
        gradient[2] = -float.MaxValue;
        gradient[3] = -float.MaxValue;
        var before = gradient.ToArray();
        GradientClipping.ClipGradNorm(new[] { gradient }, float.PositiveInfinity);
        Assert.Equal(before, gradient.ToArray());
    }
}
