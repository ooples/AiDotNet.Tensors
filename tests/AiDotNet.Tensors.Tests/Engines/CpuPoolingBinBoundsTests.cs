using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Value checks for the CPU pooling windows: adaptive bins must follow PyTorch's
/// <c>[floor(i*In/Out), ceil((i+1)*In/Out))</c> rule the GPU kernels use, and a max-pool window whose
/// elements never beat the seed must still report an index inside its own window.
/// </summary>
public class CpuPoolingBinBoundsTests
{
    [Fact]
    public void AdaptiveMaxPool2D_NonDivisibleSize_UsesCeilingBinEnds()
    {
        // 5x5 ramp 0..24 into 2x2: PyTorch bins are rows/cols [0,3) and [2,5), so each bin's max is its
        // bottom-right element. A truncated end ([0,2) and [2,5)) would give 6 for the first bin.
        var input = new Tensor<float>(new[] { 1, 1, 5, 5 });
        for (int i = 0; i < 25; i++) input[0, 0, i / 5, i % 5] = i;

        var output = new CpuEngine().TensorAdaptiveMaxPool2D(input, new[] { 2, 2 });

        Assert.Equal(new[] { 1, 1, 2, 2 }, output.Shape.ToArray());
        Assert.Equal(12f, output[0, 0, 0, 0]);
        Assert.Equal(14f, output[0, 0, 0, 1]);
        Assert.Equal(22f, output[0, 0, 1, 0]);
        Assert.Equal(24f, output[0, 0, 1, 1]);
    }

    [Fact]
    public void MaxPool2DWithTensorIndices_AllNegativeInfinity_IndexStaysInsideEachWindow()
    {
        var input = new Tensor<float>(new[] { 1, 1, 4, 4 });
        for (int i = 0; i < 16; i++) input[0, 0, i / 4, i % 4] = float.NegativeInfinity;

        new CpuEngine().MaxPool2DWithTensorIndices(input, new[] { 2, 2 }, new[] { 2, 2 }, out var indices);

        // Window origins of a 2x2/stride-2 pool over a 4x4 plane, as flat plane offsets.
        var expected = new[] { 0, 2, 8, 10 };
        var actual = indices.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int k = 0; k < expected.Length; k++) Assert.Equal(expected[k], actual[k]);
    }

    [Theory]
    [InlineData(new[] { 2 }, new[] { 2, 2 })]
    [InlineData(new[] { 0, 2 }, new[] { 2, 2 })]
    [InlineData(new[] { 2, 2 }, new[] { 2, 0 })]
    public void MaxPool2DWithTensorIndices_InvalidWindow_Throws(int[] poolSize, int[] stride)
    {
        var input = new Tensor<float>(new[] { 1, 1, 4, 4 });
        Assert.Throws<ArgumentException>(() =>
            new CpuEngine().MaxPool2DWithTensorIndices(input, poolSize, stride, out _));
    }
}
