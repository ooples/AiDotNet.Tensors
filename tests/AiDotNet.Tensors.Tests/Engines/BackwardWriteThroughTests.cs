using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Two backward write paths that skip a pass. A large float TensorCopy is split across the pool in 1 MB chunks and must
/// copy every element exactly, including into a destination longer than the source. A 1x1 stride-1 input gradient
/// overwrites its destination instead of clearing it and accumulating, so a destination full of stale values must come
/// out the same as accumulating onto zero.
/// </summary>
public sealed class BackwardWriteThroughTests
{
    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return d;
    }

    [Theory]
    [InlineData(256 * 1024)]           // the parallel threshold
    [InlineData(3 * 262144 + 12345)]   // several 1 MB chunks and a ragged tail
    public void LargeTensorCopy_CopiesEveryElement(int length)
    {
        var source = new Tensor<float>(Rand(1, length), new[] { length });
        var destination = new Tensor<float>(new float[length], new[] { length });
        new CpuEngine().TensorCopy(source, destination);
        Assert.Equal(source.ToArray(), destination.ToArray());
    }

    [Fact]
    public void LargeTensorCopy_BetweenOverlappingViews_IsAMove()
    {
        // Two views of one storage, the destination one element ahead: copied chunk by chunk, every chunk's last write
        // would land on the next chunk's first source element before it is read.
        const int length = 3 * 262144 + 12345;
        var original = Rand(5, length + 1);
        var storage = new Tensor<float>((float[])original.Clone(), new[] { length + 1 });
        new CpuEngine().TensorCopy(storage.Slice(0, 0, length), storage.Slice(0, 1, length + 1));
        var actual = storage.ToArray();
        for (int i = 0; i < length; i++)
            Assert.True(actual[i + 1] == original[i], $"[{i + 1}]: {actual[i + 1]}, expected {original[i]}");
    }

    [Theory]
    [InlineData(4, 32, 16, 8, 8)]
    [InlineData(3, 24, 40, 4, 4)]
    public void PointwiseInputGradient_OverwritesAStaleDestination(int n, int inChannels, int outChannels, int h, int w)
    {
        var e = new CpuEngine();
        var gradOutput = new Tensor<float>(Rand(2, n * outChannels * h * w), new[] { n, outChannels, h, w });
        var kernel = new Tensor<float>(Rand(3, outChannels * inChannels), new[] { outChannels, inChannels, 1, 1 });
        int[] inputShape = { n, inChannels, h, w };
        int[] one = { 1, 1 }, zero = { 0, 0 };

        var stale = new Tensor<float>(Rand(4, n * inChannels * h * w), inputShape);
        e.Conv2DBackwardInputInto(stale, gradOutput, kernel, inputShape, one, zero, one, accumulate: false);

        var zeroed = new Tensor<float>(new float[n * inChannels * h * w], inputShape);
        e.Conv2DBackwardInputInto(zeroed, gradOutput, kernel, inputShape, one, zero, one, accumulate: true);

        var expected = zeroed.ToArray();
        var actual = stale.ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i], $"[{i}]: overwrite {actual[i]}, accumulate onto zero {expected[i]}");
    }
}
