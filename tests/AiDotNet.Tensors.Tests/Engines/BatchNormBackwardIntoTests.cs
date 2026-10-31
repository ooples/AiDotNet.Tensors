using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The compiled training plan's batch-norm backward writes into its own buffers. For float 4D input the kernel writes,
/// or adds to, those buffers directly instead of computing into rented tensors and copying. It must give the allocating
/// backward's gradients bit for bit: overwriting, and accumulating onto values already in the buffers.
/// </summary>
public sealed class BatchNormBackwardIntoTests
{
    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        int length = 1;
        foreach (int s in shape) length *= s;
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    [Theory]
    [InlineData(false, 3, 5, 4, 6)]
    [InlineData(true, 3, 5, 4, 6)]
    [InlineData(true, 2, 9, 3, 3)]    // 9-element planes: a vector body and a scalar tail
    public void MatchesTheAllocatingBackward(bool accumulate, int n, int c, int h, int w)
    {
        var e = new CpuEngine();
        var gradOutput = Rand(1, n, c, h, w);
        var input = Rand(2, n, c, h, w);
        var gamma = Rand(3, c);
        var mean = Rand(4, c);
        var varData = Rand(5, c).ToArray();
        for (int i = 0; i < c; i++) varData[i] = Math.Abs(varData[i]) + 0.25f;
        var variance = new Tensor<float>(varData, new[] { c });

        var gi = e.BatchNormBackward(gradOutput, input, gamma, mean, variance, 1e-5, out var gg, out var gb);

        var prior = Rand(6, n, c, h, w).ToArray();
        var priorGamma = Rand(7, c).ToArray();
        var priorBeta = Rand(8, c).ToArray();
        var destInput = new Tensor<float>((float[])prior.Clone(), new[] { n, c, h, w });
        var destGamma = new Tensor<float>((float[])priorGamma.Clone(), new[] { c });
        var destBeta = new Tensor<float>((float[])priorBeta.Clone(), new[] { c });
        e.BatchNormBackwardInto(destInput, destGamma, destBeta, gradOutput, input, gamma, mean, variance, 1e-5,
            accumulate, accumulate, accumulate);

        AssertMerged(prior, gi.ToArray(), destInput.ToArray(), accumulate);
        AssertMerged(priorGamma, gg.ToArray(), destGamma.ToArray(), accumulate);
        AssertMerged(priorBeta, gb.ToArray(), destBeta.ToArray(), accumulate);
    }

    private static void AssertMerged(float[] prior, float[] gradient, float[] actual, bool accumulate)
    {
        Assert.Equal(gradient.Length, actual.Length);
        for (int i = 0; i < gradient.Length; i++)
        {
            float expected = accumulate ? prior[i] + gradient[i] : gradient[i];
            Assert.True(expected.Equals(actual[i]), $"[{i}]: expected {expected}, got {actual[i]}");
        }
    }
}
