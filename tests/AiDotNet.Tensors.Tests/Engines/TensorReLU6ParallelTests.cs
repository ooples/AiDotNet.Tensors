using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The float TensorReLU6 runs vectorized in 64K-element chunks across threads. Every element must equal
/// min(max(0, x), 6), including the chunk boundaries and a ragged last chunk, and NaN must stay NaN.
/// </summary>
public sealed class TensorReLU6ParallelTests
{
    [Theory]
    [InlineData(5)]                     // smaller than one vector
    [InlineData(65_536 * 3 + 37)]       // several chunks and a ragged tail
    public void EveryElement_IsClampedToZeroAndSix(int length)
    {
        var rng = new Random(7);
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(rng.NextDouble() * 20 - 7);
        data[0] = float.NaN;
        data[length - 1] = float.NaN;
        if (length > 65_536) data[65_536] = float.NaN;   // first element of the second chunk

        var y = new CpuEngine().TensorReLU6(new Tensor<float>(data, new[] { length })).ToArray();
        for (int i = 0; i < length; i++)
        {
            if (float.IsNaN(data[i])) { Assert.True(float.IsNaN(y[i]), $"[{i}]: NaN became {y[i]}"); continue; }
            Assert.Equal(Math.Min(Math.Max(data[i], 0f), 6f), y[i]);
        }
    }
}