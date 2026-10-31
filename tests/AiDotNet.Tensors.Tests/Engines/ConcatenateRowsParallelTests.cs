using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A concatenation past the parallel threshold copies its leading-index rows in parallel. Every output element must be
/// the matching input element: inputs of different widths along the channel axis, several rows.
/// </summary>
public sealed class ConcatenateRowsParallelTests
{
    [Fact]
    public void ChannelConcat_OfLargeMaps_MatchesTheInputs()
    {
        int n = 6, h = 16, w = 16;
        int[] channels = { 64, 32, 96 };
        var inputs = new Tensor<float>[channels.Length];
        for (int i = 0; i < channels.Length; i++)
        {
            var d = new float[n * channels[i] * h * w];
            for (int j = 0; j < d.Length; j++) d[j] = i * 1_000_000 + j;
            inputs[i] = new Tensor<float>(d, new[] { n, channels[i], h, w });
        }
        var y = new CpuEngine().TensorConcatenate(inputs, 1);
        Assert.Equal(new[] { n, 192, h, w }, y.Shape.ToArray());
        var yd = y.ToArray();
        int plane = h * w;
        for (int b = 0; b < n; b++)
        {
            int c0 = 0;
            for (int i = 0; i < channels.Length; i++)
            {
                for (int c = 0; c < channels[i]; c++)
                    for (int p = 0; p < plane; p++)
                        Assert.Equal(i * 1_000_000 + (b * channels[i] + c) * plane + p,
                            yd[(b * 192 + c0 + c) * plane + p]);
                c0 += channels[i];
            }
        }
    }
}