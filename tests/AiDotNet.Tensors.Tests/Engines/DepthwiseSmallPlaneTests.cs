using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A float depthwise convolution at stride 1 whose output rows are shorter than two vectors runs a padded-tile vector
/// kernel. It must equal the direct sum over in-range taps for 4-, 8- and 12-wide outputs, padded and unpadded, with a
/// channel multiplier above one.
/// </summary>
public sealed class DepthwiseSmallPlaneTests
{
    [Theory]
    [InlineData(3, 5, 1, 4, 4, 1)]     // 4x4 output, padded
    [InlineData(2, 4, 2, 8, 8, 1)]     // 8x8 output, multiplier 2
    [InlineData(2, 3, 1, 12, 10, 1)]   // 12-wide rows: one full vector and a partial one
    [InlineData(2, 3, 1, 6, 6, 0)]     // unpadded: 4x4 output
    public void EqualsTheDirectSum(int n, int c, int multiplier, int h, int w, int pad)
    {
        var rng = new Random(9);
        var x = new float[n * c * h * w]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        int oc = c * multiplier;
        var k = new float[oc * 9]; for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        var y = new CpuEngine().DepthwiseConv2D(new Tensor<float>(x, new[] { n, c, h, w }),
            new Tensor<float>(k, new[] { c, multiplier, 3, 3 }), new[] { 1, 1 }, new[] { pad, pad }).ToArray();
        int oh = h + 2 * pad - 2, ow = w + 2 * pad - 2;
        Assert.Equal(n * oc * oh * ow, y.Length);
        for (int b = 0; b < n; b++)
            for (int o = 0; o < oc; o++)
                for (int r = 0; r < oh; r++)
                    for (int q = 0; q < ow; q++)
                    {
                        double sum = 0;
                        for (int kh = 0; kh < 3; kh++)
                            for (int kw = 0; kw < 3; kw++)
                            {
                                int ih = r + kh - pad, iw = q + kw - pad;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                sum += (double)k[o * 9 + kh * 3 + kw] * x[((b * c + o / multiplier) * h + ih) * w + iw];
                            }
                        float got = y[((b * oc + o) * oh + r) * ow + q];
                        Assert.True(Math.Abs(sum - got) <= 1e-5, $"[{b},{o},{r},{q}]: {got} vs {sum}");
                    }
    }
}