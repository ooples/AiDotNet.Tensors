using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A float depthwise convolution at stride 1 whose output rows are shorter than two vectors runs a padded-tile vector
/// kernel; a same-padded one (k = 2 * pad + 1) reads each plane as one flat run. Both must equal the direct sum over
/// in-range taps for 4-, 8- and 12-wide outputs, padded and unpadded, with a channel multiplier above one.
/// </summary>
public sealed class DepthwiseSmallPlaneTests
{
    [Theory]
    [InlineData(3, 5, 1, 4, 4, 1)]     // 4x4 output, padded
    [InlineData(2, 4, 2, 8, 8, 1)]     // 8x8 output, multiplier 2
    [InlineData(2, 3, 1, 10, 12, 1)]   // 12-wide rows: one full vector and a partial one
    [InlineData(2, 3, 1, 6, 6, 0)]     // unpadded: 4x4 output
    [InlineData(4, 60, 1, 4, 4, 1)]    // 240 planes: several plane groups of the same-padded route
    [InlineData(2, 3, 1, 3, 5, 1)]     // 15 outputs: a ragged width and a partial vector
    public void EqualsTheDirectSum(int n, int c, int multiplier, int h, int w, int pad)
        => AssertDirectSum(n, c, multiplier, h, w, pad, 3, x => { });

    /// <summary>A 5x5 kernel, padded to keep the size, runs the same-padded route's general (non-3x3) loop.</summary>
    [Theory]
    [InlineData(2, 5, 1, 7, 7)]
    [InlineData(2, 3, 2, 4, 6)]
    public void FiveByFiveSamePadding_EqualsTheDirectSum(int n, int c, int multiplier, int h, int w)
        => AssertDirectSum(n, c, multiplier, h, w, 2, 5, x => { });

    /// <summary>
    /// The same-padded route reads a plane as one flat run, so a tap past a row's end lands on the next row's first
    /// pixel and must be masked. An infinite pixel at a row's end must reach only its own 3x3 neighbourhood: a masked
    /// tap that multiplied the weight instead of zeroing the input would turn the next row's outputs into NaN.
    /// </summary>
    [Fact]
    public void AnInfinitePixel_ReachesOnlyItsNeighbourhood()
        => AssertDirectSum(1, 2, 1, 4, 4, 1, 3, x => x[1 * 4 + 3] = float.PositiveInfinity);

    /// <summary>
    /// The same-padded route keeps its tile between calls and clears it only when the slot geometry changes. A 4x3
    /// plane and a 3x3 plane share lead (4) and slot (20 at 4 lanes, 24 at 8 and 16) at every vector width, but the
    /// 4x3 one writes three more pixels: run second, the 3x3 plane must still read a zero bottom margin, not the 4x3
    /// plane's last row.
    /// </summary>
    [Fact]
    public void ASmallerPlaneWithTheSameSlot_ReadsZeroMargins()
    {
        AssertDirectSum(1, 2, 1, 4, 3, 1, 3, x => { for (int i = 0; i < x.Length; i++) x[i] = 3f; });
        AssertDirectSum(1, 2, 1, 3, 3, 1, 3, x => { });
    }

    private static void AssertDirectSum(int n, int c, int multiplier, int h, int w, int pad, int kSize, Action<float[]> edit)
    {
        var rng = new Random(9);
        var x = new float[n * c * h * w]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        edit(x);
        int oc = c * multiplier, taps = kSize * kSize;
        var k = new float[oc * taps]; for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        var y = new CpuEngine().DepthwiseConv2D(new Tensor<float>(x, new[] { n, c, h, w }),
            new Tensor<float>(k, new[] { c, multiplier, kSize, kSize }), new[] { 1, 1 }, new[] { pad, pad }).ToArray();
        int oh = h + 2 * pad - kSize + 1, ow = w + 2 * pad - kSize + 1;
        Assert.Equal(n * oc * oh * ow, y.Length);
        for (int b = 0; b < n; b++)
            for (int o = 0; o < oc; o++)
                for (int r = 0; r < oh; r++)
                    for (int q = 0; q < ow; q++)
                    {
                        double sum = 0;
                        for (int kh = 0; kh < kSize; kh++)
                            for (int kw = 0; kw < kSize; kw++)
                            {
                                int ih = r + kh - pad, iw = q + kw - pad;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                sum += (double)k[o * taps + kh * kSize + kw] * x[((b * c + o / multiplier) * h + ih) * w + iw];
                            }
                        float got = y[((b * oc + o) * oh + r) * ow + q];
                        if (double.IsInfinity(sum)) Assert.Equal(sum, (double)got);
                        else Assert.True(Math.Abs(sum - got) <= 1e-5, $"[{b},{o},{r},{q}]: {got} vs {sum}");
                    }
    }
}