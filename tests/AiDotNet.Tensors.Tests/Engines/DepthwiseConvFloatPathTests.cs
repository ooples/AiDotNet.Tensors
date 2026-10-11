using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #639: the float depthwise convolution path (typed arrays, row-vectorised taps for stride 1) must equal, bit for bit,
/// the generic per-element loop it replaces -- each output summing its in-bounds taps in (kh, kw) order with separate
/// multiply and add -- across strides, kernel sizes, paddings, channel multipliers, odd widths and planes smaller than
/// the kernel (taps entirely outside the input), at any thread count.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class DepthwiseConvFloatPathTests
{
    private static float[] Reference(float[] x, float[] w, int batch, int c, int h, int wd, int mult, int k, int s, int p,
        out int oh, out int ow)
    {
        oh = (h + 2 * p - k) / s + 1;
        ow = (wd + 2 * p - k) / s + 1;
        var y = new float[batch * c * mult * oh * ow];
        for (int b = 0; b < batch; b++)
            for (int oc = 0; oc < c * mult; oc++)
                for (int i = 0; i < oh; i++)
                    for (int j = 0; j < ow; j++)
                    {
                        float sum = 0f;
                        for (int kh = 0; kh < k; kh++)
                            for (int kw = 0; kw < k; kw++)
                            {
                                int ih = i * s + kh - p, iw = j * s + kw - p;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= wd) continue;
                                sum += x[((b * c + oc / mult) * h + ih) * wd + iw] * w[(oc * k + kh) * k + kw];
                            }
                        y[((b * c * mult + oc) * oh + i) * ow + j] = sum;
                    }
        return y;
    }

    [Theory]
    [InlineData(1, 16, 16, 16, 1, 3, 1, 1)]
    [InlineData(2, 64, 16, 16, 1, 3, 2, 1)]
    [InlineData(1, 72, 8, 8, 1, 5, 2, 2)]
    [InlineData(1, 120, 4, 4, 1, 5, 1, 2)]
    [InlineData(1, 960, 1, 1, 1, 5, 1, 2)]    // a 1x1 plane under a 5x5 kernel: most taps outside
    [InlineData(1, 8, 2, 2, 1, 5, 2, 0)]      // kernel wider than the padded plane's reach
    [InlineData(1, 4, 3, 3, 1, 5, 2, 1)]      // stride 2: a tap past the right edge (reach -1) must not wrap to the next row
    [InlineData(2, 6, 13, 11, 2, 3, 1, 1)]    // channel multiplier 2, odd sizes
    [InlineData(1, 5, 17, 23, 1, 7, 3, 3)]
    public void FloatPath_EqualsGenericLoop_BitForBit(int batch, int c, int h, int w, int mult, int k, int s, int p)
    {
        var rng = new Random(c * 31 + k);
        var x = new float[batch * c * h * w];
        var kernel = new float[c * mult * k * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(rng.NextDouble() * 2 - 1);
        var expected = Reference(x, kernel, batch, c, h, w, mult, k, s, p, out int oh, out int ow);
        foreach (int dop in new[] { 1, 16 })
        {
            int previous = CpuParallelSettings.MaxDegreeOfParallelism;
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            try
            {
                var y = new CpuEngine().DepthwiseConv2D(new Tensor<float>((float[])x.Clone(), new[] { batch, c, h, w }),
                    new Tensor<float>((float[])kernel.Clone(), new[] { c, mult, k, k }), new[] { s, s }, new[] { p, p });
                Assert.Equal(new[] { batch, c * mult, oh, ow }, y.Shape.ToArray());
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(expected[i].Equals(y[i]), $"[{i}] at DOP {dop}: expected {expected[i]:R} got {y[i]:R}");
            }
            finally
            {
                CpuParallelSettings.MaxDegreeOfParallelism = previous;
            }
        }
    }
}
