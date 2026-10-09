using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The direct blocked conv kernels against a naive double-precision reference, through the engine entry points that
/// route to them. Each case asserts the route is taken, so a predicate change cannot quietly turn these into tests of
/// the im2col path.
/// </summary>
public class DirectConvAvx2Tests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
        return t;
    }

    private static int OutSize(int size, int k, int stride, int pad, int dilation) => (size + 2 * pad - dilation * (k - 1) - 1) / stride + 1;

    private static void AssertClose(double[] expected, Tensor<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i], a = actual.GetFlat(i);
            Assert.True(Math.Abs(e - a) <= 1e-4 * Math.Max(1.0, Math.Abs(e)), $"{what}[{i}]: expected {e}, got {a}");
        }
    }

    // Shapes: batch, inC, outC, size, kernel, stride, pad, dilation. 3x3 and 1x1, strides 1 and 2, padding 0-2,
    // dilation 2, and output-position counts that leave a 1- and 2-position tail of the 3-position tile.
    [Theory]
    [InlineData(2, 8, 32, 7, 3, 1, 1, 1)]
    [InlineData(3, 16, 64, 6, 3, 2, 1, 1)]
    [InlineData(2, 8, 32, 9, 1, 2, 0, 1)]
    [InlineData(2, 16, 32, 8, 3, 1, 2, 1)]
    [InlineData(2, 8, 32, 9, 3, 1, 2, 2)]
    [InlineData(4, 24, 32, 5, 3, 1, 0, 1)]
    public void Forward_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int stride, int pad, int dilation)
    {
        int outSize = OutSize(size, k, stride, pad, dilation);
        Assert.True(DirectConvAvx2.ShouldUseForward(batch, inC, outC, k, k, stride, stride, outSize, outSize) || !DirectConvAvx2.IsSupported);
        var x = Random(1, batch, inC, size, size);
        var w = Random(2, outC, inC, k, k);

        var y = _engine.Conv2D(x, w, new[] { stride, stride }, new[] { pad, pad }, new[] { dilation, dilation });

        var expected = new double[batch * outC * outSize * outSize];
        for (int b = 0; b < batch; b++)
        for (int o = 0; o < outC; o++)
        for (int oh = 0; oh < outSize; oh++)
        for (int ow = 0; ow < outSize; ow++)
        {
            double acc = 0;
            for (int c = 0; c < inC; c++)
            for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
            {
                int ih = oh * stride + i * dilation - pad, iw = ow * stride + j * dilation - pad;
                if (ih < 0 || ih >= size || iw < 0 || iw >= size) continue;
                acc += (double)w[o, c, i, j] * x[b, c, ih, iw];
            }
            expected[((b * outC + o) * outSize + oh) * outSize + ow] = acc;
        }
        AssertClose(expected, y, "output");
    }

    [Theory]
    [InlineData(2, 32, 8, 7, 3, 1, false)]
    [InlineData(3, 64, 16, 4, 3, 1, true)]
    [InlineData(2, 32, 16, 6, 3, 0, false)]
    [InlineData(2, 32, 8, 5, 1, 0, true)]
    public void BackwardInput_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int pad, bool accumulate)
    {
        int outSize = OutSize(size, k, 1, pad, 1);
        Assert.True(DirectConvAvx2.ShouldUseBackwardInput(batch, inC, outC, k, k, 1, 1, pad, pad, 1, 1) || !DirectConvAvx2.IsSupported);
        var w = Random(3, outC, inC, k, k);
        var g = Random(4, batch, outC, outSize, outSize);
        var prior = Random(5, batch, inC, size, size);
        var dx = new Tensor<float>(prior.Shape.ToArray());
        for (int i = 0; i < dx.Length; i++) dx.SetFlat(i, accumulate ? prior.GetFlat(i) : float.NaN);

        _engine.Conv2DBackwardInputInto(dx, g, w, new[] { batch, inC, size, size }, new[] { 1, 1 }, new[] { pad, pad }, new[] { 1, 1 }, accumulate);

        var expected = new double[dx.Length];
        for (int b = 0; b < batch; b++)
        for (int c = 0; c < inC; c++)
        for (int ih = 0; ih < size; ih++)
        for (int iw = 0; iw < size; iw++)
        {
            double acc = accumulate ? prior[b, c, ih, iw] : 0;
            for (int o = 0; o < outC; o++)
            for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
            {
                int oh = ih + pad - i, ow = iw + pad - j;
                if (oh < 0 || oh >= outSize || ow < 0 || ow >= outSize) continue;
                acc += (double)w[o, c, i, j] * g[b, o, oh, ow];
            }
            expected[((b * inC + c) * size + ih) * size + iw] = acc;
        }
        AssertClose(expected, dx, "input gradient");
    }

    [Theory]
    [InlineData(2, 32, 32, 7, 3, 1, 1, 1, false)]
    [InlineData(3, 16, 64, 8, 3, 2, 1, 1, true)]
    [InlineData(2, 32, 32, 9, 1, 2, 0, 1, false)]
    [InlineData(2, 32, 32, 9, 3, 1, 2, 2, false)]
    [InlineData(2, 32, 32, 6, 5, 1, 2, 1, true)]   // 5 kernel columns: a full 3-column group and a 2-column tail
    public void BackwardKernel_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int stride, int pad, int dilation, bool accumulate)
    {
        int outSize = OutSize(size, k, stride, pad, dilation);
        Assert.True(DirectConvAvx2.ShouldUseBackwardKernel(inC, outC) || !DirectConvAvx2.IsSupported);
        var x = Random(6, batch, inC, size, size);
        var g = Random(7, batch, outC, outSize, outSize);
        var prior = Random(8, outC, inC, k, k);
        var dw = new Tensor<float>(prior.Shape.ToArray());
        for (int i = 0; i < dw.Length; i++) dw.SetFlat(i, accumulate ? prior.GetFlat(i) : float.NaN);

        _engine.Conv2DBackwardKernelInto(dw, g, x, new[] { outC, inC, k, k }, new[] { stride, stride }, new[] { pad, pad }, new[] { dilation, dilation }, accumulate);

        var expected = new double[dw.Length];
        for (int o = 0; o < outC; o++)
        for (int c = 0; c < inC; c++)
        for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
        {
            double acc = accumulate ? prior[o, c, i, j] : 0;
            for (int b = 0; b < batch; b++)
            for (int oh = 0; oh < outSize; oh++)
            for (int ow = 0; ow < outSize; ow++)
            {
                int ih = oh * stride + i * dilation - pad, iw = ow * stride + j * dilation - pad;
                if (ih < 0 || ih >= size || iw < 0 || iw >= size) continue;
                acc += (double)g[b, o, oh, ow] * x[b, c, ih, iw];
            }
            expected[((o * inC + c) * k + i) * k + j] = acc;
        }
        AssertClose(expected, dw, "kernel gradient");
    }
}
