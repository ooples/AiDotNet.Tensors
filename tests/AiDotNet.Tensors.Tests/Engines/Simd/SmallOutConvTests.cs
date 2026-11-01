// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// <see cref="SmallOutConv"/> (stride-1 convolutions with one to three output channels) against the plain sum, and the
/// engine's input gradient of an RGB first layer (a forward convolution with three outputs) through it.
/// </summary>
[Collection("ConvPerfSerial")]
public class SmallOutConvTests
{
    [SkippableTheory]
    [InlineData(2, 64, 3, 32, 32, 3, 3, 1, 1)]
    [InlineData(3, 32, 1, 28, 28, 3, 3, 1, 1)]
    [InlineData(1, 16, 2, 9, 13, 5, 5, 2, 2)]
    [InlineData(2, 24, 3, 7, 5, 3, 3, 0, 0)]
    [InlineData(1, 70, 1, 3, 70, 1, 3, 0, 1)]
    public void Forward_MatchesThePlainSum(int batch, int inC, int outC, int h, int w, int kh, int kw, int ph, int pw)
    {
        Skip.IfNot(SmallOutConv.Applies(inC, outC, kh, kw, 1, 1, 1, 1, ph, pw), "AVX2 + FMA, shape in range");
        var rng = new Random(batch * 7 + inC + h * 3 + w);
        float[] x = Fill(rng, batch * inC * h * w), wt = Fill(rng, outC * inC * kh * kw);
        int oh = h + 2 * ph - kh + 1, ow = w + 2 * pw - kw + 1;
        var y = new float[batch * outC * oh * ow + 5];
        for (int i = 0; i < y.Length; i++) y[i] = float.NaN;
        SmallOutConv.Forward(x, 0, wt, 0, y, 0, batch, inC, h, w, outC, kh, kw, ph, pw);
        for (int n = 0; n < batch; n++)
            for (int o = 0; o < outC; o++)
                for (int i = 0; i < oh; i++)
                    for (int j = 0; j < ow; j++)
                    {
                        double sum = 0;
                        for (int c = 0; c < inC; c++)
                            for (int a = 0; a < kh; a++)
                                for (int b = 0; b < kw; b++)
                                {
                                    int yi = i + a - ph, xj = j + b - pw;
                                    if (yi < 0 || yi >= h || xj < 0 || xj >= w) continue;
                                    sum += (double)wt[((o * inC + c) * kh + a) * kw + b] * x[((n * inC + c) * h + yi) * w + xj];
                                }
                        float got = y[((n * outC + o) * oh + i) * ow + j];
                        Assert.True(Math.Abs(got - sum) <= 1e-4 * (1 + Math.Abs(sum)), $"[{n},{o},{i},{j}] expected {sum}, got {got}");
                    }
        // Nothing written past the output.
        for (int i = batch * outC * oh * ow; i < y.Length; i++) Assert.True(float.IsNaN(y[i]));
    }

    [SkippableFact]
    public void Engine_RgbFirstLayerInputGradient_TakesTheKernel_AndAgreesWithTheOtherRoute()
    {
        Skip.IfNot(SmallOutConv.Applies(64, 3, 3, 3, 1, 1, 1, 1, 1, 1), "AVX2 + FMA");
        const int n = 4, c = 3, k = 64, h = 32, w = 32;
        var rng = new Random(11);
        var kernel = new Tensor<float>(Fill(rng, k * c * 9), new[] { k, c, 3, 3 });
        var dy = new Tensor<float>(Fill(rng, n * k * h * w), new[] { n, k, h, w });
        var engine = new CpuEngine();
        int[] stride = { 1, 1 }, pad = { 1, 1 }, dil = { 1, 1 }, inputShape = { n, c, h, w };
        var viaKernel = new Tensor<float>(inputShape);
        int runs = SmallOutConv.t_runs;
        engine.Conv2DBackwardInputInto(viaKernel, dy, kernel, inputShape, stride, pad, dil, false);
        Assert.Equal(runs + 1, SmallOutConv.t_runs);
        var other = new Tensor<float>(inputShape);
        SmallOutConv.s_disable = true;
        try
        {
            engine.Conv2DBackwardInputInto(other, dy, kernel, inputShape, stride, pad, dil, false);
        }
        finally
        {
            SmallOutConv.s_disable = false;
        }
        float[] a = other.ToArray(), b = viaKernel.ToArray();
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(a[i] - b[i]) <= 1e-4 * (1 + Math.Abs(a[i])), $"[{i}] {a[i]} vs {b[i]}");
    }

    private static float[] Fill(Random rng, int count)
    {
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }
}
#endif
