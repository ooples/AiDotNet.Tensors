// Copyright (c) AiDotNet. All rights reserved.
// FftConv validation: output must match direct convolution to floating-point
// roundoff. Tests cover 1D, 2D single-channel, and 2D multi-channel paths.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Fft;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.FftTests;

public class FftConvTests
{
    // ── 1D FFT-conv matches direct ──────────────────────────────────────────
    [Theory]
    [InlineData(32, 3)]
    [InlineData(64, 7)]
    [InlineData(128, 31)]
    [InlineData(77, 15)]   // non-pow2 → Bluestein on the L = 77+15-1 = 91 path
    public void Conv1DSame_MatchesDirect(int n, int k)
    {
        var rng = new Random(1);
        var x = new Tensor<double>(new[] { n });
        var xd = x.GetDataArray();
        for (int i = 0; i < n; i++) xd[i] = rng.NextDouble() * 2 - 1;
        var w = new Tensor<double>(new[] { k });
        var wd = w.GetDataArray();
        for (int i = 0; i < k; i++) wd[i] = rng.NextDouble() * 2 - 1;

        // Direct "same" convolution (matches PyTorch Conv1d with padding=k//2).
        var direct = new double[n];
        int padLeft = k / 2;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
            {
                int ix = i + j - padLeft;
                if (ix >= 0 && ix < n) direct[i] += xd[ix] * wd[j];
            }

        var fft = FftConv.Conv1DSame(x, w);
        var fd = fft.GetDataArray();
        double maxErr = 0;
        for (int i = 0; i < n; i++)
        {
            double e = Math.Abs(direct[i] - fd[i]);
            if (e > maxErr) maxErr = e;
        }
        Assert.True(maxErr < 1e-9, $"Conv1D n={n} k={k}: max error {maxErr}");
    }

    // ── 2D single-channel FFT-conv matches direct ──────────────────────────
    [Theory]
    [InlineData(16, 16, 3, 3)]
    [InlineData(32, 32, 5, 5)]
    [InlineData(24, 32, 7, 5)]
    public void Conv2DSame_Single_MatchesDirect(int H, int W, int Kh, int Kw)
    {
        var rng = new Random(2);
        var input = new Tensor<double>(new[] { 1, 1, H, W });
        var inD = input.GetDataArray();
        for (int i = 0; i < inD.Length; i++) inD[i] = rng.NextDouble() * 2 - 1;
        var weight = new Tensor<double>(new[] { 1, 1, Kh, Kw });
        var wD = weight.GetDataArray();
        for (int i = 0; i < wD.Length; i++) wD[i] = rng.NextDouble() * 2 - 1;

        // Direct 2D "same" convolution.
        var direct = new double[H * W];
        int padY = Kh / 2;
        int padX = Kw / 2;
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                double s = 0;
                for (int ky = 0; ky < Kh; ky++)
                {
                    for (int kx = 0; kx < Kw; kx++)
                    {
                        int iy = y + ky - padY;
                        int ix = x + kx - padX;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                            s += inD[iy * W + ix] * wD[ky * Kw + kx];
                    }
                }
                direct[y * W + x] = s;
            }
        }

        var fft = FftConv.Conv2DSame(input, weight);
        var fD = fft.GetDataArray();
        double maxErr = 0;
        for (int i = 0; i < H * W; i++)
        {
            double e = Math.Abs(direct[i] - fD[i]);
            if (e > maxErr) maxErr = e;
        }
        Assert.True(maxErr < 1e-9, $"Conv2D H={H} W={W} Kh={Kh} Kw={Kw}: max error {maxErr}");
    }

    // ── 2D multi-channel FFT-conv matches direct ────────────────────────────
    [Fact]
    public void Conv2DSame_MultiChannel_MatchesDirect()
    {
        int N = 2, Cin = 3, Cout = 4, H = 16, W = 16, Kh = 5, Kw = 5;
        var rng = new Random(3);
        var input = new Tensor<double>(new[] { N, Cin, H, W });
        var inD = input.GetDataArray();
        for (int i = 0; i < inD.Length; i++) inD[i] = rng.NextDouble() * 2 - 1;
        var weight = new Tensor<double>(new[] { Cout, Cin, Kh, Kw });
        var wD = weight.GetDataArray();
        for (int i = 0; i < wD.Length; i++) wD[i] = rng.NextDouble() * 2 - 1;
        var bias = new Tensor<double>(new[] { Cout });
        var bD = bias.GetDataArray();
        for (int i = 0; i < Cout; i++) bD[i] = rng.NextDouble();

        // Direct conv2d same + bias.
        var direct = new double[N * Cout * H * W];
        int padY = Kh / 2;
        int padX = Kw / 2;
        for (int n = 0; n < N; n++)
            for (int co = 0; co < Cout; co++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                    {
                        double s = bD[co];
                        for (int ci = 0; ci < Cin; ci++)
                            for (int ky = 0; ky < Kh; ky++)
                                for (int kx = 0; kx < Kw; kx++)
                                {
                                    int iy = y + ky - padY;
                                    int ix = x + kx - padX;
                                    if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                        s += inD[((n * Cin + ci) * H + iy) * W + ix] * wD[((co * Cin + ci) * Kh + ky) * Kw + kx];
                                }
                        direct[((n * Cout + co) * H + y) * W + x] = s;
                    }

        var fft = FftConv.Conv2DSame(input, weight, bias);
        var fD = fft.GetDataArray();
        double maxErr = 0;
        for (int i = 0; i < direct.Length; i++)
        {
            double e = Math.Abs(direct[i] - fD[i]);
            if (e > maxErr) maxErr = e;
        }
        Assert.True(maxErr < 1e-9, $"multi-channel Conv2D: max error {maxErr}");
    }
}
