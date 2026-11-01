// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Winograd F(4x4, 3x3) against a double-precision direct sum: the forward and the input gradient, written and
/// accumulated, on planes that are and are not multiples of the 4x4 tile, with the error bounded relative to the sum of
/// the products' magnitudes (the transforms' rounding grows with it). Then the engine's routing: the forward and the
/// input gradient of a 3x3 stride-1 convolution take the kernel and agree with the direct route.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class WinogradF43Tests
{
    public static TheoryData<int, int, int, int, int> Shapes => new()
    {
        { 2, 8, 16, 8, 8 },
        { 1, 16, 8, 9, 11 },
        { 3, 24, 32, 14, 14 },
        { 2, 64, 64, 16, 16 },
        { 1, 8, 8, 13, 8 },
        { 8, 32, 16, 8, 8 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Forward_MatchesTheDirectSum(int n, int c, int k, int h, int w)
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        var rng = new Random(n * 100 + c + k + h * 7 + w);
        float[] x = Fill(rng, n * c * h * w), wt = Fill(rng, k * c * 9);
        foreach (bool accumulate in new[] { false, true })
        {
            float[] y = Fill(new Random(5), n * k * h * w);
            float[] y0 = (float[])y.Clone();
            WinogradF43.Forward(x, 0, wt, 0, y, 0, accumulate, n, c, h, w, k);
            for (int b = 0; b < n; b++)
                for (int o = 0; o < k; o++)
                    for (int i = 0; i < h; i++)
                        for (int j = 0; j < w; j++)
                        {
                            double sum = 0, mag = 0;
                            for (int ic = 0; ic < c; ic++)
                                for (int di = 0; di < 3; di++)
                                    for (int dj = 0; dj < 3; dj++)
                                    {
                                        int yi = i - 1 + di, xj = j - 1 + dj;
                                        if (yi < 0 || yi >= h || xj < 0 || xj >= w) continue;
                                        double p = (double)x[((b * c + ic) * h + yi) * w + xj] * wt[(o * c + ic) * 9 + di * 3 + dj];
                                        sum += p; mag += Math.Abs(p);
                                    }
                            int at = ((b * k + o) * h + i) * w + j;
                            double expected = (accumulate ? y0[at] : 0) + sum;
                            Assert.True(Math.Abs(y[at] - expected) <= 2e-6 * (mag + 1) * 8,
                                $"n={n} c={c} k={k} {h}x{w} acc={accumulate} [{b},{o},{i},{j}]: expected {expected}, got {y[at]}");
                        }
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void BackwardInput_MatchesTheDirectSum(int n, int c, int k, int h, int w)
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        var rng = new Random(n * 31 + c + k + h + w * 3);
        float[] dy = Fill(rng, n * k * h * w), wt = Fill(rng, k * c * 9);
        foreach (bool accumulate in new[] { false, true })
        {
            float[] dx = Fill(new Random(9), n * c * h * w);
            float[] dx0 = (float[])dx.Clone();
            WinogradF43.BackwardInput(dy, 0, wt, 0, dx, 0, accumulate, n, c, h, w, k);
            for (int b = 0; b < n; b++)
                for (int ic = 0; ic < c; ic++)
                    for (int i = 0; i < h; i++)
                        for (int j = 0; j < w; j++)
                        {
                            // dX[ic][i][j] = sum over o, di, dj of dY[o][i + 1 - di][j + 1 - dj] w[o][ic][di][dj].
                            double sum = 0, mag = 0;
                            for (int o = 0; o < k; o++)
                                for (int di = 0; di < 3; di++)
                                    for (int dj = 0; dj < 3; dj++)
                                    {
                                        int yi = i + 1 - di, xj = j + 1 - dj;
                                        if (yi < 0 || yi >= h || xj < 0 || xj >= w) continue;
                                        double p = (double)dy[((b * k + o) * h + yi) * w + xj] * wt[(o * c + ic) * 9 + di * 3 + dj];
                                        sum += p; mag += Math.Abs(p);
                                    }
                            int at = ((b * c + ic) * h + i) * w + j;
                            double expected = (accumulate ? dx0[at] : 0) + sum;
                            Assert.True(Math.Abs(dx[at] - expected) <= 2e-6 * (mag + 1) * 8,
                                $"n={n} c={c} k={k} {h}x{w} acc={accumulate} [{b},{ic},{i},{j}]: expected {expected}, got {dx[at]}");
                        }
        }
    }

    [Fact]
    public void Applies_OnlyTo3x3Stride1Padding1_WithChannelMultiplesOfEight()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        Assert.True(WinogradF43.Applies(8, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));
        Assert.False(WinogradF43.Applies(8, 64, 64, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1));   // stride 2
        Assert.False(WinogradF43.Applies(8, 64, 64, 32, 32, 3, 3, 1, 1, 0, 0, 1, 1));   // padding 0
        Assert.False(WinogradF43.Applies(8, 64, 64, 32, 32, 3, 3, 1, 1, 2, 2, 2, 2));   // dilated
        Assert.False(WinogradF43.Applies(8, 3, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));    // 3 input channels
        Assert.False(WinogradF43.Applies(64, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1));  // 32 input channels
        Assert.False(WinogradF43.Applies(8, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));   // 32 patches
        Assert.False(WinogradF43.Applies(8, 64, 60, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1));   // 60 output channels
        Assert.False(WinogradF43.Applies(8, 64, 64, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1));     // 4x4 plane
        Assert.False(WinogradF43.Applies(1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1));     // 4 patches
        Assert.False(WinogradF43.Applies(8, 64, 64, 32, 32, 5, 5, 1, 1, 2, 2, 1, 1));   // 5x5
    }

    [Fact]
    public void Engine_ForwardAndInputGradient_TakeTheKernel_AndAgreeWithTheDirectRoute()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        const int n = 8, c = 64, k = 72, h = 16, w = 16;
        var rng = new Random(77);
        var x = new Tensor<float>(Fill(rng, n * c * h * w), new[] { n, c, h, w });
        var wt = new Tensor<float>(Fill(rng, k * c * 9), new[] { k, c, 3, 3 });
        var dy = new Tensor<float>(Fill(rng, n * k * h * w), new[] { n, k, h, w });
        var engine = new CpuEngine();
        int[] stride = { 1, 1 }, pad = { 1, 1 }, dil = { 1, 1 };
        float[] yWino = engine.Conv2D(x, wt, stride, pad, dil).ToArray();
        float[] dxWino = engine.Conv2DBackwardInput(dy, wt, x._shape, stride, pad, dil).ToArray();
        float[] yDirect, dxDirect;
        WinogradF43.s_disable = true;
        try
        {
            yDirect = engine.Conv2D(x, wt, stride, pad, dil).ToArray();
            dxDirect = engine.Conv2DBackwardInput(dy, wt, x._shape, stride, pad, dil).ToArray();
        }
        finally
        {
            WinogradF43.s_disable = false;
        }
        AssertClose(yDirect, yWino, "forward");
        AssertClose(dxDirect, dxWino, "input gradient");
        // The two routes round differently: equal arrays would mean the engine never reached the kernel.
        Assert.NotEqual(yDirect, yWino);
        Assert.NotEqual(dxDirect, dxWino);
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            // 576 products per output: F(4x4)'s rounding is about 2e-6 of their summed magnitude (~0.25 each here).
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3 * (1 + Math.Abs(expected[i])),
                $"{what} [{i}]: direct {expected[i]}, Winograd {actual[i]}");
    }

    private static float[] Fill(Random rng, int length)
    {
        var a = new float[length];
        for (int i = 0; i < length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
#endif
