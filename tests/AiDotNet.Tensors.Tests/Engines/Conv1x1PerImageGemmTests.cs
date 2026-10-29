using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A batched 1x1, stride-1, unpadded float convolution runs as one sequential GEMM per image, images in parallel. It must
/// equal the direct sum out[n, o, p] = sum_i w[o, i] * x[n, i, p] (to float rounding), give bit-identical results on
/// repeated calls (each element keeps one summation order), and cover channel and pixel counts that are not multiples
/// of the vector width.
/// </summary>
public sealed class Conv1x1PerImageGemmTests
{
    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return d;
    }

    [Theory]
    [InlineData(4, 16, 96, 8, 8)]
    [InlineData(3, 96, 24, 5, 7)]     // ragged pixel count, few output channels
    [InlineData(2, 7, 13, 3, 3)]      // channel counts off the vector width
    public void EqualsTheDirectSum_AndIsDeterministic(int n, int cin, int cout, int h, int w)
    {
        var x = Rand(1, n * cin * h * w); var k = Rand(2, cout * cin);
        var e = new CpuEngine();
        var input = new Tensor<float>(x, new[] { n, cin, h, w });
        var kernel = new Tensor<float>(k, new[] { cout, cin, 1, 1 });
        var y = e.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }).ToArray();
        int hw = h * w;
        for (int b = 0; b < n; b++)
            for (int o = 0; o < cout; o++)
                for (int p = 0; p < hw; p++)
                {
                    double sum = 0;
                    for (int i = 0; i < cin; i++) sum += (double)k[o * cin + i] * x[(b * cin + i) * hw + p];
                    float got = y[(b * cout + o) * hw + p];
                    Assert.True(Math.Abs(sum - got) <= 1e-5 + 1e-5 * Math.Abs(sum), $"[{b},{o},{p}]: {got} vs {sum}");
                }
        for (int r = 0; r < 3; r++)
            Assert.Equal(y, e.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }).ToArray());
    }

    /// <summary>
    /// Deep inputs on small maps take the shared-weight route (the weight packed once for the batch). These cases span
    /// two K panels (cin over 512) and two row blocks (cout over 128), with whole and ragged pixel counts. The bound is
    /// float summation's: cin * 2^-24 times the sum of the absolute products, which a misplaced panel or block exceeds.
    /// </summary>
    [Theory]
    [InlineData(3, 960, 200, 4, 4)]
    [InlineData(5, 576, 160, 4, 4)]
    [InlineData(2, 640, 136, 3, 5)]
    public void SharedWeightRoute_EqualsTheDirectSum_AndIsDeterministic(int n, int cin, int cout, int h, int w)
    {
        var x = Rand(3, n * cin * h * w); var k = Rand(4, cout * cin);
        var e = new CpuEngine();
        var input = new Tensor<float>(x, new[] { n, cin, h, w });
        var kernel = new Tensor<float>(k, new[] { cout, cin, 1, 1 });
        var y = e.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }).ToArray();
        int hw = h * w;
        for (int b = 0; b < n; b++)
            for (int o = 0; o < cout; o++)
                for (int p = 0; p < hw; p++)
                {
                    double sum = 0, magnitude = 0;
                    for (int i = 0; i < cin; i++)
                    {
                        double term = (double)k[o * cin + i] * x[(b * cin + i) * hw + p];
                        sum += term; magnitude += Math.Abs(term);
                    }
                    float got = y[(b * cout + o) * hw + p];
                    Assert.True(Math.Abs(sum - got) <= cin * Math.Pow(2, -24) * magnitude, $"[{b},{o},{p}]: {got} vs {sum}");
                }
        for (int r = 0; r < 3; r++)
            Assert.Equal(y, e.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }).ToArray());
    }
}