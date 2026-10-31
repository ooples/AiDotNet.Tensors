using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A float FusedConv2D with a ReLU6 runs the bias and the clamp in the convolution's fused SIMD epilogue (it fell to the
/// generic activation handler before). It must equal the convolution, the bias, then min(max(0, x), 6).
/// </summary>
public sealed class FusedConv2DReLU6Tests
{
    private static Tensor<float> Rand(int seed, float scale, params int[] shape)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var s in shape) n *= s;
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(d, shape);
    }

    [Theory]
    [InlineData(2, 8, 16, 9, 7, 1, 1)]    // 3x3, plane 63: a vector body and a scalar tail
    [InlineData(4, 16, 32, 8, 8, 1, 0)]   // 1x1, plane 64, past the parallel threshold
    [InlineData(64, 16, 64, 2, 2, 1, 1)]  // 3x3 on 2x2: planes of 4, shorter than a vector (the late-stage maps)
    public void EqualsConvolutionBiasThenClamp(int n, int cin, int cout, int h, int w, int k3, int pad)
    {
        int k = k3 == 1 && pad == 1 ? 3 : 1;
        var e = new CpuEngine();
        var x = Rand(1, 3f, n, cin, h, w);
        var kernel = Rand(2, 1f, cout, cin, k, k);
        var bias = Rand(3, 2f, cout);
        var fused = e.FusedConv2D(x, kernel, bias, 1, 1, pad, pad, 1, 1, FusedActivationType.ReLU6).ToArray();
        var conv = e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { pad, pad }, new[] { 1, 1 }).ToArray();
        var b = bias.ToArray();
        int plane = conv.Length / (n * cout);
        int low = 0, high = 0;
        for (int i = 0; i < conv.Length; i++)
        {
            float v = conv[i] + b[(i / plane) % cout];
            if (v < 0) low++;
            if (v > 6) high++;
            Assert.Equal(Math.Min(Math.Max(v, 0f), 6f), fused[i]);
        }
        Assert.True(low > 0 && high > 0, $"the clamp was not exercised on both sides ({low} below 0, {high} above 6)");
    }

    [Fact]
    public void NaN_Propagates()
    {
        var e = new CpuEngine();
        var x = Rand(4, 1f, 1, 1, 3, 11);
        var xd = x.ToArray(); xd[2] = float.NaN; xd[32] = float.NaN;   // vector body and the 33-plane's scalar tail
        var kernel = new Tensor<float>(new[] { 1f }, new[] { 1, 1, 1, 1 });
        var bias = new Tensor<float>(new[] { 0.5f }, new[] { 1 });
        var y = e.FusedConv2D(new Tensor<float>(xd, new[] { 1, 1, 3, 11 }), kernel, bias, 1, 1, 0, 0, 1, 1,
            FusedActivationType.ReLU6).ToArray();
        Assert.True(float.IsNaN(y[2]) && float.IsNaN(y[32]), "NaN did not propagate");
    }
}