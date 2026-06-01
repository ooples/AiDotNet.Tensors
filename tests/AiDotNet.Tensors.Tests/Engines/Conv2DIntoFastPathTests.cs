using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The int-overload Conv2DInto used to route through Conv2DWithIm2ColFloat (the
/// SIMD-direct/Winograd cascade) — ~15× slower than the int[] overload's
/// Conv2DIm2colGemm fast path for small 3×3 stride-1 convs. It now delegates to the
/// same fast path; this guards that (a) it still matches the allocating Conv2D and
/// (b) the int and int[] Conv2DInto overloads agree.
/// </summary>
public class Conv2DIntoFastPathTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed); var t = new Tensor<float>(shape); var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    [Theory]
    [InlineData(1, 1, 16, 3, 28)]    // CNN conv1 shape
    [InlineData(1, 16, 32, 3, 14)]   // CNN conv2 shape
    [InlineData(2, 4, 6, 3, 10)]     // batched
    [InlineData(1, 3, 8, 1, 12)]     // 1×1 conv
    public void Conv2DInto_Int_MatchesAllocatingConv2D(int batch, int inC, int outC, int k, int hw)
    {
        var e = new CpuEngine();
        var x = Rand(new[] { batch, inC, hw, hw }, 11);
        var kernel = Rand(new[] { outC, inC, k, k }, 22);
        int pad = k / 2;

        var expected = e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { pad, pad }, new[] { 1, 1 }).ToArray();

        int oh = hw + 2 * pad - k + 1, ow = oh;
        var into = new Tensor<float>(new[] { batch, outC, oh, ow });
        e.Conv2DInto(into, x, kernel, stride: 1, padding: pad, dilation: 1);   // int overload (fixed fast path)
        var actual = into.ToArray();

        Assert.Equal(expected.Length, actual.Length);
        double maxDiff = 0;
        for (int i = 0; i < expected.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(expected[i] - actual[i]));
        Assert.True(maxDiff <= 1e-4, $"int Conv2DInto diverged from allocating Conv2D by {maxDiff:E3}");
    }
}
