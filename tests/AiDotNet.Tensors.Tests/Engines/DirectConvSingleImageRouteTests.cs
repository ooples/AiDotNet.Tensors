#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #653: single-image stride-1 3x3 convolutions on 40x40-80x80 planes with 64+ input channels take the direct kernels
/// (measured 1.10-2.29x over the im2col GEMM routes at 1 and 16 threads). Pins the rule's edges and checks the direct
/// result against the full im2col GEMM.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class DirectConvSingleImageRouteTests
{
    private static DirectConvShape Forward(int batch, int channels, int hw, int stride = 1) =>
        new(DirectConvPass.Forward, batch, channels, channels, hw, hw, 3, 3, stride, stride, 1, 1, 1, 1);

    [Theory]
    [InlineData(1, 64, 64, 1, true)]     // inside the rule
    [InlineData(1, 64, 40, 1, true)]     // 1600 positions: just past the batch rule's 1024
    [InlineData(1, 640, 80, 1, true)]    // the 80x80 upper bound
    [InlineData(1, 64, 81, 1, false)]    // past 80x80
    [InlineData(1, 64, 128, 1, false)]   // 128x128 planes measured slower on the direct kernels
    [InlineData(1, 32, 64, 1, false)]    // 32 channels measured slower at 16 threads
    [InlineData(1, 256, 16, 1, false)]   // a 256-position plane stays on the GEMM routes for a single image
    [InlineData(1, 64, 64, 2, false)]    // stride 2 is outside the single-image rule
    [InlineData(2, 64, 16, 1, true)]     // the batch rule is unchanged
    [InlineData(2, 64, 64, 1, false)]    // ...including its 1024-position limit for batches
    public void DefaultChoosesDirect_SingleImageRule(int batch, int channels, int hw, int stride, bool expected)
    {
        Assert.Equal(expected, DirectConvAvx2.DefaultChoosesDirect(Forward(batch, channels, hw, stride)));
    }

    [SkippableTheory]
    [InlineData(64, 64, 1)]
    [InlineData(96, 64, 16)]
    [InlineData(64, 48, 4)]
    public void SingleImageDirectConv_MatchesFullIm2Col(int channels, int hw, int threads)
    {
        Skip.IfNot(DirectConvAvx2.IsSupported, "The direct kernels need AVX2 and FMA.");
        var shape = new DirectConvShape(DirectConvPass.Forward, 1, channels, channels, hw, hw, 3, 3, 1, 1, 1, 1, 1, 1);
        Skip.IfNot(DirectConvAvx2.TryChoose(shape, out _), "A tuned entry routes this shape elsewhere on this host.");
        var e = new CpuEngine();
        var x = Rand(new[] { 1, channels, hw, hw }, 707);
        var kernel = Rand(new[] { channels, channels, 3, 3 }, 808);
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        float[] direct, full;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = threads;
            direct = e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }).ToArray();
            using (CpuEngine.ForceFullIm2ColScope())
                full = e.Conv2D(x, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }).ToArray();
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
        // Different summation order (direct vs GEMM): compare against the size of a K = 9 * channels dot product.
        double tolerance = 1e-5 * Math.Sqrt(9.0 * channels) + 1e-6;
        Assert.Equal(full.Length, direct.Length);
        bool anyDifferent = false;
        for (int i = 0; i < full.Length; i++)
        {
            Assert.True(Math.Abs(full[i] - direct[i]) <= tolerance, $"index {i}: full {full[i]} direct {direct[i]}");
            anyDifferent |= full[i] != direct[i];
        }
        // Positive control: identical bits everywhere would mean both calls ran the same route.
        Assert.True(anyDifferent, "the default call did not take a different route from the pinned full im2col");
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
#endif
