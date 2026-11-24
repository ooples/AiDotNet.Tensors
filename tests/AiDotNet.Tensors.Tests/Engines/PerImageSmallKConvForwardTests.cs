using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A short-reduction forward (an RGB stem) over a batch of at least one image per worker runs one im2col and one
/// sequential GEMM per image. Checks it against the full im2col route, that it was the route taken, and that the result
/// is the same bits at every thread count.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class PerImageSmallKConvForwardTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (int d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    [Theory]
    [InlineData(32, 3, 64, 32, 4, 2, 1)]   // DCGAN's discriminator stem at a smaller plane
    [InlineData(17, 2, 16, 19, 3, 2, 1)]   // odd batch and plane, 3x3 stride 2 (stride 1 is Winograd's)
    [InlineData(16, 1, 24, 15, 5, 2, 2)]   // one input channel, 5x5, padding 2
    public void MatchesFullIm2ColAndIsThreadCountInvariant(int batch, int cin, int cout, int hw, int k, int stride, int pad)
    {
        var e = new CpuEngine();
        var x = Rand(new[] { batch, cin, hw, hw }, 11);
        var w = Rand(new[] { cout, cin, k, k }, 12);
        int[] s = { stride, stride }, p = { pad, pad }, d = { 1, 1 };
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        float[]? first = null;
        float[] full;
        try
        {
            foreach (int threads in new[] { 1, 4, 16 })
            {
                if (threads > batch) continue;
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                long before = CpuEngine.PerImageSmallKForwards;
                var y = e.Conv2D(x, w, s, p, d).ToArray();
                Assert.True(CpuEngine.PerImageSmallKForwards > before, $"{threads} threads did not take the per-image route");
                if (first is null) first = y;
                else
                    for (int i = 0; i < y.Length; i++)
                        Assert.Equal(BitConverter.ToInt32(BitConverter.GetBytes(first[i]), 0),
                            BitConverter.ToInt32(BitConverter.GetBytes(y[i]), 0));
            }
            using (CpuEngine.ForceFullIm2ColScope())
                full = e.Conv2D(x, w, s, p, d).ToArray();
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
        Assert.NotNull(first);
        double tolerance = 1e-5 * Math.Sqrt((double)cin * k * k) + 1e-6;
        Assert.Equal(full.Length, first!.Length);
        for (int i = 0; i < full.Length; i++)
            Assert.True(Math.Abs(full[i] - first[i]) <= tolerance, $"index {i}: full {full[i]} per-image {first[i]}");
    }
}
