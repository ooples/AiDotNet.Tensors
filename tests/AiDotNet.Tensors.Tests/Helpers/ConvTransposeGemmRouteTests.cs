using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// The float transposed-convolution GEMM routes: one sequential GEMM per image when the plane has at least 64 input
/// positions and there is an image per worker, else the batch-wide GEMM on a materialized W^T. Both are checked against
/// a double-precision direct sum, the per-image route for being taken and for the same bits at every thread count.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class ConvTransposeGemmRouteTests
{
    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    // y[b, oc, oh, ow] = sum over (ic, kh, kw) with oh = ih * s - p + kh of x[b, ic, ih, iw] * w[ic, oc, kh, kw].
    private static double[] Reference(float[] x, float[] w, int batch, int cin, int h, int cout, int k, int s, int p, int oh, int ow)
    {
        var y = new double[batch * cout * oh * ow];
        for (int b = 0; b < batch; b++)
            for (int ic = 0; ic < cin; ic++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < h; iw++)
                    {
                        double xv = x[((b * cin + ic) * h + ih) * h + iw];
                        for (int oc = 0; oc < cout; oc++)
                            for (int kh = 0; kh < k; kh++)
                            {
                                int yh = ih * s - p + kh;
                                if ((uint)yh >= (uint)oh) continue;
                                for (int kw = 0; kw < k; kw++)
                                {
                                    int yw = iw * s - p + kw;
                                    if ((uint)yw >= (uint)ow) continue;
                                    y[((b * cout + oc) * oh + yh) * ow + yw] += xv * w[((ic * cout + oc) * k + kh) * k + kw];
                                }
                            }
                    }
        return y;
    }

    [Theory]
    [InlineData(32, 16, 8, 8, 4, 2, 1, true)]    // DCGAN's 256 -> 128 step at an 8x8 plane, scaled down
    [InlineData(17, 8, 16, 3, 4, 2, 1, true)]    // an RGB output layer, odd batch
    [InlineData(16, 6, 9, 5, 3, 2, 0, true)]     // 81 positions, 3x3, no padding
    [InlineData(16, 32, 4, 16, 4, 2, 1, false)]  // a 4x4 plane stays on the batch-wide GEMM
    public void MatchesDirectSum(int batch, int cin, int h, int cout, int k, int s, int p, bool perImage)
    {
        int oh = (h - 1) * s - 2 * p + k;
        var xs = Rand(batch * cin * h * h, 21);
        var ws = Rand(cin * cout * k * k, 22);
        var reference = Reference(xs, ws, batch, cin, h, cout, k, s, p, oh, oh);
        var e = new CpuEngine();
        var x = new Tensor<float>(xs, new[] { batch, cin, h, h });
        var w = new Tensor<float>(ws, new[] { cin, cout, k, k });
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        float[]? first = null;
        try
        {
            foreach (int threads in new[] { 1, 4, 16 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                long before = Im2ColHelper.ConvTransposePerImageForwards;
                var y = e.ConvTranspose2D(x, w, new[] { s, s }, new[] { p, p }, new[] { 0, 0 }).ToArray();
                Assert.Equal(perImage, Im2ColHelper.ConvTransposePerImageForwards > before);
                double tolerance = 1e-5 * Math.Sqrt((double)cin * k * k) + 1e-6;
                Assert.Equal(reference.Length, y.Length);
                for (int i = 0; i < y.Length; i++)
                    Assert.True(Math.Abs(reference[i] - y[i]) <= tolerance, $"{threads} threads, index {i}: {reference[i]} vs {y[i]}");
                if (!perImage) continue;
                if (first is null) first = y;
                else
                    for (int i = 0; i < y.Length; i++)
                        Assert.Equal(BitConverter.ToInt32(BitConverter.GetBytes(first[i]), 0),
                            BitConverter.ToInt32(BitConverter.GetBytes(y[i]), 0));
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
    }
}
