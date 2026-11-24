using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The small-channel conv backward routes: the input gradient of a short-reduction conv (an RGB stem) as the per-image
/// transposed conv, and a weight gradient of at most 64K elements as fixed image chunks summed in order. Each is checked
/// against a double-precision direct sum, written and accumulated, for being the route taken, and for the same bits at
/// every thread count.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class ConvBackwardSmallChannelRouteTests
{
    private static float[] Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    private static int Bits(float v) => BitConverter.ToInt32(BitConverter.GetBytes(v), 0);

    // x [B, C, H, H], w [O, C, k, k], dy [B, O, P, P] with P = (H + 2p - k) / s + 1.
    private static (double[] dx, double[] dw) Reference(float[] x, float[] w, float[] dy, int b, int c, int h, int o, int k, int s, int p, int oh)
    {
        var dx = new double[b * c * h * h];
        var dw = new double[o * c * k * k];
        for (int n = 0; n < b; n++)
            for (int oc = 0; oc < o; oc++)
                for (int y = 0; y < oh; y++)
                    for (int xx = 0; xx < oh; xx++)
                    {
                        double g = dy[((n * o + oc) * oh + y) * oh + xx];
                        for (int ic = 0; ic < c; ic++)
                            for (int kh = 0; kh < k; kh++)
                            {
                                int ih = y * s - p + kh;
                                if ((uint)ih >= (uint)h) continue;
                                for (int kw = 0; kw < k; kw++)
                                {
                                    int iw = xx * s - p + kw;
                                    if ((uint)iw >= (uint)h) continue;
                                    int xi = ((n * c + ic) * h + ih) * h + iw, wi = ((oc * c + ic) * k + kh) * k + kw;
                                    dx[xi] += g * w[wi];
                                    dw[wi] += g * x[xi];
                                }
                            }
                    }
        return (dx, dw);
    }

    [Theory]
    [InlineData(32, 3, 32, 16, 4, 2, 1)]   // an RGB stem, scaled down
    [InlineData(17, 1, 63, 8, 3, 2, 1)]    // grey, odd batch and plane
    [InlineData(20, 2, 44, 24, 5, 2, 2)]   // 5x5, padding 2 (the chunked route needs batch x positions >= 8192)
    public void InputAndWeightGradientsMatchDirectSum(int batch, int cin, int h, int cout, int k, int s, int p)
    {
        int oh = (h + 2 * p - k) / s + 1;
        var xs = Rand(batch * cin * h * h, 31);
        var ws = Rand(cout * cin * k * k, 32);
        var gs = Rand(batch * cout * oh * oh, 33);
        var seedX = Rand(xs.Length, 34);
        var seedW = Rand(ws.Length, 35);
        var (rdx, rdw) = Reference(xs, ws, gs, batch, cin, h, cout, k, s, p, oh);
        var e = new CpuEngine();
        var x = new Tensor<float>(xs, new[] { batch, cin, h, h });
        var w = new Tensor<float>(ws, new[] { cout, cin, k, k });
        var dy = new Tensor<float>(gs, new[] { batch, cout, oh, oh });
        int[] st = { s, s }, pd = { p, p }, dl = { 1, 1 };
        double txTol = 1e-5 * Math.Sqrt((double)cout * k * k) + 1e-6;
        double twTol = 1e-5 * Math.Sqrt((double)batch * oh * oh) + 1e-6;
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        float[]? firstDx = null, firstDw = null;
        try
        {
            foreach (int threads in new[] { 1, 4, 16 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                foreach (bool accumulate in new[] { false, true })
                {
                    long perImage = Im2ColHelper.ConvTransposePerImageForwards;
                    long chunked = CpuEngine.ChunkedWeightGradients;
                    var dx = new Tensor<float>((float[])seedX.Clone(), x.Shape.ToArray());
                    var dw = new Tensor<float>((float[])seedW.Clone(), w.Shape.ToArray());
                    e.Conv2DBackwardInputInto(dx, dy, w, x.Shape.ToArray(), st, pd, dl, accumulate);
                    e.Conv2DBackwardKernelInto(dw, dy, x, w.Shape.ToArray(), st, pd, dl, accumulate);
                    Assert.True(Im2ColHelper.ConvTransposePerImageForwards > perImage, $"{threads} threads: input gradient off the per-image route");
                    Assert.True(CpuEngine.ChunkedWeightGradients > chunked, $"{threads} threads: weight gradient off the chunked route");
                    var dxa = dx.ToArray(); var dwa = dw.ToArray();
                    for (int i = 0; i < dxa.Length; i++)
                    {
                        double expected = rdx[i] + (accumulate ? seedX[i] : 0);
                        Assert.True(Math.Abs(expected - dxa[i]) <= txTol, $"dx {threads}/{accumulate} [{i}]: {expected} vs {dxa[i]}");
                    }
                    for (int i = 0; i < dwa.Length; i++)
                    {
                        double expected = rdw[i] + (accumulate ? seedW[i] : 0);
                        Assert.True(Math.Abs(expected - dwa[i]) <= twTol, $"dw {threads}/{accumulate} [{i}]: {expected} vs {dwa[i]}");
                    }
                    if (accumulate) continue;
                    // The allocating weight-gradient entry takes the same chunked reduction.
                    var dwEager = e.Conv2DBackwardKernel(dy, x, w.Shape.ToArray(), st, pd, dl).ToArray();
                    for (int i = 0; i < dwa.Length; i++) Assert.Equal(Bits(dwa[i]), Bits(dwEager[i]));
                    if (firstDx is null) { firstDx = dxa; firstDw = dwa; continue; }
                    for (int i = 0; i < dxa.Length; i++) Assert.Equal(Bits(firstDx[i]), Bits(dxa[i]));
                    for (int i = 0; i < dwa.Length; i++) Assert.Equal(Bits(firstDw![i]), Bits(dwa[i]));
                }
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previous;
        }
    }
}
