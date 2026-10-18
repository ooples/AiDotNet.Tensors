using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1353 prototype: the persistent fused basic block (<see cref="CudaBackend.FusedBasicBlockForward"/>) against a
/// double-precision reference of conv -> BN (batch statistics, biased variance) -> ReLU -> conv -> BN -> + x -> ReLU,
/// at the issue's four ResNet-18 stage shapes (batch 8) and a small one; and bit-reproducible across runs and across
/// grid sizes, which is what the fixed-order statistics are for.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaFusedBasicBlockTests
{
    private sealed class Block : IDisposable
    {
        public readonly int N, C, H, W;
        public readonly float[] X, W1, G1, B1, W2, G2, B2;
        private readonly CudaBackend _b;
        public readonly IGpuBuffer DX, DW1, DG1, DB1, DW2, DG2, DB2, DC1, DC2, DStats, DY, DBar;

        public Block(CudaBackend b, int n, int c, int h, int w, int seed)
        {
            _b = b; N = n; C = c; H = h; W = w;
            var rng = new Random(seed);
            float[] R(int len, double scale, double shift = 0) =>
                Enumerable.Range(0, len).Select(_ => (float)(shift + (rng.NextDouble() * 2 - 1) * scale)).ToArray();
            double he = Math.Sqrt(2.0 / (c * 9));
            X = R(n * c * h * w, 1); W1 = R(c * c * 9, he); W2 = R(c * c * 9, he);
            G1 = R(c, 0.2, 1); B1 = R(c, 0.1); G2 = R(c, 0.2, 1); B2 = R(c, 0.1);
            DX = b.AllocateBuffer(X); DW1 = b.AllocateBuffer(W1); DW2 = b.AllocateBuffer(W2);
            DG1 = b.AllocateBuffer(G1); DB1 = b.AllocateBuffer(B1); DG2 = b.AllocateBuffer(G2); DB2 = b.AllocateBuffer(B2);
            DC1 = b.AllocateBuffer(X.Length); DC2 = b.AllocateBuffer(X.Length); DY = b.AllocateBuffer(X.Length);
            DStats = b.AllocateBuffer(4 * c); DBar = b.AllocateBuffer(2);
        }

        public float[] Run(int grid = 0)
        {
            _b.FusedBasicBlockForward(DX, DW1, DG1, DB1, DW2, DG2, DB2, DC1, DC2, DStats, DY, DBar, N, C, H, W, 1e-5f, grid);
            _b.Synchronize();
            return _b.DownloadBuffer(DY);
        }

        public void Dispose()
        {
            foreach (var d in new[] { DX, DW1, DG1, DB1, DW2, DG2, DB2, DC1, DC2, DStats, DY, DBar }) d.Dispose();
        }
    }

    private static double[] Conv3x3(double[] x, float[] w, int n, int c, int h, int wd)
    {
        var y = new double[x.Length];
        for (int b = 0; b < n; b++)
            for (int co = 0; co < c; co++)
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < wd; j++)
                    {
                        double acc = 0;
                        for (int ci = 0; ci < c; ci++)
                            for (int kh = 0; kh < 3; kh++)
                            {
                                int ih = i + kh - 1;
                                if (ih < 0 || ih >= h) continue;
                                for (int kw = 0; kw < 3; kw++)
                                {
                                    int iw = j + kw - 1;
                                    if (iw < 0 || iw >= wd) continue;
                                    acc += x[((b * c + ci) * h + ih) * wd + iw] * w[((co * c + ci) * 3 + kh) * 3 + kw];
                                }
                            }
                        y[((b * c + co) * h + i) * wd + j] = acc;
                    }
        return y;
    }

    private static void BatchNorm(double[] v, float[] g, float[] beta, int n, int c, int hw, bool relu, double[]? residual)
    {
        for (int ch = 0; ch < c; ch++)
        {
            double s = 0, q = 0;
            for (int b = 0; b < n; b++) for (int i = 0; i < hw; i++) s += v[(b * c + ch) * hw + i];
            double m = s / (n * hw);
            for (int b = 0; b < n; b++) for (int i = 0; i < hw; i++) { double d = v[(b * c + ch) * hw + i] - m; q += d * d; }
            double inv = 1 / Math.Sqrt(q / (n * hw) + 1e-5);
            for (int b = 0; b < n; b++)
                for (int i = 0; i < hw; i++)
                {
                    int k = (b * c + ch) * hw + i;
                    double o = (v[k] - m) * inv * g[ch] + beta[ch] + (residual is null ? 0 : residual[k]);
                    v[k] = relu ? Math.Max(0, o) : o;
                }
        }
    }

    private static double[] Reference(Block blk)
    {
        var x = blk.X.Select(v => (double)v).ToArray();
        var t = Conv3x3(x, blk.W1, blk.N, blk.C, blk.H, blk.W);
        BatchNorm(t, blk.G1, blk.B1, blk.N, blk.C, blk.H * blk.W, true, null);
        var u = Conv3x3(t, blk.W2, blk.N, blk.C, blk.H, blk.W);
        BatchNorm(u, blk.G2, blk.B2, blk.N, blk.C, blk.H * blk.W, true, x);
        return u;
    }

    private static CudaBackend RequireBackend()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable || !backend.SupportsFusedBasicBlock(8, 64, 8, 8))
        {
            backend.Dispose();
            Skip.If(true, "CUDA with cooperative launch is not available.");
        }
        return backend;
    }

    [SkippableTheory]
    [InlineData(2, 64, 4, 8)]     // small: one channel tile, one pixel tile
    [InlineData(8, 64, 32, 32)]   // ResNet-18 stage shapes at batch 8 (#1353)
    [InlineData(8, 128, 16, 16)]
    [InlineData(8, 256, 8, 8)]
    [InlineData(8, 512, 4, 4)]
    public void FusedBasicBlock_MatchesTheUnfusedReference(int n, int c, int h, int w)
    {
        using var backend = RequireBackend();
        using var blk = new Block(backend, n, c, h, w, n * 1000 + c + h);
        float[] got = blk.Run();
        double[] expected = Reference(blk);
        double num = 0, den = 0, maxErr = 0;
        for (int i = 0; i < got.Length; i++)
        {
            Assert.False(float.IsNaN(got[i]) || float.IsInfinity(got[i]), $"non-finite output at {i}");
            double d = got[i] - expected[i];
            num += d * d; den += expected[i] * expected[i]; maxErr = Math.Max(maxErr, Math.Abs(d));
        }
        double relL2 = Math.Sqrt(num / den);
        Assert.True(relL2 < 1e-5 && maxErr < 1e-3, $"[{n},{c},{h},{w}] relL2 {relL2:E3}, max |err| {maxErr:E3}");
    }

    [SkippableFact]
    public void FusedBasicBlock_IsBitReproducible_AcrossRunsAndGridSizes()
    {
        using var backend = RequireBackend();
        using var blk = new Block(backend, 8, 128, 16, 16, 7);
        float[] first = blk.Run();
        foreach (int grid in new[] { 0, 1, 5, 24 })
        {
            float[] again = blk.Run(grid);
            for (int i = 0; i < first.Length; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first[i]) == TestHelpers.MathCompat.SingleToInt32Bits(again[i]),
                    $"grid {grid}: [{i}] {again[i]:R} != {first[i]:R}");
        }
    }
}
