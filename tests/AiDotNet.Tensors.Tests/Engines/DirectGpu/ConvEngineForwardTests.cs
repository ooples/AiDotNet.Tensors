using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1325: Conv2D forward through the implicit-GEMM conv engine (conv2d_fprop_ig*) against a float64 reference, on every
/// ResNet-18 conv shape at batch 8 (128 x 128 and 64 x 64 tiles, split-K for the deep layers, the tap-uniform and the
/// generic variants) and on edge shapes, plus an interleaved per-path benchmark of the forward routes.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ConvEngineForwardTests
{
    private readonly ITestOutputHelper _out;
    public ConvEngineForwardTests(ITestOutputHelper output) => _out = output;

    // batch, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW, dH, dW
    public static IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 8, 3, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1 },      // ResNet-18 stem: generic K order (Cin = 3)
        new object[] { 8, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1 },     // stage 1: 128-wide tiles
        new object[] { 8, 64, 32, 32, 128, 3, 3, 2, 2, 1, 1, 1, 1 },    // stage 2 entry
        new object[] { 8, 64, 32, 32, 128, 1, 1, 2, 2, 0, 0, 1, 1 },    // stage 2 shortcut
        new object[] { 8, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, 1, 1 },
        new object[] { 8, 256, 8, 8, 256, 3, 3, 1, 1, 1, 1, 1, 1 },     // 64-wide tiles, split-K
        new object[] { 8, 512, 4, 4, 512, 3, 3, 1, 1, 1, 1, 1, 1 },     // deep: split-K
        new object[] { 8, 256, 8, 8, 512, 1, 1, 2, 2, 0, 0, 1, 1 },
        new object[] { 3, 5, 7, 9, 3, 3, 2, 2, 1, 1, 0, 2, 1 },         // odd: non-square, dilation, unequal strides
        new object[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 },         // one element
        new object[] { 2, 17, 13, 11, 70, 5, 3, 1, 2, 2, 1, 1, 1 },     // partial tiles in both dims, scalar stores
        new object[] { 2, 24, 9, 11, 40, 3, 3, 1, 1, 2, 2, 2, 2 },      // tap-uniform with dilation and odd sizes
        new object[] { 4, 16, 15, 15, 200, 3, 3, 2, 2, 1, 1, 1, 1 },    // Cout spans two 128 tiles, partial
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Forward_MatchesFloat64ReferenceAndIsBitReproducible(
        int batch, int inC, int h, int w, int outC, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        int outH = (h + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int outW = (w + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        Assert.True(outH > 0 && outW > 0, "invalid test shape");

        var rng = new Random(1325 + batch * 7 + inC * 3 + outC);
        var x = new float[batch * inC * h * w];
        var weights = new float[outC * inC * kH * kW];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);

        using var xBuf = backend.AllocateBuffer(x);
        using var wBuf = backend.AllocateBuffer(weights);
        // Pre-filled with NaN: every output element must be written.
        var poison = Enumerable.Repeat(float.NaN, batch * outC * outH * outW).ToArray();
        using var y1 = backend.AllocateBuffer(poison);
        using var y2 = backend.AllocateBuffer(poison);
        var a1 = new CudaConv2DArgs(xBuf, wBuf, y1, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        var a2 = new CudaConv2DArgs(xBuf, wBuf, y2, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        Assert.True(backend.Conv2DForwardViaEngine(a1), "the engine declined the shape");
        Assert.True(backend.Conv2DForwardViaEngine(a2), "the engine declined the shape");
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(y1), again = backend.DownloadBuffer(y2);

        double maxRel = 0;
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outC; co++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double sum = 0, magnitude = 0;
                        for (int ci = 0; ci < inC; ci++)
                            for (int kh = 0; kh < kH; kh++)
                            {
                                int ih = oh * sH - pH + kh * dH;
                                if (ih < 0 || ih >= h) continue;
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int iw = ow * sW - pW + kw * dW;
                                    if (iw < 0 || iw >= w) continue;
                                    double term = (double)weights[((co * inC + ci) * kH + kh) * kW + kw]
                                                  * x[((b * inC + ci) * h + ih) * w + iw];
                                    sum += term;
                                    magnitude += Math.Abs(term);
                                }
                            }
                        int idx = ((b * outC + co) * outH + oh) * outW + ow;
                        double error = Math.Abs(got[idx] - sum);
                        maxRel = Math.Max(maxRel, error / Math.Max(1.0, magnitude));
                        Assert.True(error <= 1e-5 * Math.Max(1.0, magnitude), $"y[{b},{co},{oh},{ow}] = {got[idx]} vs {sum} (|terms| {magnitude})");
                        Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(got[idx]), 0) ==
                                    BitConverter.ToInt32(BitConverter.GetBytes(again[idx]), 0),
                            $"y[{b},{co},{oh},{ow}] differs between two identical runs: {got[idx]:R} vs {again[idx]:R}");
                    }
        _out.WriteLine($"max error / sum|terms| = {maxRel:E2}");
    }

    /// <summary>
    /// Every forward route on every ResNet-18 conv (parity/networks/resnet18.json at batch 8), interleaved round by round
    /// in one process; each sample is the mean of a burst of back-to-back calls between two synchronizes. "-" marks a
    /// route that declines the shape. Gated on AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableFact]
    public void Bench_ResNet18ConvForward()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1",
            "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        const int batch = 8, rounds = 41, burst = 10;
        var layers = new (string Name, int Count, int InC, int Hw, int OutC, int K, int S, int P)[]
        {
            ("stem 3-64 3x3 @32", 1, 3, 32, 64, 3, 1, 1),
            ("64-64 3x3 @32", 4, 64, 32, 64, 3, 1, 1),
            ("64-128 3x3 s2 @32", 1, 64, 32, 128, 3, 2, 1),
            ("64-128 1x1 s2 @32", 1, 64, 32, 128, 1, 2, 0),
            ("128-128 3x3 @16", 3, 128, 16, 128, 3, 1, 1),
            ("128-256 3x3 s2 @16", 1, 128, 16, 256, 3, 2, 1),
            ("128-256 1x1 s2 @16", 1, 128, 16, 256, 1, 2, 0),
            ("256-256 3x3 @8", 3, 256, 8, 256, 3, 1, 1),
            ("256-512 3x3 s2 @8", 1, 256, 8, 512, 3, 2, 1),
            ("256-512 1x1 s2 @8", 1, 256, 8, 512, 1, 2, 0),
            ("512-512 3x3 @4", 3, 512, 4, 512, 3, 1, 1),
        };
        string[] routes = { "engine", "narrow(#1274)", "im2col+gemm", "generic" };
        var rng = new Random(1325);
        float[] Fill(int n) { var v = new float[n]; for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1); return v; }
        double Median(List<double> v) { var o = v.OrderBy(t => t).ToList(); return o[o.Count / 2]; }

        var totMin = new double[routes.Length];
        var totBest = 0.0;
        _out.WriteLine("layer                    x  | " + string.Join(" | ", routes.Select(r => $"{r,19}")) + "  (us per call, min / median)");
        foreach (var l in layers)
        {
            int outHw = (l.Hw + 2 * l.P - l.K) / l.S + 1;
            using var x = backend.AllocateBuffer(Fill(batch * l.InC * l.Hw * l.Hw));
            using var wt = backend.AllocateBuffer(Fill(l.OutC * l.InC * l.K * l.K));
            using var y = backend.AllocateBuffer(batch * l.OutC * outHw * outHw);
            var a = new CudaConv2DArgs(x, wt, y, batch, l.InC, l.Hw, l.Hw, l.OutC, outHw, outHw, l.K, l.K, l.S, l.S, l.P, l.P, 1, 1);
            Func<bool>[] run =
            {
                () => backend.Conv2DForwardViaEngine(a),
                () => backend.Conv2DForwardViaNarrowImplicit(a),
                () => backend.Conv2DForwardViaIm2colGemm(a),
                () => backend.Conv2DForwardViaGeneric(a),
            };
            var samples = routes.Select(_ => new List<double>()).ToArray();
            var declined = new bool[routes.Length];
            for (int r = 0; r < rounds; r++)
                for (int route = 0; route < routes.Length; route++)
                {
                    if (declined[route]) continue;
                    if (!run[route]()) { declined[route] = true; continue; }
                    backend.Synchronize();
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < burst; i++) run[route]();
                    backend.Synchronize();
                    if (r > 0) samples[route].Add(sw.Elapsed.TotalMilliseconds * 1000.0 / burst);
                }
            double best = double.MaxValue;
            var cells = new string[routes.Length];
            for (int route = 0; route < routes.Length; route++)
            {
                if (declined[route] || samples[route].Count == 0) { cells[route] = $"{"-",19}"; continue; }
                double mn = samples[route].Min();
                totMin[route] += l.Count * mn;
                best = Math.Min(best, mn);
                cells[route] = $"{mn,8:F1} / {Median(samples[route]),8:F1}";
            }
            totBest += l.Count * best;
            _out.WriteLine($"{l.Name,-22} {l.Count,2}  | " + string.Join(" | ", cells));
        }
        _out.WriteLine("per-step totals (min; routes that decline a shape miss it): " +
                       string.Join(", ", routes.Select((r, i) => $"{r} {totMin[i] / 1000:F3} ms")));
        _out.WriteLine($"per-step total taking the fastest route per layer: {totBest / 1000:F3} ms");
    }
}
