using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The 8x8-micro-tile forward convolution (conv2d_fprop_r8_*) against a float64 reference on ResNet-18's layers and odd
/// shapes, both tile shapes, with and without a split reduction: within the engine forward's 1e-5 of sum|terms| and
/// bit-reproducible. Input channel counts off a multiple of four are refused.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ConvFpropR8Tests
{
    private readonly ITestOutputHelper _out;
    public ConvFpropR8Tests(ITestOutputHelper output) => _out = output;

    private static float[] Rand(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    // batch, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW, dH, dW
    public static IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 8, 64, 32, 32, 128, 3, 3, 2, 2, 1, 1, 1, 1 },    // stage 2 entry
        new object[] { 8, 128, 16, 16, 256, 3, 3, 2, 2, 1, 1, 1, 1 },
        new object[] { 8, 256, 8, 8, 512, 3, 3, 2, 2, 1, 1, 1, 1 },     // deep: split-K
        new object[] { 8, 64, 32, 32, 128, 1, 1, 2, 2, 0, 0, 1, 1 },    // 1x1 shortcut
        new object[] { 8, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1 },
        new object[] { 3, 8, 7, 9, 3, 3, 2, 2, 1, 1, 0, 2, 1 },         // odd: non-square, dilation, unequal strides
        new object[] { 2, 16, 13, 11, 70, 5, 3, 1, 2, 2, 1, 1, 1 },     // partial tiles both ways, pixels not a multiple of 4
        new object[] { 4, 16, 15, 15, 200, 3, 3, 2, 2, 1, 1, 1, 1 },    // Cout spans two 128 tiles
    };

    public static IEnumerable<object[]> ShapesTilesSplits() =>
        Shapes().SelectMany(s => new[] { (64, 128, 0), (128, 64, 0), (64, 128, 3) }
            .Select(t => s.Concat(new object[] { t.Item1, t.Item2, t.Item3 }).ToArray()));

    [SkippableTheory]
    [MemberData(nameof(ShapesTilesSplits))]
    public void Forward_MatchesFloat64ReferenceAndIsBitReproducible(int batch, int inC, int h, int w, int outC, int kH, int kW,
        int sH, int sW, int pH, int pW, int dH, int dW, int tileM, int tileN, int splits)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureFpropR8Module(), "Forward r8 kernels did not compile.");
        int outH = (h + 2 * pH - dH * (kH - 1) - 1) / sH + 1, outW = (w + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        var rng = new Random(8800 + batch * 7 + inC * 3 + outC);
        var x = Rand(batch * inC * h * w, rng);
        var wt = Rand(outC * inC * kH * kW, rng);
        using var dx = backend.AllocateBuffer(x);
        using var dw = backend.AllocateBuffer(wt);
        var poison = Enumerable.Repeat(float.NaN, batch * outC * outH * outW).ToArray();
        using var y1 = backend.AllocateBuffer(poison);
        using var y2 = backend.AllocateBuffer(poison);
        foreach (var y in new[] { y1, y2 })
            Assert.True(backend.TryConv2DForwardR8(new CudaConv2DArgs(dx, dw, y, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW,
                pH, pW, dH, dW), tileM, tileN, splits), "the r8 forward refused an admitted shape");
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(y1), again = backend.DownloadBuffer(y2);
        double worst = 0;
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outC; co++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double s = 0, m = 0;
                        for (int ci = 0; ci < inC; ci++)
                            for (int i = 0; i < kH; i++)
                            {
                                int ih = oh * sH - pH + i * dH;
                                if (ih < 0 || ih >= h) continue;
                                for (int j = 0; j < kW; j++)
                                {
                                    int iw = ow * sW - pW + j * dW;
                                    if (iw < 0 || iw >= w) continue;
                                    double t = (double)x[((b * inC + ci) * h + ih) * w + iw] * wt[((co * inC + ci) * kH + i) * kW + j];
                                    s += t; m += Math.Abs(t);
                                }
                            }
                        int idx = ((b * outC + co) * outH + oh) * outW + ow;
                        Assert.False(float.IsNaN(got[idx]), $"y[{idx}] was never written");
                        Assert.True(BitConverter.SingleToInt32Bits(got[idx]) == BitConverter.SingleToInt32Bits(again[idx]),
                            $"y[{idx}] differs between two identical runs");
                        double rel = Math.Abs(got[idx] - s) / Math.Max(1.0, m);
                        worst = Math.Max(worst, rel);
                        Assert.True(rel <= 1e-5, $"y[{b},{co},{oh},{ow}] = {got[idx]} vs {s} (|terms| {m})");
                    }
        _out.WriteLine($"max err / sum|terms| = {worst:E2}");
    }

    // The wiring: the forward slot offers both r8 tiles as candidates, never as its reference.
    [SkippableFact]
    public void The_forward_slot_registers_the_r8_candidates()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CONV_R8_FPROP") == "0", "The r8 forward is switched off.");
        using var backend = CudaBackend.CreateOrThrow();
        var ids = backend.Conv2DForwardSlot.CandidateIds;
        Assert.Contains("nvrtc.conv2d.r8_64x128", ids);
        Assert.Contains("nvrtc.conv2d.r8_128x64", ids);
        Assert.DoesNotContain("nvrtc.conv2d.r8", ids[0]);
    }

    [SkippableFact]
    public void Input_channels_off_a_multiple_of_four_are_refused()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        using var x = backend.AllocateBuffer(8 * 3 * 32 * 32);
        using var w = backend.AllocateBuffer(64 * 3 * 9);
        using var y = backend.AllocateBuffer(8 * 64 * 32 * 32);
        Assert.False(backend.TryConv2DForwardR8(new CudaConv2DArgs(x, w, y, 8, 3, 32, 32, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1), 64, 128));
    }

    /// <summary>
    /// ResNet-18's strided, 1x1 and stride-1 layers at batch 8, interleaved round by round (min / median us per call): the
    /// r8 forward (both tiles) against the engine's FMA mainloop. Gated on AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableFact]
    public void Bench_R8VsEngine()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureFpropR8Module(), "Forward r8 kernels did not compile.");
        const int rounds = 41, burst = 10;
        var rng = new Random(23);
        string[] routes = { "r8 64x128", "r8 128x64", "engine" };
        foreach (var (inC, hw, outC, k, s) in new[] { (64, 32, 128, 3, 2), (128, 16, 256, 3, 2), (256, 8, 512, 3, 2),
                     (64, 32, 128, 1, 2), (128, 16, 256, 1, 2), (256, 8, 512, 1, 2), (64, 32, 64, 3, 1), (512, 4, 512, 3, 1) })
        {
            int p = k == 3 ? 1 : 0, oh = (hw + 2 * p - k) / s + 1;
            using var x = backend.AllocateBuffer(Rand(8 * inC * hw * hw, rng));
            using var w = backend.AllocateBuffer(Rand(outC * inC * k * k, rng));
            using var y = backend.AllocateBuffer(8 * outC * oh * oh);
            var a = new CudaConv2DArgs(x, w, y, 8, inC, hw, hw, outC, oh, oh, k, k, s, s, p, p, 1, 1);
            Func<bool>[] runs =
            {
                () => backend.TryConv2DForwardR8(a, 64, 128), () => backend.TryConv2DForwardR8(a, 128, 64),
                () => backend.Conv2DForwardViaEngine(a, CudaBackend.ConvMainloopFma),
            };
            var samples = routes.Select(_ => new List<double>()).ToArray();
            for (int r = 0; r < rounds; r++)
                for (int q = 0; q < runs.Length; q++)
                {
                    Assert.True(runs[q](), $"{routes[q]} declined {inC}->{outC}@{hw} k{k} s{s}");
                    backend.Synchronize();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < burst; i++) runs[q]();
                    backend.Synchronize();
                    if (r > 0) samples[q].Add(sw.Elapsed.TotalMilliseconds * 1000.0 / burst);
                }
            double Med(List<double> v) { var o = v.OrderBy(t => t).ToList(); return o[o.Count / 2]; }
            _out.WriteLine($"fwd {inC}->{outC}@{hw} k{k} s{s}: " + string.Join(" | ", routes.Select((rt, q) => $"{rt} {samples[q].Min():F1}/{Med(samples[q]):F1}")));
        }
    }
}
