using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Fused Winograd F(2x2, 3x3) (one kernel after the filter transform) against a float64 reference, for the input gradient
/// and the forward, in both launch variants (two blocks per SM and one): error as max |err| / max(1, sum|terms|), bit
/// reproducibility across two runs into NaN-poisoned outputs (every element must be stored), and an interleaved bench
/// against the unfused F(2x2) route and the engine.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class WinogradF23FusedConvTests
{
    private readonly ITestOutputHelper _out;
    public WinogradF23FusedConvTests(ITestOutputHelper output) => _out = output;

    private static float[] Rand(int n, Random rng)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private static string VariantName(int variant) => variant switch { 1 => "1 block/SM", 2 => "double-buffered", _ => "2 blocks/SM" };

    private void CheckAgainst(double[] sum, double[] magnitude, float[] got, float[] again, string what)
    {
        double maxRelTerms = 0;
        for (int i = 0; i < sum.Length; i++)
        {
            double error = Math.Abs(got[i] - sum[i]);
            Assert.False(double.IsNaN(error), $"{what}[{i}] was never written (NaN poison survived)");
            maxRelTerms = Math.Max(maxRelTerms, error / Math.Max(1.0, magnitude[i]));
            Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(got[i]), 0) ==
                        BitConverter.ToInt32(BitConverter.GetBytes(again[i]), 0),
                $"{what}[{i}] differs between two identical runs: {got[i]:R} vs {again[i]:R}");
        }
        _out.WriteLine($"{what}: max err / sum|terms| = {maxRelTerms:E2}");
        Assert.True(maxRelTerms <= 1e-6, $"{what}: max err / sum|terms| {maxRelTerms:E2} exceeds 1e-6");
    }

    [SkippableTheory]
    [InlineData(8, 64, 64, 32, 1)]
    [InlineData(8, 128, 128, 16, 1)]
    [InlineData(8, 256, 256, 8, 1)]
    [InlineData(3, 5, 7, 9, 1)]      // odd: partial tiles, channel counts below one block step
    [InlineData(2, 16, 40, 11, 0)]   // no forward padding: dX padding 2; 40 channels = a partial 32-channel block
    [InlineData(2, 8, 16, 13, 2)]    // forward padding 2: dX padding 0
    [InlineData(1, 33, 70, 7, 1)]    // tiles per image not a multiple of the 32-tile block
    public void InputGradient_MatchesFloat64ReferenceAndIsBitReproducible(int batch, int inC, int outC, int size, int pad)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradFusedModule(), "Fused Winograd kernels did not compile.");
        int h = size, w = size, outH = h + 2 * pad - 2, outW = w + 2 * pad - 2;
        var rng = new Random(2323 + batch * 7 + inC * 3 + outC);
        var weights = Rand(outC * inC * 9, rng);
        var dy = Rand(batch * outC * outH * outW, rng);

        var sum = new double[batch * inC * h * w];
        var magnitude = new double[sum.Length];
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outC; co++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double g = dy[((b * outC + co) * outH + oh) * outW + ow];
                        for (int ci = 0; ci < inC; ci++)
                            for (int kh = 0; kh < 3; kh++)
                            {
                                int ih = oh - pad + kh;
                                if (ih < 0 || ih >= h) continue;
                                for (int kw = 0; kw < 3; kw++)
                                {
                                    int iw = ow - pad + kw;
                                    if (iw < 0 || iw >= w) continue;
                                    double term = g * weights[((co * inC + ci) * 3 + kh) * 3 + kw];
                                    int idx = ((b * inC + ci) * h + ih) * w + iw;
                                    sum[idx] += term;
                                    magnitude[idx] += Math.Abs(term);
                                }
                            }
                    }

        using var dDy = backend.AllocateBuffer(dy);
        using var dW = backend.AllocateBuffer(weights);
        foreach (int variant in new[] { 0, 1, 2 })
        {
            var poison = Enumerable.Repeat(float.NaN, sum.Length).ToArray();
            using var dX1 = backend.AllocateBuffer(poison);
            using var dX2 = backend.AllocateBuffer(poison);
            foreach (var target in new[] { dX1, dX2 })
            {
                var a = new CudaConv2DBackwardArgs(dDy, dW, target, false, batch, inC, h, w, outC, outH, outW, 3, 3, 1, 1, pad, pad, 1, 1);
                Assert.True(backend.TryConv2DBackwardDataWinogradF23Fused(a, variant), "fused Winograd refused an admitted shape");
            }
            backend.Synchronize();
            CheckAgainst(sum, magnitude, backend.DownloadBuffer(dX1), backend.DownloadBuffer(dX2),
                $"dX N{batch} C{inC} K{outC} {size}x{size} pad{pad} {VariantName(variant)}");
        }
    }

    [SkippableTheory]
    [InlineData(8, 64, 64, 32, 1)]
    [InlineData(3, 5, 7, 9, 1)]
    [InlineData(2, 16, 40, 11, 0)]
    [InlineData(1, 33, 70, 7, 2)]
    public void Forward_MatchesFloat64ReferenceAndIsBitReproducible(int batch, int inC, int outC, int size, int pad)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradFusedModule(), "Fused Winograd kernels did not compile.");
        int h = size, w = size, outH = h + 2 * pad - 2, outW = w + 2 * pad - 2;
        var rng = new Random(3232 + batch * 7 + inC * 3 + outC);
        var x = Rand(batch * inC * h * w, rng);
        var weights = Rand(outC * inC * 9, rng);

        var sum = new double[batch * outC * outH * outW];
        var magnitude = new double[sum.Length];
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outC; co++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double s = 0, m = 0;
                        for (int ci = 0; ci < inC; ci++)
                            for (int kh = 0; kh < 3; kh++)
                            {
                                int ih = oh - pad + kh;
                                if (ih < 0 || ih >= h) continue;
                                for (int kw = 0; kw < 3; kw++)
                                {
                                    int iw = ow - pad + kw;
                                    if (iw < 0 || iw >= w) continue;
                                    double term = (double)x[((b * inC + ci) * h + ih) * w + iw] * weights[((co * inC + ci) * 3 + kh) * 3 + kw];
                                    s += term;
                                    m += Math.Abs(term);
                                }
                            }
                        int idx = ((b * outC + co) * outH + oh) * outW + ow;
                        sum[idx] = s;
                        magnitude[idx] = m;
                    }

        using var dx = backend.AllocateBuffer(x);
        using var dw = backend.AllocateBuffer(weights);
        foreach (int variant in new[] { 0, 1, 2 })
        {
            var poison = Enumerable.Repeat(float.NaN, sum.Length).ToArray();
            using var y1 = backend.AllocateBuffer(poison);
            using var y2 = backend.AllocateBuffer(poison);
            foreach (var target in new[] { y1, y2 })
            {
                var a = new CudaConv2DArgs(dx, dw, target, batch, inC, h, w, outC, outH, outW, 3, 3, 1, 1, pad, pad, 1, 1);
                Assert.True(backend.TryConv2DForwardWinogradF23Fused(a, variant), "fused Winograd refused an admitted shape");
            }
            backend.Synchronize();
            CheckAgainst(sum, magnitude, backend.DownloadBuffer(y1), backend.DownloadBuffer(y2),
                $"y N{batch} C{inC} K{outC} {size}x{size} pad{pad} {VariantName(variant)}");
        }
    }

    // The wiring: the input-gradient slot registers the fused route as its own candidate, never as the reference.
    [SkippableFact]
    public void The_input_gradient_slot_registers_the_fused_candidate()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradFusedModule(), "Fused Winograd kernels did not compile.");
        var ids = backend.Conv2DBackwardDataSlot.CandidateIds;
        Assert.Contains("nvrtc.conv2d_dx.winograd_f23_fused", ids);
        Assert.Contains("nvrtc.conv2d_dx.winograd_f23", ids);
        Assert.NotEqual("nvrtc.conv2d_dx.winograd_f23_fused", ids[0]);
    }

    /// <summary>
    /// Per ResNet-18 3x3 stride-1 layer at batch 8, interleaved round by round (min / median us per call): input gradient
    /// fused (2 and 1 blocks/SM) vs unfused F(2x2) vs the engine, and the same for the forward. Gated on
    /// AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableFact]
    public void Bench_FusedVsUnfusedVsEngine()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradFusedModule() && backend.EnsureWinogradModule(), "Winograd kernels did not compile.");
        const int batch = 8, rounds = 41, burst = 10;
        var rng = new Random(23);
        string[] routes = { "fused2", "fused1", "fusedDB", "F(2x2)", "engine" };
        var totalsDx = new double[routes.Length];
        var totalsFwd = new double[routes.Length];
        foreach (var (c, hw, count) in new[] { (64, 32, 4), (128, 16, 3), (256, 8, 3), (512, 4, 3) })
        {
            using var a0 = backend.AllocateBuffer(Rand(batch * c * hw * hw, rng));
            using var wt = backend.AllocateBuffer(Rand(c * c * 9, rng));
            using var o = backend.AllocateBuffer(batch * c * hw * hw);
            var b = new CudaConv2DBackwardArgs(a0, wt, o, false, batch, c, hw, hw, c, hw, hw, 3, 3, 1, 1, 1, 1, 1, 1);
            var f = new CudaConv2DArgs(a0, wt, o, batch, c, hw, hw, c, hw, hw, 3, 3, 1, 1, 1, 1, 1, 1);
            Func<bool>[] dxRuns =
            {
                () => backend.TryConv2DBackwardDataWinogradF23Fused(b, 0),
                () => backend.TryConv2DBackwardDataWinogradF23Fused(b, 1),
                () => backend.TryConv2DBackwardDataWinogradF23Fused(b, 2),
                () => backend.TryConv2DBackwardDataWinogradF23(b),
                () => backend.Conv2DBackwardInputViaEngine(b),
            };
            Func<bool>[] fwdRuns =
            {
                () => backend.TryConv2DForwardWinogradF23Fused(f, 0),
                () => backend.TryConv2DForwardWinogradF23Fused(f, 1),
                () => backend.TryConv2DForwardWinogradF23Fused(f, 2),
                () => backend.TryConv2DForwardWinogradF23(f),
                () => backend.TryConv2DForwardEngine(f),
            };
            foreach (var (label, run, totals) in new[] { ("dX", dxRuns, totalsDx), ("fwd", fwdRuns, totalsFwd) })
            {
                var samples = routes.Select(_ => new List<double>()).ToArray();
                for (int r = 0; r < rounds; r++)
                    for (int k = 0; k < run.Length; k++)
                    {
                        Assert.True(run[k](), $"{routes[k]} declined {label} {c}@{hw}");
                        backend.Synchronize();
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        for (int i = 0; i < burst; i++) run[k]();
                        backend.Synchronize();
                        if (r > 0) samples[k].Add(sw.Elapsed.TotalMilliseconds * 1000.0 / burst);
                    }
                double Med(List<double> v) { var s = v.OrderBy(t => t).ToList(); return s[s.Count / 2]; }
                for (int k = 0; k < routes.Length; k++) totals[k] += count * samples[k].Min();
                _out.WriteLine($"{label} {c}@{hw} x{count}: " + string.Join(" | ", routes.Select((rt, k) => $"{rt} {samples[k].Min():F1}/{Med(samples[k]):F1}")));
            }
        }
        _out.WriteLine("dX per step (min): " + string.Join(", ", routes.Select((rt, k) => $"{rt} {totalsDx[k] / 1000:F3} ms")));
        _out.WriteLine("fwd per step (min): " + string.Join(", ", routes.Select((rt, k) => $"{rt} {totalsFwd[k] / 1000:F3} ms")));
    }
}
