using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Fused Winograd F(2x2, 3x3) (one kernel after the filter transform) against a float64 reference, for the input gradient
/// and the forward: error as max |err| / max(1, sum|terms|), bit
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
        {
            var poison = Enumerable.Repeat(float.NaN, sum.Length).ToArray();
            using var dX1 = backend.AllocateBuffer(poison);
            using var dX2 = backend.AllocateBuffer(poison);
            foreach (var target in new[] { dX1, dX2 })
            {
                var a = new CudaConv2DBackwardArgs(dDy, dW, target, false, batch, inC, h, w, outC, outH, outW, 3, 3, 1, 1, pad, pad, 1, 1);
                Assert.True(backend.TryConv2DBackwardDataWinogradF23Fused(a), "fused Winograd refused an admitted shape");
            }
            backend.Synchronize();
            CheckAgainst(sum, magnitude, backend.DownloadBuffer(dX1), backend.DownloadBuffer(dX2),
                $"dX N{batch} C{inC} K{outC} {size}x{size} pad{pad}");
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
        {
            var poison = Enumerable.Repeat(float.NaN, sum.Length).ToArray();
            using var y1 = backend.AllocateBuffer(poison);
            using var y2 = backend.AllocateBuffer(poison);
            foreach (var target in new[] { y1, y2 })
            {
                var a = new CudaConv2DArgs(dx, dw, target, batch, inC, h, w, outC, outH, outW, 3, 3, 1, 1, pad, pad, 1, 1);
                Assert.True(backend.TryConv2DForwardWinogradF23Fused(a), "fused Winograd refused an admitted shape");
            }
            backend.Synchronize();
            CheckAgainst(sum, magnitude, backend.DownloadBuffer(y1), backend.DownloadBuffer(y2),
                $"y N{batch} C{inC} K{outC} {size}x{size} pad{pad}");
        }
    }

    // The wiring: the forward and input-gradient slots register the fused route as its own candidate, never the reference.
    [SkippableFact]
    public void The_conv_slots_register_the_fused_candidates()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradFusedModule(), "Fused Winograd kernels did not compile.");
        var ids = backend.Conv2DBackwardDataSlot.CandidateIds;
        Assert.Contains("nvrtc.conv2d_dx.winograd_f23_fused", ids);
        Assert.Contains("nvrtc.conv2d_dx.winograd_f23", ids);
        Assert.NotEqual("nvrtc.conv2d_dx.winograd_f23_fused", ids[0]);
        var forwardIds = backend.Conv2DForwardSlot.CandidateIds;
        Assert.Contains("nvrtc.conv2d.winograd_f23_fused", forwardIds);
        Assert.NotEqual("nvrtc.conv2d.winograd_f23_fused", forwardIds[0]);
    }

    // The measured choice: at ResNet-18's 64@32x32 layer (batch 8) the fused route is ~2x the unfused F(2x2) route and the
    // engine for both the forward and the input gradient, so both slots must pick it. Up to three fresh decisions each (a
    // decision can lose a round to noise on a shared GPU; see WinogradF23ConvTests).
    [SkippableFact]
    public void The_conv_slots_pick_the_fused_candidate_for_resnet_64_at_32()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_KERNEL_REGISTRY_PERSIST") != "0",
            "The measured choice is checked with persisted tuning off (AIDOTNET_KERNEL_REGISTRY_PERSIST=0).");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradFusedModule(), "Fused Winograd kernels did not compile.");
        const int batch = 8, c = 64, s = 32;
        var rng = new Random(64);
        using var a0 = backend.AllocateBuffer(Rand(batch * c * s * s, rng));
        using var wt = backend.AllocateBuffer(Rand(c * c * 9, rng));
        using var o = backend.AllocateBuffer(batch * c * s * s);
        var fwd = new CudaConv2DArgs(a0, wt, o, batch, c, s, s, c, s, s, 3, 3, 1, 1, 1, 1, 1, 1);
        var dx = new CudaConv2DBackwardArgs(a0, wt, o, false, batch, c, s, s, c, s, s, 3, 3, 1, 1, 1, 1, 1, 1);

        double budget = AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.BudgetMilliseconds;
        AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.BudgetMilliseconds =
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.SpentMilliseconds + 60_000;
        string? fwdChosen = null, dxChosen = null;
        try
        {
            for (int attempt = 1; attempt <= 3 && fwdChosen != "nvrtc.conv2d.winograd_f23_fused"; attempt++)
            {
                backend.Conv2DForwardSlot.InvalidateDecisions();
                fwdChosen = backend.Conv2DForwardSlot.Resolve(fwd.Shape(), fwd)?.Id;
                _out.WriteLine($"attempt {attempt}: forward slot picked {fwdChosen ?? "<none>"} for N8 C64 32x32; " + Describe(
                    AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelOp.Conv2DForward, fwd.Shape()));
            }
            for (int attempt = 1; attempt <= 3 && dxChosen != "nvrtc.conv2d_dx.winograd_f23_fused"; attempt++)
            {
                backend.Conv2DBackwardDataSlot.InvalidateDecisions();
                dxChosen = backend.Conv2DBackwardDataSlot.Resolve(dx.Shape(), dx)?.Id;
                _out.WriteLine($"attempt {attempt}: input-gradient slot picked {dxChosen ?? "<none>"} for N8 C64 32x32; " + Describe(
                    AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelOp.Conv2DBackwardData, dx.Shape()));
            }
        }
        finally
        {
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelPolicy.BudgetMilliseconds = budget;
        }
        Assert.Equal("nvrtc.conv2d.winograd_f23_fused", fwdChosen);
        Assert.Equal("nvrtc.conv2d_dx.winograd_f23_fused", dxChosen);

        static string Describe(AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelOp op,
            AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedShape shape)
        {
            var d = AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedKernelRegistry.Decisions
                .LastOrDefault(r => r.Op == op && r.Shape.Equals(shape));
            return $"reason {d?.Reason}, reference {d?.ReferenceMilliseconds:F3} ms, chosen {d?.CandidateMilliseconds:F3} ms, " +
                   $"max rel err {d?.MaxRelativeError:E2}, rejected [{string.Join("; ", d?.Rejected ?? Array.Empty<string>())}]";
        }
    }

    /// <summary>
    /// Per ResNet-18 3x3 stride-1 layer at batch 8, interleaved round by round (min / median us per call): input gradient
    /// fused vs unfused F(2x2) vs the engine, and the same for the forward. Gated on
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
        string[] routes = { "fused", "F(2x2)", "engine" };
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
                () => backend.TryConv2DBackwardDataWinogradF23Fused(b),
                () => backend.TryConv2DBackwardDataWinogradF23(b),
                () => backend.Conv2DBackwardInputViaEngine(b),
            };
            Func<bool>[] fwdRuns =
            {
                () => backend.TryConv2DForwardWinogradF23Fused(f),
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
