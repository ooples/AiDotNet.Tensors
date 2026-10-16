using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1335: Winograd F(4x4, 3x3) input gradient (opt-in only) against a float64 reference. The error is reported both as the
/// conv tests measure it (max |err| / max(1, sum|terms|)) and as the F(2x2) Winograd tests do (max |err| / max |ref|), and
/// the opt-in gate is checked: without CudaDispatchPolicy.ConvWinogradF4 the candidate is not registered at all.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class WinogradF43ConvTests
{
    private readonly ITestOutputHelper _out;
    public WinogradF43ConvTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(8, 64, 64, 32, 1)]
    [InlineData(8, 128, 128, 16, 1)]
    [InlineData(8, 256, 256, 8, 1)]
    [InlineData(8, 512, 512, 4, 1)]
    [InlineData(3, 5, 7, 9, 1)]      // odd: partial 4x4 tiles, channels not a tile multiple
    [InlineData(2, 16, 24, 11, 0)]   // no forward padding: dX padding 2
    [InlineData(2, 8, 16, 13, 2)]    // forward padding 2: dX padding 0
    public void InputGradient_MatchesFloat64ReferenceAndIsBitReproducible(int batch, int inC, int outC, int size, int pad)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradF43Module(), "Winograd F(4x4) kernels did not compile.");
        int h = size, w = size, outH = h + 2 * pad - 2, outW = w + 2 * pad - 2;
        Assert.True(outH > 0 && outW > 0, "invalid test shape");

        var rng = new Random(4343 + batch * 7 + inC * 3 + outC);
        var weights = new float[outC * inC * 9];
        var dy = new float[batch * outC * outH * outW];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < dy.Length; i++) dy[i] = (float)(rng.NextDouble() * 2 - 1);

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
        var poison = Enumerable.Repeat(float.NaN, sum.Length).ToArray();
        using var dX1 = backend.AllocateBuffer(poison);
        using var dX2 = backend.AllocateBuffer(poison);
        foreach (var target in new[] { dX1, dX2 })
        {
            var a = new CudaConv2DBackwardArgs(dDy, dW, target, false, batch, inC, h, w, outC, outH, outW, 3, 3, 1, 1, pad, pad, 1, 1);
            Assert.True(backend.TryConv2DBackwardDataWinogradF43(a), "Winograd F(4x4) refused an admitted shape");
        }
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(dX1), again = backend.DownloadBuffer(dX2);

        double maxRelTerms = 0, maxErr = 0, maxRef = 0;
        for (int i = 0; i < sum.Length; i++)
        {
            double error = Math.Abs(got[i] - sum[i]);
            maxRelTerms = Math.Max(maxRelTerms, error / Math.Max(1.0, magnitude[i]));
            maxErr = Math.Max(maxErr, error);
            maxRef = Math.Max(maxRef, Math.Abs(sum[i]));
            Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(got[i]), 0) ==
                        BitConverter.ToInt32(BitConverter.GetBytes(again[i]), 0),
                $"dX[{i}] differs between two identical runs: {got[i]:R} vs {again[i]:R}");
        }
        _out.WriteLine($"F(4x4) dX: max err / sum|terms| = {maxRelTerms:E2}; max err / max|ref| = {maxErr / Math.Max(1e-30, maxRef):E2}");
        Assert.True(maxRelTerms <= 1e-5, $"F(4x4) input gradient: max err / sum|terms| {maxRelTerms:E2} exceeds 1e-5");
    }

    [SkippableTheory]
    [InlineData(8, 64, 64, 32, 1)]
    [InlineData(8, 128, 128, 16, 1)]
    [InlineData(8, 256, 256, 8, 1)]
    [InlineData(8, 512, 512, 4, 1)]
    [InlineData(3, 5, 7, 9, 1)]
    [InlineData(2, 16, 24, 11, 0)]
    [InlineData(2, 8, 16, 13, 2)]
    public void WeightGradient_MatchesFloat64ReferenceAndIsBitReproducible(int batch, int inC, int outC, int size, int pad)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradF43Module(), "Winograd F(4x4) kernels did not compile.");
        int h = size, w = size, outH = h + 2 * pad - 2, outW = w + 2 * pad - 2;
        Assert.True(outH > 0 && outW > 0, "invalid test shape");

        var rng = new Random(3434 + batch * 7 + inC * 3 + outC);
        var x = new float[batch * inC * h * w];
        var dy = new float[batch * outC * outH * outW];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < dy.Length; i++) dy[i] = (float)(rng.NextDouble() * 2 - 1);

        using var dX = backend.AllocateBuffer(x);
        using var dDy = backend.AllocateBuffer(dy);
        using var dW1 = backend.AllocateBuffer(outC * inC * 9);
        using var dW2 = backend.AllocateBuffer(outC * inC * 9);
        foreach (var target in new[] { dW1, dW2 })
        {
            var a = new CudaConv2DBackwardArgs(dX, dDy, target, true, batch, inC, h, w, outC, outH, outW, 3, 3, 1, 1, pad, pad, 1, 1);
            Assert.True(backend.TryConv2DBackwardFilterWinogradF34(a), "Winograd F(3x3,4x4) refused an admitted shape");
        }
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(dW1), again = backend.DownloadBuffer(dW2);

        double maxRelTerms = 0, maxErr = 0, maxRef = 0;
        for (int co = 0; co < outC; co++)
            for (int ci = 0; ci < inC; ci++)
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                    {
                        double sum = 0, magnitude = 0;
                        for (int b = 0; b < batch; b++)
                            for (int oh = 0; oh < outH; oh++)
                            {
                                int ih = oh - pad + kh;
                                if (ih < 0 || ih >= h) continue;
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int iw = ow - pad + kw;
                                    if (iw < 0 || iw >= w) continue;
                                    double term = (double)dy[((b * outC + co) * outH + oh) * outW + ow]
                                                  * x[((b * inC + ci) * h + ih) * w + iw];
                                    sum += term;
                                    magnitude += Math.Abs(term);
                                }
                            }
                        int idx = ((co * inC + ci) * 3 + kh) * 3 + kw;
                        double error = Math.Abs(got[idx] - sum);
                        maxRelTerms = Math.Max(maxRelTerms, error / Math.Max(1.0, magnitude));
                        maxErr = Math.Max(maxErr, error);
                        maxRef = Math.Max(maxRef, Math.Abs(sum));
                        Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(got[idx]), 0) ==
                                    BitConverter.ToInt32(BitConverter.GetBytes(again[idx]), 0),
                            $"dW[{idx}] differs between two identical runs: {got[idx]:R} vs {again[idx]:R}");
                    }
        _out.WriteLine($"F(3x3,4x4) dW: max err / sum|terms| = {maxRelTerms:E2}; max err / max|ref| = {maxErr / Math.Max(1e-30, maxRef):E2}");
        Assert.True(maxRelTerms <= 1e-5, $"F(3x3,4x4) weight gradient: max err / sum|terms| {maxRelTerms:E2} exceeds 1e-5");
    }

    /// <summary>
    /// Weight gradient per ResNet-18 3x3 stride-1 layer at batch 8: F(3x3,4x4) vs F(3x3,2x2) Winograd vs #1323's implicit
    /// GEMM vs the engine, interleaved round by round (min / median us per call). Gated on AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableFact]
    public void Bench_WeightGradient_F34VsF32VsImplicitVsEngine()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradF43Module() && backend.EnsureWinogradModule(), "Winograd kernels did not compile.");
        const int batch = 8, rounds = 41, burst = 10;
        var rng = new Random(34);
        float[] Fill(int n) { var v = new float[n]; for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1); return v; }
        string[] routes = { "F(3,4)", "F(3,2)", "#1323", "engine" };
        var totals = new double[routes.Length];
        foreach (var (c, hw, count) in new[] { (64, 32, 4), (128, 16, 3), (256, 8, 3), (512, 4, 3) })
        {
            using var xb = backend.AllocateBuffer(Fill(batch * c * hw * hw));
            using var dy = backend.AllocateBuffer(Fill(batch * c * hw * hw));
            using var dw = backend.AllocateBuffer(c * c * 9);
            var a = new CudaConv2DBackwardArgs(xb, dy, dw, true, batch, c, hw, hw, c, hw, hw, 3, 3, 1, 1, 1, 1, 1, 1);
            Func<bool>[] run =
            {
                () => backend.TryConv2DBackwardFilterWinogradF34(a),
                () => backend.TryConv2DBackwardFilterWinogradF32(a),
                () => backend.Conv2DBackwardKernelViaImplicit(a),
                () => backend.Conv2DBackwardKernelViaEngine(a),
            };
            var samples = routes.Select(_ => new System.Collections.Generic.List<double>()).ToArray();
            for (int r = 0; r < rounds; r++)
                for (int k = 0; k < run.Length; k++)
                {
                    Assert.True(run[k](), $"{routes[k]} declined {c}@{hw}");
                    backend.Synchronize();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < burst; i++) run[k]();
                    backend.Synchronize();
                    if (r > 0) samples[k].Add(sw.Elapsed.TotalMilliseconds * 1000.0 / burst);
                }
            double Med(System.Collections.Generic.List<double> v) { var o = v.OrderBy(t => t).ToList(); return o[o.Count / 2]; }
            for (int k = 0; k < routes.Length; k++) totals[k] += count * samples[k].Min();
            _out.WriteLine($"dW {c}@{hw} x{count}: " + string.Join(" | ", routes.Select((rt, k) => $"{rt} {samples[k].Min():F1}/{Med(samples[k]):F1}")));
        }
        _out.WriteLine("dW per step (min): " + string.Join(", ", routes.Select((rt, k) => $"{rt} {totals[k] / 1000:F3} ms")));
    }

    /// <summary>
    /// Input gradient per ResNet-18 3x3 stride-1 layer at batch 8: F(4x4) vs F(2x2) Winograd vs the engine, interleaved
    /// round by round in one process (min / median us per call). Gated on AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableFact]
    public void Bench_InputGradient_F43VsF23VsEngine()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        Skip.IfNot(backend.EnsureWinogradF43Module() && backend.EnsureWinogradModule(), "Winograd kernels did not compile.");
        const int batch = 8, rounds = 41, burst = 10;
        var rng = new Random(43);
        float[] Fill(int n) { var v = new float[n]; for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1); return v; }
        string[] routes = { "F(4x4)", "F(2x2)", "engine" };
        double total4 = 0, total2 = 0, totalE = 0;
        foreach (var (c, hw, count) in new[] { (64, 32, 4), (128, 16, 3), (256, 8, 3), (512, 4, 3) })
        {
            using var dy = backend.AllocateBuffer(Fill(batch * c * hw * hw));
            using var wt = backend.AllocateBuffer(Fill(c * c * 9));
            using var dx = backend.AllocateBuffer(batch * c * hw * hw);
            var a = new CudaConv2DBackwardArgs(dy, wt, dx, false, batch, c, hw, hw, c, hw, hw, 3, 3, 1, 1, 1, 1, 1, 1);
            Func<bool>[] run =
            {
                () => backend.TryConv2DBackwardDataWinogradF43(a),
                () => backend.TryConv2DBackwardDataWinogradF23(a),
                () => backend.Conv2DBackwardInputViaEngine(a),
            };
            var samples = routes.Select(_ => new System.Collections.Generic.List<double>()).ToArray();
            for (int r = 0; r < rounds; r++)
                for (int k = 0; k < run.Length; k++)
                {
                    Assert.True(run[k](), $"{routes[k]} declined {c}@{hw}");
                    backend.Synchronize();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < burst; i++) run[k]();
                    backend.Synchronize();
                    if (r > 0) samples[k].Add(sw.Elapsed.TotalMilliseconds * 1000.0 / burst);
                }
            double Med(System.Collections.Generic.List<double> v) { var o = v.OrderBy(t => t).ToList(); return o[o.Count / 2]; }
            total4 += count * samples[0].Min(); total2 += count * samples[1].Min(); totalE += count * samples[2].Min();
            _out.WriteLine($"{c}@{hw} x{count}: " + string.Join(" | ", routes.Select((rt, k) => $"{rt} {samples[k].Min():F1}/{Med(samples[k]):F1}")));
        }
        _out.WriteLine($"per step (min): F(4x4) {total4 / 1000:F3} ms, F(2x2) {total2 / 1000:F3} ms, engine {totalE / 1000:F3} ms");
    }

    [SkippableFact]
    public void F43Candidate_IsRegisteredOnlyUnderTheOptIn()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        try
        {
            CudaDispatchPolicy.ConvWinogradF4 = false;
            using (var off = CudaBackend.CreateOrThrow())
                Assert.DoesNotContain("nvrtc.conv2d_dx.winograd_f43", off.Conv2DBackwardDataSlot.CandidateIds);
            using (var off2 = CudaBackend.CreateOrThrow())
                Assert.DoesNotContain("nvrtc.conv2d_dw.winograd_f34", off2.Conv2DBackwardFilterSlot.CandidateIds);
            CudaDispatchPolicy.ConvWinogradF4 = true;
            using (var on = CudaBackend.CreateOrThrow())
                Assert.Contains("nvrtc.conv2d_dx.winograd_f43", on.Conv2DBackwardDataSlot.CandidateIds);
            using (var on2 = CudaBackend.CreateOrThrow())
                Assert.Contains("nvrtc.conv2d_dw.winograd_f34", on2.Conv2DBackwardFilterSlot.CandidateIds);
        }
        finally
        {
            CudaDispatchPolicy.ConvWinogradF4 = false;
        }
    }
}
