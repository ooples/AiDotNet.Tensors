using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1313: the CUDA Conv2D weight and input gradients through the implicit-GEMM kernels (no im2col / col buffer,
/// deterministic split-K) against a float64 reference, on every ResNet-18 conv shape at batch 8 and on shapes that exercise the edges:
/// non-square input, a non-square kernel, unequal strides, dilation, and K / P / Cout that do not tile 16 or 64.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ConvImplicitGemmBackwardTests
{
    private readonly ITestOutputHelper _out;
    public ConvImplicitGemmBackwardTests(ITestOutputHelper output) => _out = output;

    // batch, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW, dH, dW
    public static System.Collections.Generic.IEnumerable<object[]> Shapes() => new[]
    {
        new object[] { 8, 3, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1 },      // ResNet-18 stem
        new object[] { 8, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1 },     // stage 1
        new object[] { 8, 64, 32, 32, 128, 3, 3, 2, 2, 1, 1, 1, 1 },    // stage 2 entry
        new object[] { 8, 64, 32, 32, 128, 1, 1, 2, 2, 0, 0, 1, 1 },    // stage 2 shortcut
        new object[] { 8, 128, 16, 16, 128, 3, 3, 1, 1, 1, 1, 1, 1 },
        new object[] { 8, 256, 8, 8, 512, 3, 3, 2, 2, 1, 1, 1, 1 },     // stage 4 entry
        new object[] { 8, 512, 4, 4, 512, 3, 3, 1, 1, 1, 1, 1, 1 },
        new object[] { 8, 256, 8, 8, 512, 1, 1, 2, 2, 0, 0, 1, 1 },
        new object[] { 3, 5, 7, 9, 3, 3, 2, 2, 1, 1, 0, 2, 1 },         // odd: non-square, dilation, unequal strides
        new object[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 },         // one element
        new object[] { 2, 17, 13, 11, 70, 5, 3, 1, 2, 2, 1, 1, 1 },     // P = 255, Cout = 70: partial tiles in both dims
        new object[] { 2, 5, 7, 9, 32, 3, 3, 1, 1, 2, 2, 2, 2 },        // Cout % 16 == 0, H*W = 63: scalar stores, dilation
        new object[] { 2, 6, 9, 11, 16, 3, 2, 2, 3, 1, 1, 1, 1 },       // Cout % 16 == 0, unequal strides, non-square kernel
        new object[] { 2, 7, 11, 13, 32, 3, 4, 2, 3, 1, 2, 1, 1 },      // phase path: strides 2 x 3, kernel 3 x 4, odd sizes
        new object[] { 3, 70, 9, 9, 48, 3, 3, 3, 3, 2, 2, 1, 1 },        // phase path: stride 3 (9 phases), Cin not a tile multiple
    };

    /// <summary>Every shape through the #1325 engine (true) and through #1323's kernels (false), its fallback.</summary>
    public static System.Collections.Generic.IEnumerable<object[]> ShapesBothPaths() =>
        Shapes().SelectMany(s => new[] { s.Concat(new object[] { true }).ToArray(), s.Concat(new object[] { false }).ToArray() });

    [SkippableTheory]
    [MemberData(nameof(ShapesBothPaths))]
    public void WeightGradient_MatchesFloat64ReferenceAndIsBitReproducible(
        int batch, int inC, int h, int w, int outC, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool engine)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        backend.UseConvEngineBackward = engine;
        int outH = (h + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int outW = (w + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        Assert.True(outH > 0 && outW > 0, "invalid test shape");

        var rng = new Random(1313 + batch * 7 + inC * 3 + outC);
        var x = new float[batch * inC * h * w];
        var dy = new float[batch * outC * outH * outW];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < dy.Length; i++) dy[i] = (float)(rng.NextDouble() * 2 - 1);
        int p = inC * kH * kW;

        using var dX = backend.AllocateBuffer(x);
        using var dDy = backend.AllocateBuffer(dy);
        using var dW1 = backend.AllocateBuffer(outC * p);
        using var dW2 = backend.AllocateBuffer(outC * p);
        backend.Conv2DBackwardKernel(dX, dDy, dW1, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        backend.Conv2DBackwardKernel(dX, dDy, dW2, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(dW1), again = backend.DownloadBuffer(dW2);

        double maxRel = 0;
        for (int co = 0; co < outC; co++)
            for (int ci = 0; ci < inC; ci++)
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++)
                    {
                        double sum = 0, magnitude = 0;
                        for (int b = 0; b < batch; b++)
                            for (int oh = 0; oh < outH; oh++)
                            {
                                int ih = oh * sH - pH + kh * dH;
                                if (ih < 0 || ih >= h) continue;
                                for (int ow = 0; ow < outW; ow++)
                                {
                                    int iw = ow * sW - pW + kw * dW;
                                    if (iw < 0 || iw >= w) continue;
                                    double term = (double)dy[((b * outC + co) * outH + oh) * outW + ow]
                                                  * x[((b * inC + ci) * h + ih) * w + iw];
                                    sum += term;
                                    magnitude += Math.Abs(term);
                                }
                            }
                        int idx = ((co * inC + ci) * kH + kh) * kW + kw;
                        // fp32 accumulation over K terms: error bounded relative to the sum of magnitudes.
                        double error = Math.Abs(got[idx] - sum);
                        double bound = 1e-5 * Math.Max(1.0, magnitude);
                        maxRel = Math.Max(maxRel, error / Math.Max(1.0, magnitude));
                        Assert.True(error <= bound, $"dW[{co},{ci},{kh},{kw}] = {got[idx]} vs {sum} (|terms| {magnitude})");
                        Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(got[idx]), 0) ==
                                    BitConverter.ToInt32(BitConverter.GetBytes(again[idx]), 0),
                            $"dW[{co},{ci},{kh},{kw}] differs between two identical runs: {got[idx]:R} vs {again[idx]:R}");
                    }
        _out.WriteLine($"max error / sum|terms| = {maxRel:E2}");
    }

    [SkippableTheory]
    [MemberData(nameof(ShapesBothPaths))]
    public void InputGradient_MatchesFloat64ReferenceAndIsBitReproducible(
        int batch, int inC, int h, int w, int outC, int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool engine)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        backend.UseConvEngineBackward = engine;
        int outH = (h + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int outW = (w + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        Assert.True(outH > 0 && outW > 0, "invalid test shape");

        var rng = new Random(7919 + batch * 7 + inC * 3 + outC);
        var weights = new float[outC * inC * kH * kW];
        var dy = new float[batch * outC * outH * outW];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < dy.Length; i++) dy[i] = (float)(rng.NextDouble() * 2 - 1);

        // Reference by scattering each dY element through the kernel, the forward conv's own index map.
        var sum = new double[batch * inC * h * w];
        var magnitude = new double[sum.Length];
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outC; co++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double g = dy[((b * outC + co) * outH + oh) * outW + ow];
                        for (int ci = 0; ci < inC; ci++)
                            for (int kh = 0; kh < kH; kh++)
                            {
                                int ih = oh * sH - pH + kh * dH;
                                if (ih < 0 || ih >= h) continue;
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int iw = ow * sW - pW + kw * dW;
                                    if (iw < 0 || iw >= w) continue;
                                    double term = g * weights[((co * inC + ci) * kH + kh) * kW + kw];
                                    int idx = ((b * inC + ci) * h + ih) * w + iw;
                                    sum[idx] += term;
                                    magnitude[idx] += Math.Abs(term);
                                }
                            }
                    }

        using var dDy = backend.AllocateBuffer(dy);
        using var dWt = backend.AllocateBuffer(weights);
        // Pre-filled with NaN: every element, including ones no output position reaches, must be written.
        var poison = new float[sum.Length];
        for (int i = 0; i < poison.Length; i++) poison[i] = float.NaN;
        using var dX1 = backend.AllocateBuffer(poison);
        using var dX2 = backend.AllocateBuffer(poison);
        backend.Conv2DBackwardInput(dDy, dWt, dX1, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        backend.Conv2DBackwardInput(dDy, dWt, dX2, batch, inC, h, w, outC, outH, outW, kH, kW, sH, sW, pH, pW, dH, dW);
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(dX1), again = backend.DownloadBuffer(dX2);

        double maxRel = 0;
        for (int i = 0; i < sum.Length; i++)
        {
            double error = Math.Abs(got[i] - sum[i]);
            maxRel = Math.Max(maxRel, error / Math.Max(1.0, magnitude[i]));
            Assert.True(error <= 1e-5 * Math.Max(1.0, magnitude[i]), $"dX[{i}] = {got[i]} vs {sum[i]} (|terms| {magnitude[i]})");
            Assert.True(BitConverter.ToInt32(BitConverter.GetBytes(got[i]), 0) ==
                        BitConverter.ToInt32(BitConverter.GetBytes(again[i]), 0),
                $"dX[{i}] differs between two identical runs: {got[i]:R} vs {again[i]:R}");
        }
        _out.WriteLine($"max error / sum|terms| = {maxRel:E2}");
    }

    /// <summary>
    /// A/B of the implicit-GEMM backward against the im2col / col2im + SGEMM path on every ResNet-18 conv (the
    /// parity/networks/resnet18.json network at batch 8): both paths interleaved round by round in one process, each
    /// sample the mean of a burst of back-to-back calls between two synchronizes. Robust to a concurrent GPU load in
    /// the way a min is: the per-layer min and median and their step totals (weighted by how often each shape occurs)
    /// are reported. Gated on AIDOTNET_PERF_RUNNER=1.
    /// </summary>
    [SkippableFact]
    public void Bench_ResNet18ConvBackward_ImplicitVsIm2col()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1",
            "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        if (int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_BENCH_DGRAD_BPS"), out int bps)) backend.ConvDgradBlocksPerSm = bps;
        _out.WriteLine($"input-gradient split target: {backend.ConvDgradBlocksPerSm} blocks per SM");
        const int batch = 8, rounds = 41, burst = 10;
        // count, Cin, H (= W), Cout, k, stride, pad; the stem's input gradient is not needed in training.
        var layers = new (string Name, int Count, int InC, int Hw, int OutC, int K, int S, int P, bool NeedsDx)[]
        {
            ("stem 3-64 3x3 @32", 1, 3, 32, 64, 3, 1, 1, false),
            ("64-64 3x3 @32", 4, 64, 32, 64, 3, 1, 1, true),
            ("64-128 3x3 s2 @32", 1, 64, 32, 128, 3, 2, 1, true),
            ("64-128 1x1 s2 @32", 1, 64, 32, 128, 1, 2, 0, true),
            ("128-128 3x3 @16", 3, 128, 16, 128, 3, 1, 1, true),
            ("128-256 3x3 s2 @16", 1, 128, 16, 256, 3, 2, 1, true),
            ("128-256 1x1 s2 @16", 1, 128, 16, 256, 1, 2, 0, true),
            ("256-256 3x3 @8", 3, 256, 8, 256, 3, 1, 1, true),
            ("256-512 3x3 s2 @8", 1, 256, 8, 512, 3, 2, 1, true),
            ("256-512 1x1 s2 @8", 1, 256, 8, 512, 1, 2, 0, true),
            ("512-512 3x3 @4", 3, 512, 4, 512, 3, 1, 1, true),
        };
        var rng = new Random(1313);
        float[] Fill(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        double Median(List<double> v) { var o = v.OrderBy(x => x).ToList(); return o[o.Count / 2]; }

        // Arms: dW via the #1325 engine / #1323 implicit / im2col, then dX via the same three.
        string[] arms = { "dW engine", "dW #1323", "dW im2col", "dX engine", "dX #1323", "dX col2im" };
        double[] totMin = new double[6], totMed = new double[6];
        _out.WriteLine("layer                    x  | " + string.Join(" | ", arms.Select(a => $"{a,19}")) + "  (us per call, min / median)");
        foreach (var l in layers)
        {
            int outHw = (l.Hw + 2 * l.P - l.K) / l.S + 1;
            using var x = backend.AllocateBuffer(Fill(batch * l.InC * l.Hw * l.Hw));
            using var dy = backend.AllocateBuffer(Fill(batch * l.OutC * outHw * outHw));
            using var w = backend.AllocateBuffer(Fill(l.OutC * l.InC * l.K * l.K));
            using var dw = backend.AllocateBuffer(l.OutC * l.InC * l.K * l.K);
            using var dx = backend.AllocateBuffer(batch * l.InC * l.Hw * l.Hw);
            var samples = arms.Select(_ => new List<double>()).ToArray();
            for (int r = 0; r < rounds; r++)
            {
                for (int arm = 0; arm < arms.Length; arm++)
                {
                    int path = arm % 3;
                    bool input = arm >= 3;
                    if (input && !l.NeedsDx) continue;
                    backend.UseConvEngineBackward = path == 0;
                    backend.UseImplicitConvWgrad = backend.UseImplicitConvDgrad = path <= 1;
                    void Call()
                    {
                        if (input)
                            backend.Conv2DBackwardInput(dy, w, dx, batch, l.InC, l.Hw, l.Hw, l.OutC, outHw, outHw, l.K, l.K, l.S, l.S, l.P, l.P, 1, 1);
                        else
                            backend.Conv2DBackwardKernel(x, dy, dw, batch, l.InC, l.Hw, l.Hw, l.OutC, outHw, outHw, l.K, l.K, l.S, l.S, l.P, l.P, 1, 1);
                    }
                    Call();
                    backend.Synchronize();
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < burst; i++) Call();
                    backend.Synchronize();
                    if (r > 0) samples[arm].Add(sw.Elapsed.TotalMilliseconds * 1000.0 / burst);
                }
            }
            backend.UseConvEngineBackward = true;
            backend.UseImplicitConvWgrad = backend.UseImplicitConvDgrad = true;
            string Cell(int arm)
            {
                if (samples[arm].Count == 0) return $"{"-",19}";
                double mn = samples[arm].Min(), md = Median(samples[arm]);
                totMin[arm] += l.Count * mn; totMed[arm] += l.Count * md;
                return $"{mn,8:F1} / {md,8:F1}";
            }
            _out.WriteLine($"{l.Name,-22} {l.Count,2}  | " + string.Join(" | ", Enumerable.Range(0, arms.Length).Select(Cell)));
        }
        _out.WriteLine("step total (weighted)       | " + string.Join(" | ", Enumerable.Range(0, arms.Length).Select(a => $"{totMin[a],8:F1} / {totMed[a],8:F1}")));
        _out.WriteLine($"conv backward per step (min): engine {(totMin[0] + totMin[3]) / 1000:F3} ms, #1323 {(totMin[1] + totMin[4]) / 1000:F3} ms, " +
                       $"im2col {(totMin[2] + totMin[5]) / 1000:F3} ms");
    }
}
