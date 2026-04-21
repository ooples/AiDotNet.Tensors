using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Empirical Winograd F(2,3) threshold audit. The current
/// <see cref="WinogradHelper.ShouldUseWinograd"/> requires output ≥ 224
/// (based on "16 small GEMMs overhead vs 1 large GEMM"), which excludes
/// every ResNet conv shape (outputs 56, 28, 14, 7). This harness runs
/// both paths on the four ResNet 3×3 stride=1 shapes and reports which
/// is faster per shape — so we can tune the threshold (or scrap it) with
/// data instead of intuition.
///
/// <para>Gated behind <c>AIDOTNET_RUN_PERF_HARNESS=1</c>.</para>
/// </summary>
public class WinogradThresholdDiag
{
    private readonly ITestOutputHelper _output;
    public WinogradThresholdDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 3;
    private const int Iters  = 10;

    [SkippableFact]
    public void WinogradVsIm2colAtResNetShapes()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        // (inC, H, W, outC, padding, label)
        (int c, int h, int w, int co, int pad, string label)[] cases =
        {
            (64,  56, 56,  64, 1, "stage1 3x3 [1,64,56,56]  output 56×56"),
            (128, 28, 28, 128, 1, "stage2 3x3 [1,128,28,28] output 28×28"),
            (256, 14, 14, 256, 1, "stage3 3x3 [1,256,14,14] output 14×14"),
            // stage4 has output 7×7 — odd, Winograd F(2,3) needs even output
            // (2x2 output tiles), so padding is required for it to apply.
            // Skip stage4 for this harness; it's out of F(2,3)'s clean
            // applicability band.
        };

        foreach (var cs in cases)
        {
            int outH = cs.h + 2 * cs.pad - 2; // kernelH - 1 = 2 for 3x3
            int outW = cs.w + 2 * cs.pad - 2;
            long flops = 2L * cs.co * cs.c * 9 * outH * outW; // 3×3 = 9

            _output.WriteLine($"=== {cs.label}  output {outH}×{outW}  (naive FLOPs={flops/1e6:F1} M) ===");

            double tIm2col = TimeConv2DIm2colGemm(cs.c, cs.h, cs.w, cs.co, cs.pad, outH, outW);
            _output.WriteLine($"  im2col + SGEMM:  {tIm2col * 1000:F1} µs  ({flops / tIm2col / 1e9:F1} GFLOP/s)");

            double tWino = TimeWinograd(cs.c, cs.h, cs.w, cs.co, cs.pad, outH, outW);
            _output.WriteLine($"  Winograd F(2,3): {tWino * 1000:F1} µs  ({flops / tWino / 1e9:F1} GFLOP/s)");
            _output.WriteLine($"  Winograd / im2col ratio: {tWino / tIm2col:F2}x  ({(tWino < tIm2col ? "WINOGRAD WINS" : "im2col wins")})");
            _output.WriteLine("");
        }
    }

    // ─── timed paths ────────────────────────────────────────────────────────

    private static double TimeConv2DIm2colGemm(int inC, int H, int W, int outC, int pad, int oH, int oW)
    {
        int K = inC * 9;          // 3×3 kernel
        int N = oH * oW;
        var input = Rand(0xAA01, inC * H * W);
        var kernel = Rand(0xAA02, outC * inC * 9);
        var col = new float[K * N];
        var output = new float[outC * N];

        // Mirrors Conv2DIm2colGemm's ProcessImage: im2col then SGEMM.
        // No Parallel.For wrapper — we're measuring the single-image cost.
        for (int i = 0; i < Warmup; i++) RunOnce();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) RunOnce();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;

        void RunOnce()
        {
            // im2col
            for (int ic = 0; ic < inC; ic++)
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                    {
                        int rowBase = (ic * 3 + kh) * 3 + kw;
                        int rowOff = rowBase * N;
                        for (int oh = 0; oh < oH; oh++)
                        {
                            int ih = oh + kh - pad;
                            if ((uint)ih >= (uint)H) { Array.Clear(col, rowOff + oh * oW, oW); continue; }
                            int inputRowBase = (ic * H + ih) * W;
                            int colRowStart = rowOff + oh * oW;
                            for (int ow = 0; ow < oW; ow++)
                            {
                                int iw = ow + kw - pad;
                                col[colRowStart + ow] = (uint)iw < (uint)W ? input[inputRowBase + iw] : 0f;
                            }
                        }
                    }

            // SGEMM
            SimdGemm.Sgemm(kernel.AsSpan(0, outC * K), col.AsSpan(0, K * N), output.AsSpan(0, outC * N), outC, K, N);
        }
    }

    private static double TimeWinograd(int inC, int H, int W, int outC, int pad, int oH, int oW)
    {
        var input = Rand(0xBB01, inC * H * W);
        var kernel = Rand(0xBB02, outC * inC * 9);
        var output = new float[outC * oH * oW];

        for (int i = 0; i < Warmup; i++)
            WinogradHelper.Conv2DWinograd(input, kernel, output, 1, inC, H, W, outC, pad, pad);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            WinogradHelper.Conv2DWinograd(input, kernel, output, 1, inC, H, W, outC, pad, pad);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
