using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Phase 2C drill-down on the remaining Conv2D gap after 564db58
/// fixed the plan specialization: the RN stage4 conv
/// <c>[1,512,7,7] 3×3→512</c> still takes 13.2 ms vs ORT's 0.72 ms
/// (18× per call). The per-call FLOPs = 231 M, so 13.2 ms = 17.5
/// GFLOPS, vs ORT's ~320 GFLOPS on the same shape — there's a ~18×
/// kernel-level headroom on this exact shape.
///
/// <para>This harness splits Conv2DIm2colGemm into its two phases
/// and times each on ResNet's four conv shapes:</para>
/// <list type="number">
///   <item><b>im2col</b> — build the <c>[K, N]</c> matrix
///   (<c>K = inC·kH·kW</c>, <c>N = oH·oW</c>) by strided reads.</item>
///   <item><b>SGEMM</b> — <c>[outC, K] × [K, N] = [outC, N]</c> via
///   SimdGemm.Sgemm.</item>
/// </list>
///
/// <para>The shape that bottlenecks is stage4 where N=49 is tiny
/// compared to K=4608 — this is a well-known SGEMM pathology (vector
/// lanes underutilised, cache-blocking doesn't amortise over N).
/// Goal: localise to im2col vs SGEMM, quantify per-phase GFLOPS,
/// and propose a specific fix.</para>
///
/// <para>Gated behind <c>AIDOTNET_RUN_PERF_HARNESS=1</c>.</para>
/// </summary>
public class Conv2DKernelSplitDiag
{
    private readonly ITestOutputHelper _output;
    public Conv2DKernelSplitDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 3;
    private const int Iters  = 10;

    [SkippableFact]
    public void SplitIm2colAndSgemm_ResNetShapes()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        _output.WriteLine($"Avx512Sgemm.CanUse = {Avx512Sgemm.CanUse}");
        _output.WriteLine($"CPU cores = {Environment.ProcessorCount}");
        _output.WriteLine("");

        // ResNet-50 hot conv shapes — same as Conv2DRootCauseDiag.
        (int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label)[] cases =
        {
            (64,  56, 56,  64, 3, 3, 1, 1, "stage1 3x3 [1,64,56,56]->[1,64,56,56]"),
            (128, 28, 28,  128, 3, 3, 1, 1, "stage2 3x3 [1,128,28,28]->[1,128,28,28]"),
            (256, 14, 14,  256, 3, 3, 1, 1, "stage3 3x3 [1,256,14,14]->[1,256,14,14]"),
            (512, 7,  7,  512, 3, 3, 1, 1, "stage4 3x3 [1,512,7,7]->[1,512,7,7]"),
        };

        foreach (var cs in cases)
        {
            int outH = (cs.h + 2 * cs.pad - cs.kH) / cs.stride + 1;
            int outW = (cs.w + 2 * cs.pad - cs.kW) / cs.stride + 1;
            int M = cs.co;                    // output channels
            int K = cs.c * cs.kH * cs.kW;     // kernel length
            int N = outH * outW;              // patches per image

            double sgemmFlops = 2.0 * M * K * N;
            // im2col does K × N reads + K × N writes = 2 K N memory ops.
            double im2colBytes = (double)K * N * sizeof(float);  // just the write side (reads are input which is smaller)

            _output.WriteLine($"=== {cs.label}  M={M} K={K} N={N}  (SGEMM FLOPs={sgemmFlops/1e6:F1}M, im2col buffer={im2colBytes/1024:F0} KB) ===");

            double tIm2col = TimeIm2col(cs.c, cs.h, cs.w, cs.kH, cs.kW, cs.stride, cs.pad, outH, outW);
            double tSgemm  = TimeSgemm(M, K, N);

            double totalKernel = tIm2col + tSgemm;
            _output.WriteLine($"  im2col only:                   {tIm2col * 1000:F1} µs  ({im2colBytes / tIm2col / 1e6:F1} MB/s write)");
            _output.WriteLine($"  SGEMM [M={M},K={K},N={N}]:      {tSgemm  * 1000:F1} µs  ({sgemmFlops / tSgemm / 1e6:F1} GFLOP/s)");
            _output.WriteLine($"  sum:                            {totalKernel * 1000:F1} µs");

            // Permutation tests — same 231M FLOPs, different (M,K,N) — to
            // characterise SimdGemm's sensitivity to shape. If one of these
            // permutations is much faster, we can transpose operands offline
            // to hit it.
            double tSwapMN = TimeSgemm(N, K, M);   // swap M ↔ N
            double tSwapMK = TimeSgemm(K, M, N);   // swap M ↔ K
            double tSwapKN = TimeSgemm(M, N, K);   // swap K ↔ N
            _output.WriteLine($"  SGEMM [M={N},K={K},N={M}] (swap M↔N): {tSwapMN * 1000:F1} µs  ({sgemmFlops / tSwapMN / 1e6:F1} GFLOP/s)");
            _output.WriteLine($"  SGEMM [M={K},K={M},N={N}] (swap M↔K): {tSwapMK * 1000:F1} µs  ({sgemmFlops / tSwapMK / 1e6:F1} GFLOP/s)");
            _output.WriteLine($"  SGEMM [M={M},K={N},N={K}] (swap K↔N): {tSwapKN * 1000:F1} µs  ({sgemmFlops / tSwapKN / 1e6:F1} GFLOP/s)");

            // Tests the "transposed-operand via transA/transB flag" path —
            // no explicit transpose buffer. Computes C^T[N,M] = col^T×kernel^T
            // by passing transA=true on col (stored [K,N]) and transB=true on
            // kernel (stored [M,K]). If SimdGemm handles transpose via cheap
            // in-place packing (not a full copy), this is the fast path we'd
            // route ResNet stage4 through.
            double tTransposedCall = TimeSgemmTransposed(M, K, N);
            _output.WriteLine($"  SGEMM transA=transB=true (C^T=col^T×kernel^T): {tTransposedCall * 1000:F1} µs  ({sgemmFlops / tTransposedCall / 1e6:F1} GFLOP/s)");

            // Transpose-result cost — what it would add to the overall
            // conv timing if we went via C^T and then flipped.
            double tTransposeOutput = TimeTranspose(N, M);
            _output.WriteLine($"  Transpose C^T [{N},{M}] → C [{M},{N}]:        {tTransposeOutput * 1000:F1} µs");
            double tKernelTranspose = TimeTranspose(M, K);
            _output.WriteLine($"  Kernel transpose [{M},{K}] → [{K},{M}]:       {tKernelTranspose * 1000:F1} µs");
            double tColTranspose = TimeTranspose(K, N);
            _output.WriteLine($"  Col transpose [{K},{N}] → [{N},{K}]:          {tColTranspose * 1000:F1} µs");

            double tWinograd = tTransposedCall + tTransposeOutput;
            _output.WriteLine($"  Projected hot path (transposed SGEMM + C^T→C): {tWinograd * 1000:F1} µs");
            _output.WriteLine("");
        }
    }

    // ─── timed paths ────────────────────────────────────────────────────────

    private static double TimeIm2col(int inC, int H, int W, int kH, int kW, int stride, int pad, int oH, int oW)
    {
        int K = inC * kH * kW;
        int N = oH * oW;
        var input = Rand(0xDD01, inC * H * W);
        var col = new float[K * N];

        for (int i = 0; i < Warmup; i++) Im2col(input, col, inC, H, W, kH, kW, stride, pad, oH, oW);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) Im2col(input, col, inC, H, W, kH, kW, stride, pad, oH, oW);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeSgemm(int M, int K, int N)
    {
        var a = Rand(0xDD02, M * K);
        var b = Rand(0xDD03, K * N);
        var c = new float[M * N];

        for (int i = 0; i < Warmup; i++) SimdGemm.Sgemm(a, b, c, M, K, N);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) SimdGemm.Sgemm(a, b, c, M, K, N);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    /// <summary>
    /// Time the transposed-call path: col stored [K, N], kernel stored [M, K],
    /// compute C^T[N, M] = col^T × kernel^T via <c>Sgemm(col, ldb=N, transA=true,
    /// kernel, lda=K, transB=true, ...)</c>. If this is as fast as the swapped
    /// M↔N timing, we can use the transpose overload directly in Conv2D.
    /// </summary>
    private static double TimeSgemmTransposed(int M, int K, int N)
    {
        // col: originally [K, N] row-major. lda=N. transA=true makes op(col)=col^T [N, K].
        // kernel: originally [M, K] row-major. ldb=K. transB=true makes op(kernel)=kernel^T [K, M].
        // Result C^T [N, M].
        var col = Rand(0xDD04, K * N);
        var kernel = Rand(0xDD05, M * K);
        var cT = new float[N * M];

        for (int i = 0; i < Warmup; i++)
            SimdGemm.Sgemm(col, N, true, kernel, K, true, cT, N, K, M);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            SimdGemm.Sgemm(col, N, true, kernel, K, true, cT, N, K, M);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeTranspose(int rows, int cols)
    {
        var src = Rand(0xDD06, rows * cols);
        var dst = new float[rows * cols];

        for (int i = 0; i < Warmup; i++) Transpose(src, dst, rows, cols);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) Transpose(src, dst, rows, cols);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static void Transpose(float[] src, float[] dst, int rows, int cols)
    {
        // Blocked transpose mirror of the CpuEngine TransposeFloat helper —
        // test benchmarks the identical algorithm, so timings correspond to
        // the production hot-path cost.
        const int BLK = 32;
        for (int rb = 0; rb < rows; rb += BLK)
        {
            int rEnd = Math.Min(rb + BLK, rows);
            for (int cb = 0; cb < cols; cb += BLK)
            {
                int cEnd = Math.Min(cb + BLK, cols);
                for (int r = rb; r < rEnd; r++)
                {
                    int srcRowBase = r * cols;
                    for (int c = cb; c < cEnd; c++)
                        dst[c * rows + r] = src[srcRowBase + c];
                }
            }
        }
    }

    /// <summary>
    /// Reference im2col — mirrors the inner loops in CpuEngine.Conv2DIm2colGemm.
    /// Deliberately uses the same structure so we can measure apples-to-apples
    /// what the Conv2D call pays for this phase.
    /// </summary>
    private static void Im2col(
        float[] input, float[] col,
        int inC, int H, int W, int kH, int kW, int stride, int pad,
        int oH, int oW)
    {
        int N = oH * oW;
        for (int ic = 0; ic < inC; ic++)
        {
            for (int kh = 0; kh < kH; kh++)
            {
                for (int kw = 0; kw < kW; kw++)
                {
                    int rowBase = (ic * kH + kh) * kW + kw;
                    int rowOff = rowBase * N;
                    for (int oh = 0; oh < oH; oh++)
                    {
                        int ih = oh * stride + kh - pad;
                        if ((uint)ih >= (uint)H)
                        {
                            Array.Clear(col, rowOff + oh * oW, oW);
                            continue;
                        }
                        int inputRowBase = (ic * H + ih) * W;
                        int colRowStart = rowOff + oh * oW;
                        for (int ow = 0; ow < oW; ow++)
                        {
                            int iw = ow * stride + kw - pad;
                            col[colRowStart + ow] = (uint)iw < (uint)W ? input[inputRowBase + iw] : 0f;
                        }
                    }
                }
            }
        }
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
