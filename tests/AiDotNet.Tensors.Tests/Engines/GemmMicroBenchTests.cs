using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Isolated single-GEMM A/B benchmark for the AIsEval MLP shapes (#475). The
/// MlpForward gap to torch is dominated by the first layer GEMM
/// [128×784]@[784×512]; this harness measures the managed <see cref="SimdGemm"/>
/// kernel (and native OpenBLAS, when available) in isolation — best-of-N + the
/// achieved GFLOP/s — so microkernel changes can be A/B'd without the full-MLP
/// noise (activations, 3 chained layers, allocation). Env-gated.
/// </summary>
public class GemmMicroBenchTests
{
    private readonly ITestOutputHelper _output;
    public GemmMicroBenchTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Sgemm_MlpShapes_AbBench()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        // (m, k, n): the three MLP layer GEMMs at bs=128, plus the dominant one
        // isolated. m = batch rows, k = in-features, n = out-features.
        (int m, int k, int n)[] shapes =
        {
            (128, 784, 512),   // layer 0 — ~85% of MLP FLOPs
            (128, 512, 128),   // layer 1
            (128, 128, 10),    // layer 2 (classifier head)
        };

        _output.WriteLine($"native OpenBLAS available: HasRawSgemm={BlasProvider.HasRawSgemm}, IsMklVerified={SafeMkl()}, cores={Environment.ProcessorCount}");

        // Scaling curve for layer-0 — distinguishes bandwidth-bound (early plateau)
        // from framework-overhead (sub-linear from the start). Fresh B per thread
        // count because the pack cache's col-sub count is thread-dependent.
        {
            int m = 128, k = 784, n = 512;
            double flops = 2.0 * m * k * n;
            int saved = CpuParallelSettings.MaxDegreeOfParallelism;
            foreach (int t in new[] { 1, 2, 4, 8, 16 })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = t;
                var a = MakeRandom(m * k);
                var b = MakeRandom(k * n);   // fresh identity → rebuild pack at this thread count
                var c = new float[m * n];
                double best = Bench(() => SimdGemm.SgemmWithCachedB(a, b, c, m, k, n));
                _output.WriteLine($"  scaling t={t,2}: {best:F4} ms ({flops / (best * 1e6):F0} GF)");
            }
            CpuParallelSettings.MaxDegreeOfParallelism = saved;
        }

        foreach (var (m, k, n) in shapes)
        {
            var a = MakeRandom(m * k);
            var b = MakeRandom(k * n);
            var c = new float[m * n];
            double flops = 2.0 * m * k * n;

            // ---- Managed SgemmWithCachedB (the path FusedLinear actually uses) ----
            double bestCached = Bench(() => SimdGemm.SgemmWithCachedB(a, b, c, m, k, n));
            double gCached = flops / (bestCached * 1e6);

            // ---- Single-thread managed (decompose microkernel vs parallel scaling) ----
            bool savedParallel = SimdGemm.UseParallelGemm;
            SimdGemm.UseParallelGemm = false;
            double bestSerial = Bench(() => SimdGemm.SgemmWithCachedB(a, b, c, m, k, n));
            SimdGemm.UseParallelGemm = savedParallel;
            double gSerial = flops / (bestSerial * 1e6);
            double scaling = bestSerial / bestCached;
            _output.WriteLine(
                $"[{m}x{k}]@[{k}x{n}]  serial {gSerial:F0} GF  parallel {gCached:F0} GF  scaling {scaling:F1}x");

            // ---- Managed Sgemm lda-overload (no-trans, parallel direct path) ----
            double bestLda = Bench(() => SimdGemm.Sgemm(a, k, false, b, n, false, c, m, k, n));
            double gLda = flops / (bestLda * 1e6);

            // ---- Native (MKL/OpenBLAS SgemmRaw), if present ----
            string nativeStr = "n/a";
            if (BlasProvider.HasRawSgemm)
            {
                double bestNative = BenchNative(a, b, c, m, k, n);
                nativeStr = $"{bestNative:F4} ms ({flops / (bestNative * 1e6):F0} GFLOP/s)";
            }

            _output.WriteLine(
                $"[{m}x{k}]@[{k}x{n}]  cachedB {bestCached:F4} ms ({gCached:F0} GF)  |  Sgemm-lda {bestLda:F4} ms ({gLda:F0} GF)  |  native {nativeStr}");
        }
    }

    private static double Bench(Action f)
    {
        for (int w = 0; w < 20; w++) f();
        double best = double.MaxValue;
        for (int r = 0; r < 50; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            f();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static unsafe double BenchNative(float[] a, float[] b, float[] c, int m, int k, int n)
    {
        double best = double.MaxValue;
        fixed (float* pa = a, pb = b, pc = c)
        {
            for (int w = 0; w < 20; w++) BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n);
            for (int r = 0; r < 50; r++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n);
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
        }
        return best;
    }

    private static string SafeMkl()
    {
        try { return BlasProvider.IsMklVerified.ToString(); }
        catch { return "?"; }
    }

    private static float[] MakeRandom(int n)
    {
        var rng = new Random(2026);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        return a;
    }
}
