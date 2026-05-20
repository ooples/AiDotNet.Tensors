using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Perf-Serial")]
public class ConvTranspose2DL2PerfTest
{
    private readonly ITestOutputHelper _output;

    public ConvTranspose2DL2PerfTest(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Issue #358 headline target: M=4096, N=16, K=512, transA=true FP64.
    /// Spec target: ≤ 1 ms on x64 AVX-512.
    /// This first-run benchmark uses a relaxed 5 ms gate to allow margin
    /// for the initial measurement; future PRs can tighten the gate.
    /// </summary>
    [Fact]
    public void L2Shape_FP64_BlasManagedGemm_BelowPerfTarget()
    {
        const int M = 4096;
        const int N = 16;
        const int K = 512;
        const int warmupIters = 5;
        const int measureIters = 100;

        // Allocate buffers. A is stored [K, M] for transA=true.
        var rng = new Random(42);
        double[] a = new double[K * M];  // [K, M] row-major
        double[] b = new double[K * N];  // [K, N] row-major
        double[] c = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        // Warmup — JIT + cache.
        for (int i = 0; i < warmupIters; i++)
        {
            BlasManagedLib.Gemm<double>(
                a, lda: M, transA: true,
                b, ldb: N, transB: false,
                c, ldc: N,
                M, N, K);
        }

        // Measure.
        double[] timesMs = new double[measureIters];
        var sw = new Stopwatch();
        for (int i = 0; i < measureIters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<double>(
                a, lda: M, transA: true,
                b, ldb: N, transB: false,
                c, ldc: N,
                M, N, K);
            sw.Stop();
            timesMs[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(timesMs);
        double median = timesMs[measureIters / 2];
        double p95 = timesMs[(int)(measureIters * 0.95)];
        double max = timesMs[measureIters - 1];

        // Detect available SIMD level.
        bool hasAvx512 = Avx512Fp64_8x16.IsSupported;
        bool hasAvx2 = Avx2Fp64_4x8.IsSupported;
        string archLabel = hasAvx512 ? "AVX-512" : (hasAvx2 ? "AVX2" : "scalar");

        _output.WriteLine($"#358 L2-shape FP64 benchmark — M={M} N={N} K={K} transA=true");
        _output.WriteLine($"  Arch:   {archLabel}");
        _output.WriteLine($"  Median: {median:F3} ms");
        _output.WriteLine($"  p95:    {p95:F3} ms");
        _output.WriteLine($"  Max:    {max:F3} ms");

        // Pre-K5 baselines from issue #358:
        //   - cblas_dgemm transA=true (OpenBLAS): 215 ms
        //   - cblas_dgemm transA=true (MKL):     559 ms
        //   - Naive 7-nested loop:                100 ms
        // Spec target (peak): 0.44 ms (75 GFLOPS)
        // L1 gate (relaxed first measurement):
        //   - AVX-512: 5 ms
        //   - AVX2:    25 ms
        //   - Scalar:  no gate, just report

        double gate = hasAvx512 ? 5.0 : (hasAvx2 ? 25.0 : double.MaxValue);
        if (gate < double.MaxValue)
        {
            _output.WriteLine($"  Gate:   {gate:F1} ms ({archLabel})");
            Assert.True(median <= gate,
                $"L2-shape median {median:F3} ms exceeds gate {gate:F1} ms for {archLabel}. " +
                $"Pre-K5 baseline was 215-559 ms (BLAS) or 100 ms (naive). If the perf has regressed " +
                $"significantly, investigate microkernel selection and parallelism.");
        }
        else
        {
            _output.WriteLine($"  Gate:   none (scalar-only host)");
        }
    }
}
