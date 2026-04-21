using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// B7 benchmark deltas — reports the measured throughput of the current
/// dispatch (AVX-512 if the host has it, AVX2 fallback otherwise) on a
/// shape representative of BERT FFN (768 × 3072 × 768). Not an assertion;
/// the test emits the numbers into test output for operator review.
///
/// <para>On an AVX-512-capable host the reported GFLOPS should be ~1.5-2×
/// the AVX2 baseline. On an AVX2-only host the two paths are identical so
/// the delta is 0. Either way this test runs green — it prints, it
/// doesn't enforce.</para>
/// </summary>
public class Avx512BenchmarkDelta
{
    private readonly ITestOutputHelper _output;

    public Avx512BenchmarkDelta(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Sgemm_BertFfnShape_ReportsGFLOPS()
    {
        // BERT-base FFN: hidden=768 → 4×hidden=3072 → 768. The two matmuls
        // in one FFN layer are the dominant cost of a transformer block on
        // CPU, so this shape is the canonical "did AVX-512 help" signal.
        const int M = 64, K = 768, N = 3072;
        var a = Random(M * K, 0xBE);
        var b = Random(K * N, 0xEF);
        var c = new float[M * N];

        // Warm up JIT + cache.
        SimdGemm.Sgemm(a, b, c, M, K, N);
        SimdGemm.Sgemm(a, b, c, M, K, N);

        const int iters = 10;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            SimdGemm.Sgemm(a, b, c, M, K, N);
        sw.Stop();

        double secsPerIter = sw.Elapsed.TotalSeconds / iters;
        double flopsPerIter = 2.0 * M * K * N;
        double gflops = flopsPerIter / 1e9 / secsPerIter;
        bool avx512Available = Avx512Sgemm.CanUse;

        _output.WriteLine(
            $"SGEMM[{M}x{K}x{N}]  iters={iters}  t/iter={secsPerIter * 1000:F2}ms  {gflops:F1} GFLOPS  Avx512.CanUse={avx512Available}");

        // Sanity: non-zero GFLOPS and finite timing.
        Assert.True(gflops > 0.1, "SGEMM is effectively not running");
        Assert.True(secsPerIter < 60.0, "Single iteration took more than 60s");
    }

    [Fact]
    public void Sgemm_ResNet50BlockShape_ReportsGFLOPS()
    {
        // ResNet-50 bottleneck Conv lowered to GEMM via im2col:
        //   [outC=64, inC×kH×kW=64*9=576] × [576, oH*oW=56*56=3136]
        // is a typical mid-network conv. Reports whether AVX-512 Conv2D
        // delivers on CNN workloads too.
        const int M = 64, K = 576, N = 3136;
        var a = Random(M * K, 0x12);
        var b = Random(K * N, 0x34);
        var c = new float[M * N];

        SimdGemm.Sgemm(a, b, c, M, K, N);
        SimdGemm.Sgemm(a, b, c, M, K, N);

        const int iters = 10;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            SimdGemm.Sgemm(a, b, c, M, K, N);
        sw.Stop();

        double secsPerIter = sw.Elapsed.TotalSeconds / iters;
        double flopsPerIter = 2.0 * M * K * N;
        double gflops = flopsPerIter / 1e9 / secsPerIter;

        _output.WriteLine(
            $"SGEMM[{M}x{K}x{N}]  iters={iters}  t/iter={secsPerIter * 1000:F2}ms  {gflops:F1} GFLOPS  Avx512.CanUse={Avx512Sgemm.CanUse}");

        Assert.True(gflops > 0.1);
    }

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
