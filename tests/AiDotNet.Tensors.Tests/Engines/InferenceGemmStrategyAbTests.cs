using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Phase 1 A/B (plan: compiled-inference-engine). The inference FusedLinear path
/// prefers native BLAS (SgemmRaw/MKL) then applies bias+activation in a SEPARATE
/// O(MN) pass, and the Tier-1 path copies the BLAS output into a managed buffer.
/// Neither our persistent prepacked-B cache (SgemmWithCachedB) nor any epilogue
/// fusion reaches that path. The question this bench answers: for the small/medium
/// MLP inference shapes, does routing through our managed cached-B GEMM
/// (prepack reused across calls, no pinned→managed copy) + the bias/activation
/// pass BEAT native BLAS + separate pass + copy? The answer dictates whether
/// Phase 1 should route inference onto the managed (fusable) path or whether we
/// need oneDNN post-ops / an AVX-512 managed kernel first.
///
/// Env-gated (AIDOTNET_RUN_JIT_PERF=1); CI-safe no-op otherwise.
/// </summary>
public class InferenceGemmStrategyAbTests
{
    private readonly ITestOutputHelper _output;
    public InferenceGemmStrategyAbTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void InferenceGemm_NativeBlasPlusSeparatePass_vs_ManagedCachedBPlusEpilogue()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        _output.WriteLine($"HasRawSgemm={BlasProvider.HasRawSgemm}, IsMklVerified={BlasProvider.IsMklVerified}, cores={Environment.ProcessorCount}");

        // MLP layer shapes (m=batch rows, k=in-features, n=out-features) at the
        // benchmark batch sizes. Layer 0 dominates FLOPs; layers 1/2 are where
        // packing/dispatch overhead dominates compute.
        (int k, int n)[] layers = { (784, 512), (512, 128), (128, 10) };
        int[] batches = { 1, 8, 32, 128 };

        foreach (int m in batches)
        {
            foreach (var (k, n) in layers)
            {
                var a = MakeRandom(m * k);
                var b = MakeRandom(k * n);
                var bias = MakeRandom(n);
                var outNative = new float[m * n];
                var scratch = new float[m * n];
                var outManaged = new float[m * n];

                // Strategy A — native BLAS GEMM → copy → separate bias+ReLU pass
                // (mirrors the current FusedLinear Tier-1 inference path).
                double bestNative = Bench(() =>
                {
                    if (BlasProvider.HasRawSgemm)
                    {
                        unsafe
                        {
                            fixed (float* pa = a, pb = b, ps = scratch)
                                BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, ps, n);
                        }
                        Array.Copy(scratch, outNative, m * n);   // pinned→managed copy
                    }
                    else
                    {
                        SimdGemm.SgemmWithCachedB(a.AsSpan(0, m * k), b, outNative.AsSpan(0, m * n), m, k, n);
                    }
                    CpuFusedOperations.ApplyBiasActivationInPlace(outNative, bias, m, n, FusedActivationType.ReLU);
                });

                // Strategy B — managed cached-B GEMM (prepack reused) → bias+ReLU
                // pass directly on the output (no copy).
                double bestManaged = Bench(() =>
                {
                    SimdGemm.SgemmWithCachedB(a.AsSpan(0, m * k), b, outManaged.AsSpan(0, m * n), m, k, n);
                    CpuFusedOperations.ApplyBiasActivationInPlace(outManaged, bias, m, n, FusedActivationType.ReLU);
                });

                // Correctness cross-check (same math both strategies).
                double maxDiff = 0;
                for (int i = 0; i < m * n; i++) maxDiff = Math.Max(maxDiff, Math.Abs(outNative[i] - outManaged[i]));

                string verdict = bestManaged < bestNative ? "MANAGED wins" : "native wins";
                _output.WriteLine(
                    $"[m={m,4} {k}x{n}]  native+sep+copy {bestNative * 1000:F2}us  managed+epi {bestManaged * 1000:F2}us  " +
                    $"ratio {bestNative / bestManaged:F2}x  {verdict}  (maxDiff {maxDiff:E2})");
            }
        }
    }

    private static double Bench(Action f)
    {
        for (int w = 0; w < 30; w++) f();
        double best = double.MaxValue;
        for (int r = 0; r < 100; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            f();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static float[] MakeRandom(int n)
    {
        var rng = new Random(2026);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        return a;
    }
}
