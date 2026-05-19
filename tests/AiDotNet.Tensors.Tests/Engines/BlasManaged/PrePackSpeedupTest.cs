using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-E (#373) speedup gate — pre-packing a stable weight B once and reusing
/// the packed buffer across many GEMMs must beat re-packing B per call.
/// <para>
/// Shape: M=8, N=1024, K=1024 — small batch (typical of per-token transformer
/// inference) where pack-B (4 MB at this size) is a meaningful fraction of
/// total time. Repeated 200×.
/// </para>
/// <para>
/// Gate: pre-packed mean wall time MUST be at least 1.1× lower than the
/// no-pre-pack baseline. Fires only on Release builds (Debug timings are
/// too noisy); under Debug the test still asserts numerical correctness.
/// </para>
/// </summary>
public class PrePackSpeedupTest
{
    private readonly ITestOutputHelper _output;

    public PrePackSpeedupTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void PrePackedB_Reused_Across_Repeated_Gemms_Is_Faster()
    {
        const int M = 8, N = 1024, K = 1024;
        const int iterations = 200;
        const int warmup = 20;
        var rng = new Random(42);

        var a = new float[M * K];
        var b = new float[K * N];
        var cBaseline = new float[M * N];
        var cPrePack = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // ── Baseline: re-pack B every call ────────────────────────────────
        for (int w = 0; w < warmup; w++)
        {
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBaseline, N, M, N, K);
        }
        var swBaseline = Stopwatch.StartNew();
        for (int it = 0; it < iterations; it++)
        {
            BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cBaseline, N, M, N, K);
        }
        swBaseline.Stop();
        double baselineUs = (swBaseline.Elapsed.TotalMilliseconds * 1000.0) / iterations;

        // ── Pre-packed: one pack, many reuses ─────────────────────────────
        var handle = BlasManagedLib.PrePackB<float>(b, N, false, K, N);
        try
        {
            var opts = new BlasOptions<float> { PackedB = handle };
            for (int w = 0; w < warmup; w++)
            {
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cPrePack, N, M, N, K, opts);
            }
            var swPrePack = Stopwatch.StartNew();
            for (int it = 0; it < iterations; it++)
            {
                BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cPrePack, N, M, N, K, opts);
            }
            swPrePack.Stop();
            double prePackUs = (swPrePack.Elapsed.TotalMilliseconds * 1000.0) / iterations;

            // ── Correctness — both paths must agree numerically ─────────────
            double maxDelta = 0;
            for (int i = 0; i < cBaseline.Length; i++)
                maxDelta = Math.Max(maxDelta, Math.Abs((double)cBaseline[i] - cPrePack[i]));
            Assert.True(maxDelta < 1e-3,
                $"Pre-pack vs baseline drift {maxDelta:G6} exceeds 1e-3 — pre-pack output is incorrect");

            double speedup = baselineUs / Math.Max(prePackUs, 1e-9);
            _output.WriteLine($"M={M} N={N} K={K} iters={iterations}");
            _output.WriteLine($"  baseline (live pack): {baselineUs:F1} us/call");
            _output.WriteLine($"  pre-pack (reused):    {prePackUs:F1} us/call");
            _output.WriteLine($"  speedup:              {speedup:F2}x");

#if !DEBUG
            // Release-only gate. Debug timings are too noisy for ratio asserts.
            // Empirical: pre-pack typically delivers ~1.2-1.4x at this shape on
            // x64-amd-avx2-cpu16. 1.1x leaves headroom for noisy CI hosts while
            // still flagging real regressions (e.g., handle not being consumed —
            // pre-fix measurements showed 0.92x because handle.TileMc mismatched
            // the autotuner's pick and consume fell back to live pack).
            Assert.True(speedup >= 1.1,
                $"Pre-pack should be ≥1.1x faster than live-pack baseline; got {speedup:F2}x");
#endif
        }
        finally
        {
            handle.Dispose();
        }
    }
}
