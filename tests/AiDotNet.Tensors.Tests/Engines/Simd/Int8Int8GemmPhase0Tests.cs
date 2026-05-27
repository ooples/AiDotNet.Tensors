using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Issue #465 Phase 0 (de-risk) — correctness of the AVX2 int8×int8 micro-kernel
/// (<see cref="SimdGemm.MatMulInt8Int8Avx2"/>) and a viability microbenchmark vs
/// fp32 <c>Sgemm</c>. The W8A8 program only proceeds to the full fused entry point
/// if int8 actually wins on AVX2-only hardware (see the env-gated bench below).
/// </summary>
public class Int8Int8GemmPhase0Tests
{
    private readonly ITestOutputHelper _output;
    public Int8Int8GemmPhase0Tests(ITestOutputHelper output) { _output = output; }

    [Theory]
    [InlineData(1, 64, 8)]
    [InlineData(4, 128, 16)]
    [InlineData(16, 512, 32)]
    [InlineData(8, 100, 12)]   // k not a multiple of 32 → exercises the scalar tail
    [InlineData(3, 37, 5)]     // tiny / odd k
    public void MatMulInt8Int8_Avx2_MatchesScalarAndDequantReference(int m, int k, int n)
    {
        var (aU8, aScale) = QuantizeActivationsPerRow(RandomFloat(m * k, 1, m, k), m, k);
        var (bI8, bRowSum, bScale) = QuantizeWeightsPerRow(RandomFloat(n * k, 2, n, k), n, k);

        var cAvx = new float[m * n];
        var cScalar = new float[m * n];
        SimdGemm.MatMulInt8Int8Avx2(aU8, bI8, bRowSum, aScale, bScale, cAvx, m, k, n);
        SimdGemm.MatMulInt8Int8Scalar(aU8, bI8, bRowSum, aScale, bScale, cScalar, m, k, n);

        // (1) AVX2 path must equal the scalar path exactly (same int32 → same float).
        for (int i = 0; i < m * n; i++)
            Assert.True(cAvx[i] == cScalar[i], $"AVX2 vs scalar mismatch at {i}: {cAvx[i]} != {cScalar[i]}");

        // (2) Both must equal the dequantize-then-fp32-matmul reference (proves the
        //     +128 unsigned-shift correction is right), within int8 rounding noise.
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double expected = 0;
                for (int t = 0; t < k; t++)
                {
                    double aq = (aU8[i * k + t] - 128) * aScale[i];   // dequantized activation
                    double bq = bI8[j * k + t] * bScale[j];           // dequantized weight
                    expected += aq * bq;
                }
                double tol = 1e-3 * (1 + Math.Abs(expected));
                Assert.True(Math.Abs(cAvx[i * n + j] - expected) <= tol,
                    $"c[{i},{j}]={cAvx[i * n + j]:E5} vs dequant-ref {expected:E5} (m={m},k={k},n={n})");
            }
    }

    [Trait("Category", "Performance")]
    [Fact]
    public void Int8Int8_vs_Fp32_ViabilityBench()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        if (!SimdGemm.Int8Int8Avx2Available) { _output.WriteLine("AVX2 not available — skipping."); return; }

        // Fair per-core comparison: the int8 kernel is single-threaded, so pin fp32
        // Sgemm to a single thread too. The per-core instruction-throughput ratio is
        // the fundamental "does int8 win on AVX2" question; parallel scaling is
        // orthogonal (both paths parallelize over M/N similarly).
        int savedMdop = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = 1;
        try
        {
            _output.WriteLine($"AVX-VNNI: {IsAvxVnniSupported()} (false ⇒ AVX2-emulation path — the host this de-risk targets); single-threaded");
            _output.WriteLine("shape [m,k,n]          fp32 ms  fp32 GF/s   int8 ms  int8 GF/s   int8/fp32  verdict");

            // decode (memory-bound m≈1), small-compute, and compute-bound shapes.
            foreach (var (m, k, n) in new[] { (1, 2048, 2048), (16, 512, 512), (128, 2048, 2048) })
            {
                var af = RandomFloat(m * k, 1, m, k);
                var bfKN = RandomFloat(k * n, 2, k, n);   // fp32 B as [k, n] for Sgemm
                var (aU8, aScale) = QuantizeActivationsPerRow(af, m, k);
                var (bI8, bRowSum, bScale) = QuantizeWeightsPerRow(RandomFloat(n * k, 2, n, k), n, k);

                var cF = new float[m * n];
                var cI = new float[m * n];

                double Time(Action f)
                {
                    f(); f();                                   // warmup
                    double best = double.MaxValue;
                    for (int r = 0; r < 5; r++)
                    {
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        f();
                        sw.Stop();
                        best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                    }
                    return best;
                }

                double flops = 2.0 * m * k * n;
                double tF = Time(() => SimdGemm.Sgemm(af, bfKN, cF, m, k, n));
                double tI = Time(() => SimdGemm.MatMulInt8Int8Avx2(aU8, bI8, bRowSum, aScale, bScale, cI, m, k, n));
                double ratio = tF / tI;
                double gfF = flops / (tF * 1e6), gfI = flops / (tI * 1e6);
                string verdict = ratio >= 1.15 ? "int8 WINS" : (ratio >= 0.9 ? "~parity" : "int8 LOSES");
                _output.WriteLine($"[{m},{k},{n}]".PadRight(22) +
                    $"{tF,8:F2} {gfF,9:F1}  {tI,8:F2} {gfI,9:F1}   {ratio,6:F2}×   {verdict}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = savedMdop;
        }
    }

    private static bool IsAvxVnniSupported()
    {
#if NET8_0_OR_GREATER
        return System.Runtime.Intrinsics.X86.AvxVnni.IsSupported;
#else
        return false;
#endif
    }

    // ---- per-row symmetric int8 quant helpers (the W8A8 activation/weight scheme) ----

    // Activations: per-row symmetric int8, then +128 → uint8 for VPMADDUBSW.
    private static (byte[] u8, float[] scale) QuantizeActivationsPerRow(float[] a, int m, int k)
    {
        var u8 = new byte[m * k];
        var scale = new float[m];
        for (int i = 0; i < m; i++)
        {
            float maxAbs = 0;
            for (int t = 0; t < k; t++) { float x = Math.Abs(a[i * k + t]); if (x > maxAbs) maxAbs = x; }
            float s = maxAbs == 0 ? 1f : maxAbs / 127f;
            scale[i] = s;
            float inv = 1f / s;
            for (int t = 0; t < k; t++)
            {
                int q = (int)Math.Round(a[i * k + t] * inv);
                if (q < -127) q = -127; if (q > 127) q = 127;
                u8[i * k + t] = (byte)(q + 128);
            }
        }
        return (u8, scale);
    }

    // Weights: per-row symmetric int8 + per-row sum (for the −128·Σb correction).
    // NOTE: clamp the quant level to ±63 (≈6-bit) rather than ±127. The AVX2
    // VPMADDUBSW path sums two adjacent uint8×sbyte products into a SATURATING
    // int16; with u≤255 and |b|≤63 the pair-max is 2·255·63 = 32130 < 32767, so
    // no saturation — this isolates the kernel's logic (shift correction + dequant)
    // for the Phase-0 correctness check. Full-range int8 weights CAN saturate the
    // int16 intermediate (a known AVX2-emulation accuracy limit vs VNNI's VPDPBUSD);
    // quantifying that SNR loss is Phase 3 of #465, not this de-risk.
    private const int WeightQuantMax = 63;
    private static (sbyte[] i8, int[] rowSum, float[] scale) QuantizeWeightsPerRow(float[] b, int n, int k)
    {
        var i8 = new sbyte[n * k];
        var rowSum = new int[n];
        var scale = new float[n];
        for (int j = 0; j < n; j++)
        {
            float maxAbs = 0;
            for (int t = 0; t < k; t++) { float x = Math.Abs(b[j * k + t]); if (x > maxAbs) maxAbs = x; }
            float s = maxAbs == 0 ? 1f : maxAbs / WeightQuantMax;
            scale[j] = s;
            float inv = 1f / s;
            int sum = 0;
            for (int t = 0; t < k; t++)
            {
                int q = (int)Math.Round(b[j * k + t] * inv);
                if (q < -WeightQuantMax) q = -WeightQuantMax; if (q > WeightQuantMax) q = WeightQuantMax;
                i8[j * k + t] = (sbyte)q;
                sum += q;
            }
            rowSum[j] = sum;
        }
        return (i8, rowSum, scale);
    }

    private static float[] RandomFloat(int len, int seed, int a, int b)
    {
        var r = new Random(seed * 7919 + a * 31 + b);
        var arr = new float[len];
        for (int i = 0; i < len; i++) arr[i] = (float)(r.NextDouble() - 0.5) * 2f;
        return arr;
    }
}
