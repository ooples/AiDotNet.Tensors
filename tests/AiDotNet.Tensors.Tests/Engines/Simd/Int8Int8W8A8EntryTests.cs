using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Issue #465 Phases 1 &amp; 3 — the fused W8A8 entry point
/// <see cref="SimdGemm.SgemmA8W8RowScaledCachedB"/> (internal per-row activation
/// quant → int8×int8 GEMM → dequant) plus the W8A8 accuracy (SNR) characterization.
///
/// <para>W8A8 quantizes <b>both</b> operands, so its SNR is lower than the
/// weight-only path — the bars here are <b>measured</b> on this kernel, not the
/// weight-only <c>&gt; 10 dB</c> bar reused blindly. On AVX2 the <c>VPMADDUBSW</c>
/// int16 intermediate also saturates for full-range int8 weights, which is included
/// in the measured GEMM SNR below.</para>
/// </summary>
public class Int8Int8W8A8EntryTests
{
    private readonly ITestOutputHelper _output;
    public Int8Int8W8A8EntryTests(ITestOutputHelper output) { _output = output; }

    [Theory]
    [InlineData(1, 256, 32)]
    [InlineData(8, 512, 64)]
    [InlineData(16, 128, 48)]
    [InlineData(4, 100, 12)]   // k%32 tail
    public void SgemmA8W8_EndToEnd_SnrAboveMeasuredBar(int m, int k, int n)
    {
        var a = Gaussian(m * k, 11);
        var bf = Gaussian(n * k, 22);                       // fp32 weights [n, k]
        var (bI8, rowScales) = QuantizeWeightsPerRowFull(bf, n, k);

        var c = new float[m * n];
        SimdGemm.SgemmA8W8RowScaledCachedB(a, bI8, rowScales, c, m, k, n);

        // Reference: true fp32 matmul against the ORIGINAL fp32 weights (end-to-end
        // error = weight-quant + activation-quant + any AVX2 saturation).
        double sigSq = 0, errSq = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double refv = 0;
                for (int t = 0; t < k; t++) refv += (double)a[i * k + t] * bf[j * k + t];
                double e = c[i * n + j] - refv;
                sigSq += refv * refv; errSq += e * e;
                Assert.True(!float.IsNaN(c[i * n + j]) && !float.IsInfinity(c[i * n + j]));
            }
        double snr = errSq > 0 ? 10.0 * Math.Log10(sigSq / errSq) : double.PositiveInfinity;
        _output.WriteLine($"[{m},{k},{n}] W8A8 end-to-end SNR = {snr:F1} dB");
        // Measured bar: per-row symmetric W8A8 on gaussian data lands ~25-40 dB; the
        // AVX2 saturation occasionally trims it. 18 dB is a justified, comfortable
        // floor (≈6.5 effective bits) — well below the typical measured value, so it
        // catches a real wiring/correction regression without masking quant noise.
        Assert.True(snr >= 18.0, $"W8A8 end-to-end SNR {snr:F1} dB below the 18 dB bar (m={m},k={k},n={n}).");
    }

    [Fact]
    public void SgemmA8W8_BoundedWeights_MatchesDequantReference_Tightly()
    {
        // With ±63-bounded weights the AVX2 path does not saturate, so the only error
        // is int8 rounding — verifies the entry point's quant+cache+kernel+dequant
        // wiring is exact (tight tolerance), independent of saturation noise.
        const int m = 6, k = 256, n = 40;
        var a = Gaussian(m * k, 5);
        var bf = Gaussian(n * k, 6);
        var (bI8, rowScales) = QuantizeWeightsPerRowBounded(bf, n, k, 63);

        var c = new float[m * n];
        SimdGemm.SgemmA8W8RowScaledCachedB(a, bI8, rowScales, c, m, k, n);

        // Reference: dequantize BOTH operands exactly as the kernel sees them.
        var aU8 = new byte[m * k]; var actScale = new float[m];
        Int8Quantizer.QuantizeActivationsPerRowToUint8(a, m, k, aU8, actScale);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double exp = 0;
                for (int t = 0; t < k; t++)
                    exp += (aU8[i * k + t] - 128) * (double)actScale[i] * (bI8[j * k + t] * (double)rowScales[j]);
                Assert.True(Math.Abs(c[i * n + j] - exp) <= 1e-3 * (1 + Math.Abs(exp)),
                    $"c[{i},{j}]={c[i * n + j]:E5} vs {exp:E5}");
            }
    }

    [Fact]
    public void SgemmA8W8_CachedBRowSum_StableAcrossCalls()
    {
        // The per-weight-row sum is cached on the weight-array identity; repeated
        // calls with the same weights must give identical results (cache correctness).
        const int m = 4, k = 128, n = 24;
        var a = Gaussian(m * k, 7);
        var bf = Gaussian(n * k, 8);
        var (bI8, rowScales) = QuantizeWeightsPerRowFull(bf, n, k);

        var c1 = new float[m * n];
        var c2 = new float[m * n];
        SimdGemm.SgemmA8W8RowScaledCachedB(a, bI8, rowScales, c1, m, k, n);
        SimdGemm.SgemmA8W8RowScaledCachedB(a, bI8, rowScales, c2, m, k, n);  // cache hit
        Assert.Equal(c1, c2);
    }

    [Theory]
    [InlineData(256)]
    [InlineData(2048)]
    public void ActivationQuant_PerRow_RoundTripSnr_IsHealthy(int k)
    {
        const int m = 8;
        var a = Gaussian(m * k, 13);
        var u8 = new byte[m * k]; var scale = new float[m];
        Int8Quantizer.QuantizeActivationsPerRowToUint8(a, m, k, u8, scale);

        double sigSq = 0, errSq = 0;
        for (int i = 0; i < m; i++)
            for (int t = 0; t < k; t++)
            {
                double deq = (u8[i * k + t] - 128) * (double)scale[i];   // dequantized
                double e = deq - a[i * k + t];
                sigSq += (double)a[i * k + t] * a[i * k + t]; errSq += e * e;
                Assert.InRange((int)u8[i * k + t], 1, 255);             // valid uint8 (+128 shift)
            }
        double snr = errSq > 0 ? 10.0 * Math.Log10(sigSq / errSq) : double.PositiveInfinity;
        _output.WriteLine($"k={k}: per-row activation round-trip SNR = {snr:F1} dB");
        // Symmetric int8 round-trip on gaussian data ≈ 40 dB; 30 dB is a safe floor.
        Assert.True(snr >= 30.0, $"activation round-trip SNR {snr:F1} dB below 30 dB (k={k}).");
    }

    // ---- helpers ----

    private static (sbyte[] i8, float[] scale) QuantizeWeightsPerRowFull(float[] b, int n, int k)
        => QuantizeWeightsPerRowBounded(b, n, k, 127);

    private static (sbyte[] i8, float[] scale) QuantizeWeightsPerRowBounded(float[] b, int n, int k, int qMax)
    {
        var i8 = new sbyte[n * k];
        var scale = new float[n];
        for (int j = 0; j < n; j++)
        {
            float maxAbs = 0;
            for (int t = 0; t < k; t++) { float x = Math.Abs(b[j * k + t]); if (x > maxAbs) maxAbs = x; }
            float s = maxAbs == 0 ? 1f : maxAbs / qMax;
            scale[j] = s;
            float inv = 1f / s;
            for (int t = 0; t < k; t++)
            {
                int q = (int)Math.Round(b[j * k + t] * inv, MidpointRounding.ToEven);
                if (q < -qMax) q = -qMax; if (q > qMax) q = qMax;
                i8[j * k + t] = (sbyte)q;
            }
        }
        return (i8, scale);
    }

    private static float[] Gaussian(int len, int seed)
    {
        var r = new Random(seed);
        var arr = new float[len];
        for (int i = 0; i < len; i++)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = 1.0 - r.NextDouble();
            arr[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return arr;
    }
}
