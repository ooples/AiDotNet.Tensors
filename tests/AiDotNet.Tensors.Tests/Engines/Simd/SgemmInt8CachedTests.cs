using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Correctness for SimdGemm.SgemmWithInt8CachedB (Path D weight-only int8).
/// Verifies the int8 path matches the float path within the expected
/// quantization SNR band (35-40 dB on typical transformer-shape inputs).
/// </summary>
public class SgemmInt8CachedTests
{
    [Theory]
    [InlineData(64, 256, 256)]   // small matmul fast path
    [InlineData(256, 768, 768)]  // BERT QKV proj
    [InlineData(256, 768, 3072)] // BERT FFN up
    [InlineData(256, 3072, 768)] // BERT FFN down
    public void Int8MatchesFloat32WithinSnrBudget(int m, int k, int n)
    {
        var rng = new Random(0xD42 + m + k + n);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var cFloat = new float[m * n];
        SimdGemm.Sgemm(a, b, cFloat, m, k, n);

        var cInt8 = new float[m * n];
        SimdGemm.SgemmWithInt8CachedB(a, b, cInt8, m, k, n);

        // Compute SNR vs float reference. Per-tensor symmetric int8 of B
        // adds quantization noise of magnitude ~|B|/254. Output noise
        // scales with |A| · sqrt(K) · |B|/254 (random-walk accumulation).
        // For BERT shapes (uniform [-1,1]), the SNR comes out around 35-40 dB.
        double sumSqSig = 0, sumSqErr = 0;
        for (int i = 0; i < cFloat.Length; i++)
        {
            sumSqSig += (double)cFloat[i] * cFloat[i];
            double err = cInt8[i] - cFloat[i];
            sumSqErr += err * err;
        }
        double snrDb = sumSqErr > 0 ? 10.0 * Math.Log10(sumSqSig / sumSqErr) : 100.0;
        Assert.True(snrDb > 30.0, $"Int8 vs float SNR {snrDb:F1} dB below 30 dB (m={m},k={k},n={n})");
    }

    [Fact]
    public void Int8ResultsAreCachedAcrossCalls()
    {
        // Calling twice with the same B array should hit the cache on the
        // second call. We can't directly observe the cache, but if it
        // works, the second call's output should match the first exactly.
        int m = 256, k = 768, n = 768;
        var rng = new Random(0xCACE);
        var a1 = new float[m * k];
        var a2 = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a1.Length; i++) { a1[i] = (float)rng.NextDouble(); a2[i] = (float)rng.NextDouble(); }
        for (int i = 0; i < b.Length; i++) b[i] = (float)rng.NextDouble();

        var c1 = new float[m * n];
        var c2 = new float[m * n];
        SimdGemm.SgemmWithInt8CachedB(a1, b, c1, m, k, n);
        SimdGemm.SgemmWithInt8CachedB(a2, b, c2, m, k, n);

        // Re-run with a1 — if the cache is consistent, output matches c1 exactly.
        var c1Again = new float[m * n];
        SimdGemm.SgemmWithInt8CachedB(a1, b, c1Again, m, k, n);
        for (int i = 0; i < c1.Length; i++)
            Assert.Equal(c1[i], c1Again[i]);
    }
}
