using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The two float shape classes BlasManaged hands to <see cref="SimdGemm"/> (a tall A^T over a short K, and a small output
/// over a long K) take that route for a DisableAutotune request as for an Auto one: the engine's linear-layer weight
/// gradients ask for DisableAutotune, and without the route the CNN's 3136 x 64 x 128 A^T weight gradient tail-split on
/// AVX-512 hosts and ran 15x slower. Routed, the product is SimdGemm's to the bit; a request naming a plan keeps it.
/// </summary>
public sealed class BlasManagedSimdGemmRouteTests
{
    private static float[] Data(int length, int seed)
    {
        var rng = new Random(seed);
        var d = new float[length];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return d;
    }

    // (transA, m, n, k): the CNN classifier's weight gradient (A^T, 3136 x 128 over K = 64) and its forward product
    // (64 x 128 over K = 3136).
    [Theory]
    [InlineData(true, 3136, 128, 64, PackingMode.DisableAutotune)]
    [InlineData(true, 3136, 128, 64, PackingMode.Auto)]
    [InlineData(false, 64, 128, 3136, PackingMode.DisableAutotune)]
    [InlineData(false, 64, 128, 3136, PackingMode.Auto)]
    public void RoutedShape_IsSimdGemmToTheBit(bool transA, int m, int n, int k, PackingMode mode)
    {
        int lda = transA ? m : k;
        var a = Data(m * k, 1);
        var b = Data(k * n, 2);
        var expected = new float[m * n];
        SimdGemm.Sgemm(a, lda, transA, b, n, false, expected, m, k, n);

        var c = new float[m * n];
        for (int i = 0; i < c.Length; i++) c[i] = float.NaN;   // BlasManaged overwrites C
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            a, lda, transA, b, n, false, c, n, m, n, k, new BlasOptions<float> { PackingMode = mode });

        for (int i = 0; i < c.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(c[i]),
                $"C[{i / n},{i % n}] = {c[i]:R}, SimdGemm {expected[i]:R}");
    }

    [Fact]
    public void NamedPlan_IsNotRerouted_AndStillRight()
    {
        const int m = 3136, n = 128, k = 64;
        var a = Data(m * k, 3);
        var b = Data(k * n, 4);
        var c = new float[m * n];
        AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            a, m, true, b, n, false, c, n, m, n, k, new BlasOptions<float> { PackingMode = PackingMode.ForceStreaming });
        for (int row = 0; row < m; row += 97)
            for (int col = 0; col < n; col += 13)
            {
                double s = 0;
                for (int p = 0; p < k; p++) s += (double)a[p * m + row] * b[p * n + col];
                Assert.True(Math.Abs(s - c[row * n + col]) <= 1e-4 * (1 + Math.Abs(s)), $"C[{row},{col}] = {c[row * n + col]}, expected {s}");
            }
    }
}
