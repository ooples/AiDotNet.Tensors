using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Stage 1 parity tests for SimdGemmDouble's new packed-tiled path (PackADouble
/// + PackBDouble + DgemmMicroKernel4x8 + DgemmMacroKernel + DgemmTiledSequential).
/// Each test compares against a scalar reference at FP64 ulp ≤ a small tolerance
/// (1e-9 absolute) to catch accumulator-order drift while still detecting any
/// real correctness bug.
///
/// Per the Stage 0 gate (DgemmShouldUsePackedTiled), the packed path fires when
/// work ≥ 1M FMAs, m ≥ Mr=4, n ≥ Nr=8, k ≥ 8. Tests below cover both inside-gate
/// shapes (use packed path) and edge shapes (full-Mr/Nr tiles + scalar remainder).
/// </summary>
public class SimdGemmDoublePackedTests
{
    private static double[] MakeRandom(int len, int seed)
    {
        var rng = new Random(seed);
        var arr = new double[len];
        for (int i = 0; i < len; i++) arr[i] = rng.NextDouble() - 0.5;
        return arr;
    }

    private static void ScalarReference(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c,
        int m, int k, int n)
    {
        c.Clear();
        for (int i = 0; i < m; i++)
            for (int kk = 0; kk < k; kk++)
            {
                double aik = a[i * k + kk];
                for (int j = 0; j < n; j++)
                    c[i * n + j] += aik * b[kk * n + j];
            }
    }

    private static void AssertClose(double[] expected, double[] actual, double tol = 1e-9)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            Assert.True(diff < tol,
                $"[{i}] expected={expected[i]:F12} actual={actual[i]:F12} diff={diff:E3}");
        }
    }

    [Theory]
    // Inside the packed-tiled gate (≥1M FMAs). Mix of full-tile + edge shapes.
    [InlineData(64, 64, 64)]     // exact full Mr/Nr tiles (16 Mr-rows × 8 Nr-cols)
    [InlineData(100, 128, 200)]  // mc%Mr != 0, nc%Nr != 0
    [InlineData(96, 256, 96)]    // square at the LargeMc boundary
    [InlineData(128, 64, 64)]    // tall-skinny
    [InlineData(64, 256, 256)]   // ResNet head shape
    [InlineData(4, 256, 8)]      // minimum Mr × minimum Nr × medium K
    public void DgemmTiled_MatchesScalarReference(int m, int k, int n)
    {
        var a = MakeRandom(m * k, 1);
        var b = MakeRandom(k * n, 2);
        var actual = new double[m * n];
        var expected = new double[m * n];

        SimdGemm.Dgemm(a, b, actual, m, k, n);
        ScalarReference(a, b, expected, m, k, n);

        AssertClose(expected, actual);
    }

    [Theory]
    // Below the packed-tiled gate — exercises the inline 64-block path
    // unchanged from pre-Stage-1. Confirms Stage 1 didn't regress small shapes.
    [InlineData(16, 16, 16)]   // 4K FMAs
    [InlineData(8, 32, 8)]     // 2K FMAs
    [InlineData(4, 8, 8)]      // minimum tile
    public void DgemmInline_SmallShapes_MatchesScalarReference(int m, int k, int n)
    {
        var a = MakeRandom(m * k, 11);
        var b = MakeRandom(k * n, 22);
        var actual = new double[m * n];
        var expected = new double[m * n];

        SimdGemm.Dgemm(a, b, actual, m, k, n);
        ScalarReference(a, b, expected, m, k, n);

        AssertClose(expected, actual);
    }

    [Fact]
    public void DgemmSequential_MatchesScalarReference()
    {
        // Cover the sequential entry too (same kernel, different parallel
        // dispatch toggle). Shape inside the packed gate.
        int m = 64, k = 128, n = 64;
        var a = MakeRandom(m * k, 7);
        var b = MakeRandom(k * n, 9);
        var actual = new double[m * n];
        var expected = new double[m * n];

        SimdGemm.DgemmSequential(a, b, actual, m, k, n);
        ScalarReference(a, b, expected, m, k, n);

        AssertClose(expected, actual);
    }
}
