// Issue #358 Task G6 — verifies that BlasProvider.IsDeterministicMode is
// readable and that SetDeterministicMode toggles it, and that enabling
// deterministic mode does not break BlasManaged.Gemm correctness.

using System;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

// Serialized with the other BlasManaged global-state mutators (ScalarKernelTests,
// PrePackSpeedupTest, StatsCounterTests, …). These tests toggle the process-wide
// BlasProvider.IsDeterministicMode flag, which switches the GEMM reduction/packing
// path. Without serialization this class runs in the parallel pool and can flip the
// flag mid-GEMM in a concurrent pre-pack / scalar-kernel test, drifting that test's
// packed output away from its live-pack baseline (the intermittent CI "drift"
// failures in PrePackedB_Output_BitMatches_LivePack and the cached-packed-buffer
// tests) — and SetDeterministicMode_Toggles can itself observe a concurrent flip.
[Collection("BlasManaged-Stats-Serial")]
public class DeterministicModeTests
{
    [Fact]
    public void BlasProvider_SetDeterministicMode_Toggles()
    {
        bool before = BlasProvider.IsDeterministicMode;
        try
        {
            BlasProvider.SetDeterministicMode(true);
            Assert.True(BlasProvider.IsDeterministicMode);
            BlasProvider.SetDeterministicMode(false);
            Assert.False(BlasProvider.IsDeterministicMode);
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);  // restore
        }
    }

    [Fact]
    public void BlasManaged_Gemm_RespectsDeterministicMode_StillCorrect()
    {
        // Sanity: enabling deterministic mode doesn't break correctness.
        bool before = BlasProvider.IsDeterministicMode;
        try
        {
            BlasProvider.SetDeterministicMode(true);

            int m = 8, n = 8, k = 16;
            var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
            double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);

            double[] actual = new double[m * n];
            BlasManagedLib.Gemm<double>(
                a, lda: k, transA: false,
                b, ldb: n, transB: false,
                actual, ldc: n,
                m, n, k);

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], precision: 10);
        }
        finally
        {
            BlasProvider.SetDeterministicMode(before);
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static (double[] a, double[] b) GenerateRandomMatrices(
        int m, int n, int k, bool transA, bool transB, int seed)
    {
        var rng = new Random(seed);
        int aLen = m * k;
        var a = new double[aLen];
        for (int i = 0; i < aLen; i++) a[i] = rng.NextDouble() * 2 - 1;
        int bLen = k * n;
        var b = new double[bLen];
        for (int i = 0; i < bLen; i++) b[i] = rng.NextDouble() * 2 - 1;
        return (a, b);
    }

    private static double[] NaiveGemm(
        double[] a, int aRows, int aCols, bool transA,
        double[] b, int bRows, int bCols, bool transB)
    {
        int m = transA ? aCols : aRows;
        int kA = transA ? aRows : aCols;
        int kB = transB ? bCols : bRows;
        int n = transB ? bRows : bCols;
        if (kA != kB) throw new ArgumentException($"Inner dim mismatch: A K={kA}, B K={kB}");
        int k = kA;

        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int kk = 0; kk < k; kk++)
                {
                    double aval = transA ? a[kk * aCols + i] : a[i * aCols + kk];
                    double bval = transB ? b[j * bCols + kk] : b[kk * bCols + j];
                    sum += aval * bval;
                }
                c[i * n + j] = sum;
            }
        return c;
    }
}
