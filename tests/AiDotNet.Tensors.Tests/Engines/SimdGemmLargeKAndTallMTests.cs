using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// #1307: the large-K route (a deep reduction into a small output, the weight gradient Xᵀ·dY of a linear layer, A
/// optionally stored transposed) and the tall-M route (dX = dY·Wᵀ with an A too large for the column-panel route).
/// Each must match a double reference, overwrite and accumulate, through the public Sgemm entry points that take it, and
/// give the same bits for any thread budget and on the serial path.
/// </summary>
[Collection("GemmDeterminismSerial")]
public sealed class SimdGemmLargeKAndTallMTests
{
    private static float[] Fill(int n, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() - 0.5);
        return v;
    }

    // A is [m x k] (or stored [k x m] when transA), B is [k x n] (or stored [n x k] when transB).
    private static double[] Reference(float[] a, bool transA, float[] b, bool transB, float[]? c0, int m, int k, int n)
    {
        var r = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = c0 is null ? 0 : c0[i * n + j];
                for (int p = 0; p < k; p++)
                    s += (double)(transA ? a[p * m + i] : a[i * k + p]) * (transB ? b[j * k + p] : b[p * n + j]);
                r[i * n + j] = s;
            }
        return r;
    }

    private static void AssertClose(double[] expected, float[] actual, int k, string what)
    {
        double tol = 2e-6 * Math.Sqrt(k) * 8;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    private static int Bits(float v) => BitConverter.ToInt32(BitConverter.GetBytes(v), 0);

    [Theory]
    [InlineData(256, 4096, 256, true)]     // Cᵀ·dY of a 256-wide layer over 4096 rows
    [InlineData(128, 4096, 384, true)]     // Xᵀ·dQKV, n past one 256-column panel
    [InlineData(70, 3001, 50, true)]       // a K the slices do not divide; partial tiles in m and n
    [InlineData(256, 4096, 256, false)]
    [InlineData(70, 3001, 300, false)]
    public void LargeK_MatchesDoubleReference(int m, int k, int n, bool transA)
    {
        int lda = transA ? m : k;
        Assert.True(SimdGemm.QualifiesForLargeK(transA, false, lda, m, k, n), "the shape must take the large-K route");
        var a = Fill(m * k, 3);
        var b = Fill(k * n, 5);
        var c = new float[m * n];
        int before = SimdGemm.LargeKRunsOnThisThread;
        SimdGemm.Sgemm(a, lda, transA, b, n, false, c, m, k, n);
        Assert.Equal(before + 1, SimdGemm.LargeKRunsOnThisThread);
        AssertClose(Reference(a, transA, b, false, null, m, k, n), c, k, $"overwrite {m}x{k}x{n} transA={transA}");

        var c0 = Fill(m * n, 9);
        var acc = (float[])c0.Clone();
        SimdGemm.SgemmAdd(a, lda, transA, b, n, false, acc, m, k, n);
        AssertClose(Reference(a, transA, b, false, c0, m, k, n), acc, k, $"accumulate {m}x{k}x{n} transA={transA}");
    }

    [Theory]
    [InlineData(4096, 768, 256, false)]    // dQKV·W with W already [k x n]
    [InlineData(4096, 768, 256, true)]     // dQKV·Wᵀ, W stored [n x k]: B transposed once, then this route
    [InlineData(4096, 384, 128, true)]
    [InlineData(1501, 400, 70, false)]     // partial row block and partial panel
    public void TallM_MatchesDoubleReference(int m, int k, int n, bool transB)
    {
        Assert.True(SimdGemm.QualifiesForTallM(false, false, m, k, n), "the shape must take the tall-M route");
        var a = Fill(m * k, 11);
        var b = Fill(k * n, 13);
        var c = new float[m * n];
        int before = SimdGemm.TallMRunsOnThisThread;
        SimdGemm.Sgemm(a, k, false, b, transB ? k : n, transB, c, m, k, n);
        Assert.Equal(before + 1, SimdGemm.TallMRunsOnThisThread);
        AssertClose(Reference(a, false, b, transB, null, m, k, n), c, k, $"{m}x{k}x{n} transB={transB}");
    }

    // A tall product with an A of 2 MB or more takes the tall-M route ahead of the column-panel route, at any K.
    [Theory]
    [InlineData(4096, 256, 768, false)]    // the fused attention's QKV projection
    [InlineData(4096, 128, 128, true)]     // dY·Woᵀ, Wo stored [n x k]
    [InlineData(2049, 300, 70, false)]
    public void BigA_TakesTallM_AndMatchesDoubleReference(int m, int k, int n, bool transB)
    {
        Assert.True(SimdGemm.PrefersTallMForBigA(m, k, n), "the shape must prefer the tall-M route");
        var a = Fill(m * k, 31);
        var b = Fill(k * n, 33);
        var c = new float[m * n];
        int before = SimdGemm.TallMRunsOnThisThread;
        SimdGemm.Sgemm(a, k, false, b, transB ? k : n, transB, c, m, k, n);
        Assert.Equal(before + 1, SimdGemm.TallMRunsOnThisThread);
        AssertClose(Reference(a, false, b, transB, null, m, k, n), c, k, $"{m}x{k}x{n} transB={transB}");
    }
    [Theory]
    [InlineData(256, 4096, 768, true, false)]
    [InlineData(70, 3001, 300, false, false)]
    [InlineData(4096, 768, 256, false, true)]
    public void NewRoutes_AreBitIdentical_AcrossThreadBudgetsAndSerial(int m, int k, int n, bool transA, bool transB)
    {
        int lda = transA ? m : k, ldb = transB ? k : n;
        var a = Fill(m * k, 21);
        var b = Fill(k * n, 23);
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            float[] Run(int threads)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var c = new float[m * n];
                SimdGemm.Sgemm(a, lda, transA, b, ldb, transB, c, m, k, n);
                return c;
            }
            var baseline = Run(saved);
            foreach (int threads in new[] { 2, 4, 7, 16 })
            {
                var other = Run(threads);
                for (int i = 0; i < baseline.Length; i++)
                    Assert.True(Bits(baseline[i]) == Bits(other[i]), $"[{i}] at {threads} threads: {other[i]:R} vs {baseline[i]:R}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = saved;
        }
        if (!transB)
        {
            // The serial entry point (used inside parallel regions) must take the same route in the same order.
            var serial = new float[m * n];
            var reference = new float[m * n];
            SimdGemm.Sgemm(a, lda, transA, b, ldb, false, reference, m, k, n);
            if (!transA)
            {
                SimdGemm.SgemmSequential(a, b, serial, m, k, n);
                for (int i = 0; i < reference.Length; i++)
                    Assert.True(Bits(reference[i]) == Bits(serial[i]), $"serial [{i}]: {serial[i]:R} vs {reference[i]:R}");
            }
        }
    }
}