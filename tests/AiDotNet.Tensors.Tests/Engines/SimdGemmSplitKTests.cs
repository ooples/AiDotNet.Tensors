using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Split-K float GEMM (a small output with a long K, e.g. a flatten-to-dense layer): correct against a double
/// reference for overwrite and accumulate, including a K the slices do not divide, and bit-identical whatever the
/// thread budget (its slice plan depends on the shape alone).
/// </summary>
[Collection("GemmDeterminismSerial")]
public sealed class SimdGemmSplitKTests
{
    private static float[] Fill(int n, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() - 0.5);
        return v;
    }

    private static double[] Reference(float[] a, float[] b, float[]? c0, int m, int k, int n)
    {
        var r = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = c0 is null ? 0 : c0[i * n + j];
                for (int p = 0; p < k; p++) s += (double)a[i * k + p] * b[p * n + j];
                r[i * n + j] = s;
            }
        return r;
    }

    private static void AssertClose(double[] expected, float[] actual, int k, string what)
    {
        double tol = 2e-6 * k;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Theory]
    [InlineData(64, 3136, 128)]
    [InlineData(32, 2048, 64)]
    [InlineData(12, 1531, 40)]
    // Slices of 1250 exceed the 2-D bound, so each slice's own dispatch must not split again (re-entry guard).
    [InlineData(32, 20000, 64)]
    public void SplitK_MatchesDoubleReference_ForOverwriteAndAccumulate(int m, int k, int n)
    {
        var a = Fill(m * k, 3);
        var b = Fill(k * n, 5);
        var c = new float[m * n];
#pragma warning disable CS0618
        SimdGemm.Sgemm(a, b, c, m, k, n);
#pragma warning restore CS0618
        AssertClose(Reference(a, b, null, m, k, n), c, k, $"overwrite {m}x{k}x{n}");

        var c0 = Fill(m * n, 9);
        var acc = (float[])c0.Clone();
        SimdGemm.SgemmAdd(a, k, false, b, n, false, acc, m, k, n);
        AssertClose(Reference(a, b, c0, m, k, n), acc, k, $"accumulate {m}x{k}x{n}");
    }

    // BitConverter.SingleToInt32Bits is not on .NET Framework 4.7.1.
    private static int Bits(float v) => BitConverter.ToInt32(BitConverter.GetBytes(v), 0);

    [Fact]
    public void SplitK_IsBitIdentical_AcrossThreadBudgets()
    {
        const int m = 64, k = 3136, n = 128;
        var a = Fill(m * k, 21);
        var b = Fill(k * n, 23);
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            float[] Run(int threads)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var c = new float[m * n];
#pragma warning disable CS0618
                SimdGemm.Sgemm(a, b, c, m, k, n);
#pragma warning restore CS0618
                return c;
            }
            var baseline = Run(saved);
            // Budget 1 is left out: with one thread this shape is routed above SimdGemm (BlasManaged) to a different
            // kernel, so its bits differ from the multi-threaded result with or without split-K (7446 of 8192
            // elements, measured with split-K switched off). Every parallel budget must agree exactly.
            foreach (int threads in new[] { 2, 4, 7 })
            {
                var other = Run(threads);
                for (int i = 0; i < baseline.Length; i++)
                    Assert.True(Bits(baseline[i]) == Bits(other[i]),
                        $"[{i}] at {threads} threads: {other[i]:R} vs {baseline[i]:R}");
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = saved;
        }
    }
}
