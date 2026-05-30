using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class TrsmDeterminismTests
{
    [Theory]
    [InlineData(64, 32)]
    [InlineData(128, 16)]
    [InlineData(256, 8)]
    public void Trsm_FP64_BitExactAcrossThreadCounts(int m, int n)
    {
        var rng = new Random(42);
        double[] a = new double[m * m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j <= i; j++) a[i * m + j] = rng.NextDouble() * 2 - 1;
            a[i * m + i] += m;
        }
        double[] b0 = new double[m * n];
        for (int i = 0; i < b0.Length; i++) b0[i] = rng.NextDouble() * 2 - 1;

        double[]? baseline = null;
        foreach (int threads in new[] { 1, 2, 4, 8 })
        {
            double[] actual = (double[])b0.Clone();
            var opts = new BlasOptions<double> { NumThreads = threads, Mode = BlasMode.Deterministic };
            BlasManagedLib.Trsm<double>(Side.Left, Uplo.Lower, false, Diag.NonUnit, m, n, 1.0, a, m, actual, n, opts);
            if (baseline is null) baseline = actual;
            else
                for (int i = 0; i < actual.Length; i++)
                    Assert.Equal(baseline[i], actual[i]); // EXACT bit equality, no tolerance
        }
    }
}
