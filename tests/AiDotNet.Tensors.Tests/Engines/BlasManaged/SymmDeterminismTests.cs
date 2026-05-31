using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class SymmDeterminismTests
{
    [Theory]
    [InlineData(64, 48)]
    [InlineData(128, 96)]
    [InlineData(192, 64)]
    public void Symm_FP64_BitExactAcrossThreadCounts(int m, int n)
    {
        var rng = new Random(42);
        double[] a = new double[m * m];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] b = new double[m * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double[] c0 = new double[m * n];
        for (int i = 0; i < c0.Length; i++) c0[i] = rng.NextDouble() * 2 - 1;

        double[]? baseline = null;
        foreach (int threads in new[] { 1, 2, 4, 8 })
        {
            double[] actual = (double[])c0.Clone();
            var opts = new BlasOptions<double> { NumThreads = threads, Mode = BlasMode.Deterministic };
            BlasManagedLib.Symm<double>(Side.Left, Uplo.Lower, m, n, 1.3, a, m, b, n, 0.7, actual, n, opts);
            if (baseline is null) baseline = actual;
            else for (int i = 0; i < actual.Length; i++) Assert.Equal(baseline[i], actual[i]);
        }
    }
}
