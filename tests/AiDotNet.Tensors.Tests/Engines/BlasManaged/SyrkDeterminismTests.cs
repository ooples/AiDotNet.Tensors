using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class SyrkDeterminismTests
{
    [Theory]
    [InlineData(64, 48)]
    [InlineData(128, 96)]
    [InlineData(192, 64)]
    public void Syrk_FP64_BitExactAcrossThreadCounts(int n, int k)
    {
        var rng = new Random(42);
        double[] a = new double[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        double[] c0 = new double[n * n];
        for (int i = 0; i < c0.Length; i++) c0[i] = rng.NextDouble() * 2 - 1;

        double[]? baseline = null;
        foreach (int threads in new[] { 1, 2, 4, 8 })
        {
            double[] actual = (double[])c0.Clone();
            var opts = new BlasOptions<double> { NumThreads = threads, Mode = BlasMode.Deterministic };
            BlasManagedLib.Syrk<double>(Uplo.Lower, false, n, k, 1.3, a, k, 0.7, actual, n, opts);
            if (baseline is null) baseline = actual;
            else for (int i = 0; i < actual.Length; i++) Assert.Equal(baseline[i], actual[i]);
        }
    }
}
