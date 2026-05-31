using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class GbmvDeterminismTests
{
    // GBMV is level-2 (each output element is an independent fixed-order band dot),
    // so it is bit-exact regardless of thread count. This guards that contract.
    [Theory]
    [InlineData(256, 2, 3)]
    [InlineData(512, 4, 4)]
    public void Gbmv_FP64_BitExactAcrossThreadCounts(int nn, int kl, int ku)
    {
        int lda = kl + ku + 1;
        var rng = new Random(42);
        double[] ab = new double[lda * nn];
        for (int i = 0; i < ab.Length; i++) ab[i] = rng.NextDouble() * 2 - 1;
        double[] x = new double[nn];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextDouble() * 2 - 1;
        double[] y0 = new double[nn];
        for (int i = 0; i < y0.Length; i++) y0[i] = rng.NextDouble() * 2 - 1;

        double[]? baseline = null;
        foreach (int threads in new[] { 1, 2, 4, 8 })
        {
            double[] actual = (double[])y0.Clone();
            var opts = new BlasOptions<double> { NumThreads = threads, Mode = BlasMode.Deterministic };
            BlasManagedLib.Gbmv<double>(false, nn, nn, kl, ku, 1.3, ab, lda, x, 1, 0.7, actual, 1, opts);
            if (baseline is null) baseline = actual;
            else for (int i = 0; i < actual.Length; i++) Assert.Equal(baseline[i], actual[i]);
        }
    }
}
