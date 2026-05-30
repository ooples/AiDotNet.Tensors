using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class SpMMDeterminismTests
{
    [Theory]
    [InlineData(128, 96, 32)]
    [InlineData(256, 128, 16)]
    public void SpMM_FP64_Csr_BitExactAcrossThreadCounts(int rows, int cols, int n)
    {
        var rng = new Random(42);
        // Build a CSR matrix at ~10% density.
        var ptr = new int[rows + 1];
        var ind = new List<int>();
        var val = new List<double>();
        for (int i = 0; i < rows; i++)
        {
            for (int k = 0; k < cols; k++)
                if (rng.NextDouble() < 0.10) { ind.Add(k); val.Add(rng.NextDouble() * 2 - 1); }
            ptr[i + 1] = ind.Count;
        }
        int[] indA = ind.ToArray();
        double[] valA = val.ToArray();
        double[] b = new double[cols * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double[] c0 = new double[rows * n];
        for (int i = 0; i < c0.Length; i++) c0[i] = rng.NextDouble() * 2 - 1;

        double[]? baseline = null;
        foreach (int threads in new[] { 1, 2, 4, 8 })
        {
            double[] actual = (double[])c0.Clone();
            var opts = new BlasOptions<double> { NumThreads = threads, Mode = BlasMode.Deterministic };
            var layout = new SparseLayout<double>
            { Rows = rows, Cols = cols, Pointers = ptr, Indices = indA, Values = valA, Format = SparseLayoutFormat.Csr };
            BlasManagedLib.SpMM<double>(1.3, layout, b, n, n, 0.7, actual, n, opts);
            if (baseline is null) baseline = actual;
            else for (int i = 0; i < actual.Length; i++) Assert.Equal(baseline[i], actual[i]);
        }
    }
}
