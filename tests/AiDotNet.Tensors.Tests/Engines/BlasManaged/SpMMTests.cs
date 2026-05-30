using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class SpMMTests
{
    // Build CSR arrays from a dense rows×cols matrix (drop exact zeros).
    private static (int[] ptr, int[] ind, double[] val) DenseToCsr(double[] a, int rows, int cols)
    {
        var ptr = new int[rows + 1];
        var ind = new List<int>();
        var val = new List<double>();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double v = a[i * cols + j];
                if (v != 0.0) { ind.Add(j); val.Add(v); }
            }
            ptr[i + 1] = ind.Count;
        }
        return (ptr, ind.ToArray(), val.ToArray());
    }

    // Dense reference: C = alpha*A*B + beta*C.
    private static double[] ReferenceSpMM(
        double[] a, int rows, int cols, double[] b, int n, double alpha, double beta, double[] c)
    {
        var outC = (double[])c.Clone();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int k = 0; k < cols; k++) acc += a[i * cols + k] * b[k * n + j];
                outC[i * n + j] = alpha * acc + beta * c[i * n + j];
            }
        return outC;
    }

    [Fact]
    public void SpMM_FP64_Csr_MatchesDenseReference()
    {
        const int rows = 5, cols = 4, n = 3;
        var rng = new Random(42);
        double[] a = new double[rows * cols];
        // ~50% sparse
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() < 0.5 ? 0.0 : rng.NextDouble() * 2 - 1;
        double[] b = new double[cols * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[rows * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        var (ptr, ind, val) = DenseToCsr(a, rows, cols);
        double[] expected = ReferenceSpMM(a, rows, cols, b, n, 1.0, 0.0, c);

        double[] actual = (double[])c.Clone();
        var layout = new SparseLayout<double>
        {
            Rows = rows, Cols = cols, Pointers = ptr, Indices = ind, Values = val,
            Format = SparseLayoutFormat.Csr,
        };
        BlasManagedLib.SpMM<double>(1.0, layout, b, n, n, 0.0, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }
}
