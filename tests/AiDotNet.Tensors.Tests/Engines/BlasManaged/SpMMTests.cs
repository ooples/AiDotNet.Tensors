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

    private static (int[] ptr, int[] ind, double[] val) DenseToCsc(double[] a, int rows, int cols)
    {
        var ptr = new int[cols + 1];
        var ind = new List<int>();
        var val = new List<double>();
        for (int k = 0; k < cols; k++)
        {
            for (int i = 0; i < rows; i++)
            {
                double v = a[i * cols + k];
                if (v != 0.0) { ind.Add(i); val.Add(v); }
            }
            ptr[k + 1] = ind.Count;
        }
        return (ptr, ind.ToArray(), val.ToArray());
    }

    [Fact]
    public void SpMM_FP64_Csc_AlphaBeta_MatchesDenseReference()
    {
        const int rows = 6, cols = 5, n = 4;
        var rng = new Random(123);
        double[] a = new double[rows * cols];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() < 0.6 ? 0.0 : rng.NextDouble() * 2 - 1;
        double[] b = new double[cols * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[rows * n];
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 2 - 1;

        double alpha = 1.7, beta = -0.5;
        var (ptr, ind, val) = DenseToCsc(a, rows, cols);
        double[] expected = ReferenceSpMM(a, rows, cols, b, n, alpha, beta, c);

        double[] actual = (double[])c.Clone();
        var layout = new SparseLayout<double>
        {
            Rows = rows, Cols = cols, Pointers = ptr, Indices = ind, Values = val,
            Format = SparseLayoutFormat.Csc,
        };
        BlasManagedLib.SpMM<double>(alpha, layout, b, n, n, beta, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 9);
    }

    [Fact]
    public void SpMM_FP64_Csr_EmptyRow_LeavesScaledBeta()
    {
        // Row 1 is all-zero → output row 1 must equal beta * original.
        const int rows = 3, cols = 3, n = 2;
        double[] a = { 1, 0, 0,  0, 0, 0,  0, 2, 3 };
        double[] b = { 1, 1,  2, 2,  3, 3 };
        double[] c = { 5, 6,  7, 8,  9, 10 };
        double beta = 2.0;

        var (ptr, ind, val) = DenseToCsr(a, rows, cols);
        double[] expected = ReferenceSpMM(a, rows, cols, b, n, 1.0, beta, c);
        double[] actual = (double[])c.Clone();
        var layout = new SparseLayout<double>
        { Rows = rows, Cols = cols, Pointers = ptr, Indices = ind, Values = val, Format = SparseLayoutFormat.Csr };
        BlasManagedLib.SpMM<double>(1.0, layout, b, n, n, beta, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 12);
        // Explicit empty-row check: row 1 == beta * original.
        Assert.Equal(beta * 7, actual[1 * n + 0], 12);
        Assert.Equal(beta * 8, actual[1 * n + 1], 12);
    }

    [Fact]
    public void SpMM_FP32_Csr_MatchesReference()
    {
        const int rows = 5, cols = 4, n = 3;
        var rng = new Random(7);
        float[] a = new float[rows * cols];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() < 0.5 ? 0f : (float)(rng.NextDouble() * 2 - 1);
        float[] b = new float[cols * n];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] c = new float[rows * n];
        for (int i = 0; i < c.Length; i++) c[i] = (float)(rng.NextDouble() * 2 - 1);

        double[] a64 = Array.ConvertAll(a, x => (double)x);
        double[] b64 = Array.ConvertAll(b, x => (double)x);
        double[] c64 = Array.ConvertAll(c, x => (double)x);
        var (ptr, ind, valD) = DenseToCsr(a64, rows, cols);
        float[] valF = Array.ConvertAll(valD, x => (float)x);
        double[] expected = ReferenceSpMM(a64, rows, cols, b64, n, 1.0, 0.0, c64);

        float[] actual = (float[])c.Clone();
        var layout = new SparseLayout<float>
        { Rows = rows, Cols = cols, Pointers = ptr, Indices = ind, Values = valF, Format = SparseLayoutFormat.Csr };
        BlasManagedLib.SpMM<float>(1f, layout, b, n, n, 0f, actual, n);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 3);
    }

    [Fact]
    public void SpMM_FP64_Csr_FusedBiasReLU_EqualsUnfusedPipeline()
    {
        // Exceed-vendor lever: SpMM + bias + ReLU in one call == SpMM then bias then ReLU.
        const int rows = 6, cols = 5, n = 4;
        var rng = new Random(2024);
        double[] a = new double[rows * cols];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() < 0.5 ? 0.0 : rng.NextDouble() * 2 - 1;
        double[] b = new double[cols * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double[] bias = new double[n];
        for (int i = 0; i < n; i++) bias[i] = rng.NextDouble() * 2 - 1;

        var (ptr, ind, val) = DenseToCsr(a, rows, cols);
        var layout = new SparseLayout<double>
        { Rows = rows, Cols = cols, Pointers = ptr, Indices = ind, Values = val, Format = SparseLayoutFormat.Csr };

        // Unfused reference: SpMM (beta=0) then +bias[j] then ReLU.
        double[] unfused = new double[rows * n];
        BlasManagedLib.SpMM<double>(1.0, layout, b, n, n, 0.0, unfused, n);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < n; j++)
            {
                double v = unfused[i * n + j] + bias[j];
                unfused[i * n + j] = v > 0 ? v : 0; // ReLU
            }

        // Fused: bias + ReLU via the epilogue.
        double[] fused = new double[rows * n];
        var opts = new BlasOptions<double>
        {
            Epilogue = new Epilogue<double> { BiasN = bias, Activation = AiDotNet.Tensors.Engines.FusedActivationType.ReLU },
        };
        BlasManagedLib.SpMM<double>(1.0, layout, b, n, n, 0.0, fused, n, opts);

        for (int i = 0; i < fused.Length; i++) Assert.Equal(unfused[i], fused[i], 10);
    }
}
