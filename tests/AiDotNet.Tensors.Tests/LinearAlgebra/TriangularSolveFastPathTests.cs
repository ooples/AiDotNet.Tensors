using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Issue #379 — float/double fast path in the triangular solver
/// (<c>LinearSolvers.TriangularSolveSingle</c>): native typed arithmetic with the
/// row-update / diagonal-divide vectorized over the RHS columns. Verifies it solves
/// correctly across upper/lower, multi-RHS and single-RHS, for float and double, and
/// that the transpose path (exercised by Cholesky's two-phase L·y=b / Lᵀ·x=y solve)
/// stays correct. Correctness is checked by reconstruction (A·X' ≈ B), independent of
/// the scalar reference.
/// </summary>
public class TriangularSolveFastPathTests
{
    [Theory]
    [InlineData(true, 12)]   // upper, multi-RHS (hits the vectorized column loop)
    [InlineData(false, 12)]  // lower, multi-RHS
    [InlineData(true, 1)]    // upper, single-RHS
    [InlineData(false, 1)]   // lower, single-RHS
    public void SolveTriangular_Double_MatchesConstructedSolution(bool upper, int nrhs)
    {
        const int n = 7;
        var a = MakeTriangularDouble(n, upper, seed: 11);
        var x0 = RandomDouble(n * nrhs, seed: 22);
        var b = MatMulDouble(a, x0, n, nrhs);

        var aT = ToTensorD(a, n, n);
        var bT = ToTensorD(b, n, nrhs);
        var xT = Linalg.SolveTriangular(aT, bT, upper);

        var xd = xT.AsSpan();
        for (int i = 0; i < n * nrhs; i++)
            Assert.True(Math.Abs(xd[i] - x0[i]) <= 1e-9 * (1 + Math.Abs(x0[i])),
                $"x[{i}]={xd[i]:E6} vs expected {x0[i]:E6} (upper={upper}, nrhs={nrhs})");
    }

    [Theory]
    [InlineData(true, 16)]
    [InlineData(false, 16)]
    [InlineData(false, 1)]
    public void SolveTriangular_Float_MatchesConstructedSolution(bool upper, int nrhs)
    {
        const int n = 7;
        var a = MakeTriangularFloat(n, upper, seed: 33);
        var x0 = RandomFloat(n * nrhs, seed: 44);
        var b = MatMulFloat(a, x0, n, nrhs);

        var aT = ToTensorF(a, n, n);
        var bT = ToTensorF(b, n, nrhs);
        var xT = Linalg.SolveTriangular(aT, bT, upper);

        var xs = xT.AsSpan();
        for (int i = 0; i < n * nrhs; i++)
            Assert.True(Math.Abs(xs[i] - x0[i]) <= 1e-3f * (1 + Math.Abs(x0[i])),
                $"x[{i}]={xs[i]:E5} vs expected {x0[i]:E5} (upper={upper}, nrhs={nrhs})");
    }

    [Fact]
    public void CholeskySolve_Double_ExercisesTransposePath()
    {
        // Cholesky solve does L·y=b then Lᵀ·x=y → covers the transpose=true fast path.
        const int n = 5, nrhs = 8;
        // SPD A = M·Mᵀ + n·I (well-conditioned, positive-definite).
        var m = RandomDouble(n * n, seed: 55);
        var a = new double[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int k = 0; k < n; k++) s += m[i * n + k] * m[j * n + k];
                a[i * n + j] = s + (i == j ? n : 0);
            }
        var x0 = RandomDouble(n * nrhs, seed: 66);
        var b = MatMulDouble(a, x0, n, nrhs);

        var xT = Linalg.Solve(ToTensorD(a, n, n), ToTensorD(b, n, nrhs));  // routes SPD → Cholesky
        var xd = xT.AsSpan();
        for (int i = 0; i < n * nrhs; i++)
            Assert.True(Math.Abs(xd[i] - x0[i]) <= 1e-7 * (1 + Math.Abs(x0[i])),
                $"x[{i}]={xd[i]:E6} vs expected {x0[i]:E6}");
    }

    // ---- helpers ----

    private static double[] MakeTriangularDouble(int n, bool upper, int seed)
    {
        var a = RandomDouble(n * n, seed);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if ((upper && i > j) || (!upper && i < j)) a[i * n + j] = 0.0;
                else if (i == j) a[i * n + j] = 2.0 + Math.Abs(a[i * n + j]); // non-tiny diagonal
        return a;
    }

    private static float[] MakeTriangularFloat(int n, bool upper, int seed)
    {
        var d = MakeTriangularDouble(n, upper, seed);
        var f = new float[d.Length];
        for (int i = 0; i < d.Length; i++) f[i] = (float)d[i];
        return f;
    }

    private static double[] MatMulDouble(double[] a, double[] x, int n, int nrhs)
    {
        var b = new double[n * nrhs];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < nrhs; c++)
            {
                double s = 0;
                for (int k = 0; k < n; k++) s += a[i * n + k] * x[k * nrhs + c];
                b[i * nrhs + c] = s;
            }
        return b;
    }

    private static float[] MatMulFloat(float[] a, float[] x, int n, int nrhs)
    {
        var b = new float[n * nrhs];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < nrhs; c++)
            {
                double s = 0;
                for (int k = 0; k < n; k++) s += (double)a[i * n + k] * x[k * nrhs + c];
                b[i * nrhs + c] = (float)s;
            }
        return b;
    }

    private static double[] RandomDouble(int len, int seed)
    {
        var r = new Random(seed);
        var arr = new double[len];
        for (int i = 0; i < len; i++) arr[i] = r.NextDouble() * 2 - 1;
        return arr;
    }

    private static float[] RandomFloat(int len, int seed)
    {
        var r = new Random(seed);
        var arr = new float[len];
        for (int i = 0; i < len; i++) arr[i] = (float)(r.NextDouble() * 2 - 1);
        return arr;
    }

    private static Tensor<double> ToTensorD(double[] data, int rows, int cols)
    {
        var t = new Tensor<double>(new[] { rows, cols });
        var dst = t.AsWritableSpan();
        for (int i = 0; i < data.Length; i++) dst[i] = data[i];
        return t;
    }

    private static Tensor<float> ToTensorF(float[] data, int rows, int cols)
    {
        var t = new Tensor<float>(new[] { rows, cols });
        var dst = t.AsWritableSpan();
        for (int i = 0; i < data.Length; i++) dst[i] = data[i];
        return t;
    }
}
