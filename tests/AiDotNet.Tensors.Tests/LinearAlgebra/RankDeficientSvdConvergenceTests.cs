using System;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

public sealed class RankDeficientSvdConvergenceTests
{
    private static readonly double[] RankOne3x3 =
    {
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        3.0, 6.0, 9.0,
    };

    private static Tensor<TElem> Matrix<TElem>(int rows, int cols, TElem[] values)
        where TElem : unmanaged, IEquatable<TElem>, IComparable<TElem>
    {
        var tensor = new Tensor<TElem>(new[] { rows, cols });
        Array.Copy(values, tensor.GetDataArray(), values.Length);
        return tensor;
    }

    private static void AssertOrthonormalAndReconstructs(
        double[] original, int m, int n, double[] u, int uCols, double[] s, double[] vh, int vRows, double tol)
    {
        int k = Math.Min(m, n);
        for (int a = 0; a < uCols; a++)
        {
            for (int b = 0; b < uCols; b++)
            {
                double dot = 0;
                for (int i = 0; i < m; i++) dot += u[i * uCols + a] * u[i * uCols + b];
                Assert.True(Math.Abs(dot - (a == b ? 1.0 : 0.0)) <= tol, $"(U^T U)[{a},{b}] = {dot:R}");
            }
        }

        for (int a = 0; a < vRows; a++)
        {
            for (int b = 0; b < vRows; b++)
            {
                double dot = 0;
                for (int j = 0; j < n; j++) dot += vh[a * n + j] * vh[b * n + j];
                Assert.True(Math.Abs(dot - (a == b ? 1.0 : 0.0)) <= tol, $"(Vh Vh^T)[{a},{b}] = {dot:R}");
            }
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double recon = 0;
                for (int r = 0; r < k; r++) recon += u[i * uCols + r] * s[r] * vh[r * n + j];
                Assert.True(Math.Abs(recon - original[i * n + j]) <= tol * Math.Max(1.0, Math.Abs(original[i * n + j])),
                    $"A[{i},{j}] expected {original[i * n + j]:R}, got {recon:R}");
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Double_RankOne3x3_ConvergesWithRankOneAndNullityTwo(bool fullMatrices)
    {
        var (u, s, vh) = Linalg.Svd(Matrix(3, 3, RankOne3x3), fullMatrices);
        double[] sd = s.GetDataArray();
        double rankTol = sd[0] * 3 * 2.220446049250313e-16;
        int numericalRank = sd.Count(v => v > rankTol);

        Assert.Equal(1, numericalRank);
        Assert.Equal(2, 3 - numericalRank);
        Assert.True(Math.Abs(sd[0] - 14.0) < 1e-13, $"sigma_0 = {sd[0]:R}");
        AssertOrthonormalAndReconstructs(RankOne3x3, 3, 3, u.GetDataArray(), 3, sd, vh.GetDataArray(), 3, 1e-13);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Float_RankOne3x3_ConvergesAndReconstructs(bool fullMatrices)
    {
        float[] input = RankOne3x3.Select(v => (float)v).ToArray();
        var (u, s, vh) = Linalg.Svd(Matrix(3, 3, input), fullMatrices);
        float[] sf = s.GetDataArray();
        float rankTol = sf[0] * 3 * 1.1920929e-7f;

        Assert.Equal(1, sf.Count(v => v > rankTol));
        Assert.True(Math.Abs(sf[0] - 14f) < 1e-5f, $"sigma_0 = {sf[0]:R}");
        AssertOrthonormalAndReconstructs(
            RankOne3x3, 3, 3,
            u.GetDataArray().Select(v => (double)v).ToArray(), 3,
            sf.Select(v => (double)v).ToArray(),
            vh.GetDataArray().Select(v => (double)v).ToArray(), 3,
            1e-5);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decimal_RankOne3x3_ConvergesToDecimalPrecision(bool fullMatrices)
    {
        decimal[] input = RankOne3x3.Select(v => (decimal)v).ToArray();
        var (u, s, vh) = Linalg.Svd(Matrix(3, 3, input), fullMatrices);
        decimal[] sm = s.GetDataArray();

        Assert.True(Math.Abs(sm[0] - 14m) < 1e-25m, $"sigma_0 = {sm[0]}");
        Assert.True(sm[1] < 1e-25m, $"sigma_1 = {sm[1]}");
        Assert.True(sm[2] < 1e-25m, $"sigma_2 = {sm[2]}");
        AssertOrthonormalAndReconstructs(
            RankOne3x3, 3, 3,
            u.GetDataArray().Select(v => (double)v).ToArray(), 3,
            sm.Select(v => (double)v).ToArray(),
            vh.GetDataArray().Select(v => (double)v).ToArray(), 3,
            1e-14);
    }

#if NET5_0_OR_GREATER
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HalfAndBFloat16_RankOne3x3_ConvergeWithoutSweepFailure(bool bfloat16)
    {
        if (bfloat16)
        {
            BFloat16[] input = RankOne3x3.Select(v => (BFloat16)(float)v).ToArray();
            var (u, s, vh) = Linalg.Svd(Matrix(3, 3, input), fullMatrices: false);
            double[] sd = s.GetDataArray().Select(v => (double)(float)v).ToArray();
            Assert.True(Math.Abs(sd[0] - 14.0) < 0.25, $"sigma_0 = {sd[0]:R}");
            AssertOrthonormalAndReconstructs(
                RankOne3x3, 3, 3,
                u.GetDataArray().Select(v => (double)(float)v).ToArray(), 3,
                sd,
                vh.GetDataArray().Select(v => (double)(float)v).ToArray(), 3,
                0.1);
        }
        else
        {
            Half[] input = RankOne3x3.Select(v => (Half)v).ToArray();
            var (u, s, vh) = Linalg.Svd(Matrix(3, 3, input), fullMatrices: false);
            double[] sd = s.GetDataArray().Select(v => (double)v).ToArray();
            Assert.True(Math.Abs(sd[0] - 14.0) < 0.05, $"sigma_0 = {sd[0]:R}");
            AssertOrthonormalAndReconstructs(
                RankOne3x3, 3, 3,
                u.GetDataArray().Select(v => (double)v).ToArray(), 3,
                sd,
                vh.GetDataArray().Select(v => (double)v).ToArray(), 3,
                0.02);
        }
    }
#endif

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TallAndWide_MultiNullityRankDeficientMatrices_ConvergeAndCompleteBases(bool fullMatrices)
    {
        // 5x4 of exact rank 2 (columns 2 and 3 are linear combinations of columns 0 and 1 -> nullity 2).
        var c0 = new[] { 1.0, -2.0, 0.5, 3.0, -1.5 };
        var c1 = new[] { -0.5, 1.0, 2.0, -1.0, 2.5 };
        var tall = new double[5 * 4];
        for (int i = 0; i < 5; i++)
        {
            tall[i * 4 + 0] = c0[i];
            tall[i * 4 + 1] = c1[i];
            tall[i * 4 + 2] = 2.0 * c0[i] - 3.0 * c1[i];
            tall[i * 4 + 3] = -1.5 * c0[i] + 0.5 * c1[i];
        }

        var (ut, st, vht) = Linalg.Svd(Matrix(5, 4, tall), fullMatrices);
        double[] std = st.GetDataArray();
        double tallTol = std[0] * 5 * 2.220446049250313e-16;
        Assert.Equal(2, std.Count(v => v > tallTol));
        AssertOrthonormalAndReconstructs(tall, 5, 4, ut.GetDataArray(), fullMatrices ? 5 : 4, std, vht.GetDataArray(), 4, 1e-13);

        // 3x5 of exact rank 1 (nullity 4 in R^5, transpose path has nullity 2 in R^3).
        var wide = new double[3 * 5];
        double[] rowScale = { 1.0, -2.0, 3.0 };
        double[] colScale = { 2.0, -1.0, 4.0, 0.5, -3.0 };
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 5; j++)
                wide[i * 5 + j] = rowScale[i] * colScale[j];

        var (uw, sw, vhw) = Linalg.Svd(Matrix(3, 5, wide), fullMatrices);
        double[] swd = sw.GetDataArray();
        double wideTol = swd[0] * 5 * 2.220446049250313e-16;
        Assert.Equal(1, swd.Count(v => v > wideTol));
        AssertOrthonormalAndReconstructs(wide, 3, 5, uw.GetDataArray(), 3, swd, vhw.GetDataArray(), fullMatrices ? 5 : 3, 1e-13);
    }

    [Fact]
    public void RowGradedMatrix_PreservesTinyGenuineSingularValueAboveDeflation()
    {
        // A = [[1, 1], [0, 1e-20]]: det(A) = 1e-20, sigma_0 = sqrt(2), sigma_1 = 1e-20 / sqrt(2).
        // Row 0 cancels at 45 degrees, while row 1 carries no cancellation and must not be deflated.
        var a = new[] { 1.0, 1.0, 0.0, 1e-20 };
        var (_, s, _) = Linalg.Svd(Matrix(2, 2, a), fullMatrices: false);
        double[] sd = s.GetDataArray();
        double expectedSmall = 1e-20 / Math.Sqrt(2.0);

        Assert.True(Math.Abs(sd[0] - Math.Sqrt(2.0)) < 1e-15, $"sigma_0 = {sd[0]:R}");
        Assert.True(Math.Abs(sd[1] - expectedSmall) <= 1e-14 * expectedSmall, $"sigma_1 = {sd[1]:R}, expected {expectedSmall:R}");
    }
}
