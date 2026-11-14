using System;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Accuracy of the float and double SVD on inputs where the singular values span many orders of magnitude.
/// <para>
/// The SVD used to eigendecompose the Gram matrix AᵀA, which squares the condition number: a singular value
/// below about sqrt(eps)·σ_max came back as rounding noise. For the rank-2 matrix [[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]]
/// the third value was 9.2e-9 instead of about 1e-17 (numpy.linalg.svd: 3.4e-17), so numpy's matrix_rank tolerance
/// (σ_max·max(m,n)·eps ≈ 1.1e-15) reported rank 3. The one-sided Jacobi SVD works on A and keeps each singular
/// value's relative accuracy (Demmel and Veselić 1992).
/// </para>
/// </summary>
public class SvdRelativeAccuracyTests
{
    private static readonly double[] RankTwo = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };

    private static Tensor<TElem> Matrix<TElem>(int rows, int cols, TElem[] values)
        where TElem : unmanaged, IEquatable<TElem>, IComparable<TElem>
    {
        var t = new Tensor<TElem>(new[] { rows, cols });
        Array.Copy(values, t.GetDataArray(), values.Length);
        return t;
    }

    /// <summary>Q·diag(1, 1e-10) with Q a rotation by 30 degrees: singular values exactly 1 and 1e-10.</summary>
    private static double[] Graded()
    {
        double c = Math.Cos(Math.PI / 6), s = Math.Sin(Math.PI / 6);
        return new[] { c, -s * 1e-10, s, c * 1e-10 };
    }

    [Fact]
    public void Double_RankDeficient_SmallestSingularValueIsBelowTheRankTolerance()
    {
        var (_, s, _) = Linalg.Svd(Matrix(3, 3, RankTwo), false);
        double[] values = s.GetDataArray();
        double tolerance = values[0] * 3 * 2.220446049250313e-16;
        Assert.True(values[2] < tolerance, $"third singular value {values[2]:R} is not below {tolerance:R}");
        Assert.True(values[1] > tolerance, $"second singular value {values[1]:R} should be well above {tolerance:R}");
        // The squared singular values sum to the squared Frobenius norm, 2.85.
        Assert.Equal(2.85, values.Sum(v => v * v), 13);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Double_RankDeficient_ReconstructsAndUIsOrthonormal(bool fullMatrices)
    {
        var (u, s, vh) = Linalg.Svd(Matrix(3, 3, RankTwo), fullMatrices);
        double[] ud = u.GetDataArray(), sd = s.GetDataArray(), vd = vh.GetDataArray();
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double a = 0, uu = 0;
                for (int k = 0; k < 3; k++)
                {
                    a += ud[i * 3 + k] * sd[k] * vd[k * 3 + j];
                    uu += ud[k * 3 + i] * ud[k * 3 + j];
                }

                Assert.True(Math.Abs(a - RankTwo[i * 3 + j]) < 1e-14, $"A[{i},{j}] reconstructs as {a:R}");
                // The column for the zero singular value must be completed to an orthonormal basis, not left zero.
                Assert.True(Math.Abs(uu - (i == j ? 1.0 : 0.0)) < 1e-14, $"(UᵀU)[{i},{j}] = {uu:R}");
            }
        }
    }

    [Fact]
    public void Double_Graded_SmallSingularValueKeepsItsRelativeAccuracy()
    {
        var (_, s, _) = Linalg.Svd(Matrix(2, 2, Graded()), false);
        double[] values = s.GetDataArray();
        Assert.True(Math.Abs(values[0] - 1.0) < 1e-15, $"σ1 = {values[0]:R}");
        // Through AᵀA the 1e-20 eigenvalue sits below 1·eps, so σ2 came back with an error near 1e-8.
        Assert.True(Math.Abs(values[1] - 1e-10) < 1e-24, $"σ2 = {values[1]:R}");
    }

    [Fact]
    public void Float_Graded_SmallSingularValueKeepsItsRelativeAccuracy()
    {
        float[] graded = Graded().Select(v => (float)v).ToArray();
        var (_, s, _) = Linalg.Svd(Matrix(2, 2, graded), false);
        float[] values = s.GetDataArray();
        Assert.True(Math.Abs(values[0] - 1f) < 1e-6f, $"σ1 = {values[0]:R}");
        // The float inputs carry a relative rounding of about 6e-8 per entry, so σ2 is accurate to about 1e-7
        // relative; through a float or double AᵀA it was lost entirely.
        Assert.True(Math.Abs(values[1] - 1e-10f) < 1e-15f, $"σ2 = {values[1]:R}");
    }
}
