using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Correctness tests for the <see cref="Linalg"/> namespace (issue #211). Each
/// test verifies a single op against a known-value reference computed by hand
/// or extracted from a trusted library (NumPy/SciPy). Tolerances are tuned per
/// decomposition based on the algorithm's theoretical conditioning.
/// </summary>
public class LinalgTests
{
    private const float FloatTol = 1e-4f;
    private const double DoubleTol = 1e-8;

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static Tensor<double> FromRows(double[,] rows)
    {
        int m = rows.GetLength(0);
        int n = rows.GetLength(1);
        var t = new Tensor<double>(new[] { m, n });
        var d = t.GetDataArray();
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) d[i * n + j] = rows[i, j];
        return t;
    }

    private static void AssertClose(Tensor<double> actual, double[,] expected, double tol = DoubleTol)
    {
        int m = expected.GetLength(0);
        int n = expected.GetLength(1);
        Assert.Equal(m, actual.Shape[actual.Rank - 2]);
        Assert.Equal(n, actual.Shape[actual.Rank - 1]);
        var d = actual.GetDataArray();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                Assert.True(Math.Abs(d[i * n + j] - expected[i, j]) < tol,
                    $"[{i},{j}]: expected {expected[i, j]}, actual {d[i * n + j]} (tol={tol})");
    }

    private static double[,] MatMul(double[,] a, double[,] b)
    {
        int m = a.GetLength(0), k = a.GetLength(1), n = b.GetLength(1);
        var r = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int l = 0; l < k; l++) s += a[i, l] * b[l, j];
                r[i, j] = s;
            }
        return r;
    }

    private static double[,] Transpose(double[,] a)
    {
        int m = a.GetLength(0), n = a.GetLength(1);
        var r = new double[n, m];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) r[j, i] = a[i, j];
        return r;
    }

    private static double[,] ToArray2D(Tensor<double> t)
    {
        int m = t.Shape[t.Rank - 2];
        int n = t.Shape[t.Rank - 1];
        var r = new double[m, n];
        var d = t.GetDataArray();
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) r[i, j] = d[i * n + j];
        return r;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DECOMPOSITIONS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void Cholesky_SmallSpd_ReconstructsInput()
    {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] — a classic SPD example.
        var A = FromRows(new double[,] { { 4, 12, -16 }, { 12, 37, -43 }, { -16, -43, 98 } });
        var L = Linalg.Cholesky(A);
        var reconstructed = MatMul(ToArray2D(L), Transpose(ToArray2D(L)));
        AssertClose(FromRows(reconstructed), new double[,] { { 4, 12, -16 }, { 12, 37, -43 }, { -16, -43, 98 } });
    }

    [Fact]
    public void Cholesky_Upper_ReconstructsInput()
    {
        var A = FromRows(new double[,] { { 2, 1 }, { 1, 2 } });
        var U = Linalg.Cholesky(A, upper: true);
        var reconstructed = MatMul(Transpose(ToArray2D(U)), ToArray2D(U));
        AssertClose(FromRows(reconstructed), new double[,] { { 2, 1 }, { 1, 2 } });
    }

    [Fact]
    public void Lu_3x3_ProducesValidFactorization()
    {
        var A = FromRows(new double[,] { { 2, 1, 1 }, { 4, 3, 3 }, { 8, 7, 9 } });
        var (P, L, U) = Linalg.LU(A);

        // Verify P·A == L·U.
        var pa = MatMul(ToArray2D(P), ToArray2D(A));
        var lu = MatMul(ToArray2D(L), ToArray2D(U));
        AssertClose(FromRows(pa), lu);
    }

    [Fact]
    public void Qr_Reduced_ProducesOrthogonalQAndUpperR()
    {
        var A = FromRows(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var (Q, R) = Linalg.QR(A, "reduced");

        // Q should be 3×2, R should be 2×2.
        Assert.Equal(new[] { 3, 2 }, Q.Shape.ToArray());
        Assert.Equal(new[] { 2, 2 }, R.Shape.ToArray());

        // Q·R == A.
        var qr = MatMul(ToArray2D(Q), ToArray2D(R));
        AssertClose(FromRows(qr), new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }, tol: 1e-6);

        // Qᵀ·Q == I (orthogonality).
        var qtq = MatMul(Transpose(ToArray2D(Q)), ToArray2D(Q));
        AssertClose(FromRows(qtq), new double[,] { { 1, 0 }, { 0, 1 } }, tol: 1e-6);

        // R is upper triangular.
        var rD = R.GetDataArray();
        Assert.True(Math.Abs(rD[1 * 2 + 0]) < 1e-6, "R[1,0] should be zero (upper triangular).");
    }

    [Fact]
    public void Qr_Complete_QIsMxM()
    {
        var A = FromRows(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var (Q, R) = Linalg.QR(A, "complete");
        Assert.Equal(new[] { 3, 3 }, Q.Shape.ToArray());
        Assert.Equal(new[] { 3, 2 }, R.Shape.ToArray());
    }

    [Fact]
    public void Eigh_SymmetricMatrix_ReturnsExpectedEigenvalues()
    {
        // [[2, 0], [0, 3]] — diagonal, eigenvalues are {2, 3}.
        var A = FromRows(new double[,] { { 2, 0 }, { 0, 3 } });
        var (w, v) = Linalg.Eigh(A);
        var wD = w.GetDataArray();
        // Sorted ascending.
        Assert.Equal(2.0, wD[0], 6);
        Assert.Equal(3.0, wD[1], 6);
    }

    [Fact]
    public void Eigh_2x2_ReconstructsInput()
    {
        // Symmetric A = [[4, 2], [2, 5]] has eigenvalues (λ = 3, 6).
        var A = FromRows(new double[,] { { 4, 2 }, { 2, 5 } });
        var (w, V) = Linalg.Eigh(A);

        // Verify V·diag(w)·Vᵀ == A.
        int n = 2;
        var Vmat = ToArray2D(V);
        var VT = Transpose(Vmat);
        var wD = w.GetDataArray();
        var diagW = new double[n, n];
        for (int i = 0; i < n; i++) diagW[i, i] = wD[i];
        var reconstructed = MatMul(MatMul(Vmat, diagW), VT);
        AssertClose(FromRows(reconstructed), new double[,] { { 4, 2 }, { 2, 5 } }, tol: 1e-6);
    }

    [Fact]
    public void Svd_2x2_ReconstructsInput()
    {
        var A = FromRows(new double[,] { { 3, 0 }, { 4, 5 } });
        var (U, S, Vh) = Linalg.Svd(A, fullMatrices: false);
        // Reconstruct A = U · diag(S) · Vh.
        var Umat = ToArray2D(U);
        var VhMat = ToArray2D(Vh);
        int k = S.Shape[0];
        var diagS = new double[k, k];
        for (int i = 0; i < k; i++) diagS[i, i] = S.GetDataArray()[i];
        var reconstructed = MatMul(MatMul(Umat, diagS), VhMat);
        AssertClose(FromRows(reconstructed), new double[,] { { 3, 0 }, { 4, 5 } }, tol: 1e-4);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SOLVERS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void Solve_3x3_MatchesInverseTimesB()
    {
        var A = FromRows(new double[,] { { 3, 1, 1 }, { 1, 3, 1 }, { 1, 1, 3 } });
        var b = new Tensor<double>(new[] { 3 });
        b.GetDataArray()[0] = 5; b.GetDataArray()[1] = 6; b.GetDataArray()[2] = 7;
        var x = Linalg.Solve(A, b);
        // Verify A·x == b.
        var xD = x.GetDataArray();
        var aD = A.GetDataArray();
        for (int i = 0; i < 3; i++)
        {
            double axi = 0;
            for (int j = 0; j < 3; j++) axi += aD[i * 3 + j] * xD[j];
            Assert.True(Math.Abs(axi - b.GetDataArray()[i]) < 1e-8, $"A·x[{i}] = {axi}, expected {b.GetDataArray()[i]}");
        }
    }

    [Fact]
    public void SolveTriangular_Upper_ReturnsCorrect()
    {
        // [[2, 1], [0, 3]] · x = [5, 9]:
        //   3·x₁ = 9 → x₁ = 3
        //   2·x₀ + x₁ = 5 → 2·x₀ = 2 → x₀ = 1
        var A = FromRows(new double[,] { { 2, 1 }, { 0, 3 } });
        var b = new Tensor<double>(new[] { 2 });
        b.GetDataArray()[0] = 5; b.GetDataArray()[1] = 9;
        var x = Linalg.SolveTriangular(A, b, upper: true);
        Assert.Equal(1.0, x.GetDataArray()[0], 6);
        Assert.Equal(3.0, x.GetDataArray()[1], 6);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // INVERSES, SCALARS, NORMS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void Inv_2x2_ReconstructsIdentity()
    {
        var A = FromRows(new double[,] { { 4, 7 }, { 2, 6 } });
        var inv = Linalg.Inv(A);
        var prod = MatMul(ToArray2D(A), ToArray2D(inv));
        AssertClose(FromRows(prod), new double[,] { { 1, 0 }, { 0, 1 } }, tol: 1e-8);
    }

    [Fact]
    public void Det_2x2_MatchesFormula()
    {
        // det([[a, b], [c, d]]) = a·d − b·c.
        var A = FromRows(new double[,] { { 3, 2 }, { 1, 4 } });
        var det = Linalg.Det(A);
        Assert.Equal(10.0, det.GetDataArray()[0], 8);
    }

    [Fact]
    public void SlogDet_2x2_ReturnsSignAndLog()
    {
        var A = FromRows(new double[,] { { 3, 2 }, { 1, 4 } });
        var (sign, logAbs) = Linalg.SlogDet(A);
        Assert.Equal(1.0, sign.GetDataArray()[0], 6);
        Assert.Equal(Math.Log(10.0), logAbs.GetDataArray()[0], 6);
    }

    [Fact]
    public void MatrixRank_FullRank_ReturnsN()
    {
        var A = FromRows(new double[,] { { 2, 0 }, { 0, 3 } });
        var rank = Linalg.MatrixRank(A);
        Assert.Equal(2, rank.GetDataArray()[0]);
    }

    [Fact]
    public void MatrixRank_SingularMatrix_ReturnsLessThanN()
    {
        // [[1, 2], [2, 4]] is rank-1 (second row = 2× first).
        var A = FromRows(new double[,] { { 1, 2 }, { 2, 4 } });
        var rank = Linalg.MatrixRank(A);
        Assert.Equal(1, rank.GetDataArray()[0]);
    }

    [Fact]
    public void VectorNorm_L2_MatchesEuclidean()
    {
        var v = new Tensor<double>(new[] { 3 });
        v.GetDataArray()[0] = 3; v.GetDataArray()[1] = 4; v.GetDataArray()[2] = 0;
        var n = Linalg.VectorNorm(v, 2.0);
        Assert.Equal(5.0, n.GetDataArray()[0], 6);
    }

    [Fact]
    public void VectorNorm_L1_MatchesSumAbs()
    {
        var v = new Tensor<double>(new[] { 3 });
        v.GetDataArray()[0] = -2; v.GetDataArray()[1] = 3; v.GetDataArray()[2] = -1;
        var n = Linalg.VectorNorm(v, 1.0);
        Assert.Equal(6.0, n.GetDataArray()[0], 6);
    }

    [Fact]
    public void VectorNorm_LInf_MatchesMaxAbs()
    {
        var v = new Tensor<double>(new[] { 3 });
        v.GetDataArray()[0] = -2; v.GetDataArray()[1] = 3; v.GetDataArray()[2] = -5;
        var n = Linalg.VectorNorm(v, double.PositiveInfinity);
        Assert.Equal(5.0, n.GetDataArray()[0], 6);
    }

    [Fact]
    public void MatrixNorm_Fro_MatchesSqrtSumSq()
    {
        var A = FromRows(new double[,] { { 1, 2 }, { 3, 4 } });
        var n = Linalg.MatrixNorm(A, "fro");
        Assert.Equal(Math.Sqrt(30.0), n.GetDataArray()[0], 6);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STRUCTURAL
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void MatrixPower_Zero_ReturnsIdentity()
    {
        var A = FromRows(new double[,] { { 2, 3 }, { 1, 4 } });
        var p = Linalg.MatrixPower(A, 0);
        AssertClose(FromRows(ToArray2D(p)), new double[,] { { 1, 0 }, { 0, 1 } });
    }

    [Fact]
    public void MatrixPower_Three_MatchesTripleMatMul()
    {
        var A = FromRows(new double[,] { { 2, 1 }, { 0, 2 } });
        var A2 = MatMul(ToArray2D(A), ToArray2D(A));
        var A3 = MatMul(A2, ToArray2D(A));
        var p = Linalg.MatrixPower(A, 3);
        AssertClose(FromRows(ToArray2D(p)), A3, tol: 1e-8);
    }

    [Fact]
    public void MultiDot_Three_MatchesSequentialMatMul()
    {
        var a = FromRows(new double[,] { { 1, 2 } });        // 1×2
        var b = FromRows(new double[,] { { 3, 4, 5 }, { 6, 7, 8 } }); // 2×3
        var c = FromRows(new double[,] { { 1 }, { 2 }, { 3 } });     // 3×1

        var result = Linalg.MultiDot(new List<Tensor<double>> { a, b, c });

        // Reference: a·b·c = [15, 18, 21] · c = 15+36+63 = 114
        var ab = MatMul(ToArray2D(a), ToArray2D(b));
        var abc = MatMul(ab, ToArray2D(c));
        AssertClose(FromRows(ToArray2D(result)), abc, tol: 1e-8);
    }

    [Fact]
    public void Cross_3d_MatchesFormula()
    {
        // [1,0,0] × [0,1,0] = [0,0,1]
        var a = new Tensor<double>(new[] { 3 });
        var b = new Tensor<double>(new[] { 3 });
        a.GetDataArray()[0] = 1;
        b.GetDataArray()[1] = 1;
        var c = Linalg.Cross(a, b);
        var cd = c.GetDataArray();
        Assert.Equal(0.0, cd[0], 6);
        Assert.Equal(0.0, cd[1], 6);
        Assert.Equal(1.0, cd[2], 6);
    }

    [Fact]
    public void Vander_MatchesFormula()
    {
        var x = new Tensor<double>(new[] { 3 });
        x.GetDataArray()[0] = 1; x.GetDataArray()[1] = 2; x.GetDataArray()[2] = 3;
        var v = Linalg.Vander(x, n: 3, increasing: false);
        // Expected (descending): [[1,1,1],[4,2,1],[9,3,1]]
        AssertClose(FromRows(ToArray2D(v)),
            new double[,] { { 1, 1, 1 }, { 4, 2, 1 }, { 9, 3, 1 } });
    }

    [Fact]
    public void VecDot_MatchesSum()
    {
        var a = new Tensor<double>(new[] { 3 });
        var b = new Tensor<double>(new[] { 3 });
        a.GetDataArray()[0] = 1; a.GetDataArray()[1] = 2; a.GetDataArray()[2] = 3;
        b.GetDataArray()[0] = 4; b.GetDataArray()[1] = 5; b.GetDataArray()[2] = 6;
        var d = Linalg.VecDot(a, b);
        Assert.Equal(32.0, d.GetDataArray()[0], 6);
    }

    [Fact]
    public void MatrixExp_Zero_ReturnsIdentity()
    {
        var A = FromRows(new double[,] { { 0, 0 }, { 0, 0 } });
        var e = Linalg.MatrixExp(A);
        AssertClose(FromRows(ToArray2D(e)),
            new double[,] { { 1, 0 }, { 0, 1 } }, tol: 1e-8);
    }

    [Fact]
    public void MatrixExp_DiagonalInput_MatchesElementwiseExp()
    {
        var A = FromRows(new double[,] { { 1, 0 }, { 0, 2 } });
        var e = Linalg.MatrixExp(A);
        AssertClose(FromRows(ToArray2D(e)),
            new double[,] { { Math.E, 0 }, { 0, Math.E * Math.E } }, tol: 1e-4);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ITERATIVE SOLVERS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void CG_SpdSystem_ConvergesToSolution()
    {
        var A = FromRows(new double[,] { { 4, 1 }, { 1, 3 } });
        var b = new Tensor<double>(new[] { 2 });
        b.GetDataArray()[0] = 1; b.GetDataArray()[1] = 2;
        var x = Linalg.CG(A, b, maxIter: 100, tol: 1e-10);
        // Verify A·x ≈ b.
        var xD = x.GetDataArray();
        double r0 = 4 * xD[0] + 1 * xD[1] - 1;
        double r1 = 1 * xD[0] + 3 * xD[1] - 2;
        Assert.True(Math.Abs(r0) < 1e-6);
        Assert.True(Math.Abs(r1) < 1e-6);
    }

    [Fact]
    public void GMRES_NonSymmetric_ConvergesToSolution()
    {
        var A = FromRows(new double[,] { { 2, 1 }, { 1, 3 } });
        var b = new Tensor<double>(new[] { 2 });
        b.GetDataArray()[0] = 3; b.GetDataArray()[1] = 4;
        var x = Linalg.GMRES(A, b, maxIter: 100, tol: 1e-10);
        var xD = x.GetDataArray();
        double r0 = 2 * xD[0] + 1 * xD[1] - 3;
        double r1 = 1 * xD[0] + 3 * xD[1] - 4;
        Assert.True(Math.Abs(r0) < 1e-6);
        Assert.True(Math.Abs(r1) < 1e-6);
    }

    [Fact]
    public void BiCGSTAB_ConvergesToSolution()
    {
        var A = FromRows(new double[,] { { 4, 1 }, { 1, 3 } });
        var b = new Tensor<double>(new[] { 2 });
        b.GetDataArray()[0] = 1; b.GetDataArray()[1] = 2;
        var x = Linalg.BiCGSTAB(A, b, maxIter: 100, tol: 1e-10);
        var xD = x.GetDataArray();
        double r0 = 4 * xD[0] + 1 * xD[1] - 1;
        double r1 = 1 * xD[0] + 3 * xD[1] - 2;
        Assert.True(Math.Abs(r0) < 1e-6);
        Assert.True(Math.Abs(r1) < 1e-6);
    }
}
