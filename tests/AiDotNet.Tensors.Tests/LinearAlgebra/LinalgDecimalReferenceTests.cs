using System;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Reference tests for the decimal (generic <c>INumericOperations&lt;T&gt;</c>) Linalg kernels.
/// <para>
/// Expected values were computed with mpmath 1.4.1 at 50 significant digits (mp.dps = 50:
/// mp.cholesky, mp.det, mp.inverse, mp.eigsy, mp.eig, mp.svd_r, mp.qr) and rounded to 28
/// significant digits. The QR R factor was cross-checked against NumPy 2.5.4 numpy.linalg.qr
/// (LAPACK geqrf sign convention: R[i,i] = −sign(a)·‖a‖), which mpmath's qr matches.
/// The partial-pivot LU reference follows the getrf rule (first row of maximal |a|).
/// </para>
/// <para>
/// Tolerances: direct factorizations 1e-24, iterative ones (Jacobi, Francis QR, one-sided Jacobi
/// SVD) 1e-21. Both are far below double precision (≈1e-16), so they prove the computation runs in
/// decimal rather than through a double round trip.
/// </para>
/// </summary>
public class LinalgDecimalReferenceTests
{
    private const decimal DirectTol = 1e-24m;
    private const decimal IterTol = 1e-21m;

    private static readonly decimal[] AValues = { 4m, 1m, 2m, 1m, 5m, 3m, 2m, 3m, 6m };
    private static readonly decimal[] BValues = { 1m, 2m, 2m, -1m, 3m, 0.5m };
    private static readonly decimal[] GValues = { 1m, 2m, 0m, -3m, 1m, 1m, 0m, 1m, 2m };
    private static readonly decimal[] LowerValues = { 2m, 0m, 0m, 1m, 3m, 0m, 2m, 1m, 4m };
    private static readonly decimal[] R43Values = { 1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 10m, 1m, 0m, 1m };

    private static readonly decimal[] CholeskyL = { 2.0m, 0.0m, 0.0m, 0.5m, 2.179449471770336776118490992m, 0.0m, 1.0m, 1.14707866935280882953604789m, 1.919429739874786214802666347m };
    private static readonly decimal[] InvA = { 0.3m, 0.0m, -0.1m, 0.0m, 0.2857142857142857142857142857m, -0.1428571428571428571428571429m, -0.1m, -0.1428571428571428571428571429m, 0.2714285714285714285714285714m };
    private static readonly decimal[] SolveAB = { 0.0m, 0.55m, 0.1428571428571428571428571429m, -0.3571428571428571428571428571m, 0.4285714285714285714285714286m, 0.07857142857142857142857142857m };
    private static readonly decimal[] LuPackedA = { 4.0m, 1.0m, 2.0m, 0.25m, 4.75m, 2.5m, 0.5m, 0.5263157894736842105263157895m, 3.684210526315789473684210526m };
    private static readonly decimal[] EighW = { 2.19439716742240855460793064m, 3.386770156607549226057975534m, 9.418832675970042219334093825m };
    private static readonly decimal[] EigGRe = { 0.921139595569813022600602355m, 0.921139595569813022600602355m, 2.15772080886037395479879529m };
    private static readonly decimal[] EigGIm = { 2.275165422342088055924332205m, -2.275165422342088055924332205m, 0.0m };
    private static readonly decimal[] SvdR43 = { 17.45089558463664042628199024m, 0.9869391657365875151929653046m, 0.7015656613921296935525897352m };
    private static readonly decimal[] SvdA = { 9.418832675970042219334093825m, 3.386770156607549226057975534m, 2.19439716742240855460793064m };
    private static readonly decimal[] QrR43 = { -8.185352771872449969953703725m, -9.529216659791807427707296874m, -11.97260554691791189635019351m, 0.0m, 1.481225793303056364487900445m, 1.28977484042715112009830787m, 0.0m, 0.0m, 0.9965928350693499872721494708m };
    private static readonly decimal[] TriLowerSolve = { 0.5m, 1.0m, 0.5m, -0.6666666666666666666666666667m, 0.375m, -0.2083333333333333333333333333m };
    private static readonly decimal[] CholSolveLT = { 0.08854166666666666666666666667m, 0.6545138888888888888888888889m, 0.1354166666666666666666666667m, -0.2048611111111111111111111111m, 0.09375m, -0.05208333333333333333333333333m };
    private static readonly decimal[] CholInvLT = { 0.3211805555555555555555555556m, -0.03819444444444444444444444444m, -0.05208333333333333333333333333m, -0.03819444444444444444444444444m, 0.1180555555555555555555555556m, -0.02083333333333333333333333333m, -0.05208333333333333333333333333m, -0.02083333333333333333333333333m, 0.0625m };

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static Tensor<decimal> T(int rows, int cols, decimal[] values)
    {
        var t = new Tensor<decimal>(new[] { rows, cols });
        Array.Copy(values, t.GetDataArray(), values.Length);
        return t;
    }

    private static void AssertClose(decimal[] expected, decimal[] actual, decimal tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"{what}[{i}]: expected {expected[i]}, actual {actual[i]}, |diff| = {Math.Abs(expected[i] - actual[i])} > {tol}");
    }

    private static decimal[] MatMul(decimal[] a, decimal[] b, int m, int k, int n)
    {
        var r = new decimal[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                decimal s = 0m;
                for (int l = 0; l < k; l++) s += a[i * k + l] * b[l * n + j];
                r[i * n + j] = s;
            }
        return r;
    }

    private static decimal[] Transpose(decimal[] a, int m, int n)
    {
        var r = new decimal[n * m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) r[j * m + i] = a[i * n + j];
        return r;
    }

    private static decimal[] Identity(int n)
    {
        var r = new decimal[n * n];
        for (int i = 0; i < n; i++) r[i * n + i] = 1m;
        return r;
    }

    // ── Direct factorizations ───────────────────────────────────────────────

    [Fact]
    public void CholeskyEx_MatchesMpmath()
    {
        var (l, info) = Linalg.CholeskyEx(T(3, 3, AValues));
        Assert.Equal(0, info.GetDataArray()[0]);
        AssertClose(CholeskyL, l.GetDataArray(), DirectTol, "L");
    }

    [Fact]
    public void CholeskyEx_NotPositiveDefinite_ReportsLapackInfo()
    {
        // [[1,2],[2,1]] has a negative second pivot: potrf info = 2.
        var (_, info) = Linalg.CholeskyEx(T(2, 2, new[] { 1m, 2m, 2m, 1m }));
        Assert.Equal(2, info.GetDataArray()[0]);
    }

    [Fact]
    public void Det_MatchesMpmath()
        => AssertClose(new[] { 70m }, Linalg.Det(T(3, 3, AValues)).GetDataArray(), DirectTol, "det");

    [Fact]
    public void Inv_MatchesMpmath()
        => AssertClose(InvA, Linalg.Inv(T(3, 3, AValues)).GetDataArray(), DirectTol, "inv");

    [Fact]
    public void Solve_And_SolveEx_MatchMpmath()
    {
        var a = T(3, 3, AValues);
        var b = T(3, 2, BValues);
        AssertClose(SolveAB, Linalg.Solve(a, b).GetDataArray(), DirectTol, "solve");
        var (x, info) = Linalg.SolveEx(a, b);
        Assert.Equal(0, info.GetDataArray()[0]);
        AssertClose(SolveAB, x.GetDataArray(), DirectTol, "solveEx");
    }

    [Fact]
    public void SingularSystem_SolveExReportsInfo_SolveThrows()
    {
        // getrf on [[1,2],[2,4]] pivots row 1 first and leaves U[1,1] = 0: info = 2.
        var a = T(2, 2, new[] { 1m, 2m, 2m, 4m });
        var b = T(2, 1, new[] { 1m, 1m });
        var (_, info) = Linalg.SolveEx(a, b);
        Assert.Equal(2, info.GetDataArray()[0]);
        Assert.Throws<ArgumentException>(() => Linalg.Solve(a, b));
    }

    [Fact]
    public void LuFactor_LuSolve_And_LU_MatchMpmath()
    {
        var a = T(3, 3, AValues);
        var b = T(3, 2, BValues);
        var (lu, piv) = Linalg.LuFactor(a);
        AssertClose(LuPackedA, lu.GetDataArray(), DirectTol, "LU");
        Assert.Equal(new[] { 0, 1, 2 }, piv.GetDataArray());
        AssertClose(SolveAB, Linalg.LuSolve(lu, piv, b).GetDataArray(), DirectTol, "luSolve");

        var (p, l, u) = Linalg.LU(a);
        var pa = MatMul(p.GetDataArray(), AValues, 3, 3, 3);
        AssertClose(pa, MatMul(l.GetDataArray(), u.GetDataArray(), 3, 3, 3), DirectTol, "PA - LU");
    }

    [Fact]
    public void LuFactor_PivotsLikeGetrf()
    {
        // Column 0 max is row 2 (|7|): getrf pivot[0] = 2; P·A = L·U must hold.
        var a = T(3, 3, new[] { 1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 10m });
        var (lu, piv) = Linalg.LuFactor(a);
        Assert.Equal(2, piv.GetDataArray()[0]);
        var (p, l, u) = Linalg.LU(a);
        AssertClose(MatMul(p.GetDataArray(), a.GetDataArray(), 3, 3, 3),
            MatMul(l.GetDataArray(), u.GetDataArray(), 3, 3, 3), DirectTol, "PA - LU");
        Assert.Equal(7m, lu.GetDataArray()[0]);
    }

    [Fact]
    public void SolveTriangular_CholeskySolve_CholeskyInverse_MatchMpmath()
    {
        var l = T(3, 3, LowerValues);
        var b = T(3, 2, BValues);
        AssertClose(TriLowerSolve, Linalg.SolveTriangular(l, b, upper: false).GetDataArray(), DirectTol, "trsm");
        AssertClose(CholSolveLT, Linalg.CholeskySolve(b, l).GetDataArray(), DirectTol, "cholesky_solve");
        AssertClose(CholInvLT, Linalg.CholeskyInverse(l).GetDataArray(), DirectTol, "cholesky_inverse");
    }

    [Fact]
    public void Diagonal_ReturnsDecimalDiagonal()
        => Assert.Equal(new[] { 4m, 5m, 6m }, Linalg.Diagonal(T(3, 3, AValues)).GetDataArray());

    // ── Iterative factorizations ────────────────────────────────────────────

    [Fact]
    public void QR_MatchesMpmathAndLapackSign()
    {
        var a = T(4, 3, R43Values);
        var (q, r) = Linalg.QR(a);
        Assert.Equal(new[] { 4, 3 }, q.Shape.ToArray());
        AssertClose(QrR43, r.GetDataArray(), IterTol, "R");
        AssertClose(R43Values, MatMul(q.GetDataArray(), r.GetDataArray(), 4, 3, 3), IterTol, "QR - A");
        var qd = q.GetDataArray();
        AssertClose(Identity(3), MatMul(Transpose(qd, 4, 3), qd, 3, 4, 3), IterTol, "QtQ");

        var (qc, rc) = Linalg.QR(a, "complete");
        var qcd = qc.GetDataArray();
        AssertClose(Identity(4), MatMul(Transpose(qcd, 4, 4), qcd, 4, 4, 4), IterTol, "complete QtQ");
        AssertClose(R43Values, MatMul(qcd, rc.GetDataArray(), 4, 4, 3), IterTol, "complete QR - A");
    }

    [Fact]
    public void Eigh_MatchesMpmath_AscendingOrthonormal()
    {
        var (w, v) = Linalg.Eigh(T(3, 3, AValues));
        AssertClose(EighW, w.GetDataArray(), IterTol, "w");
        var vd = v.GetDataArray();
        AssertClose(Identity(3), MatMul(Transpose(vd, 3, 3), vd, 3, 3, 3), IterTol, "VtV");
        var av = MatMul(AValues, vd, 3, 3, 3);
        var vl = new decimal[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) vl[i * 3 + j] = vd[i * 3 + j] * w.GetDataArray()[j];
        AssertClose(av, vl, IterTol, "AV - VΛ");
        // Upper-triangle mode reads the same symmetric matrix.
        AssertClose(EighW, Linalg.Eigh(T(3, 3, AValues), upper: true).Eigenvalues.GetDataArray(), IterTol, "w upper");
    }

    [Fact]
    public void Eig_ComplexPair_MatchesMpmath()
    {
        var (w, v) = Linalg.Eig(T(3, 3, GValues));
        Assert.Equal(new[] { 3, 2 }, w.Shape.ToArray());
        Assert.Equal(new[] { 3, 3, 2 }, v.Shape.ToArray());
        var wd = w.GetDataArray();
        var vd = v.GetDataArray();

        var order = Enumerable.Range(0, 3).OrderBy(i => wd[i * 2]).ThenByDescending(i => wd[i * 2 + 1]).ToArray();
        AssertClose(EigGRe, order.Select(i => wd[i * 2]).ToArray(), IterTol, "Re λ");
        AssertClose(EigGIm, order.Select(i => wd[i * 2 + 1]).ToArray(), IterTol, "Im λ");

        // LAPACK geev convention: the positive-imaginary member of a conjugate pair comes first.
        for (int k = 0; k < 2; k++)
            if (wd[k * 2 + 1] != 0m)
            {
                Assert.True(wd[k * 2 + 1] > 0m, "positive imaginary part first");
                Assert.Equal(-wd[k * 2 + 1], wd[(k + 1) * 2 + 1]);
                break;
            }

        // A·(xr + i·xi) = (λr + i·λi)(xr + i·xi), each vector of unit 2-norm.
        for (int k = 0; k < 3; k++)
        {
            decimal lr = wd[k * 2], li = wd[k * 2 + 1], norm2 = 0m;
            for (int r = 0; r < 3; r++)
            {
                decimal axr = 0m, axi = 0m;
                for (int c = 0; c < 3; c++)
                {
                    axr += GValues[r * 3 + c] * vd[(c * 3 + k) * 2];
                    axi += GValues[r * 3 + c] * vd[(c * 3 + k) * 2 + 1];
                }
                decimal xr = vd[(r * 3 + k) * 2], xi = vd[(r * 3 + k) * 2 + 1];
                Assert.True(Math.Abs(axr - (lr * xr - li * xi)) <= IterTol, $"Re residual k={k} r={r}");
                Assert.True(Math.Abs(axi - (li * xr + lr * xi)) <= IterTol, $"Im residual k={k} r={r}");
                norm2 += xr * xr + xi * xi;
            }
            Assert.True(Math.Abs(norm2 - 1m) <= IterTol, $"|v{k}| = {norm2}");
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Svd_Tall_MatchesMpmath_AndReconstructs(bool fullMatrices)
    {
        var (u, s, vh) = Linalg.Svd(T(4, 3, R43Values), fullMatrices);
        int uCols = fullMatrices ? 4 : 3;
        Assert.Equal(new[] { 4, uCols }, u.Shape.ToArray());
        Assert.Equal(new[] { 3, 3 }, vh.Shape.ToArray());
        AssertClose(SvdR43, s.GetDataArray(), IterTol, "S");
        AssertSvd(R43Values, 4, 3, u.GetDataArray(), uCols, s.GetDataArray(), vh.GetDataArray(), 3);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Svd_Wide_MatchesMpmath_AndReconstructs(bool fullMatrices)
    {
        var at = Transpose(R43Values, 4, 3);
        var (u, s, vh) = Linalg.Svd(T(3, 4, at), fullMatrices);
        int vRows = fullMatrices ? 4 : 3;
        Assert.Equal(new[] { 3, 3 }, u.Shape.ToArray());
        Assert.Equal(new[] { vRows, 4 }, vh.Shape.ToArray());
        AssertClose(SvdR43, s.GetDataArray(), IterTol, "S");
        AssertSvd(at, 3, 4, u.GetDataArray(), 3, s.GetDataArray(), vh.GetDataArray(), vRows);
    }

    [Fact]
    public void Svd_SymmetricPositiveDefinite_SingularValuesAreEigenvalues()
        => AssertClose(SvdA, Linalg.Svd(T(3, 3, AValues)).S.GetDataArray(), IterTol, "S");

    private static void AssertSvd(decimal[] a, int m, int n, decimal[] u, int uCols, decimal[] s, decimal[] vh, int vRows)
    {
        int k = Math.Min(m, n);
        for (int i = 1; i < k; i++) Assert.True(s[i - 1] >= s[i], "singular values descending");
        AssertClose(Identity(uCols), MatMul(Transpose(u, m, uCols), u, uCols, m, uCols), IterTol, "UtU");
        AssertClose(Identity(vRows), MatMul(vh, Transpose(vh, vRows, n), vRows, n, vRows), IterTol, "VhVht");
        var us = new decimal[m * k];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++) us[i * k + j] = u[i * uCols + j] * s[j];
        var vhk = new decimal[k * n];
        Array.Copy(vh, vhk, k * n);
        AssertClose(a, MatMul(us, vhk, m, k, n), IterTol, "USVh - A");
    }

    // ── Cross-check against the float64 path ────────────────────────────────

    [Fact]
    public void Decimal_AgreesWithFloat64Path()
    {
        var ad = new Tensor<double>(new[] { 3, 3 });
        for (int i = 0; i < 9; i++) ad.GetDataArray()[i] = (double)AValues[i];
        var a = T(3, 3, AValues);
        void Near(double[] d, decimal[] m, string what)
        {
            Assert.Equal(d.Length, m.Length);
            for (int i = 0; i < d.Length; i++)
                Assert.True(Math.Abs(d[i] - (double)m[i]) <= 1e-9, $"{what}[{i}]: float64 {d[i]} vs decimal {m[i]}");
        }
        Near(Linalg.Eigh(ad).Eigenvalues.GetDataArray(), Linalg.Eigh(a).Eigenvalues.GetDataArray(), "eigh");
        Near(Linalg.Svd(ad).S.GetDataArray(), Linalg.Svd(a).S.GetDataArray(), "svd");
        Near(Linalg.Det(ad).GetDataArray(), Linalg.Det(a).GetDataArray(), "det");
        Near(Linalg.Inv(ad).GetDataArray(), Linalg.Inv(a).GetDataArray(), "inv");
        Near(Linalg.QR(ad).R.GetDataArray(), Linalg.QR(a).R.GetDataArray(), "qr R");
        Near(Linalg.Cholesky(ad).GetDataArray(), Linalg.Cholesky(a).GetDataArray(), "cholesky");
    }

    // ── Narrow floating types (same generic kernels) ───────────────────────

    [Fact]
    public void BFloat16_Eigh_And_Det_WithinBFloat16Precision()
    {
        var t = new Tensor<BFloat16>(new[] { 3, 3 });
        for (int i = 0; i < 9; i++) t.GetDataArray()[i] = BFloat16.FromFloat((float)AValues[i]);
        var w = Linalg.Eigh(t).Eigenvalues.GetDataArray();
        for (int i = 0; i < 3; i++)
            Assert.True(Math.Abs((double)(float)w[i] - (double)EighW[i]) <= 0.1, $"bf16 w[{i}] = {(float)w[i]}");
        Assert.True(Math.Abs((float)Linalg.Det(t).GetDataArray()[0] - 70f) <= 1f);
    }

    [Fact]
    public void Half_Eigh_And_Solve_WithinHalfPrecision()
    {
        var t = new Tensor<Half>(new[] { 3, 3 });
        for (int i = 0; i < 9; i++) t.GetDataArray()[i] = (Half)(float)AValues[i];
        var w = Linalg.Eigh(t).Eigenvalues.GetDataArray();
        for (int i = 0; i < 3; i++)
            Assert.True(Math.Abs((double)(float)w[i] - (double)EighW[i]) <= 0.02, $"half w[{i}] = {(float)w[i]}");
        var b = new Tensor<Half>(new[] { 3, 2 });
        for (int i = 0; i < 6; i++) b.GetDataArray()[i] = (Half)(float)BValues[i];
        var x = Linalg.Solve(t, b).GetDataArray();
        for (int i = 0; i < 6; i++)
            Assert.True(Math.Abs((double)(float)x[i] - (double)SolveAB[i]) <= 0.01, $"half x[{i}] = {(float)x[i]}");
    }
}
