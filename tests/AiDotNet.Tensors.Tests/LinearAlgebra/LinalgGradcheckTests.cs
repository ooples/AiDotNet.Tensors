using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Gradcheck tests for <see cref="Linalg"/> backward functions — compares each
/// analytical gradient against a numerical finite-difference reference. This
/// is the acceptance criterion from issue #211: "Gradcheck passes for all
/// differentiable variants".
///
/// <para>The tests invoke <see cref="LinalgBackward"/> directly rather than
/// routing through a <see cref="GradientTape{T}"/> so they work as focused
/// unit checks on the math. A separate integration test set exercises the
/// same backward through the tape.</para>
/// </summary>
public class LinalgGradcheckTests
{
    private const double Eps = 1e-5;
    private const double Tolerance = 1e-3;

    // ── Finite-difference helper ────────────────────────────────────────────

    /// <summary>
    /// Computes ∂f/∂A[i,j] via central differences. Returned shape matches A.
    /// </summary>
    private static Tensor<double> NumericalGradient(
        Func<Tensor<double>, Tensor<double>> f, Tensor<double> A)
    {
        var shape = (int[])A._shape.Clone();
        var grad = new Tensor<double>(shape);
        var gradData = grad.GetDataArray();
        var aData = A.GetDataArray();
        for (int idx = 0; idx < aData.Length; idx++)
        {
            double orig = aData[idx];
            aData[idx] = orig + Eps;
            var fPlus = f(A).GetDataArray()[0];
            aData[idx] = orig - Eps;
            var fMinus = f(A).GetDataArray()[0];
            aData[idx] = orig;
            gradData[idx] = (fPlus - fMinus) / (2 * Eps);
        }
        return grad;
    }

    private static void AssertTensorClose(Tensor<double> a, Tensor<double> b, double tol)
    {
        Assert.Equal(a.Length, b.Length);
        var aD = a.GetDataArray();
        var bD = b.GetDataArray();
        for (int i = 0; i < a.Length; i++)
        {
            double diff = Math.Abs(aD[i] - bD[i]);
            double rel = diff / (Math.Max(Math.Abs(aD[i]), Math.Abs(bD[i])) + 1e-12);
            Assert.True(diff < tol || rel < tol,
                $"mismatch at [{i}]: analytical={aD[i]}, numerical={bD[i]}, diff={diff}");
        }
    }

    private static Tensor<double> MakeSpdMatrix(int n, int seed)
    {
        var rng = new Random(seed);
        var M = new Tensor<double>(new[] { n, n });
        var d = M.GetDataArray();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                d[i * n + j] = rng.NextDouble() - 0.5;
        // A = M · Mᵀ + n·I  →  SPD
        var A = new Tensor<double>(new[] { n, n });
        var ad = A.GetDataArray();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double s = i == j ? n : 0;
                for (int k = 0; k < n; k++) s += d[i * n + k] * d[j * n + k];
                ad[i * n + j] = s;
            }
        return A;
    }

    private static Tensor<double> Ones(int[] shape)
    {
        var t = new Tensor<double>(shape);
        var d = t.GetDataArray();
        for (int i = 0; i < d.Length; i++) d[i] = 1.0;
        return t;
    }

    // ── Gradcheck tests ─────────────────────────────────────────────────────

    [Fact]
    public void Det_GradMatchesNumerical()
    {
        var A = MakeSpdMatrix(3, seed: 42);
        // analytical: d det/dA = det·A⁻ᵀ
        var det = Linalg.Det(A);
        var grads = new Dictionary<Tensor<double>, Tensor<double>>();
        var gradOut = new Tensor<double>(new[] { 1 });
        gradOut.GetDataArray()[0] = 1.0;
        LinalgBackward.DetBackward<double>()(gradOut, new[] { A }, det, Array.Empty<object>(),
            new CpuEngine(), grads);
        var analytical = grads[A];

        var numerical = NumericalGradient(a => Linalg.Det(a), A);
        AssertTensorClose(analytical, numerical, Tolerance);
    }

    [Fact]
    public void SlogDet_GradMatchesNumerical()
    {
        var A = MakeSpdMatrix(3, seed: 17);
        var (sign, logAbs) = Linalg.SlogDet(A);
        var grads = new Dictionary<Tensor<double>, Tensor<double>>();
        var gradOut = new Tensor<double>(new[] { 1 });
        gradOut.GetDataArray()[0] = 1.0;
        LinalgBackward.SlogDetBackward<double>()(gradOut, new[] { A }, logAbs, Array.Empty<object>(),
            new CpuEngine(), grads);
        var analytical = grads[A];

        var numerical = NumericalGradient(a => Linalg.SlogDet(a).LogAbsDet, A);
        AssertTensorClose(analytical, numerical, Tolerance);
    }

    [Fact]
    public void Inv_GradMatchesNumerical()
    {
        // Scalar objective: sum of inverse. grad(sum(inv(A))) = -A⁻ᵀ · 1 · A⁻ᵀ (per element).
        var A = MakeSpdMatrix(3, seed: 7);
        var invA = Linalg.Inv(A);
        int n = A.Shape[0];

        var grads = new Dictionary<Tensor<double>, Tensor<double>>();
        // gradOutput is d(sum(invA))/d(invA) = ones_like(invA).
        var gradOut = Ones(new[] { n, n });
        LinalgBackward.InvBackward<double>()(gradOut, new[] { A }, invA, Array.Empty<object>(),
            new CpuEngine(), grads);
        var analytical = grads[A];

        var numerical = NumericalGradient(a =>
        {
            var inv = Linalg.Inv(a);
            double s = 0;
            foreach (var v in inv.GetDataArray()) s += v;
            var r = new Tensor<double>(new[] { 1 });
            r.GetDataArray()[0] = s;
            return r;
        }, A);
        AssertTensorClose(analytical, numerical, Tolerance);
    }

    [Fact]
    public void Solve_GradA_MatchesNumerical()
    {
        var A = MakeSpdMatrix(3, seed: 99);
        var b = new Tensor<double>(new[] { 3 });
        b.GetDataArray()[0] = 1; b.GetDataArray()[1] = 2; b.GetDataArray()[2] = 3;

        var x = Linalg.Solve(A, b);
        var grads = new Dictionary<Tensor<double>, Tensor<double>>();
        var gradOut = Ones(new[] { 3 });
        LinalgBackward.SolveBackward<double>()(gradOut, new[] { A, b }, x, Array.Empty<object>(),
            new CpuEngine(), grads);
        var gradA = grads[A];

        var numericalA = NumericalGradient(a =>
        {
            var sol = Linalg.Solve(a, b);
            double s = 0;
            foreach (var v in sol.GetDataArray()) s += v;
            var r = new Tensor<double>(new[] { 1 });
            r.GetDataArray()[0] = s;
            return r;
        }, A);
        AssertTensorClose(gradA, numericalA, Tolerance);
    }

    [Fact]
    public void MatrixPower_GradMatchesNumerical()
    {
        // A² gradient: grad(sum(A²))/dA = A + Aᵀ  for symmetric A.
        var A = new Tensor<double>(new[] { 3, 3 });
        var ad = A.GetDataArray();
        var rng = new Random(13);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) ad[i * 3 + j] = rng.NextDouble() - 0.5;

        int power = 2;
        var result = Linalg.MatrixPower(A, power);
        var grads = new Dictionary<Tensor<double>, Tensor<double>>();
        var gradOut = Ones(new[] { 3, 3 });
        LinalgBackward.MatrixPowerBackward<double>()(gradOut, new[] { A }, result,
            new object[] { power }, new CpuEngine(), grads);
        var analytical = grads[A];

        var numerical = NumericalGradient(a =>
        {
            var p = Linalg.MatrixPower(a, power);
            double s = 0;
            foreach (var v in p.GetDataArray()) s += v;
            var r = new Tensor<double>(new[] { 1 });
            r.GetDataArray()[0] = s;
            return r;
        }, A);
        AssertTensorClose(analytical, numerical, Tolerance);
    }

    [Fact]
    public void VectorNormL2_GradMatchesNumerical()
    {
        var v = new Tensor<double>(new[] { 4 });
        var d = v.GetDataArray();
        d[0] = 1; d[1] = -2; d[2] = 3; d[3] = 0.5;

        var norm = Linalg.VectorNorm(v, 2.0);
        var grads = new Dictionary<Tensor<double>, Tensor<double>>();
        var gradOut = Ones(new[] { 1 });
        LinalgBackward.VectorNormL2Backward<double>()(gradOut, new[] { v }, norm, Array.Empty<object>(),
            new CpuEngine(), grads);
        var analytical = grads[v];

        var numerical = NumericalGradient(a => Linalg.VectorNorm(a, 2.0), v);
        AssertTensorClose(analytical, numerical, Tolerance);
    }
}
