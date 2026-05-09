// PR #321 / #319 follow-up: numerical correctness of double-precision
// FusedOptimizer kernels. Each Adam/AdamW/SGD update is run via the new
// SIMD double path AND a scalar reference, then compared per-element
// to confirm the SIMD path's reduction order produces matching results
// at float64 precision tolerances.

using System;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class FusedOptimizerDoubleTests
{
    private const double Tolerance = 1e-12;

    [Fact]
    public unsafe void SgdUpdateSimd_Double_MatchesScalarReference()
    {
        const int N = 64; // multiple of 16 to exercise unrolled path
        const double Lr = 0.01;
        var rng = new Random(42);
        var paramSimd = new double[N];
        var paramScalar = new double[N];
        var grad = new double[N];
        for (int i = 0; i < N; i++)
        {
            double p = (rng.NextDouble() - 0.5) * 0.1;
            paramSimd[i] = paramScalar[i] = p;
            grad[i] = (rng.NextDouble() - 0.5) * 0.05;
        }
        fixed (double* pP = paramSimd, pG = grad)
            FusedOptimizer.SgdUpdateSimd(pP, pG, N, Lr);
        for (int i = 0; i < N; i++) paramScalar[i] -= Lr * grad[i];
        for (int i = 0; i < N; i++)
            Assert.True(Math.Abs(paramSimd[i] - paramScalar[i]) < Tolerance,
                $"[{i}] SIMD={paramSimd[i]:F15} scalar={paramScalar[i]:F15}");
    }

    [Fact]
    public unsafe void AdamUpdateSimd_Double_MatchesScalarReference()
    {
        const int N = 64;
        const double Lr = 0.001, B1 = 0.9, B2 = 0.999, Eps = 1e-8;
        const int Step = 5;
        var rng = new Random(42);
        var paramSimd = new double[N]; var paramScalar = new double[N];
        var grad = new double[N];
        var mSimd = new double[N]; var mScalar = new double[N];
        var vSimd = new double[N]; var vScalar = new double[N];
        for (int i = 0; i < N; i++)
        {
            double p = (rng.NextDouble() - 0.5) * 0.1;
            paramSimd[i] = paramScalar[i] = p;
            grad[i] = (rng.NextDouble() - 0.5) * 0.05;
            // Pre-populate m/v with prior-step values (Adam state isn't fresh).
            mSimd[i] = mScalar[i] = (rng.NextDouble() - 0.5) * 0.001;
            vSimd[i] = vScalar[i] = rng.NextDouble() * 0.001;
        }

        fixed (double* pP = paramSimd, pG = grad, pM = mSimd, pV = vSimd)
            FusedOptimizer.AdamUpdateSimd(pP, pG, pM, pV, N, Lr, B1, B2, Eps, Step);

        // Scalar reference matching the inner loop of AdamUpdateSimd.
        double bc1 = 1.0 - Math.Pow(B1, Step);
        double bc2 = 1.0 - Math.Pow(B2, Step);
        for (int i = 0; i < N; i++)
        {
            mScalar[i] = B1 * mScalar[i] + (1.0 - B1) * grad[i];
            vScalar[i] = B2 * vScalar[i] + (1.0 - B2) * grad[i] * grad[i];
            double mHat = mScalar[i] / bc1;
            double vHat = vScalar[i] / bc2;
            paramScalar[i] -= Lr * mHat / (Math.Sqrt(vHat) + Eps);
        }

        for (int i = 0; i < N; i++)
        {
            Assert.True(Math.Abs(paramSimd[i] - paramScalar[i]) < Tolerance,
                $"param[{i}] SIMD={paramSimd[i]:F15} scalar={paramScalar[i]:F15}");
            Assert.True(Math.Abs(mSimd[i] - mScalar[i]) < Tolerance,
                $"m[{i}] SIMD={mSimd[i]:F15} scalar={mScalar[i]:F15}");
            Assert.True(Math.Abs(vSimd[i] - vScalar[i]) < Tolerance,
                $"v[{i}] SIMD={vSimd[i]:F15} scalar={vScalar[i]:F15}");
        }
    }

    [Fact]
    public unsafe void AdamWUpdateSimd_Double_AppliesWeightDecayThenAdam()
    {
        const int N = 32;
        const double Lr = 0.001, B1 = 0.9, B2 = 0.999, Eps = 1e-8, Wd = 0.01;
        const int Step = 3;
        var rng = new Random(42);
        var paramSimd = new double[N]; var paramScalar = new double[N];
        var grad = new double[N];
        var mSimd = new double[N]; var mScalar = new double[N];
        var vSimd = new double[N]; var vScalar = new double[N];
        for (int i = 0; i < N; i++)
        {
            double p = (rng.NextDouble() - 0.5) * 0.1;
            paramSimd[i] = paramScalar[i] = p;
            grad[i] = (rng.NextDouble() - 0.5) * 0.05;
        }

        fixed (double* pP = paramSimd, pG = grad, pM = mSimd, pV = vSimd)
            FusedOptimizer.AdamWUpdateSimd(pP, pG, pM, pV, N, Lr, B1, B2, Eps, Wd, Step);

        // Scalar reference: weight-decay multiplicative update, then Adam.
        for (int i = 0; i < N; i++) paramScalar[i] *= (1.0 - Wd * Lr);
        double bc1 = 1.0 - Math.Pow(B1, Step);
        double bc2 = 1.0 - Math.Pow(B2, Step);
        for (int i = 0; i < N; i++)
        {
            mScalar[i] = B1 * mScalar[i] + (1.0 - B1) * grad[i];
            vScalar[i] = B2 * vScalar[i] + (1.0 - B2) * grad[i] * grad[i];
            double mHat = mScalar[i] / bc1;
            double vHat = vScalar[i] / bc2;
            paramScalar[i] -= Lr * mHat / (Math.Sqrt(vHat) + Eps);
        }
        for (int i = 0; i < N; i++)
            Assert.True(Math.Abs(paramSimd[i] - paramScalar[i]) < Tolerance,
                $"param[{i}] SIMD={paramSimd[i]:F15} scalar={paramScalar[i]:F15}");
    }

    [Fact]
    public unsafe void Double_KernelHandlesNonAlignedTail()
    {
        // Length not a multiple of 4 — exercise the scalar-tail path.
        const int N = 13;
        var rng = new Random(7);
        var p1 = new double[N]; var p2 = new double[N]; var g = new double[N];
        for (int i = 0; i < N; i++) { p1[i] = p2[i] = rng.NextDouble(); g[i] = rng.NextDouble(); }
        fixed (double* pP = p1, pG = g)
            FusedOptimizer.SgdUpdateSimd(pP, pG, N, 0.1);
        for (int i = 0; i < N; i++) p2[i] -= 0.1 * g[i];
        for (int i = 0; i < N; i++)
            Assert.True(Math.Abs(p1[i] - p2[i]) < Tolerance, $"[{i}] {p1[i]} vs {p2[i]}");
    }
}
