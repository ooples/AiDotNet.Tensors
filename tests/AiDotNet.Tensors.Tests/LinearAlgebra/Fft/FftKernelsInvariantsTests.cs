// Copyright (c) AiDotNet. All rights reserved.
// Algebraic / analytical invariants that MUST hold for any correct DFT kernel,
// independent of implementation. Failures here mean the numeric core is wrong —
// no amount of higher-level API polish can paper over them.
//
// Coverage:
//   - Round trip:   IFFT(FFT(x)) == x   (all 3 norm modes, pow-of-2 AND Bluestein lengths)
//   - Linearity:    FFT(a·x + b·y) == a·FFT(x) + b·FFT(y)
//   - Parseval:     Σ|x|² == (1/N) Σ|X|²  (backward norm) or  Σ|X|²  (ortho norm)
//   - Shift:        time-shift by m ↔ multiply X[k] by e^{−2πi k m / N}
//   - Real-input:   X[N−k] == conj(X[k])
//   - DC bin:       X[0] == Σ x[n]
//   - Analytical:   delta → flat spectrum;  single sinusoid → single bin

using System;
using AiDotNet.Tensors.LinearAlgebra.Fft;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.Fft;

public class FftKernelsInvariantsTests
{
    // Include non-power-of-2 sizes to exercise Bluestein.
    public static TheoryData<int> TransformLengths => new()
    {
        { 2 }, { 4 }, { 8 }, { 16 }, { 64 }, { 256 },          // radix-2
        { 3 }, { 5 }, { 7 }, { 15 }, { 17 }, { 100 }, { 255 }, // Bluestein
    };

    // ── Round trip ──────────────────────────────────────────────────────────
    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void RoundTrip_Backward(int n)
    {
        var x = MakeRandomComplex(n, seed: 42);
        var y = (double[])x.Clone();
        FftKernels.Transform1D(y, n, inverse: false, FftNorm.Backward);
        FftKernels.Transform1D(y, n, inverse: true, FftNorm.Backward);
        AssertClose(x, y, tol: 1e-10, $"backward round-trip n={n}");
    }

    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void RoundTrip_Forward(int n)
    {
        var x = MakeRandomComplex(n, seed: 43);
        var y = (double[])x.Clone();
        FftKernels.Transform1D(y, n, inverse: false, FftNorm.Forward);
        FftKernels.Transform1D(y, n, inverse: true, FftNorm.Forward);
        AssertClose(x, y, tol: 1e-10, $"forward round-trip n={n}");
    }

    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void RoundTrip_Ortho(int n)
    {
        var x = MakeRandomComplex(n, seed: 44);
        var y = (double[])x.Clone();
        FftKernels.Transform1D(y, n, inverse: false, FftNorm.Ortho);
        FftKernels.Transform1D(y, n, inverse: true, FftNorm.Ortho);
        AssertClose(x, y, tol: 1e-10, $"ortho round-trip n={n}");
    }

    // ── Linearity ──────────────────────────────────────────────────────────
    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void Linearity(int n)
    {
        var x = MakeRandomComplex(n, seed: 100);
        var y = MakeRandomComplex(n, seed: 101);
        double a = 0.3, b = -1.7;

        // Left side: FFT(a·x + b·y)
        var left = new double[2 * n];
        for (int i = 0; i < 2 * n; i++) left[i] = a * x[i] + b * y[i];
        FftKernels.Transform1D(left, n, inverse: false, FftNorm.Backward);

        // Right side: a·FFT(x) + b·FFT(y)
        var Fx = (double[])x.Clone();
        var Fy = (double[])y.Clone();
        FftKernels.Transform1D(Fx, n, inverse: false, FftNorm.Backward);
        FftKernels.Transform1D(Fy, n, inverse: false, FftNorm.Backward);
        var right = new double[2 * n];
        for (int i = 0; i < 2 * n; i++) right[i] = a * Fx[i] + b * Fy[i];

        AssertClose(left, right, tol: 1e-10, $"linearity n={n}");
    }

    // ── Parseval's theorem ─────────────────────────────────────────────────
    // Backward norm: Σ|x[n]|² = (1/N) Σ|X[k]|²
    // Ortho norm:    Σ|x[n]|² = Σ|X[k]|² (unitary)
    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void Parseval_Backward(int n)
    {
        var x = MakeRandomComplex(n, seed: 200);
        double energyTime = Magnitude2Sum(x);
        var X = (double[])x.Clone();
        FftKernels.Transform1D(X, n, inverse: false, FftNorm.Backward);
        double energyFreq = Magnitude2Sum(X);
        Assert.True(Math.Abs(energyTime - energyFreq / n) < 1e-10 * Math.Max(1, energyTime),
            $"Parseval backward failed at n={n}: time={energyTime}, freq/N={energyFreq / n}");
    }

    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void Parseval_Ortho(int n)
    {
        var x = MakeRandomComplex(n, seed: 201);
        double energyTime = Magnitude2Sum(x);
        var X = (double[])x.Clone();
        FftKernels.Transform1D(X, n, inverse: false, FftNorm.Ortho);
        double energyFreq = Magnitude2Sum(X);
        Assert.True(Math.Abs(energyTime - energyFreq) < 1e-10 * Math.Max(1, energyTime),
            $"Parseval ortho failed at n={n}: time={energyTime}, freq={energyFreq}");
    }

    // ── Shift theorem ──────────────────────────────────────────────────────
    // Circular time shift by m ↔ frequency-domain multiplication by e^{-2πi k m / N}.
    [Theory]
    [InlineData(8, 3)]
    [InlineData(16, 5)]
    [InlineData(15, 4)]   // Bluestein path
    [InlineData(100, 27)] // Bluestein path
    public void TimeShift_To_PhaseRotation(int n, int m)
    {
        var x = MakeRandomComplex(n, seed: 300);

        // Apply circular time shift: y[n] = x[(n - m) mod N]
        var y = new double[2 * n];
        for (int i = 0; i < n; i++)
        {
            int src = ((i - m) % n + n) % n;
            y[2 * i] = x[2 * src];
            y[2 * i + 1] = x[2 * src + 1];
        }

        var Fx = (double[])x.Clone();
        FftKernels.Transform1D(Fx, n, inverse: false, FftNorm.Backward);
        var Fy = (double[])y.Clone();
        FftKernels.Transform1D(Fy, n, inverse: false, FftNorm.Backward);

        // Expected: Fy[k] = Fx[k] * e^{-2πi k m / N}.
        var expected = new double[2 * n];
        for (int k = 0; k < n; k++)
        {
            double theta = -2.0 * Math.PI * k * m / n;
            double pRe = Math.Cos(theta);
            double pIm = Math.Sin(theta);
            double xRe = Fx[2 * k];
            double xIm = Fx[2 * k + 1];
            expected[2 * k] = xRe * pRe - xIm * pIm;
            expected[2 * k + 1] = xRe * pIm + xIm * pRe;
        }
        AssertClose(Fy, expected, tol: 1e-10, $"shift theorem n={n}, m={m}");
    }

    // ── Conjugate symmetry for real input ──────────────────────────────────
    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void RealInput_ConjugateSymmetric(int n)
    {
        var x = new double[2 * n];
        var rng = new Random(400);
        for (int i = 0; i < n; i++)
        {
            x[2 * i] = rng.NextDouble() * 2 - 1;
            x[2 * i + 1] = 0.0; // strictly real
        }
        FftKernels.Transform1D(x, n, inverse: false, FftNorm.Backward);

        // X[N-k] == conj(X[k]) for k = 1..N-1.
        for (int k = 1; k < n; k++)
        {
            int mirror = n - k;
            double reK = x[2 * k], imK = x[2 * k + 1];
            double reM = x[2 * mirror], imM = x[2 * mirror + 1];
            Assert.True(Math.Abs(reK - reM) < 1e-10,
                $"real-input Re mismatch at k={k} (n={n}): {reK} vs {reM}");
            Assert.True(Math.Abs(imK + imM) < 1e-10,
                $"real-input Im mismatch at k={k} (n={n}): {imK} vs {-imM}");
        }
    }

    // ── DC bin equals sum ──────────────────────────────────────────────────
    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void DcBin_EqualsSum(int n)
    {
        var x = MakeRandomComplex(n, seed: 500);
        double sumRe = 0, sumIm = 0;
        for (int i = 0; i < n; i++)
        {
            sumRe += x[2 * i];
            sumIm += x[2 * i + 1];
        }
        var X = (double[])x.Clone();
        FftKernels.Transform1D(X, n, inverse: false, FftNorm.Backward);
        Assert.True(Math.Abs(X[0] - sumRe) < 1e-10, $"DC Re mismatch n={n}");
        Assert.True(Math.Abs(X[1] - sumIm) < 1e-10, $"DC Im mismatch n={n}");
    }

    // ── Analytical: Kronecker delta → flat spectrum ────────────────────────
    [Theory]
    [MemberData(nameof(TransformLengths))]
    public void Delta_To_FlatSpectrum(int n)
    {
        var x = new double[2 * n];
        x[0] = 1.0;
        FftKernels.Transform1D(x, n, inverse: false, FftNorm.Backward);
        for (int k = 0; k < n; k++)
        {
            Assert.True(Math.Abs(x[2 * k] - 1.0) < 1e-10, $"delta@k={k} Re mismatch n={n}");
            Assert.True(Math.Abs(x[2 * k + 1]) < 1e-10, $"delta@k={k} Im mismatch n={n}");
        }
    }

    // ── Analytical: complex exponential → single bin ───────────────────────
    // x[n] = e^{2πi · k0 · n / N}  ⇒  X[k] = N · δ[k − k0]
    [Theory]
    [InlineData(8, 3)]
    [InlineData(16, 7)]
    [InlineData(17, 5)]   // Bluestein path
    [InlineData(100, 25)] // Bluestein path
    public void Sinusoid_To_SingleBin(int n, int k0)
    {
        var x = new double[2 * n];
        for (int i = 0; i < n; i++)
        {
            double theta = 2.0 * Math.PI * k0 * i / n;
            x[2 * i] = Math.Cos(theta);
            x[2 * i + 1] = Math.Sin(theta);
        }
        FftKernels.Transform1D(x, n, inverse: false, FftNorm.Backward);
        for (int k = 0; k < n; k++)
        {
            double mag = Math.Sqrt(x[2 * k] * x[2 * k] + x[2 * k + 1] * x[2 * k + 1]);
            double expected = (k == k0) ? (double)n : 0.0;
            Assert.True(Math.Abs(mag - expected) < 1e-9 * n,
                $"sinusoid spike at k={k} (expected bin {k0}, n={n}): {mag} vs {expected}");
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    private static double[] MakeRandomComplex(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new double[2 * n];
        for (int i = 0; i < 2 * n; i++) x[i] = rng.NextDouble() * 2 - 1;
        return x;
    }

    private static double Magnitude2Sum(double[] x)
    {
        double s = 0;
        for (int i = 0; i < x.Length; i += 2)
        {
            s += x[i] * x[i] + x[i + 1] * x[i + 1];
        }
        return s;
    }

    private static void AssertClose(double[] a, double[] b, double tol, string context)
    {
        Assert.Equal(a.Length, b.Length);
        double maxAbsErr = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double err = Math.Abs(a[i] - b[i]);
            if (err > maxAbsErr) maxAbsErr = err;
        }
        Assert.True(maxAbsErr < tol,
            $"{context}: maxAbsErr={maxAbsErr}, tol={tol}");
    }
}
