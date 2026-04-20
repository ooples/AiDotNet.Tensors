// Copyright (c) AiDotNet. All rights reserved.
// Parity tests: compare our output against known analytical/reference values.
// Every golden value here is hand-computed or pulled from numpy.fft / scipy —
// citations are in each test's comment.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Fft;
using Xunit;
using Fft = AiDotNet.Tensors.LinearAlgebra.Fft.Fft;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.FftTests;

public class FftParityTests
{
    // ── np.fft.fft([1, 2, 3, 4]) == [10, -2+2j, -2, -2-2j] ─────────────────
    [Fact]
    public void Fft_N4_MatchesNumpyGolden()
    {
        var x = new Tensor<double>(new[] { 8 }); // 4 complex = 8 interleaved doubles
        var d = x.GetDataArray();
        d[0] = 1; d[1] = 0;
        d[2] = 2; d[3] = 0;
        d[4] = 3; d[5] = 0;
        d[6] = 4; d[7] = 0;
        var X = Fft.Fft1(x);
        var r = X.GetDataArray();
        double[] expectedRe = { 10, -2, -2, -2 };
        double[] expectedIm = { 0, 2, 0, -2 };
        for (int k = 0; k < 4; k++)
        {
            Assert.Equal(expectedRe[k], r[2 * k], precision: 10);
            Assert.Equal(expectedIm[k], r[2 * k + 1], precision: 10);
        }
    }

    // ── np.fft.rfft([1, 0, -1, 0]) == [0, 2, 0] ────────────────────────────
    // (Signal is cos(πn/2) at n=0..3 → peak at k=1 bin.)
    [Fact]
    public void RFft_Cosine_MatchesNumpyGolden()
    {
        var x = new Tensor<double>(new[] { 4 });
        var d = x.GetDataArray();
        d[0] = 1; d[1] = 0; d[2] = -1; d[3] = 0;
        var X = Fft.RFft(x);
        var r = X.GetDataArray();
        // Expect 3 complex bins = 6 doubles: [0,0, 2,0, 0,0]
        double[] expected = { 0, 0, 2, 0, 0, 0 };
        for (int i = 0; i < 6; i++)
            Assert.Equal(expected[i], r[i], precision: 10);
    }

    // ── Ortho norm on length-4 [1,2,3,4]: forward scale = 1/2 ──────────────
    [Fact]
    public void Fft_N4_Ortho_MatchesGolden()
    {
        var x = new Tensor<double>(new[] { 8 });
        var d = x.GetDataArray();
        d[0] = 1; d[2] = 2; d[4] = 3; d[6] = 4;
        var X = Fft.Fft1(x, norm: FftNorm.Ortho);
        var r = X.GetDataArray();
        // Expected from backward-norm FFT divided by √4 = 2.
        double[] expectedRe = { 5, -1, -1, -1 };
        double[] expectedIm = { 0, 1, 0, -1 };
        for (int k = 0; k < 4; k++)
        {
            Assert.Equal(expectedRe[k], r[2 * k], precision: 10);
            Assert.Equal(expectedIm[k], r[2 * k + 1], precision: 10);
        }
    }

    // ── Window functions match hand-computed values ────────────────────────
    [Fact]
    public void Hann_Periodic_N4()
    {
        // Hann periodic of length 4: 0.5·(1 − cos(2πn/4)) at n=0..3.
        var w = Windows.Hann<double>(4, periodic: true);
        double[] expected = { 0.0, 0.5, 1.0, 0.5 };
        for (int i = 0; i < 4; i++)
            Assert.Equal(expected[i], w[i], precision: 10);
    }

    [Fact]
    public void Hann_Symmetric_N5()
    {
        // Hann symmetric of length 5: 0.5·(1 − cos(2πn/4)) at n=0..4.
        var w = Windows.Hann<double>(5, periodic: false);
        double[] expected = { 0.0, 0.5, 1.0, 0.5, 0.0 };
        for (int i = 0; i < 5; i++)
            Assert.Equal(expected[i], w[i], precision: 10);
    }

    [Fact]
    public void Hamming_Periodic_N4()
    {
        // Hamming periodic of length 4: 0.54 − 0.46·cos(2πn/4) at n=0..3.
        var w = Windows.Hamming<double>(4, periodic: true);
        double[] expected = { 0.08, 0.54, 1.0, 0.54 };
        for (int i = 0; i < 4; i++)
            Assert.Equal(expected[i], w[i], precision: 10);
    }

    [Fact]
    public void Bartlett_Symmetric_N5()
    {
        // Bartlett symmetric length 5: 1 − |2n − 4|/4 at n=0..4 = [0, 0.5, 1, 0.5, 0].
        var w = Windows.Bartlett<double>(5, periodic: false);
        double[] expected = { 0.0, 0.5, 1.0, 0.5, 0.0 };
        for (int i = 0; i < 5; i++)
            Assert.Equal(expected[i], w[i], precision: 10);
    }

    [Fact]
    public void Kaiser_Beta_Zero_IsConstant()
    {
        // Kaiser(β=0) = constant 1 (I₀(0) = 1).
        var w = Windows.Kaiser<double>(8, beta: 0.0, periodic: false);
        for (int i = 0; i < 8; i++)
            Assert.Equal(1.0, w[i], precision: 10);
    }

    [Fact]
    public void BesselI0_KnownValues()
    {
        // I₀(0) = 1, I₀(1) ≈ 1.2660658778, I₀(5) ≈ 27.239871823.
        Assert.Equal(1.0, Windows.BesselI0(0.0), precision: 10);
        Assert.Equal(1.2660658777520084, Windows.BesselI0(1.0), precision: 10);
        Assert.Equal(27.239871823604442, Windows.BesselI0(5.0), precision: 8);
    }

    // ── Pure sinusoid spectral peaks match known amplitude ─────────────────
    // x[n] = cos(2πk₀n/N) for real input: RFFT magnitude at bin k₀ is N/2
    // (by Euler, cos splits into two conjugate exponentials of magnitude N/2
    // each — but for real FFT the negative half is folded, leaving N/2 at k₀).
    [Fact]
    public void RFft_PureCosine_PeakMagnitude()
    {
        int n = 64;
        int k0 = 5;
        var x = new Tensor<double>(new[] { n });
        var d = x.GetDataArray();
        for (int i = 0; i < n; i++) d[i] = Math.Cos(2.0 * Math.PI * k0 * i / n);
        var X = Fft.RFft(x);
        var r = X.GetDataArray();

        for (int k = 0; k <= n / 2; k++)
        {
            double mag = Math.Sqrt(r[2 * k] * r[2 * k] + r[2 * k + 1] * r[2 * k + 1]);
            double expected = (k == k0) ? n / 2.0 : 0.0;
            Assert.True(Math.Abs(mag - expected) < 1e-9 * n,
                $"cosine peak bin {k}: mag={mag}, expected={expected}");
        }
    }
}
