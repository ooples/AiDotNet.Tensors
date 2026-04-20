// Copyright (c) AiDotNet. All rights reserved.
// Tensor-level FFT API tests. The kernel is already covered by
// FftKernelsInvariantsTests — these exercise the surface wiring:
// batching, axis handling, real ↔ complex conversion, fftshift, fftfreq.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Fft = AiDotNet.Tensors.LinearAlgebra.Fft.Fft;
using FftNorm = AiDotNet.Tensors.LinearAlgebra.Fft.FftNorm;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.FftTests;

public class FftApiTests
{
    // ── 1D complex round-trip on a batched tensor ───────────────────────────
    [Fact]
    public void Fft1_RoundTrip_Batched()
    {
        const int batch = 4;
        const int n = 8;
        var input = MakeBatchedComplex(batch, n, seed: 1);
        var X = Fft.Fft1(input);
        var back = Fft.IFft1(X);
        AssertClose(input, back, tol: 1e-10);
    }

    // ── RFFT → IRFFT round-trip ─────────────────────────────────────────────
    [Theory]
    [InlineData(8)]
    [InlineData(32)]
    [InlineData(100)] // non-power-of-2 → Bluestein
    public void RFft_IRFft_RoundTrip(int n)
    {
        // Build a batched real tensor, shape [3, n].
        const int batch = 3;
        var x = new Tensor<double>(new[] { batch, n });
        var data = x.GetDataArray();
        var rng = new Random(123);
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 2 - 1;

        var X = Fft.RFft(x);
        Assert.Equal(batch, X.Shape[0]);
        Assert.Equal(2 * (n / 2 + 1), X.Shape[1]);

        var back = Fft.IRFft(X, n);
        Assert.Equal(batch, back.Shape[0]);
        Assert.Equal(n, back.Shape[1]);
        AssertClose(x, back, tol: 1e-9);
    }

    // ── Ortho norm is unitary (both sides have the same scale) ──────────────
    [Fact]
    public void Ortho_Norm_Unitary()
    {
        const int n = 32;
        var input = MakeBatchedComplex(1, n, seed: 7);
        var X1 = Fft.Fft1(input, norm: FftNorm.Ortho);
        var back1 = Fft.IFft1(X1, norm: FftNorm.Ortho);
        AssertClose(input, back1, tol: 1e-10);

        // Parseval in ortho: Σ|x|² == Σ|X|².
        double eTime = EnergySquared(input);
        double eFreq = EnergySquared(X1);
        Assert.True(Math.Abs(eTime - eFreq) < 1e-10 * Math.Max(1, eTime));
    }

    // ── fftshift / ifftshift round-trip is identity ─────────────────────────
    [Theory]
    [InlineData(8)]
    [InlineData(9)] // odd length
    public void FftShift_IFftShift_Identity(int n)
    {
        // Real-tensor semantics: treat the last axis as regular (not complex-interleaved).
        var x = new Tensor<double>(new[] { n });
        var data = x.GetDataArray();
        for (int i = 0; i < n; i++) data[i] = i;
        var shifted = Fft.FftShift(x, lastAxisIsComplex: false);
        var back = Fft.IFftShift(shifted, lastAxisIsComplex: false);
        var backData = back.GetDataArray();
        for (int i = 0; i < n; i++)
            Assert.Equal(i, backData[i], precision: 10);
    }

    // ── fftshift actually moves DC to the center ────────────────────────────
    [Fact]
    public void FftShift_Even_MovesDcToCenter()
    {
        var x = new Tensor<double>(new[] { 8 });
        var d = x.GetDataArray();
        for (int i = 0; i < 8; i++) d[i] = i; // [0,1,2,3,4,5,6,7]
        var s = Fft.FftShift(x, lastAxisIsComplex: false);
        var sd = s.GetDataArray();
        // After fftshift: [4,5,6,7,0,1,2,3] — zero-freq index (0) lands at position 4.
        double[] expected = { 4, 5, 6, 7, 0, 1, 2, 3 };
        for (int i = 0; i < 8; i++) Assert.Equal(expected[i], sd[i], precision: 10);
    }

    // ── fftshift preserves re/im pairing for odd-N complex (Critical fix) ───
    [Fact]
    public void FftShift_Complex_OddN_PreservesReImPairing()
    {
        // 5 complex bins → 10 doubles interleaved re/im. After fftshift with
        // lastAxisIsComplex=true, the result should be a pair-rotation (every
        // pair stays intact), NOT a 5-double rotation that splits pairs.
        int n = 5;
        var x = new Tensor<double>(new[] { 2 * n });
        var d = x.GetDataArray();
        // Re_k = k, Im_k = 100 + k so we can spot any re/im split.
        for (int k = 0; k < n; k++) { d[2 * k] = k; d[2 * k + 1] = 100 + k; }

        var s = Fft.FftShift(x); // default lastAxisIsComplex = true
        var sd = s.GetDataArray();

        // For n=5 complex, fftshift rotates by n/2 = 2 complex pairs → [2,3,4,0,1].
        for (int k = 0; k < n; k++)
        {
            int sourceK = (k + (n - n / 2)) % n; // inverse of n/2 rotation
            // Re at position k came from logical index (k - 2 mod 5)
            double expectedRe = (k + n - n / 2) % n;
            double expectedIm = 100 + (k + n - n / 2) % n;
            Assert.Equal(expectedRe, sd[2 * k], precision: 10);
            Assert.Equal(expectedIm, sd[2 * k + 1], precision: 10);
        }
    }

    // ── fftfreq conventions match numpy ──────────────────────────────────────
    [Fact]
    public void FftFreq_Even()
    {
        var f = Fft.FftFreq<double>(8, d: 1.0);
        // numpy.fft.fftfreq(8): [0, 1, 2, 3, -4, -3, -2, -1] / 8
        double[] expected = { 0, 1, 2, 3, -4, -3, -2, -1 };
        for (int i = 0; i < 8; i++) Assert.Equal(expected[i] / 8.0, f[i], precision: 10);
    }

    [Fact]
    public void FftFreq_Odd()
    {
        var f = Fft.FftFreq<double>(7, d: 1.0);
        // numpy.fft.fftfreq(7): [0, 1, 2, 3, -3, -2, -1] / 7
        double[] expected = { 0, 1, 2, 3, -3, -2, -1 };
        for (int i = 0; i < 7; i++) Assert.Equal(expected[i] / 7.0, f[i], precision: 10);
    }

    [Fact]
    public void RFftFreq()
    {
        var f = Fft.RFftFreq<double>(8, d: 1.0);
        // Length 5: [0, 1, 2, 3, 4] / 8
        Assert.Equal(5, f.Shape[0]);
        for (int i = 0; i < 5; i++) Assert.Equal(i / 8.0, f[i], precision: 10);
    }

    // ── 2D FFT round-trip ──────────────────────────────────────────────────
    [Fact]
    public void Fft2_RoundTrip()
    {
        // Complex tensor shape [H, 2W] for W complex columns.
        const int H = 8, W = 8;
        var x = new Tensor<double>(new[] { H, 2 * W });
        var d = x.GetDataArray();
        var rng = new Random(321);
        for (int i = 0; i < d.Length; i++) d[i] = rng.NextDouble() * 2 - 1;

        var X = Fft.Fft2(x);
        Assert.Equal(new[] { H, 2 * W }, X.Shape.ToArray());
        var back = Fft.IFft2(X);
        AssertClose(x, back, tol: 1e-9);
    }

    // ── 2D RFFT round-trip ─────────────────────────────────────────────────
    [Fact]
    public void RFft2_IRFft2_RoundTrip()
    {
        const int H = 8, W = 16;
        var x = new Tensor<double>(new[] { H, W });
        var d = x.GetDataArray();
        var rng = new Random(444);
        for (int i = 0; i < d.Length; i++) d[i] = rng.NextDouble() * 2 - 1;

        var X = Fft.RFft2(x);
        Assert.Equal(new[] { H, 2 * (W / 2 + 1) }, X.Shape.ToArray());
        var back = Fft.IRFft2(X, s: new[] { H, W });
        AssertClose(x, back, tol: 1e-9);
    }

    // ── HFFT ↔ IHFFT round trip ────────────────────────────────────────────
    [Fact]
    public void HFft_IHFft_RoundTrip_Real()
    {
        const int n = 16;
        var x = new Tensor<double>(new[] { n });
        var d = x.GetDataArray();
        var rng = new Random(555);
        for (int i = 0; i < n; i++) d[i] = rng.NextDouble() * 2 - 1;

        var Xc = Fft.IHFft(x);                 // real → Hermitian complex
        var back = Fft.HFft(Xc, n: n);         // Hermitian → real
        AssertClose(x, back, tol: 1e-9);
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    private static Tensor<double> MakeBatchedComplex(int batch, int n, int seed)
    {
        var t = new Tensor<double>(new[] { batch, 2 * n });
        var d = t.GetDataArray();
        var rng = new Random(seed);
        for (int i = 0; i < d.Length; i++) d[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static double EnergySquared(Tensor<double> t)
    {
        var d = t.GetDataArray();
        double s = 0;
        for (int i = 0; i < d.Length; i += 2) s += d[i] * d[i] + d[i + 1] * d[i + 1];
        return s;
    }

    private static void AssertClose(Tensor<double> a, Tensor<double> b, double tol)
    {
        Assert.Equal(a.Length, b.Length);
        var ad = a.GetDataArray();
        var bd = b.GetDataArray();
        double maxErr = 0;
        for (int i = 0; i < ad.Length; i++)
        {
            double e = Math.Abs(ad[i] - bd[i]);
            if (e > maxErr) maxErr = e;
        }
        Assert.True(maxErr < tol, $"max error {maxErr} > tol {tol}");
    }
}
