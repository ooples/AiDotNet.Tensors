// Copyright (c) AiDotNet. All rights reserved.
// STFT / ISTFT integration tests: end-to-end roundtrip reconstruction,
// FFT-based convolution matches direct convolution, and windowed overlap-add
// reconstructs exactly for COLA-compliant window/hop pairs.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Fft;
using Xunit;
using Fft = AiDotNet.Tensors.LinearAlgebra.Fft.Fft;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.FftTests;

public class StftIntegrationTests
{
    // ── COLA (constant-overlap-add) reconstruction ─────────────────────────
    // Hann window with hop = nFft / 4 is COLA — STFT → ISTFT must recover the
    // original signal exactly (up to interior samples; edge tolerance looser
    // because reflection-pad interacts with windowing).
    [Theory]
    [InlineData(256, 64, 1.0)]
    [InlineData(512, 128, 1.0)]
    [InlineData(256, 32, 2.0)] // wider hop within COLA; still reconstructs at 32/256
    public void Stft_Hann_Cola_Reconstruction(int nFft, int hop, double _)
    {
        int n = 4000;
        var x = MakeSignal(n, seed: 42);
        var window = Windows.Hann<double>(nFft, periodic: true);

        var S = Stft.Forward(x, nFft, hop, winLength: nFft, window, center: true, PadMode.Reflect);
        var y = Stft.Inverse(S, nFft, hop, winLength: nFft, window, center: true, length: n);

        // Interior region reconstructs to high precision; edge region (first /
        // last nFft samples) is slightly affected by reflection padding.
        int edge = nFft;
        AssertCloseInterior(x, y, edge, tol: 1e-6, $"nFft={nFft} hop={hop}");
    }

    // ── FFT-based convolution matches direct convolution ───────────────────
    // For length-N signal a and length-M filter b, (a ⊛ b) via
    // IRFFT(RFFT(a_pad) · RFFT(b_pad)) at length N+M-1 should match direct
    // linear convolution.
    [Theory]
    [InlineData(16, 4)]
    [InlineData(63, 15)]    // non-pow2 → Bluestein
    [InlineData(128, 32)]
    public void FftConv_Matches_DirectConv(int n, int m)
    {
        var rng = new Random(1);
        double[] a = new double[n];
        double[] b = new double[m];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < m; i++) b[i] = rng.NextDouble() * 2 - 1;

        // Direct linear convolution.
        int cLen = n + m - 1;
        double[] direct = new double[cLen];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                direct[i + j] += a[i] * b[j];

        // FFT-based: pad both to cLen, RFFT, multiply, IRFFT.
        var aPad = new Tensor<double>(new[] { cLen });
        var bPad = new Tensor<double>(new[] { cLen });
        var aD = aPad.GetDataArray();
        var bD = bPad.GetDataArray();
        Array.Copy(a, aD, n);
        Array.Copy(b, bD, m);

        var A = Fft.RFft(aPad);
        var B = Fft.RFft(bPad);
        // Multiply complex RFFTs element-wise (stored as interleaved re/im).
        var aDat = A.GetDataArray();
        var bDat = B.GetDataArray();
        var prodT = new Tensor<double>((int[])A._shape.Clone());
        var prodD = prodT.GetDataArray();
        for (int i = 0; i < aDat.Length; i += 2)
        {
            double aRe = aDat[i], aIm = aDat[i + 1];
            double brRe = bDat[i], brIm = bDat[i + 1];
            prodD[i] = aRe * brRe - aIm * brIm;
            prodD[i + 1] = aRe * brIm + aIm * brRe;
        }
        var viaFft = Fft.IRFft(prodT, cLen);
        var viaD = viaFft.GetDataArray();

        double maxErr = 0;
        for (int i = 0; i < cLen; i++)
        {
            double e = Math.Abs(direct[i] - viaD[i]);
            if (e > maxErr) maxErr = e;
        }
        Assert.True(maxErr < 1e-9, $"FFT-conv vs direct max error {maxErr} > 1e-9 for n={n}, m={m}");
    }

    // ── Spectral energy match (Parseval) through STFT ──────────────────────
    // Summing |STFT|² · 1/N over all freqs and frames must equal the time-
    // domain energy (up to windowing normalization).
    [Fact]
    public void Stft_Normalized_Preserves_Energy()
    {
        int n = 1024;
        int nFft = 256;
        int hop = nFft; // no overlap → each sample contributes exactly once (given rectangular window)
        var x = MakeSignal(n, seed: 99);
        var window = new Tensor<double>(new[] { nFft });
        var wd = window.GetDataArray();
        for (int i = 0; i < nFft; i++) wd[i] = 1.0; // rectangular, exact energy accounting

        var S = Stft.Forward(x, nFft, hop, winLength: nFft, window, center: false, PadMode.Constant, normalized: true, onesided: false);

        // Parseval per frame: Σ_k |X[k, f]|² == Σ_i |x_windowed[i, f]|² (normalized: both sides scaled by 1/N).
        double energyFreq = 0;
        var sd = S.GetDataArray();
        for (int i = 0; i < sd.Length; i += 2)
            energyFreq += sd[i] * sd[i] + sd[i + 1] * sd[i + 1];

        double energyTime = 0;
        int frames = 1 + Math.Max(0, (n - nFft) / hop);
        var xd = x.GetDataArray();
        for (int f = 0; f < frames; f++)
        {
            int start = f * hop;
            for (int i = 0; i < nFft && start + i < n; i++)
                energyTime += xd[start + i] * xd[start + i];
        }

        // normalized=true scales each frame's spectrum by 1/√N, so Σ|X|² == Σ|x|².
        Assert.True(Math.Abs(energyTime - energyFreq) < 1e-9 * Math.Max(1, energyTime),
            $"Parseval: time={energyTime}, freq={energyFreq}");
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    private static Tensor<double> MakeSignal(int n, int seed)
    {
        var x = new Tensor<double>(new[] { n });
        var d = x.GetDataArray();
        var rng = new Random(seed);
        for (int i = 0; i < n; i++) d[i] = rng.NextDouble() * 2 - 1;
        return x;
    }

    private static void AssertCloseInterior(Tensor<double> expected, Tensor<double> actual, int edge, double tol, string context)
    {
        var ed = expected.GetDataArray();
        var ad = actual.GetDataArray();
        Assert.Equal(ed.Length, ad.Length);
        double maxErr = 0;
        for (int i = edge; i < ed.Length - edge; i++)
        {
            double e = Math.Abs(ed[i] - ad[i]);
            if (e > maxErr) maxErr = e;
        }
        Assert.True(maxErr < tol, $"{context}: interior max error {maxErr} > tol {tol}");
    }
}
