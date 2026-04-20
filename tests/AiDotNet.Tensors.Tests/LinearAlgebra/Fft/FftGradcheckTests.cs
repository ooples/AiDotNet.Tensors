// Copyright (c) AiDotNet. All rights reserved.
// Gradient check via direct closed-form verification against finite-difference
// directional derivatives.
//
// Strategy: for each op y = f(x) with analytic backward rule grad_x = g(grad_y),
// we pick a fixed "probe vector" w (size of y), define the scalar loss
//   L(x) = Σ wᵢ · yᵢ(x)
// and verify
//   dL/dx · x   (from the analytic backward)  ≈   (L(x·(1+ε)) − L(x·(1−ε))) / (2ε).
// This tests every norm mode in a tape-free way that doesn't require the
// rest of the autograd infrastructure to be linear-loss-aware.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Fft = AiDotNet.Tensors.LinearAlgebra.Fft.Fft;
using FftNorm = AiDotNet.Tensors.LinearAlgebra.Fft.FftNorm;

namespace AiDotNet.Tensors.Tests.LinearAlgebra.FftTests;

public class FftGradcheckTests
{
    // ── Complex FFT backward rule ──────────────────────────────────────────
    [Theory]
    [InlineData(8, FftNorm.Backward)]
    [InlineData(8, FftNorm.Forward)]
    [InlineData(8, FftNorm.Ortho)]
    [InlineData(16, FftNorm.Backward)]
    [InlineData(16, FftNorm.Ortho)]
    public void Fft1_BackwardRule_MatchesFiniteDifference(int n, FftNorm norm)
    {
        var x = MakeRandom(2 * n, seed: 1);
        var w = MakeRandom(2 * n, seed: 2);

        // Analytic: dL/dx = IFft1(w, dualNorm).  Then dot with x gives the directional derivative along x.
        var wT = new Tensor<double>(new[] { 2 * n });
        Array.Copy(w, wT.GetDataArray(), w.Length);
        var analyticGrad = Fft.IFft1(wT, n, DualNorm(norm));
        double analyticDD = Dot(analyticGrad.GetDataArray(), x);

        double numericDD = NumericDirectionalDerivative(x, xPerturbed =>
        {
            var t = new Tensor<double>(new[] { 2 * n });
            Array.Copy(xPerturbed, t.GetDataArray(), xPerturbed.Length);
            var y = Fft.Fft1(t, n, norm);
            return Dot(w, y.GetDataArray());
        });

        Assert.True(Math.Abs(analyticDD - numericDD) < 1e-5 * (1 + Math.Abs(analyticDD)),
            $"Fft1 gradcheck n={n} norm={norm}: analytic={analyticDD}, numeric={numericDD}");
    }

    // ── IFft1 (inverse complex FFT) backward rule ──────────────────────────
    [Theory]
    [InlineData(8, FftNorm.Backward)]
    [InlineData(8, FftNorm.Ortho)]
    [InlineData(16, FftNorm.Forward)]
    public void IFft1_BackwardRule_MatchesFiniteDifference(int n, FftNorm norm)
    {
        var x = MakeRandom(2 * n, seed: 10);
        var w = MakeRandom(2 * n, seed: 11);

        // Analytic: dL/dx = Fft1(w, dualNorm).
        var wT = new Tensor<double>(new[] { 2 * n });
        Array.Copy(w, wT.GetDataArray(), w.Length);
        var analyticGrad = Fft.Fft1(wT, n, DualNorm(norm));
        double analyticDD = Dot(analyticGrad.GetDataArray(), x);

        double numericDD = NumericDirectionalDerivative(x, xPerturbed =>
        {
            var t = new Tensor<double>(new[] { 2 * n });
            Array.Copy(xPerturbed, t.GetDataArray(), xPerturbed.Length);
            var y = Fft.IFft1(t, n, norm);
            return Dot(w, y.GetDataArray());
        });

        Assert.True(Math.Abs(analyticDD - numericDD) < 1e-5 * (1 + Math.Abs(analyticDD)),
            $"IFft1 gradcheck n={n} norm={norm}: analytic={analyticDD}, numeric={numericDD}");
    }

    // ── Real FFT backward rule ─────────────────────────────────────────────
    [Theory]
    [InlineData(8, FftNorm.Backward)]
    [InlineData(8, FftNorm.Ortho)]
    [InlineData(16, FftNorm.Forward)]
    public void RFft_BackwardRule_MatchesFiniteDifference(int n, FftNorm norm)
    {
        var x = MakeRandom(n, seed: 20);
        int outLen = 2 * (n / 2 + 1);
        var w = MakeRandom(outLen, seed: 21);

        // Analytic: dL/dx = IRFft(halveInterior(w), n, dualNorm).
        // The halving compensates for the Hermitian-mirror doubling that
        // IRFft applies on the forward path — a unit change in a packed
        // interior bin only corresponds to a unit change in ONE real gradient
        // direction, not the doubled mirror the unpacking would imply.
        var wHalved = HalveInteriorBins(w, n);
        var wT = new Tensor<double>(new[] { outLen });
        Array.Copy(wHalved, wT.GetDataArray(), wHalved.Length);
        var analyticGrad = Fft.IRFft(wT, n, DualNorm(norm));
        double analyticDD = Dot(analyticGrad.GetDataArray(), x);

        double numericDD = NumericDirectionalDerivative(x, xPerturbed =>
        {
            var t = new Tensor<double>(new[] { n });
            Array.Copy(xPerturbed, t.GetDataArray(), xPerturbed.Length);
            var y = Fft.RFft(t, n, norm);
            return Dot(w, y.GetDataArray());
        });

        // RFft conjugate-symmetric packing: non-edge bins contribute twice to
        // the real-domain dot product when unpacked. We either match by
        // symmetry-aware comparison or accept that analytic vs numeric differ
        // by exactly a factor-of-2 for the interior bins. Easier: use an
        // "even-length Nyquist and DC have factor 1, everything else 2" rule
        // directly in the numeric computation:
        //   dL/dx = IRFft( W_doubled ) where W_doubled scales non-edge bins by 2.
        // But the finite-difference test we wrote uses the user-observed loss
        // L = Σ wᵢ · yᵢ over the PACKED output (without the doubling), so the
        // analytic rule IRFft(w) is actually what maps to that L. No doubling
        // needed — verify below.
        Assert.True(Math.Abs(analyticDD - numericDD) < 1e-5 * (1 + Math.Abs(analyticDD)),
            $"RFft gradcheck n={n} norm={norm}: analytic={analyticDD}, numeric={numericDD}");
    }

    // ── IRFft backward rule ────────────────────────────────────────────────
    [Theory]
    [InlineData(8, FftNorm.Backward)]
    [InlineData(16, FftNorm.Ortho)]
    public void IRFft_BackwardRule_MatchesFiniteDifference(int n, FftNorm norm)
    {
        // Input is (K = n/2+1)-complex = 2K doubles.
        int k = n / 2 + 1;
        var x = MakeRandomHermitian(n, seed: 30);  // Hermitian-symmetric complex input
        int inLen = 2 * k;
        var w = MakeRandom(n, seed: 31);

        // Analytic: dL/dx = doubleInterior(RFft(w, n, dualNorm)).
        var wT = new Tensor<double>(new[] { n });
        Array.Copy(w, wT.GetDataArray(), w.Length);
        var rfft = Fft.RFft(wT, n, DualNorm(norm));
        var analyticGrad = DoubleInteriorBinsTensor(rfft, n);
        double analyticDD = Dot(analyticGrad.GetDataArray(), x);

        double numericDD = NumericDirectionalDerivative(x, xPerturbed =>
        {
            var t = new Tensor<double>(new[] { inLen });
            Array.Copy(xPerturbed, t.GetDataArray(), xPerturbed.Length);
            var y = Fft.IRFft(t, n, norm);
            return Dot(w, y.GetDataArray());
        });

        Assert.True(Math.Abs(analyticDD - numericDD) < 1e-5 * (1 + Math.Abs(analyticDD)),
            $"IRFft gradcheck n={n} norm={norm}: analytic={analyticDD}, numeric={numericDD}");
    }

    // ── helpers ────────────────────────────────────────────────────────────
    private static double[] MakeRandom(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.NextDouble() * 2 - 1;
        return a;
    }

    // Generate a Hermitian-symmetric packed complex spectrum (so that
    // IRFft of it is real-valued) for use as IRFft input.
    private static double[] MakeRandomHermitian(int n, int seed)
    {
        // Shortcut: take any real length-n signal and run it through RFft.
        // The result has Hermitian symmetry by construction and produces a
        // real IRFft output for gradcheck purposes.
        var rng = new Random(seed);
        var real = new double[n];
        for (int i = 0; i < n; i++) real[i] = rng.NextDouble() * 2 - 1;
        var t = new Tensor<double>(new[] { n });
        Array.Copy(real, t.GetDataArray(), n);
        var X = Fft.RFft(t);
        return X.GetDataArray();
    }

    private static double Dot(double[] a, double[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double s = 0;
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    private static double NumericDirectionalDerivative(double[] x, Func<double[], double> loss)
    {
        const double eps = 1e-5;
        var xp = new double[x.Length];
        var xm = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            xp[i] = x[i] * (1 + eps);
            xm[i] = x[i] * (1 - eps);
        }
        double lp = loss(xp);
        double lm = loss(xm);
        return (lp - lm) / (2 * eps);
    }

    private static FftNorm DualNorm(FftNorm norm) => norm switch
    {
        FftNorm.Backward => FftNorm.Forward,
        FftNorm.Forward => FftNorm.Backward,
        FftNorm.Ortho => FftNorm.Ortho,
        _ => throw new ArgumentOutOfRangeException(nameof(norm)),
    };

    private static double[] HalveInteriorBins(double[] packed, int n)
    {
        int K = n / 2 + 1;
        bool evenN = n % 2 == 0;
        var result = (double[])packed.Clone();
        int interiorEnd = evenN ? K - 1 : K;
        for (int k = 1; k < interiorEnd; k++)
        {
            result[2 * k] *= 0.5;
            result[2 * k + 1] *= 0.5;
        }
        return result;
    }

    // ── Fft2 gradcheck ──────────────────────────────────────────────────────
    [Theory]
    [InlineData(4, 4, FftNorm.Backward)]
    [InlineData(4, 4, FftNorm.Ortho)]
    [InlineData(8, 4, FftNorm.Forward)]
    public void Fft2_BackwardRule_MatchesFiniteDifference(int H, int W, FftNorm norm)
    {
        // Complex tensor with shape [H, 2W] interleaved re/im.
        var x = MakeRandom(H * 2 * W, seed: 40);
        var w = MakeRandom(H * 2 * W, seed: 41);

        var wT = new Tensor<double>(new[] { H, 2 * W });
        Array.Copy(w, wT.GetDataArray(), w.Length);
        var analyticGrad = Fft.IFft2(wT, norm: DualNorm(norm));
        double analyticDD = Dot(analyticGrad.GetDataArray(), x);

        double numericDD = NumericDirectionalDerivative(x, xPerturbed =>
        {
            var t = new Tensor<double>(new[] { H, 2 * W });
            Array.Copy(xPerturbed, t.GetDataArray(), xPerturbed.Length);
            var y = Fft.Fft2(t, norm: norm);
            return Dot(w, y.GetDataArray());
        });

        Assert.True(Math.Abs(analyticDD - numericDD) < 1e-4 * (1 + Math.Abs(analyticDD)),
            $"Fft2 gradcheck H={H} W={W} norm={norm}: analytic={analyticDD}, numeric={numericDD}");
    }

    // ── RFft2 gradcheck ─────────────────────────────────────────────────────
    [Theory]
    [InlineData(4, 4, FftNorm.Backward)]
    [InlineData(4, 4, FftNorm.Ortho)]
    public void RFft2_BackwardRule_MatchesFiniteDifference(int H, int W, FftNorm norm)
    {
        var x = MakeRandom(H * W, seed: 50);
        int freqW = W / 2 + 1;
        var wRfft = MakeRandom(H * 2 * freqW, seed: 51);

        // Analytic: halve interior last-axis bins of grad, then IRFft2 with dual norm.
        var wHalved = HalveInteriorLastAxisArray(wRfft, H, W);
        var wT = new Tensor<double>(new[] { H, 2 * freqW });
        Array.Copy(wHalved, wT.GetDataArray(), wHalved.Length);
        var analyticGrad = Fft.IRFft2(wT, s: new[] { H, W }, norm: DualNorm(norm));
        double analyticDD = Dot(analyticGrad.GetDataArray(), x);

        double numericDD = NumericDirectionalDerivative(x, xPerturbed =>
        {
            var t = new Tensor<double>(new[] { H, W });
            Array.Copy(xPerturbed, t.GetDataArray(), xPerturbed.Length);
            var y = Fft.RFft2(t, norm: norm);
            return Dot(wRfft, y.GetDataArray());
        });

        Assert.True(Math.Abs(analyticDD - numericDD) < 1e-4 * (1 + Math.Abs(analyticDD)),
            $"RFft2 gradcheck H={H} W={W} norm={norm}: analytic={analyticDD}, numeric={numericDD}");
    }

    private static double[] HalveInteriorLastAxisArray(double[] packed, int H, int W)
    {
        int freqW = W / 2 + 1;
        bool evenW = W % 2 == 0;
        var result = (double[])packed.Clone();
        int interiorEnd = evenW ? freqW - 1 : freqW;
        int rowLen = 2 * freqW;
        for (int y = 0; y < H; y++)
        {
            for (int k = 1; k < interiorEnd; k++)
            {
                result[y * rowLen + 2 * k] *= 0.5;
                result[y * rowLen + 2 * k + 1] *= 0.5;
            }
        }
        return result;
    }

    private static Tensor<double> DoubleInteriorBinsTensor(Tensor<double> packed, int n)
    {
        int K = n / 2 + 1;
        bool evenN = n % 2 == 0;
        var result = new Tensor<double>((int[])packed._shape.Clone());
        var src = packed.GetDataArray();
        var dst = result.GetDataArray();
        Array.Copy(src, dst, src.Length);
        int last = packed.Shape[packed.Rank - 1];
        int batch = src.Length / last;
        int interiorEnd = evenN ? K - 1 : K;
        for (int b = 0; b < batch; b++)
        {
            int off = b * last;
            for (int k = 1; k < interiorEnd; k++)
            {
                dst[off + 2 * k] *= 2.0;
                dst[off + 2 * k + 1] *= 2.0;
            }
        }
        return result;
    }
}
