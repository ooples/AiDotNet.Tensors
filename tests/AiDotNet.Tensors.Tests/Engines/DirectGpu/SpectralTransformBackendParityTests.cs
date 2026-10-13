#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #941: the spectral transform ops of the #850 family that no test called on any backend (<c>AnalyticSignalMask</c>,
/// <c>ApplyMelFilterbank</c>, <c>BatchedFFT2D</c>, <c>BuildSpectrum</c>, <c>IstftFromSpectrum</c>, <c>IstftNormalize</c>,
/// <c>PhaseVocoder</c>, <c>SpectralFilter</c>, <c>StftMagPhase</c>), called directly on each backend and compared with a
/// CPU reference written from the CUDA kernels' definitions. Outputs are pre-filled with NaN, so an element a kernel
/// does not write fails; every op that disagrees is reported, not only the first. WebGPU runs on Dawn
/// (AIDOTNET_WEBGPU_NODE_DIR).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class SpectralTransformBackendParityTests
{
    [ThreadStatic] private static List<string>? t_mismatches;

    private static float[] Values(int count, int seed, double lo = -2, double hi = 2)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)(lo + (hi - lo) * rng.NextDouble());
        return a;
    }

    private static IGpuBuffer Filled(IDirectGpuBackend backend, int n, float value)
    {
        var buffer = backend.AllocateBuffer(n);
        backend.Fill(buffer, value, n);
        return buffer;
    }

    // |expected - actual| <= tolerance * max(1, scale), where scale defaults to the largest |expected| (a transform's
    // outputs are sums of many terms, so their error tracks the magnitude of the whole output, not of each element).
    private static void Compare(double[] expected, IDirectGpuBackend backend, IGpuBuffer actualBuffer, double tolerance, string what)
    {
        float[] actual = backend.DownloadBuffer(actualBuffer);
        if (actual.Length < expected.Length) { t_mismatches!.Add($"{what}: {actual.Length} outputs for {expected.Length} expected"); return; }
        double scale = 1;
        foreach (double e in expected) scale = Math.Max(scale, Math.Abs(e));
        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(expected[i] - actual[i]) <= tolerance * scale) continue;
            t_mismatches!.Add($"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
            return;
        }
    }

    private static void Check(IDirectGpuBackend backend)
    {
        t_mismatches = new List<string>();
        foreach (var (name, op) in new (string, Action<IDirectGpuBackend>)[]
        {
            ("AnalyticSignalMask", AnalyticSignalMask), ("ApplyMelFilterbank", ApplyMelFilterbank), ("BatchedFFT2D", BatchedFFT2D),
            ("BuildSpectrum", BuildSpectrum), ("Istft", Istft), ("PhaseVocoder", PhaseVocoder), ("SpectralFilter", SpectralFilter),
            ("StftMagPhase", StftMagPhase),
        })
        {
            try { op(backend); }
            catch (Exception ex) when (ex is not Xunit.SkipException) { t_mismatches.Add($"{name} threw {ex.GetType().Name}: {ex.Message}"); }
        }
        Assert.True(t_mismatches.Count == 0, string.Join(Environment.NewLine, t_mismatches));
    }

    // gain 0/1/2: DC and Nyquist keep 1, positive frequencies double, negative ones vanish; bins outside
    // [binLow, binHigh) vanish.
    private static void AnalyticSignalMask(IDirectGpuBackend backend)
    {
        foreach (var (batch, n, low, high) in new[] { (1, 8, 0, 8), (3, 16, 2, 6), (2, 64, 0, 33), (4, 15, 1, 7) })
        {
            float[] re = Values(batch * n, 71 + n), im = Values(batch * n, 72 + n);
            var expR = new double[batch * n];
            var expI = new double[batch * n];
            int half = n >> 1;
            for (int idx = 0; idx < batch * n; idx++)
            {
                int k = idx % n;
                double gain = (k == 0 || k == half) ? (k < low || k >= high ? 0 : 1) : k < half ? (k < low || k >= high ? 0 : 2) : 0;
                expR[idx] = re[idx] * gain;
                expI[idx] = im[idx] * gain;
            }
            using var dRe = backend.AllocateBuffer(re);
            using var dIm = backend.AllocateBuffer(im);
            using var oRe = Filled(backend, batch * n, float.NaN);
            using var oIm = Filled(backend, batch * n, float.NaN);
            backend.AnalyticSignalMask(dRe, dIm, oRe, oIm, batch, n, low, high);
            Compare(expR, backend, oRe, 0, $"AnalyticSignalMask.re batch={batch} n={n} [{low},{high})");
            Compare(expI, backend, oIm, 0, $"AnalyticSignalMask.im batch={batch} n={n} [{low},{high})");
        }
    }

    // melSpec[frame, mel] = sum_f powerSpec[frame, f] * filterbank[mel, f].
    private static void ApplyMelFilterbank(IDirectGpuBackend backend)
    {
        foreach (var (frames, freqs, mels) in new[] { (1, 1, 1), (7, 33, 10), (40, 257, 64) })
        {
            float[] power = Values(frames * freqs, 73 + freqs, 0, 5), bank = Values(mels * freqs, 74 + mels, 0, 1);
            var expected = new double[frames * mels];
            for (int t = 0; t < frames; t++)
                for (int m = 0; m < mels; m++)
                {
                    double s = 0;
                    for (int f = 0; f < freqs; f++) s += (double)power[t * freqs + f] * bank[m * freqs + f];
                    expected[t * mels + m] = s;
                }
            using var dPower = backend.AllocateBuffer(power);
            using var dBank = backend.AllocateBuffer(bank);
            using var o = Filled(backend, frames * mels, float.NaN);
            backend.ApplyMelFilterbank(dPower, dBank, o, frames, freqs, mels);
            Compare(expected, backend, o, 1e-5, $"ApplyMelFilterbank frames={frames} freqs={freqs} mels={mels}");
        }
    }

    private static (double[] Re, double[] Im) Dft2D(float[] re, float[] im, int batch, int h, int w)
    {
        var outR = new double[batch * h * w];
        var outI = new double[batch * h * w];
        for (int b = 0; b < batch; b++)
            for (int u = 0; u < h; u++)
                for (int v = 0; v < w; v++)
                {
                    double sr = 0, si = 0;
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                        {
                            double a = -2 * Math.PI * ((double)u * y / h + (double)v * x / w);
                            int i = b * h * w + y * w + x;
                            sr += re[i] * Math.Cos(a) - im[i] * Math.Sin(a);
                            si += re[i] * Math.Sin(a) + im[i] * Math.Cos(a);
                        }
                    outR[b * h * w + u * w + v] = sr;
                    outI[b * h * w + u * w + v] = si;
                }
        return (outR, outI);
    }

    // Forward against a direct 2-D DFT; inverse(forward(x)) == x, which holds whatever scaling convention the pair uses.
    private static void BatchedFFT2D(IDirectGpuBackend backend)
    {
        foreach (var (batch, h, w) in new[] { (1, 4, 8), (3, 8, 16), (2, 16, 4) })
        {
            int total = batch * h * w;
            float[] re = Values(total, 75 + total), im = Values(total, 76 + total);
            var (expR, expI) = Dft2D(re, im, batch, h, w);
            using var dRe = backend.AllocateBuffer(re);
            using var dIm = backend.AllocateBuffer(im);
            using var fRe = Filled(backend, total, float.NaN);
            using var fIm = Filled(backend, total, float.NaN);
            backend.BatchedFFT2D(dRe, dIm, fRe, fIm, batch, h, w, inverse: false);
            Compare(expR, backend, fRe, 1e-5, $"BatchedFFT2D.re batch={batch} {h}x{w}");
            Compare(expI, backend, fIm, 1e-5, $"BatchedFFT2D.im batch={batch} {h}x{w}");
            using var bRe = Filled(backend, total, float.NaN);
            using var bIm = Filled(backend, total, float.NaN);
            backend.BatchedFFT2D(fRe, fIm, bRe, bIm, batch, h, w, inverse: true);
            Compare(Array.ConvertAll(re, v => (double)v), backend, bRe, 1e-5, $"BatchedFFT2D round trip.re batch={batch} {h}x{w}");
            Compare(Array.ConvertAll(im, v => (double)v), backend, bIm, 1e-5, $"BatchedFFT2D round trip.im batch={batch} {h}x{w}");
        }
    }

    // spec[b, frame, k] = mag * e^{i phase} for k < numFreqs (mag/phase are [b, numFreqs, numFrames]), then the
    // conjugate-symmetric half: spec[nFft - k] = conj(spec[k]) for 1 <= k < numFreqs - 1.
    private static (double[] Re, double[] Im) BuildSpectrumReference(float[] mag, float[] phase, int batch, int freqs, int frames, int nFft)
    {
        var re = new double[batch * frames * nFft];
        var im = new double[batch * frames * nFft];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < frames; t++)
            {
                int spec = (b * frames + t) * nFft, magOff = b * freqs * frames;
                for (int k = 0; k < freqs && k < nFft; k++)
                {
                    float m = mag[magOff + k * frames + t], p = phase[magOff + k * frames + t];
                    re[spec + k] = m * (double)MathF.Cos(p);
                    im[spec + k] = m * (double)MathF.Sin(p);
                }
                for (int k = 1; k < freqs - 1; k++)
                {
                    int dst = nFft - k;
                    if (dst >= 0 && dst < nFft && k < nFft) { re[spec + dst] = re[spec + k]; im[spec + dst] = -im[spec + k]; }
                }
            }
        return (re, im);
    }

    private static void BuildSpectrum(IDirectGpuBackend backend)
    {
        foreach (var (batch, frames, nFft) in new[] { (1, 1, 4), (2, 5, 16), (3, 7, 32) })
        {
            int freqs = nFft / 2 + 1;
            float[] mag = Values(batch * freqs * frames, 77 + nFft, 0, 3), phase = Values(batch * freqs * frames, 78 + nFft, -3, 3);
            var (expR, expI) = BuildSpectrumReference(mag, phase, batch, freqs, frames, nFft);
            using var dMag = backend.AllocateBuffer(mag);
            using var dPhase = backend.AllocateBuffer(phase);
            using var oRe = Filled(backend, batch * frames * nFft, float.NaN);
            using var oIm = Filled(backend, batch * frames * nFft, float.NaN);
            backend.BuildSpectrum(dMag, dPhase, oRe, oIm, batch, freqs, frames, nFft);
            Compare(expR, backend, oRe, 1e-5, $"BuildSpectrum.re batch={batch} frames={frames} nFft={nFft}");
            Compare(expI, backend, oIm, 1e-5, $"BuildSpectrum.im batch={batch} frames={frames} nFft={nFft}");
        }
    }

    // result[b, n] = sum over frames covering n of window[i] * (1/nFft) * Re(IDFT(spec[b, frame]))[i], with
    // i = n - (frame * hop - (center ? nFft / 2 : 0)); windowSum[b, n] = sum of window[i]^2 over the same frames. Then
    // IstftNormalize divides result by windowSum where windowSum > 1e-8.
    private static void Istft(IDirectGpuBackend backend)
    {
        foreach (var (batch, frames, nFft, hop, center) in new[] { (1, 4, 8, 2, 0), (2, 6, 16, 4, 1), (2, 3, 8, 8, 1) })
        {
            int outLen = center == 1 ? (frames - 1) * hop : (frames - 1) * hop + nFft;
            float[] specRe = Values(batch * frames * nFft, 79 + nFft), specIm = Values(batch * frames * nFft, 80 + nFft);
            float[] window = Values(nFft, 81 + nFft, 0.1, 1);
            var expResult = new double[batch * outLen];
            var expSum = new double[batch * outLen];
            for (int b = 0; b < batch; b++)
                for (int n = 0; n < outLen; n++)
                {
                    double acc = 0, wsum = 0;
                    for (int t = 0; t < frames; t++)
                    {
                        int i = n - (center == 1 ? t * hop - nFft / 2 : t * hop);
                        if (i < 0 || i >= nFft) continue;
                        int spec = (b * frames + t) * nFft;
                        double s = 0;
                        for (int k = 0; k < nFft; k++)
                        {
                            double a = 2 * Math.PI * k * i / nFft;
                            s += specRe[spec + k] * Math.Cos(a) - specIm[spec + k] * Math.Sin(a);
                        }
                        acc += s / nFft * window[i];
                        wsum += (double)window[i] * window[i];
                    }
                    expResult[b * outLen + n] = acc;
                    expSum[b * outLen + n] = wsum;
                }
            using var dRe = backend.AllocateBuffer(specRe);
            using var dIm = backend.AllocateBuffer(specIm);
            using var dWin = backend.AllocateBuffer(window);
            using var result = Filled(backend, batch * outLen, float.NaN);
            using var wsumBuf = Filled(backend, batch * outLen, float.NaN);
            backend.IstftFromSpectrum(dRe, dIm, dWin, result, wsumBuf, batch, frames, nFft, hop, outLen, center);
            string what = $"batch={batch} frames={frames} nFft={nFft} hop={hop} center={center}";
            Compare(expResult, backend, result, 1e-5, "IstftFromSpectrum.result " + what);
            Compare(expSum, backend, wsumBuf, 1e-5, "IstftFromSpectrum.windowSum " + what);
            var normalized = new double[expResult.Length];
            for (int i = 0; i < normalized.Length; i++) normalized[i] = expSum[i] > 1e-8 ? expResult[i] / expSum[i] : expResult[i];
            backend.IstftNormalize(result, wsumBuf, batch * outLen);
            Compare(normalized, backend, result, 1e-4, "IstftNormalize " + what);
        }
    }

    // Per (leading, frequency): linear interpolation of the magnitude along time at t * rate, and the phase advance of the
    // source frames, wrapped to (-pi, pi], accumulated from 0.
    private static void PhaseVocoder(IDirectGpuBackend backend)
    {
        foreach (var (leading, freqs, frames, outFrames, rate) in new[] { (1, 1, 2, 3, 0.5f), (2, 5, 9, 13, 0.7f), (3, 17, 20, 8, 2.5f) })
        {
            int inCount = leading * freqs * frames, outCount = leading * freqs * outFrames;
            float[] mag = Values(inCount, 82 + inCount, 0, 3), phase = Values(inCount, 83 + inCount, -3, 3);
            var expMag = new double[outCount];
            var expPhase = new double[outCount];
            float twoPi = 2f * MathF.PI;
            for (int b = 0; b < leading; b++)
                for (int f = 0; f < freqs; f++)
                {
                    int row = b * freqs * frames + f * frames, outRow = b * freqs * outFrames + f * outFrames;
                    float acc = 0f;
                    for (int t = 0; t < outFrames; t++)
                    {
                        float src = t * rate;
                        int t0 = (int)MathF.Floor(src);
                        if (t0 > frames - 1) t0 = frames - 1;
                        int t1 = Math.Min(t0 + 1, frames - 1);
                        float frac = src - t0;
                        expMag[outRow + t] = (1f - frac) * mag[row + t0] + frac * mag[row + t1];
                        float dp = 0f;
                        if (t0 + 1 < frames)
                        {
                            dp = phase[row + t0 + 1] - phase[row + t0];
                            dp -= twoPi * MathF.Round(dp / twoPi, MidpointRounding.AwayFromZero);
                        }
                        acc += dp;
                        expPhase[outRow + t] = acc;
                    }
                }
            using var dMag = backend.AllocateBuffer(mag);
            using var dPhase = backend.AllocateBuffer(phase);
            using var oMag = Filled(backend, outCount, float.NaN);
            using var oPhase = Filled(backend, outCount, float.NaN);
            backend.PhaseVocoder(dMag, dPhase, oMag, oPhase, leading, frames, freqs, outFrames, rate);
            string what = $"leading={leading} freqs={freqs} frames={frames} out={outFrames} rate={rate}";
            Compare(expMag, backend, oMag, 1e-5, "PhaseVocoder.mag " + what);
            Compare(expPhase, backend, oPhase, 1e-4, "PhaseVocoder.phase " + what);
        }
    }

    // Re(IFFT2(FFT2(x) * filter)) per slice, the filter shared (one slice) or one per slice; the inverse carries the
    // 1/(H*W) of a round trip.
    private static void SpectralFilter(IDirectGpuBackend backend)
    {
        foreach (var (batch, h, w, shared) in new[] { (1, 4, 4, true), (3, 8, 8, true), (2, 4, 16, false) })
        {
            int slice = h * w, total = batch * slice, filterSlices = shared ? 1 : batch;
            float[] x = Values(total, 84 + total);
            float[] fRe = Values(filterSlices * slice, 85 + total), fIm = Values(filterSlices * slice, 86 + total);
            var (xr, xi) = Dft2D(x, new float[total], batch, h, w);
            var prodR = new float[total];
            var prodI = new float[total];
            for (int i = 0; i < total; i++)
            {
                int fi = shared ? i % slice : i;
                prodR[i] = (float)(xr[i] * fRe[fi] - xi[i] * fIm[fi]);
                prodI[i] = (float)(xr[i] * fIm[fi] + xi[i] * fRe[fi]);
            }
            // inverse DFT = conj(DFT(conj(z))) / N
            var (ir, _) = Dft2D(prodR, Array.ConvertAll(prodI, v => -v), batch, h, w);
            var expected = new double[total];
            for (int i = 0; i < total; i++) expected[i] = ir[i] / slice;
            using var dX = backend.AllocateBuffer(x);
            using var dFRe = backend.AllocateBuffer(fRe);
            using var dFIm = backend.AllocateBuffer(fIm);
            using var o = Filled(backend, total, float.NaN);
            backend.SpectralFilter(dX, dFRe, dFIm, o, batch, h, w, filterSlices);
            Compare(expected, backend, o, 1e-4, $"SpectralFilter batch={batch} {h}x{w} filterSlices={filterSlices}");
        }
    }

    // Frame f of row b covers padded[b, f*hop .. f*hop + nFft) times the window; bin k of its DFT goes to
    // [b, k, frame] as magnitude and atan2 phase.
    private static void StftMagPhase(IDirectGpuBackend backend)
    {
        foreach (var (batch, nFft, hop, frames) in new[] { (1, 4, 2, 1), (2, 16, 4, 6), (3, 32, 8, 5) })
        {
            int lp = (frames - 1) * hop + nFft, freqs = nFft / 2 + 1;
            float[] padded = Values(batch * lp, 87 + lp), window = Values(nFft, 88 + nFft, 0.1, 1);
            var expMag = new double[batch * freqs * frames];
            var expPhase = new double[batch * freqs * frames];
            for (int b = 0; b < batch; b++)
                for (int k = 0; k < freqs; k++)
                    for (int t = 0; t < frames; t++)
                    {
                        double re = 0, im = 0;
                        for (int i = 0; i < nFft; i++)
                        {
                            double v = (double)padded[b * lp + t * hop + i] * window[i], a = -2 * Math.PI * k * i / nFft;
                            re += v * Math.Cos(a);
                            im += v * Math.Sin(a);
                        }
                        int o = b * freqs * frames + k * frames + t;
                        expMag[o] = Math.Sqrt(re * re + im * im);
                        // A bin whose imaginary part is a rounding residue around zero (k = 0, k = nFft/2) has a phase that
                        // flips between 0 and +-pi with the sign of that residue; compare its magnitude only.
                        expPhase[o] = Math.Abs(im) < 1e-4 * Math.Max(1, Math.Abs(re)) ? double.NaN : Math.Atan2(im, re);
                    }
            using var dPadded = backend.AllocateBuffer(padded);
            using var dWin = backend.AllocateBuffer(window);
            using var oMag = Filled(backend, batch * freqs * frames, float.NaN);
            using var oPhase = Filled(backend, batch * freqs * frames, float.NaN);
            backend.StftMagPhase(dPadded, dWin, oMag, oPhase, batch, lp, nFft, hop, frames, freqs);
            string what = $"batch={batch} nFft={nFft} hop={hop} frames={frames}";
            Compare(expMag, backend, oMag, 1e-5, "StftMagPhase.mag " + what);
            float[] phase = backend.DownloadBuffer(oPhase);
            for (int i = 0; i < expPhase.Length; i++)
            {
                if (double.IsNaN(expPhase[i])) { if (float.IsNaN(phase[i])) t_mismatches!.Add($"StftMagPhase.phase {what}[{i}] was not written"); continue; }
                double d = Math.Abs(expPhase[i] - phase[i]);
                d = Math.Min(d, 2 * Math.PI - d);
                if (d > 1e-3) { t_mismatches!.Add($"StftMagPhase.phase {what}[{i}]: expected {expPhase[i]}, got {phase[i]}"); break; }
            }
        }
    }

    [SkippableFact]
    public void OpenCl_SpectralTransform_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_SpectralTransform_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_SpectralTransform_MatchesCpu()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_SpectralTransform_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Metal_SpectralTransform_MatchesCpu()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void WebGpu_SpectralTransform_MatchesCpu() => Check(DawnHostedWebGpu.BackendOrSkip());
}
#endif