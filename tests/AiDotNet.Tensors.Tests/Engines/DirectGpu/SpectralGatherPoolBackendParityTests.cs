#if NET6_0_OR_GREATER
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
/// #1238 (part of #941): the gather, pooling and filterbank ops of the #850 spectral family (<c>BispectrumGather</c>,
/// <c>TrispectrumGather</c>, <c>CavityBounceInplace</c>, <c>WidebandLogBinPool</c>, <c>PacPhaseBinMi</c>,
/// <c>MelFilterbankApply</c>), called directly on each backend and compared with a CPU reference written from the
/// documented definition. Every backend defines these and none had a test that called them on any backend.
/// Outputs are pre-filled with NaN, so an element a kernel does not write fails; every op that disagrees is
/// reported, not only the first.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class SpectralGatherPoolBackendParityTests
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

    private static void Compare(double[] expected, IDirectGpuBackend backend, IGpuBuffer actualBuffer, double tolerance, string what)
    {
        float[] actual = backend.DownloadBuffer(actualBuffer);
        if (actual.Length < expected.Length)
        {
            t_mismatches!.Add($"{what}: {actual.Length} outputs for {expected.Length} expected");
            return;
        }
        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1.0, Math.Abs(expected[i]))) continue;
            t_mismatches!.Add($"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
            return;
        }
    }

    private static void Check(IDirectGpuBackend backend)
    {
        t_mismatches = new List<string>();
        Bispectrum(backend);
        Trispectrum(backend);
        CavityBounce(backend);
        WidebandLogBinPool(backend);
        PacPhaseBinMi(backend);
        MelFilterbank(backend);
        Assert.True(t_mismatches.Count == 0, string.Join(Environment.NewLine, t_mismatches));
    }

    // B(f1, f2) = X(f1) X(f2) conj(X(f1 + f2)), output [maxF1, maxF2].
    private static void Bispectrum(IDirectGpuBackend backend)
    {
        foreach (var (f1N, f2N) in new[] { (1, 1), (5, 7), (40, 30) })
        {
            int specLength = f1N + f2N - 1, total = f1N * f2N;
            float[] re = Values(specLength, 11 + total), im = Values(specLength, 12 + total);
            double[] expR = new double[total], expI = new double[total];
            for (int f1 = 0; f1 < f1N; f1++)
                for (int f2 = 0; f2 < f2N; f2++)
                {
                    var b = new System.Numerics.Complex(re[f1], im[f1]) * new System.Numerics.Complex(re[f2], im[f2])
                        * System.Numerics.Complex.Conjugate(new System.Numerics.Complex(re[f1 + f2], im[f1 + f2]));
                    expR[f1 * f2N + f2] = b.Real;
                    expI[f1 * f2N + f2] = b.Imaginary;
                }
            using var dRe = backend.AllocateBuffer(re);
            using var dIm = backend.AllocateBuffer(im);
            using var oR = Filled(backend, total, float.NaN);
            using var oI = Filled(backend, total, float.NaN);
            backend.BispectrumGather(dRe, dIm, oR, oI, f1N, f2N);
            Compare(expR, backend, oR, 1e-5, $"BispectrumGather re {f1N}x{f2N}");
            Compare(expI, backend, oI, 1e-5, $"BispectrumGather im {f1N}x{f2N}");
        }
    }

    // T(f1, f2, f3) = X(f1) X(f2) X(f3) conj(X(f1 + f2 + f3)), output [maxF1, maxF2, maxF3].
    private static void Trispectrum(IDirectGpuBackend backend)
    {
        foreach (var (f1N, f2N, f3N) in new[] { (1, 1, 1), (3, 4, 5), (12, 11, 10) })
        {
            int specLength = f1N + f2N + f3N - 2, total = f1N * f2N * f3N;
            float[] re = Values(specLength, 21 + total), im = Values(specLength, 22 + total);
            double[] expR = new double[total], expI = new double[total];
            for (int f1 = 0; f1 < f1N; f1++)
                for (int f2 = 0; f2 < f2N; f2++)
                    for (int f3 = 0; f3 < f3N; f3++)
                    {
                        var t = new System.Numerics.Complex(re[f1], im[f1]) * new System.Numerics.Complex(re[f2], im[f2])
                            * new System.Numerics.Complex(re[f3], im[f3])
                            * System.Numerics.Complex.Conjugate(new System.Numerics.Complex(re[f1 + f2 + f3], im[f1 + f2 + f3]));
                        int idx = (f1 * f2N + f2) * f3N + f3;
                        expR[idx] = t.Real;
                        expI[idx] = t.Imaginary;
                    }
            using var dRe = backend.AllocateBuffer(re);
            using var dIm = backend.AllocateBuffer(im);
            using var oR = Filled(backend, total, float.NaN);
            using var oI = Filled(backend, total, float.NaN);
            backend.TrispectrumGather(dRe, dIm, oR, oI, f1N, f2N, f3N);
            Compare(expR, backend, oR, 2e-5, $"TrispectrumGather re {f1N}x{f2N}x{f3N}");
            Compare(expI, backend, oI, 2e-5, $"TrispectrumGather im {f1N}x{f2N}x{f3N}");
        }
    }

    // In place: real = tanh(clamp(real * invN, -20, 20)), imag = 0.
    private static void CavityBounce(IDirectGpuBackend backend)
    {
        const float invN = 1f / 64f;
        foreach (int total in new[] { 1, 7, 1000 })
        {
            float[] re = Values(total, 31 + total, -200, 200);
            re[0] = 5000f;                       // far past the clamp: tanh(20)
            if (total > 1) re[1] = -5000f;
            if (total > 2) re[2] = 0f;
            using var dRe = backend.AllocateBuffer(re);
            using var dIm = backend.AllocateBuffer(Values(total, 32 + total));   // non-zero: the kernel must clear it
            backend.CavityBounceInplace(dRe, dIm, total, invN);
            var expR = new double[total];
            for (int i = 0; i < total; i++) expR[i] = Math.Tanh(Math.Min(Math.Max((double)re[i] * invN, -20.0), 20.0));
            Compare(expR, backend, dRe, 2e-5, $"CavityBounceInplace re n={total}");
            Compare(new double[total], backend, dIm, 0, $"CavityBounceInplace im n={total}");
        }
    }

    // Per segment, bin k averages mag[binStart .. binEnd) and takes log(1 + avg), with
    // binStart = 1 + trunc((k / numBins)^2 (usable - 1)) and binEnd the same for k + 1, at least one wide, capped at usable.
    private static void WidebandLogBinPool(IDirectGpuBackend backend)
    {
        foreach (var (segments, fftSize, numBins, usable) in new[] { (1, 64, 8, 32), (3, 256, 16, 128), (40, 128, 20, 64), (2, 16, 12, 5),
            // shapes whose exact integer edges float (k=7 of 10 at usable 101; k=5 of 6 at 181) or double (k=1 of 7 at 50)
            // arithmetic truncated to the bin below
            (2, 202, 10, 101), (1, 362, 6, 181), (3, 100, 7, 50) })
        {
            float[] mag = Values(segments * fftSize, 41 + segments * fftSize, 0, 30);
            var expected = new double[segments * numBins];
            for (int seg = 0; seg < segments; seg++)
                for (int k = 0; k < numBins; k++)
                {
                    // The definition, in exact integers: 1 + floor(k^2 (usable-1) / numBins^2). Float or double (k/numBins)^2 truncates
                    // exact integer edges to the bin below, which is what every backend did before.
                    long den = (long)numBins * numBins;
                    int binStart = 1 + (int)((long)k * k * (usable - 1) / den);
                    int binEnd = 1 + (int)((long)(k + 1) * (k + 1) * (usable - 1) / den);
                    if (binEnd <= binStart) binEnd = binStart + 1;
                    if (binEnd > usable) binEnd = usable;
                    double sum = 0;
                    int count = 0;
                    for (int i = binStart; i < binEnd; i++) { sum += mag[seg * fftSize + i]; count++; }
                    expected[seg * numBins + k] = Math.Log(1.0 + (count > 0 ? sum / count : 0.0));
                }
            using var dMag = backend.AllocateBuffer(mag);
            using var o = Filled(backend, segments * numBins, float.NaN);
            backend.WidebandLogBinPool(dMag, o, segments, fftSize, numBins, usable);
            Compare(expected, backend, o, 2e-5, $"WidebandLogBinPool segments={segments} fft={fftSize} bins={numBins} usable={usable}");
        }
    }

    // Per batch row: amplitudes averaged within 18 phase bins over [-pi, pi), normalized to a distribution p, and
    // MI = (log 18 - H(p)) / log 18. Only output[b, gammaIdx] is written.
    private static void PacPhaseBinMi(IDirectGpuBackend backend)
    {
        const int phaseBins = 18;
        const float sentinel = 7f;
        foreach (var (batch, samples, bands, gammaIdx) in new[] { (1, 1, 1, 0), (3, 100, 4, 2), (5, 2000, 3, 0), (300, 50, 2, 1) })
        {
            var rng = new Random(51 + batch * samples);
            var theta = new float[batch * samples];
            var amp = new float[bands * batch * samples];
            for (int i = 0; i < theta.Length; i++)
            {
                // Well inside a bin, so the bin a sample lands in does not depend on a device's last bit.
                int bin = rng.Next(phaseBins);
                theta[i] = (float)(-Math.PI + (bin + 0.2 + 0.6 * rng.NextDouble()) * (2 * Math.PI / phaseBins));
            }
            if (samples >= 100)
            {
                theta[0] = -4f;      // below -pi: clamps to the first bin
                theta[1] = 4f;       // above pi: clamps to the last
            }
            for (int i = 0; i < amp.Length; i++) amp[i] = (float)(0.1 + 1.9 * rng.NextDouble());
            // Make the distribution uneven, or every MI is near zero and a kernel returning zero would pass.
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < samples; i++)
                    if (theta[b * samples + i] < 0f) amp[(gammaIdx * batch + b) * samples + i] *= 4f;

            var expected = new double[batch * bands];
            for (int i = 0; i < expected.Length; i++) expected[i] = sentinel;
            double largest = 0;
            for (int b = 0; b < batch; b++)
            {
                var sums = new double[phaseBins];
                var counts = new double[phaseBins];
                for (int i = 0; i < samples; i++)
                {
                    float fbin = (theta[b * samples + i] + 3.14159265358979f) / (2.0f * 3.14159265358979f) * phaseBins;
                    int bin = Math.Min(Math.Max((int)fbin, 0), phaseBins - 1);
                    sums[bin] += amp[(gammaIdx * batch + b) * samples + i];
                    counts[bin] += 1;
                }
                double totalAmp = 0;
                for (int k = 0; k < phaseBins; k++) totalAmp += counts[k] > 0 ? sums[k] / counts[k] : 0;
                double mi = 0;
                if (totalAmp > 0)
                {
                    double entropy = 0;
                    for (int k = 0; k < phaseBins; k++)
                    {
                        double p = (counts[k] > 0 ? sums[k] / counts[k] : 0) / totalAmp;
                        if (p > 1e-12) entropy -= p * Math.Log(p);
                    }
                    mi = (Math.Log(phaseBins) - entropy) / Math.Log(phaseBins);
                }
                expected[b * bands + gammaIdx] = mi;
                largest = Math.Max(largest, mi);
            }
            Assert.True(largest > 0.01, $"degenerate case: the largest modulation index is {largest}");

            using var dTheta = backend.AllocateBuffer(theta);
            using var dAmp = backend.AllocateBuffer(amp);
            using var o = Filled(backend, batch * bands, sentinel);
            backend.PacPhaseBinMi(dTheta, dAmp, o, batch, samples, bands, gammaIdx);
            Compare(expected, backend, o, 1e-4, $"PacPhaseBinMi batch={batch} samples={samples} bands={bands} gammaIdx={gammaIdx}");
        }
    }

    // melEnergy[seg, m] = sum_i powerSpec[seg, i] * melFilters[m, i].
    private static void MelFilterbank(IDirectGpuBackend backend)
    {
        foreach (var (segments, specBins, melBins) in new[] { (1, 1, 1), (3, 33, 8), (20, 129, 40) })
        {
            float[] power = Values(segments * specBins, 61 + segments * specBins, 0, 5);
            float[] filters = Values(melBins * specBins, 62 + melBins * specBins, 0, 1);
            var expected = new double[segments * melBins];
            for (int seg = 0; seg < segments; seg++)
                for (int m = 0; m < melBins; m++)
                {
                    double sum = 0;
                    for (int i = 0; i < specBins; i++) sum += (double)power[seg * specBins + i] * filters[m * specBins + i];
                    expected[seg * melBins + m] = sum;
                }
            using var dPower = backend.AllocateBuffer(power);
            using var dFilters = backend.AllocateBuffer(filters);
            using var o = Filled(backend, segments * melBins, float.NaN);
            backend.MelFilterbankApply(dPower, dFilters, o, segments, specBins, melBins);
            Compare(expected, backend, o, 1e-5, $"MelFilterbankApply segments={segments} specBins={specBins} melBins={melBins}");
        }
    }

    [SkippableFact]
    public void OpenCl_SpectralGatherPool_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_SpectralGatherPool_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_SpectralGatherPool_MatchesCpu()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_SpectralGatherPool_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Metal_SpectralGatherPool_MatchesCpu()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_SpectralGatherPool_MatchesCpu()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();   // the real backend, executing on Dawn; shared, so not disposed
        Check(backend);
    }
#endif
}
#endif
