using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // Split the real signal into even and odd components. Their spectra are recoverable
    // from one half-length complex transform, including DC and Nyquist.
    private static void RfftMagnitudeFloatRows(ReadOnlyMemory<float> input, ReadOnlyMemory<float> upstream,
        Memory<float> destination, int rows, int length, int bins, float epsilon, bool backward, bool accumulate)
    {
        int workers = Math.Max(1, Math.Min(rows, CpuParallelSettings.MaxDegreeOfParallelism));
        var roots = CpuStorageAllocationDiagnostics.RentWorkspace<float>(length, "workspace.rfft_magnitude_twiddles");
        try
        {
            for (int i = 0; i < length / 2; i++)
            {
                double angle = -2 * Math.PI * i / length;
                roots[2 * i] = (float)Math.Cos(angle);
                roots[2 * i + 1] = (float)Math.Sin(angle);
            }
            CpuParallelSettings.ParallelForOrSerial(0, workers, (long)rows * length * 16, worker =>
            {
                int scratchLength = backward ? checked(2 * length) : length + 2;
                var scratch = CpuStorageAllocationDiagnostics.RentWorkspace<float>(scratchLength, "workspace.rfft_magnitude_fft");
                try
                {
                    var spectrum = scratch.AsSpan(0, scratchLength);
                    for (int row = worker; row < rows; row += workers)
                    {
                        input.Span.Slice(row * length, length).CopyTo(spectrum);
                        ComplexFftFloat(spectrum, length / 2, roots, length, inverse: false);
                        UnpackRealSpectrum(spectrum, roots, length);
                        var destinationRow = destination.Span.Slice(row * (backward ? length : bins), backward ? length : bins);
                        var gradient = backward ? upstream.Span.Slice(row * bins, bins) : default;
                        for (int k = 0; k < bins; k++)
                        {
                            float real = spectrum[2 * k], imaginary = spectrum[2 * k + 1];
                            float magnitude = (float)Math.Sqrt(real * real + imaginary * imaginary + epsilon);
                            if (backward)
                            {
                                float scale = gradient[k] / magnitude;
                                spectrum[2 * k] = real * scale;
                                spectrum[2 * k + 1] = imaginary * scale;
                            }
                            else destinationRow[k] = magnitude;
                        }
                        if (!backward) continue;
                        spectrum.Slice(2 * bins).Clear();
                        ComplexFftFloat(spectrum, length, roots, length, inverse: true);
                        for (int j = 0; j < length; j++)
                            destinationRow[j] = accumulate ? destinationRow[j] + spectrum[2 * j] : spectrum[2 * j];
                    }
                }
                finally { ArrayPool<float>.Shared.Return(scratch); }
            }, deterministicSafe: true);
        }
        finally { ArrayPool<float>.Shared.Return(roots); }
    }

    private static void UnpackRealSpectrum(Span<float> z, float[] roots, int n)
    {
        int half = n / 2;
        float dcReal = z[0], dcImag = z[1];
        z[0] = dcReal + dcImag;
        z[1] = 0;
        z[n] = dcReal - dcImag;
        z[n + 1] = 0;
        for (int k = 1; k <= half / 2; k++)
        {
            int lo = 2 * k, hi = 2 * (half - k);
            float ar = z[lo], ai = z[lo + 1], br = z[hi], bi = z[hi + 1];
            float evenReal = .5f * (ar + br), evenImag = .5f * (ai - bi);
            float oddReal = .5f * (ai + bi), oddImag = .5f * (br - ar);
            float wr = roots[lo], wi = roots[lo + 1];
            float tr = wr * oddReal - wi * oddImag, ti = wr * oddImag + wi * oddReal;
            z[lo] = evenReal + tr;
            z[lo + 1] = evenImag + ti;
            if (lo == hi) continue;
            z[hi] = evenReal - tr;
            z[hi + 1] = ti - evenImag;
        }
    }

    private static void ComplexFftFloat(Span<float> z, int count, float[] roots, int rootLength, bool inverse)
    {
        for (int i = 1, j = 0; i < count; i++)
        {
            int bit = count >> 1;
            while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
            j ^= bit;
            if (i >= j) continue;
            (z[2 * i], z[2 * j]) = (z[2 * j], z[2 * i]);
            (z[2 * i + 1], z[2 * j + 1]) = (z[2 * j + 1], z[2 * i + 1]);
        }
        for (int half = 1; half < count; half <<= 1)
        {
            int stride = half * 2, step = rootLength / stride;
            for (int start = 0; start < count; start += stride)
                for (int j = 0; j < half; j++)
                {
                    int a = 2 * (start + j), b = a + stride, root = 2 * j * step;
                    float wr = roots[root], wi = inverse ? -roots[root + 1] : roots[root + 1];
                    float tr = wr * z[b] - wi * z[b + 1], ti = wr * z[b + 1] + wi * z[b];
                    float ar = z[a], ai = z[a + 1];
                    z[a] = ar + tr;
                    z[a + 1] = ai + ti;
                    z[b] = ar - tr;
                    z[b + 1] = ai - ti;
                }
        }
    }
}
