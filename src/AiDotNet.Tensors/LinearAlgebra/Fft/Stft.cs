// Copyright (c) AiDotNet. All rights reserved.
// Short-Time Fourier Transform + inverse (overlap-add reconstruction).
// Mirrors torch.stft / torch.istft parameters and output shape.

using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

/// <summary>
/// Padding mode for <see cref="Stft.Forward{T}"/> centered framing.
/// </summary>
public enum PadMode
{
    /// <summary>Reflect around the boundary without repeating the edge sample. <c>x[-k] = x[k]</c>.</summary>
    Reflect,
    /// <summary>Pad with zeros (constant 0).</summary>
    Constant,
    /// <summary>Replicate the edge sample. <c>x[-k] = x[0]</c>.</summary>
    Replicate,
}

/// <summary>
/// STFT / ISTFT. Complex output is interleaved real/imag in the last axis
/// (same convention as <see cref="Fft"/>).
/// </summary>
public static class Stft
{
    /// <summary>
    /// Forward STFT. Produces a complex spectrogram with shape
    /// <c>[..., freqs, frames, 2]</c> flattened in the last axis as
    /// <c>[..., freqs, 2·frames]</c> — frame index is the second-to-last
    /// logical axis, complex real/imag pair is the last physical axis doubled.
    /// </summary>
    /// <param name="input">Real input signal of shape <c>[..., samples]</c>.</param>
    /// <param name="nFft">FFT size. Each frame is (zero-padded to) this length.</param>
    /// <param name="hopLength">Distance between successive frame starts. Default <c>nFft / 4</c>.</param>
    /// <param name="winLength">Window length (<c>≤ nFft</c>). Default equals <paramref name="nFft"/>.</param>
    /// <param name="window">Window tensor of length <paramref name="winLength"/>; null → rectangular.</param>
    /// <param name="center">If true, pad input so frame <c>t</c> is centered at sample <c>t · hop</c> (adds <c>nFft / 2</c> padding each side).</param>
    /// <param name="padMode">Padding used when <paramref name="center"/> is true.</param>
    /// <param name="normalized">If true, scales each frame's spectrum by <c>1/√nFft</c> (ortho-style).</param>
    /// <param name="onesided">If true (default for real input), return only the non-negative frequency bins (length <c>nFft / 2 + 1</c>).</param>
    /// <param name="returnComplex">When <c>true</c> (default), complex bins are
    /// stored interleaved in the last axis (shape <c>[..., freqs, 2·frames]</c>).
    /// When <c>false</c>, an extra trailing size-2 axis holds the (re, im) pair
    /// — shape <c>[..., freqs, frames, 2]</c> — matching torch.stft's
    /// <c>return_complex=False</c> layout.</param>
    /// <returns>Complex tensor: either interleaved
    /// <c>[..., freqs, 2·frames]</c> (<paramref name="returnComplex"/>=true)
    /// or paired <c>[..., freqs, frames, 2]</c> (false).</returns>
    public static Tensor<T> Forward<T>(
        Tensor<T> input,
        int nFft,
        int? hopLength = null,
        int? winLength = null,
        Tensor<T>? window = null,
        bool center = true,
        PadMode padMode = PadMode.Reflect,
        bool normalized = false,
        bool onesided = true,
        bool returnComplex = true)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (nFft <= 0) throw new ArgumentException("nFft must be positive.", nameof(nFft));
        int hop = hopLength ?? (nFft / 4);
        if (hop <= 0) throw new ArgumentException("hopLength must be positive.", nameof(hopLength));
        int win = winLength ?? nFft;
        if (win <= 0 || win > nFft) throw new ArgumentException("winLength must be in (0, nFft].", nameof(winLength));

        var ops = MathHelper.GetNumericOperations<T>();

        // Materialize window (length nFft, zero-padded on both sides if winLength < nFft).
        var winPadded = new double[nFft];
        if (window is not null)
        {
            if (window.Rank != 1 || window.Shape[0] != win)
                throw new ArgumentException($"window must be 1D with length {win}.", nameof(window));
            int pad = (nFft - win) / 2;
            var wSrc = window.GetDataArray();
            for (int i = 0; i < win; i++) winPadded[pad + i] = ops.ToDouble(wSrc[i]);
        }
        else
        {
            for (int i = 0; i < nFft; i++) winPadded[i] = 1.0; // rectangular
        }

        // Pad input if `center`.
        int rank = input.Rank;
        int sigLen = input.Shape[rank - 1];
        int batch = 1;
        for (int i = 0; i < rank - 1; i++) batch *= input._shape[i];
        int padAmt = center ? nFft / 2 : 0;
        int paddedLen = sigLen + 2 * padAmt;
        int nFrames = center
            ? 1 + (paddedLen - nFft) / hop
            : 1 + Math.Max(0, (sigLen - nFft) / hop);
        if (nFrames < 1) nFrames = 1;

        int nFreqs = onesided ? (nFft / 2 + 1) : nFft;

        // Output shape: [..., freqs, 2·frames] (frame index packed into last axis alongside re/im).
        var outShape = new int[rank + 1];
        for (int i = 0; i < rank - 1; i++) outShape[i] = input._shape[i];
        outShape[rank - 1] = nFreqs;
        outShape[rank] = 2 * nFrames;
        var output = new Tensor<T>(outShape);
        var inD = input.GetDataArray();
        var outD = output.GetDataArray();

        double frameScale = normalized ? 1.0 / Math.Sqrt(nFft) : 1.0;

        Parallel.For(0, batch, b =>
        {
            // Build padded view of this batch's signal.
            var padded = new double[paddedLen];
            int srcOff = b * sigLen;
            if (center)
            {
                for (int i = 0; i < sigLen; i++) padded[padAmt + i] = ops.ToDouble(inD[srcOff + i]);
                // Apply padding on each side.
                FillPad(padded, padAmt, sigLen, padMode);
            }
            else
            {
                for (int i = 0; i < sigLen; i++) padded[i] = ops.ToDouble(inD[srcOff + i]);
            }

            var frame = new double[2 * nFft];
            for (int f = 0; f < nFrames; f++)
            {
                int frameStart = f * hop;
                // Load windowed real frame into the real part of `frame`; imag = 0.
                for (int i = 0; i < nFft; i++)
                {
                    int idx = frameStart + i;
                    double x = (idx < paddedLen) ? padded[idx] : 0.0;
                    frame[2 * i] = x * winPadded[i];
                    frame[2 * i + 1] = 0.0;
                }
                FftKernels.Transform1D(frame, nFft, inverse: false, FftNorm.Backward);

                // Write (freqs, frame f) into output. The stored layout is:
                //   output[..., k, 2*f]     = Re{X[k, f]}
                //   output[..., k, 2*f + 1] = Im{X[k, f]}
                for (int k = 0; k < nFreqs; k++)
                {
                    double re = frame[2 * k] * frameScale;
                    double im = frame[2 * k + 1] * frameScale;
                    int outIdx = b * nFreqs * 2 * nFrames + k * 2 * nFrames + 2 * f;
                    outD[outIdx] = ops.FromDouble(re);
                    outD[outIdx + 1] = ops.FromDouble(im);
                }
            }
        });

        if (!returnComplex)
        {
            // Reshape from [..., freqs, 2·frames] to [..., freqs, frames, 2].
            var reshapedShape = new int[outShape.Length + 1];
            for (int i = 0; i < outShape.Length - 1; i++) reshapedShape[i] = outShape[i];
            reshapedShape[outShape.Length - 1] = nFrames;
            reshapedShape[outShape.Length] = 2;
            var reshaped = new Tensor<T>(reshapedShape);
            var srcD = output.GetDataArray();
            var dstD = reshaped.GetDataArray();
            int outerCount = batch * nFreqs;
            // srcD layout: outer * 2 * nFrames (re/im interleaved per frame)
            // dstD layout: outer * nFrames * 2 (re/im pair per frame)
            // Both are the same byte order — it's just a reshape of the last axis.
            Array.Copy(srcD, dstD, srcD.Length);
            return reshaped;
        }

        return output;
    }

    /// <summary>
    /// Inverse STFT. Takes a complex spectrogram (shape <c>[..., freqs, 2·frames]</c>
    /// as emitted by <see cref="Forward{T}"/>) and reconstructs the real signal via
    /// weighted overlap-add. Requires the window to satisfy COLA with the chosen
    /// hop (e.g., Hann with <c>hop = nFft / 4</c>) for exact reconstruction.
    /// </summary>
    public static Tensor<T> Inverse<T>(
        Tensor<T> input,
        int nFft,
        int? hopLength = null,
        int? winLength = null,
        Tensor<T>? window = null,
        bool center = true,
        bool normalized = false,
        bool onesided = true,
        int? length = null)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (nFft <= 0) throw new ArgumentException("nFft must be positive.", nameof(nFft));
        int hop = hopLength ?? (nFft / 4);
        int win = winLength ?? nFft;
        var ops = MathHelper.GetNumericOperations<T>();

        var winPadded = new double[nFft];
        if (window is not null)
        {
            int pad = (nFft - win) / 2;
            var wSrc = window.GetDataArray();
            for (int i = 0; i < win; i++) winPadded[pad + i] = ops.ToDouble(wSrc[i]);
        }
        else
        {
            for (int i = 0; i < nFft; i++) winPadded[i] = 1.0;
        }

        int rank = input.Rank;
        // Input shape: [..., freqs, 2·frames]
        int nFreqsIn = input.Shape[rank - 2];
        int frames2 = input.Shape[rank - 1];
        if (frames2 % 2 != 0) throw new ArgumentException("Input last axis must be 2·frames (interleaved re/im).");
        int nFrames = frames2 / 2;

        int nFreqsExpected = onesided ? (nFft / 2 + 1) : nFft;
        if (nFreqsIn != nFreqsExpected)
            throw new ArgumentException($"Input has {nFreqsIn} frequency bins, expected {nFreqsExpected} for nFft={nFft} onesided={onesided}.");

        int paddedLen = (nFrames - 1) * hop + nFft;
        int padAmt = center ? nFft / 2 : 0;
        int outLen = length ?? (paddedLen - 2 * padAmt);
        if (outLen <= 0) outLen = 1;

        // Flatten leading dims.
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        var outShape = new int[rank - 1];
        for (int i = 0; i < rank - 2; i++) outShape[i] = input._shape[i];
        outShape[rank - 2] = outLen;
        var output = new Tensor<T>(outShape);
        var inD = input.GetDataArray();
        var outD = output.GetDataArray();

        double frameScale = normalized ? Math.Sqrt(nFft) : 1.0;

        Parallel.For(0, batch, b =>
        {
            var reconstructed = new double[paddedLen];
            var windowSum = new double[paddedLen];
            var buf = new double[2 * nFft];

            for (int f = 0; f < nFrames; f++)
            {
                Array.Clear(buf, 0, buf.Length);
                // Load full complex spectrum. For onesided, mirror conjugate into negative freqs.
                for (int k = 0; k < nFreqsIn; k++)
                {
                    int inIdx = b * nFreqsIn * 2 * nFrames + k * 2 * nFrames + 2 * f;
                    buf[2 * k] = ops.ToDouble(inD[inIdx]) * frameScale;
                    buf[2 * k + 1] = ops.ToDouble(inD[inIdx + 1]) * frameScale;
                }
                if (onesided)
                {
                    // Hermitian mirror: X[n-k] = conj(X[k]) for k = 1 .. nFft/2 - 1 (even)
                    // or k = 1 .. (nFft-1)/2 (odd). The nFft/2 bin (if even) is real.
                    int lastIndependent = (nFft % 2 == 0) ? nFft / 2 : (nFft - 1) / 2;
                    for (int k = 1; k <= lastIndependent - (nFft % 2 == 0 ? 1 : 0); k++)
                    {
                        buf[2 * (nFft - k)] = buf[2 * k];
                        buf[2 * (nFft - k) + 1] = -buf[2 * k + 1];
                    }
                    if (nFft % 2 != 0)
                    {
                        // For odd nFft, the loop above already covered the full mirror range.
                    }
                }
                FftKernels.Transform1D(buf, nFft, inverse: true, FftNorm.Backward);

                int frameStart = f * hop;
                for (int i = 0; i < nFft; i++)
                {
                    int idx = frameStart + i;
                    if (idx >= paddedLen) break;
                    reconstructed[idx] += buf[2 * i] * winPadded[i];
                    windowSum[idx] += winPadded[i] * winPadded[i];
                }
            }

            // Normalize by squared-window overlap sum.
            for (int i = 0; i < paddedLen; i++)
            {
                if (windowSum[i] > 1e-12) reconstructed[i] /= windowSum[i];
            }

            // Crop padding.
            int outOff = b * outLen;
            for (int i = 0; i < outLen; i++)
            {
                int srcIdx = padAmt + i;
                double v = (srcIdx < paddedLen) ? reconstructed[srcIdx] : 0.0;
                outD[outOff + i] = ops.FromDouble(v);
            }
        });

        return output;
    }

    // ── Padding helpers ─────────────────────────────────────────────────────
    private static void FillPad(double[] padded, int padAmt, int sigLen, PadMode mode)
    {
        // Fill padded[0..padAmt) and padded[padAmt + sigLen .. padded.Length).
        // Inner real data sits in padded[padAmt .. padAmt + sigLen).
        switch (mode)
        {
            case PadMode.Constant:
                // Default-zeroed array already satisfies.
                break;
            case PadMode.Replicate:
                for (int i = 0; i < padAmt; i++) padded[i] = padded[padAmt];
                for (int i = 0; i < padAmt; i++) padded[padAmt + sigLen + i] = padded[padAmt + sigLen - 1];
                break;
            case PadMode.Reflect:
                // x[-k] = x[k] (does not duplicate boundary). Guard against very short signals.
                for (int i = 0; i < padAmt; i++)
                {
                    int src = padAmt + Math.Min(i + 1, sigLen - 1);
                    padded[padAmt - 1 - i] = padded[src];
                }
                for (int i = 0; i < padAmt; i++)
                {
                    int src = padAmt + Math.Max(0, sigLen - 2 - i);
                    padded[padAmt + sigLen + i] = padded[src];
                }
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }
}
