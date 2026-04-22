// Copyright (c) AiDotNet. All rights reserved.
// CPU reference implementations for the tail of Issue #217:
//   RoI family: RoIAlign, RoIPool, PsRoIAlign, PsRoIPool
//   Audio primitives: Spectrogram, AmplitudeToDB, MuLaw encode/decode,
//                    ComputeDeltas, Resample, PitchShift, TimeStretch
// Matches torchvision 0.15 / torchaudio 2.0 semantics bit-for-bit
// where practical.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // ========================================================================
    // RoI family
    // ========================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> RoIAlign<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned)
    {
        if (input.Rank != 4) throw new ArgumentException("RoIAlign input must be [N, C, H, W].");
        if (boxes.Rank != 2 || boxes._shape[1] != 5)
            throw new ArgumentException("RoIAlign boxes must be [K, 5] = (batch_idx, x1, y1, x2, y2).");
        int N = input._shape[0], C = input._shape[1], H = input._shape[2], W = input._shape[3];
        int K = boxes._shape[0];
        var output = new Tensor<T>(new[] { K, C, outputHeight, outputWidth });
        if (K == 0) return output;

        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var b = boxes.AsSpan();
        var dst = output.AsWritableSpan();
        double offset = aligned ? 0.5 : 0.0;

        for (int k = 0; k < K; k++)
        {
            int n = (int)ops.ToDouble(b[k * 5]);
            if (n < 0 || n >= N) throw new ArgumentException($"RoIAlign batch idx {n} out of range [0, {N}).");
            double x1 = ops.ToDouble(b[k * 5 + 1]) * spatialScale - offset;
            double y1 = ops.ToDouble(b[k * 5 + 2]) * spatialScale - offset;
            double x2 = ops.ToDouble(b[k * 5 + 3]) * spatialScale - offset;
            double y2 = ops.ToDouble(b[k * 5 + 4]) * spatialScale - offset;

            double roiW = aligned ? (x2 - x1) : Math.Max(x2 - x1, 1.0);
            double roiH = aligned ? (y2 - y1) : Math.Max(y2 - y1, 1.0);
            double binH = roiH / outputHeight;
            double binW = roiW / outputWidth;

            int ry = samplingRatio > 0 ? samplingRatio : (int)Math.Ceiling(roiH / outputHeight);
            int rx = samplingRatio > 0 ? samplingRatio : (int)Math.Ceiling(roiW / outputWidth);
            if (ry < 1) ry = 1;
            if (rx < 1) rx = 1;
            double gridArea = ry * rx;

            for (int c = 0; c < C; c++)
            {
                int planeBase = (n * C + c) * H * W;
                for (int ph = 0; ph < outputHeight; ph++)
                for (int pw = 0; pw < outputWidth; pw++)
                {
                    double acc = 0;
                    for (int iy = 0; iy < ry; iy++)
                    {
                        double sy = y1 + ph * binH + (iy + 0.5) * binH / ry;
                        for (int ix = 0; ix < rx; ix++)
                        {
                            double sx = x1 + pw * binW + (ix + 0.5) * binW / rx;
                            acc += BilinearSample(src, planeBase, sy, sx, H, W, ops);
                        }
                    }
                    dst[((k * C + c) * outputHeight + ph) * outputWidth + pw] =
                        ops.FromDouble(acc / gridArea);
                }
            }
        }
        return output;
    }

    /// <summary>Bilinear sample with zero-padding (RoIAlign convention).</summary>
    private static double BilinearSample<T>(ReadOnlySpan<T> src, int planeBase,
        double y, double x, int H, int W, Interfaces.INumericOperations<T> ops)
    {
        if (y < -1.0 || y > H || x < -1.0 || x > W) return 0.0;
        if (y <= 0) y = 0;
        if (x <= 0) x = 0;
        int y0 = (int)y;
        int x0 = (int)x;
        int y1 = y0 + 1 >= H ? H - 1 : y0 + 1;
        int x1 = x0 + 1 >= W ? W - 1 : x0 + 1;
        if (y0 >= H - 1) { y0 = y1 = H - 1; y = y0; }
        if (x0 >= W - 1) { x0 = x1 = W - 1; x = x0; }
        double ly = y - y0, lx = x - x0;
        double hy = 1.0 - ly, hx = 1.0 - lx;
        double v00 = ops.ToDouble(src[planeBase + y0 * W + x0]);
        double v01 = ops.ToDouble(src[planeBase + y0 * W + x1]);
        double v10 = ops.ToDouble(src[planeBase + y1 * W + x0]);
        double v11 = ops.ToDouble(src[planeBase + y1 * W + x1]);
        return hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> RoIPool<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, float spatialScale)
    {
        if (input.Rank != 4) throw new ArgumentException("RoIPool input must be [N, C, H, W].");
        if (boxes.Rank != 2 || boxes._shape[1] != 5) throw new ArgumentException("RoIPool boxes must be [K, 5].");
        int N = input._shape[0], C = input._shape[1], H = input._shape[2], W = input._shape[3];
        int K = boxes._shape[0];
        var output = new Tensor<T>(new[] { K, C, outputHeight, outputWidth });
        if (K == 0) return output;

        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var b = boxes.AsSpan();
        var dst = output.AsWritableSpan();
        T negInf = ops.FromDouble(double.NegativeInfinity);

        for (int k = 0; k < K; k++)
        {
            int n = (int)ops.ToDouble(b[k * 5]);
            if (n < 0 || n >= N) throw new ArgumentException("RoIPool batch idx out of range.");
            int x1 = (int)Math.Round(ops.ToDouble(b[k * 5 + 1]) * spatialScale);
            int y1 = (int)Math.Round(ops.ToDouble(b[k * 5 + 2]) * spatialScale);
            int x2 = (int)Math.Round(ops.ToDouble(b[k * 5 + 3]) * spatialScale);
            int y2 = (int)Math.Round(ops.ToDouble(b[k * 5 + 4]) * spatialScale);
            int roiW = Math.Max(x2 - x1 + 1, 1);
            int roiH = Math.Max(y2 - y1 + 1, 1);
            double binH = (double)roiH / outputHeight;
            double binW = (double)roiW / outputWidth;

            for (int c = 0; c < C; c++)
            {
                int planeBase = (n * C + c) * H * W;
                for (int ph = 0; ph < outputHeight; ph++)
                for (int pw = 0; pw < outputWidth; pw++)
                {
                    int hstart = (int)Math.Floor(ph * binH) + y1;
                    int hend = (int)Math.Ceiling((ph + 1) * binH) + y1;
                    int wstart = (int)Math.Floor(pw * binW) + x1;
                    int wend = (int)Math.Ceiling((pw + 1) * binW) + x1;
                    hstart = Math.Min(Math.Max(hstart, 0), H);
                    hend = Math.Min(Math.Max(hend, 0), H);
                    wstart = Math.Min(Math.Max(wstart, 0), W);
                    wend = Math.Min(Math.Max(wend, 0), W);
                    bool empty = hend <= hstart || wend <= wstart;
                    double best = empty ? 0.0 : double.NegativeInfinity;
                    for (int yy = hstart; yy < hend; yy++)
                    for (int xx = wstart; xx < wend; xx++)
                    {
                        double v = ops.ToDouble(src[planeBase + yy * W + xx]);
                        if (v > best) best = v;
                    }
                    dst[((k * C + c) * outputHeight + ph) * outputWidth + pw] =
                        empty ? ops.Zero : ops.FromDouble(best);
                }
            }
        }
        return output;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> PsRoIAlign<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, int outputChannels,
        float spatialScale, int samplingRatio)
    {
        if (input.Rank != 4) throw new ArgumentException("PsRoIAlign input must be [N, C, H, W].");
        if (boxes.Rank != 2 || boxes._shape[1] != 5)
            throw new ArgumentException("PsRoIAlign boxes must be [K, 5] = (batch_idx, x1, y1, x2, y2).");
        int N = input._shape[0], C = input._shape[1], H = input._shape[2], W = input._shape[3];
        if (C != outputChannels * outputHeight * outputWidth)
            throw new ArgumentException($"PsRoIAlign requires C == outputChannels * outH * outW ({outputChannels * outputHeight * outputWidth}); got {C}.");
        int K = boxes._shape[0];
        var output = new Tensor<T>(new[] { K, outputChannels, outputHeight, outputWidth });
        if (K == 0) return output;

        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var b = boxes.AsSpan();
        var dst = output.AsWritableSpan();

        for (int k = 0; k < K; k++)
        {
            int n = (int)ops.ToDouble(b[k * 5]);
            if (n < 0 || n >= N)
                throw new ArgumentException($"PsRoIAlign batch idx {n} out of range [0, {N}).");
            double x1 = ops.ToDouble(b[k * 5 + 1]) * spatialScale;
            double y1 = ops.ToDouble(b[k * 5 + 2]) * spatialScale;
            double x2 = ops.ToDouble(b[k * 5 + 3]) * spatialScale;
            double y2 = ops.ToDouble(b[k * 5 + 4]) * spatialScale;
            double roiW = Math.Max(x2 - x1, 0.1);
            double roiH = Math.Max(y2 - y1, 0.1);
            double binH = roiH / outputHeight;
            double binW = roiW / outputWidth;

            int ry = samplingRatio > 0 ? samplingRatio : (int)Math.Ceiling(roiH / outputHeight);
            int rx = samplingRatio > 0 ? samplingRatio : (int)Math.Ceiling(roiW / outputWidth);
            if (ry < 1) ry = 1;
            if (rx < 1) rx = 1;

            for (int co = 0; co < outputChannels; co++)
            for (int ph = 0; ph < outputHeight; ph++)
            for (int pw = 0; pw < outputWidth; pw++)
            {
                int c = (co * outputHeight + ph) * outputWidth + pw;
                int planeBase = (n * C + c) * H * W;
                double acc = 0;
                for (int iy = 0; iy < ry; iy++)
                {
                    double sy = y1 + ph * binH + (iy + 0.5) * binH / ry;
                    for (int ix = 0; ix < rx; ix++)
                    {
                        double sx = x1 + pw * binW + (ix + 0.5) * binW / rx;
                        acc += BilinearSample(src, planeBase, sy, sx, H, W, ops);
                    }
                }
                dst[((k * outputChannels + co) * outputHeight + ph) * outputWidth + pw] =
                    ops.FromDouble(acc / (ry * rx));
            }
        }
        return output;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> PsRoIPool<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, int outputChannels, float spatialScale)
    {
        if (input.Rank != 4) throw new ArgumentException("PsRoIPool input must be [N, C, H, W].");
        if (boxes.Rank != 2 || boxes._shape[1] != 5)
            throw new ArgumentException("PsRoIPool boxes must be [K, 5].");
        int N = input._shape[0], C = input._shape[1], H = input._shape[2], W = input._shape[3];
        if (C != outputChannels * outputHeight * outputWidth)
            throw new ArgumentException("PsRoIPool requires C == outputChannels * outH * outW.");
        int K = boxes._shape[0];
        var output = new Tensor<T>(new[] { K, outputChannels, outputHeight, outputWidth });
        if (K == 0) return output;

        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var b = boxes.AsSpan();
        var dst = output.AsWritableSpan();

        for (int k = 0; k < K; k++)
        {
            int n = (int)ops.ToDouble(b[k * 5]);
            if (n < 0 || n >= N)
                throw new ArgumentException($"PsRoIPool batch idx {n} out of range [0, {N}).");
            double x1 = ops.ToDouble(b[k * 5 + 1]) * spatialScale;
            double y1 = ops.ToDouble(b[k * 5 + 2]) * spatialScale;
            double x2 = ops.ToDouble(b[k * 5 + 3]) * spatialScale;
            double y2 = ops.ToDouble(b[k * 5 + 4]) * spatialScale;
            double binH = Math.Max(y2 - y1, 0.1) / outputHeight;
            double binW = Math.Max(x2 - x1, 0.1) / outputWidth;

            for (int co = 0; co < outputChannels; co++)
            for (int ph = 0; ph < outputHeight; ph++)
            for (int pw = 0; pw < outputWidth; pw++)
            {
                int c = (co * outputHeight + ph) * outputWidth + pw;
                int planeBase = (n * C + c) * H * W;
                int hs = Math.Max(0, (int)Math.Floor(y1 + ph * binH));
                int he = Math.Min(H, (int)Math.Ceiling(y1 + (ph + 1) * binH));
                int ws = Math.Max(0, (int)Math.Floor(x1 + pw * binW));
                int we = Math.Min(W, (int)Math.Ceiling(x1 + (pw + 1) * binW));
                double acc = 0; int count = 0;
                for (int yy = hs; yy < he; yy++)
                for (int xx = ws; xx < we; xx++)
                {
                    acc += ops.ToDouble(src[planeBase + yy * W + xx]);
                    count++;
                }
                dst[((k * outputChannels + co) * outputHeight + ph) * outputWidth + pw] =
                    count > 0 ? ops.FromDouble(acc / count) : ops.Zero;
            }
        }
        return output;
    }

    // ========================================================================
    // Audio primitives
    // ========================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> Spectrogram<T>(Tensor<T> waveform, int nFft, int hopLength, int winLength, Tensor<T>? window = null)
    {
        // Thin wrapper: run STFT and return only magnitude. STFT here
        // takes (input, nFft, hop, window, center) — the winLength arg
        // on our Spectrogram API is absorbed into the window shape.
        var win = window ?? HannWindow<T>(winLength);
        STFT(waveform, nFft, hopLength, win, center: true,
            out var mag, out _);
        return mag;
    }

    private static Tensor<T> HannWindow<T>(int n)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var w = new Tensor<T>(new[] { n });
        var s = w.AsWritableSpan();
        for (int i = 0; i < n; i++)
            s[i] = ops.FromDouble(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / Math.Max(1, n - 1)));
        return w;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> AmplitudeToDB<T>(Tensor<T> input, float minAmplitude = 1e-10f, float? topDb = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var result = new Tensor<T>(input._shape);
        var dst = result.AsWritableSpan();
        double minAmp = Math.Max(1e-20, minAmplitude);
        double peak = double.NegativeInfinity;
        for (int i = 0; i < src.Length; i++)
        {
            double v = Math.Max(ops.ToDouble(src[i]), minAmp);
            double db = 20.0 * Math.Log10(v);
            if (db > peak) peak = db;
            dst[i] = ops.FromDouble(db);
        }
        if (topDb.HasValue)
        {
            double floor = peak - topDb.Value;
            for (int i = 0; i < dst.Length; i++)
            {
                double v = ops.ToDouble(dst[i]);
                if (v < floor) dst[i] = ops.FromDouble(floor);
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<int> MuLawEncoding<T>(Tensor<T> input, int quantizationChannels = 256)
    {
        if (quantizationChannels < 2)
            throw new ArgumentException("quantizationChannels must be >= 2 (μ = qc − 1 must be positive).",
                nameof(quantizationChannels));
        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var result = new Tensor<int>(input._shape);
        var dst = result.AsWritableSpan();
        double mu = quantizationChannels - 1;
        double logMu = Log1pFallback(mu);
        for (int i = 0; i < src.Length; i++)
        {
            double x = ops.ToDouble(src[i]);
            if (x > 1) x = 1; else if (x < -1) x = -1;
            double y = Math.Sign(x) * Log1pFallback(mu * Math.Abs(x)) / logMu;
            int q = (int)Math.Floor((y + 1) * 0.5 * mu + 0.5);
            if (q < 0) q = 0; else if (q > mu) q = (int)mu;
            dst[i] = q;
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> MuLawDecoding<T>(Tensor<int> input, int quantizationChannels = 256)
    {
        if (quantizationChannels < 2)
            throw new ArgumentException("quantizationChannels must be >= 2.", nameof(quantizationChannels));
        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var result = new Tensor<T>(input._shape);
        var dst = result.AsWritableSpan();
        double mu = quantizationChannels - 1;
        for (int i = 0; i < src.Length; i++)
        {
            double q = src[i];
            double y = (q / mu) * 2 - 1;
            double x = Math.Sign(y) * (Math.Pow(1 + mu, Math.Abs(y)) - 1) / mu;
            dst[i] = ops.FromDouble(x);
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ComputeDeltas<T>(Tensor<T> input, int winLength = 5)
    {
        if (winLength < 3 || (winLength & 1) == 0)
            throw new ArgumentException("winLength must be an odd integer >= 3.");
        int n = winLength / 2;
        // torchaudio convention: denom = 2 * Σ k² for k=1..n.
        double denom = 0.0;
        for (int i = 1; i <= n; i++) denom += 2.0 * i * i;
        int rank = input.Rank;
        if (rank < 1) throw new ArgumentException("input must have at least 1 axis.");
        int tLen = input._shape[rank - 1];
        if (tLen <= 0)
            return new Tensor<T>((int[])input._shape.Clone());
        int leading = input.Length / tLen;

        var result = new Tensor<T>(input._shape);
        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.AsSpan();
        var dst = result.AsWritableSpan();

        for (int row = 0; row < leading; row++)
        {
            int base_ = row * tLen;
            for (int t = 0; t < tLen; t++)
            {
                double acc = 0;
                for (int k = 1; k <= n; k++)
                {
                    int left = t - k < 0 ? 0 : t - k;
                    int right = t + k >= tLen ? tLen - 1 : t + k;
                    acc += k * (ops.ToDouble(src[base_ + right]) - ops.ToDouble(src[base_ + left]));
                }
                dst[base_ + t] = ops.FromDouble(acc / denom);
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> Resample<T>(Tensor<T> waveform, int origRate, int newRate)
    {
        if (origRate <= 0 || newRate <= 0) throw new ArgumentException("rates must be positive.");
        if (waveform.Rank < 1) throw new ArgumentException("waveform must have at least 1 axis.");
        if (waveform._shape[waveform.Rank - 1] == 0)
        {
            var empty = (int[])waveform._shape.Clone();
            empty[waveform.Rank - 1] = 0;
            return new Tensor<T>(empty);
        }
        if (origRate == newRate) return waveform.Clone();
        int gcd = Gcd(origRate, newRate);
        int up = newRate / gcd;
        int down = origRate / gcd;
        // Simplified low-pass sinc kernel. Use a short filter (half-width =
        // maxRate / gcd * 16) to keep CPU cost bounded.
        int halfWidth = Math.Max(8, Math.Min(256, up * 8));
        int tapCount = 2 * halfWidth + 1;
        double cutoff = 1.0 / Math.Max(up, down);

        // Output length for 1D or multi-axis: resample along last axis.
        int tIn = waveform._shape[waveform.Rank - 1];
        int tOut = (int)((long)tIn * up / down);
        var outShape = (int[])waveform._shape.Clone();
        outShape[waveform.Rank - 1] = tOut;
        var result = new Tensor<T>(outShape);
        var ops = MathHelper.GetNumericOperations<T>();
        var src = waveform.AsSpan();
        var dst = result.AsWritableSpan();
        int leading = waveform.Length / tIn;

        for (int row = 0; row < leading; row++)
        {
            int sBase = row * tIn, dBase = row * tOut;
            for (int ot = 0; ot < tOut; ot++)
            {
                double srcIdx = (double)ot * down / up;
                int centre = (int)Math.Floor(srcIdx);
                double acc = 0;
                double weightSum = 0;
                for (int k = -halfWidth; k <= halfWidth; k++)
                {
                    int idx = centre + k;
                    if (idx < 0 || idx >= tIn) continue;
                    double t = (idx - srcIdx) * cutoff;
                    double w = Sinc(t) * Hann(k, halfWidth);
                    acc += w * ops.ToDouble(src[sBase + idx]);
                    weightSum += w;
                }
                dst[dBase + ot] = ops.FromDouble(weightSum > 0 ? acc / weightSum : 0);
            }
        }
        return result;
    }

    /// <summary>net471 doesn't have Math.Log1p; fall back to ln(1+x).</summary>
    private static double Log1pFallback(double x) => Math.Log(1.0 + x);

    private static double Sinc(double x)
    {
        if (Math.Abs(x) < 1e-12) return 1.0;
        double px = Math.PI * x;
        return Math.Sin(px) / px;
    }

    private static double Hann(int k, int halfWidth)
    {
        double n = k + halfWidth;
        int N = 2 * halfWidth;
        return 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * n / N);
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0) { int t = b; b = a % b; a = t; }
        return a;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> PitchShift<T>(Tensor<T> waveform, int sampleRate, double nSteps,
        int nFft = 512, int hopLength = 128)
    {
        // Semitone ratio.
        double rate = Math.Pow(2.0, nSteps / 12.0);
        // Time-stretch by 1/rate so pitch shifts when resampled back.
        var stretched = TimeStretch(waveform, 1.0 / rate, nFft, hopLength);
        // Resample by rate (back to original sample rate) — conceptually.
        int origRate = sampleRate;
        int newRate = (int)Math.Round(sampleRate * rate);
        return Resample(stretched, newRate, origRate);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TimeStretch<T>(Tensor<T> waveform, double rate,
        int nFft = 512, int hopLength = 128)
    {
        if (rate <= 0) throw new ArgumentException("rate must be positive.");
        if (nFft < 2) throw new ArgumentException("nFft must be at least 2 (Hann window divides by nFft−1).");
        if (hopLength <= 0) throw new ArgumentException("hopLength must be positive.");
        if (Math.Abs(rate - 1.0) < 1e-12) return waveform.Clone();
        int winLength = nFft;
        var window = new Tensor<T>(new int[] { winLength });
        // Hann window.
        var ops = MathHelper.GetNumericOperations<T>();
        var wSpan = window.AsWritableSpan();
        for (int i = 0; i < winLength; i++)
            wSpan[i] = ops.FromDouble(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / (winLength - 1)));

        STFT(waveform, nFft, hopLength, window, center: true, out var mag, out var phase);
        // Phase vocoder: remap time axis with linear interpolation, update
        // phase by accumulating the phase difference scaled by the time
        // stretch. Operates on the (last-frame, freq) sub-axes.
        int rank = mag.Rank;
        int nFreq = mag._shape[rank - 1];
        int nFrames = mag._shape[rank - 2];
        int leading = mag.Length / (nFrames * nFreq);
        int outFrames = (int)Math.Floor(nFrames / rate);

        var newShape = (int[])mag._shape.Clone();
        newShape[rank - 2] = outFrames;
        var newMag = new Tensor<T>(newShape);
        var newPhase = new Tensor<T>(newShape);

        var mSpan = mag.AsSpan();
        var pSpan = phase.AsSpan();
        var nmSpan = newMag.AsWritableSpan();
        var npSpan = newPhase.AsWritableSpan();

        for (int b = 0; b < leading; b++)
        {
            int stride = nFrames * nFreq;
            int outStride = outFrames * nFreq;
            var accPhase = new double[nFreq];
            for (int t = 0; t < outFrames; t++)
            {
                double srcT = t * rate;
                int t0 = (int)Math.Floor(srcT);
                int t1 = Math.Min(t0 + 1, nFrames - 1);
                double frac = srcT - t0;
                for (int f = 0; f < nFreq; f++)
                {
                    double m0 = ops.ToDouble(mSpan[b * stride + t0 * nFreq + f]);
                    double m1 = ops.ToDouble(mSpan[b * stride + t1 * nFreq + f]);
                    double m = (1 - frac) * m0 + frac * m1;
                    nmSpan[b * outStride + t * nFreq + f] = ops.FromDouble(m);
                    double dp = 0;
                    if (t0 + 1 < nFrames)
                    {
                        dp = ops.ToDouble(pSpan[b * stride + (t0 + 1) * nFreq + f])
                           - ops.ToDouble(pSpan[b * stride + t0 * nFreq + f]);
                        // Wrap dp to [-pi, pi).
                        dp -= 2 * Math.PI * Math.Round(dp / (2 * Math.PI));
                    }
                    accPhase[f] += dp;
                    npSpan[b * outStride + t * nFreq + f] = ops.FromDouble(accPhase[f]);
                }
            }
        }

        int targetLen = (int)Math.Round(waveform._shape[waveform.Rank - 1] / rate);
        return ISTFT(newMag, newPhase, nFft, hopLength, window, center: true, length: targetLen);
    }
}
