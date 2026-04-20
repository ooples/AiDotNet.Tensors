// Copyright (c) AiDotNet. All rights reserved.
// FFT-based convolution: RFFT → elementwise multiply → IRFFT.
// Wins over direct convolution when kernel is large (K ≳ 31) because direct
// cost is O(H·W·K²) whereas FFT cost is O(H·W·log(H·W)) + per-output-channel
// elementwise mul. Crossover is hardware-dependent; the companion
// FftConvolutionPass has the autotune heuristic.
//
// IMPORTANT: This implementation matches the **cross-correlation** semantics
// that PyTorch's Conv1d / Conv2d use (kernel NOT mathematically flipped) —
// we pre-flip the kernel along its spatial axes so that the FFT-domain
// multiplication corresponds to cross-correlation, not true convolution.
// Result: numerically identical to a direct Conv1d/Conv2d loop.

using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

/// <summary>
/// FFT-based convolution helpers. Produces numerically-identical output to
/// direct linear convolution (up to floating-point roundoff).
/// </summary>
public static class FftConv
{
    /// <summary>
    /// 1D FFT-based convolution. Computes <c>y = conv1d(x, kernel, padding=same)</c>
    /// by padding to <c>N + K − 1</c>, RFft-ing both, multiplying spectra, and
    /// IRFft-ing back. Output shape matches <c>x</c>.
    /// </summary>
    /// <param name="input">Real input of shape <c>[..., N]</c>.</param>
    /// <param name="kernel">Real kernel of shape <c>[K]</c> (no batch).</param>
    /// <returns>Convolved output of shape <c>[..., N]</c> (same padding).</returns>
    public static Tensor<T> Conv1DSame<T>(Tensor<T> input, Tensor<T> kernel)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (kernel.Rank != 1) throw new ArgumentException("kernel must be 1D.", nameof(kernel));
        int N = input.Shape[input.Rank - 1];
        int K = kernel.Shape[0];
        int L = N + K - 1;

        // Zero-pad input and kernel to length L.
        var xPadShape = (int[])input._shape.Clone();
        xPadShape[^1] = L;
        var xPad = new Tensor<T>(xPadShape);
        var xInSrc = input.GetDataArray();
        var xInDst = xPad.GetDataArray();
        int leading = 1;
        for (int i = 0; i < input.Rank - 1; i++) leading *= input._shape[i];
        for (int b = 0; b < leading; b++)
            Array.Copy(xInSrc, b * N, xInDst, b * L, N);

        // Flip kernel along its single axis so that the FFT multiplication
        // corresponds to cross-correlation (ML convention), not pure math
        // convolution.
        var kPad = new Tensor<T>(new[] { L });
        var kSrc = kernel.GetDataArray();
        var kDst = kPad.GetDataArray();
        for (int i = 0; i < K; i++) kDst[i] = kSrc[K - 1 - i];

        var X = Fft.RFft(xPad);
        var KSpec = Fft.RFft(kPad);

        // Broadcast multiply: X has shape [..., 2·(L/2+1)], KSpec has shape [2·(L/2+1)].
        var ops = MathHelper.GetNumericOperations<T>();
        var XDat = X.GetDataArray();
        var KDat = KSpec.GetDataArray();
        int freq = L / 2 + 1;
        var prodT = new Tensor<T>((int[])X._shape.Clone());
        var prodD = prodT.GetDataArray();
        int rowLen = 2 * freq;
        for (int b = 0; b < leading; b++)
        {
            for (int k = 0; k < freq; k++)
            {
                double xRe = ops.ToDouble(XDat[b * rowLen + 2 * k]);
                double xIm = ops.ToDouble(XDat[b * rowLen + 2 * k + 1]);
                double kRe = ops.ToDouble(KDat[2 * k]);
                double kIm = ops.ToDouble(KDat[2 * k + 1]);
                prodD[b * rowLen + 2 * k] = ops.FromDouble(xRe * kRe - xIm * kIm);
                prodD[b * rowLen + 2 * k + 1] = ops.FromDouble(xRe * kIm + xIm * kRe);
            }
        }

        var full = Fft.IRFft(prodT, L);
        // Crop to "same" padding — center-align K−1 samples onto the border.
        int startOffset = K / 2; // floor division: matches PyTorch conv1d with padding=(K-1)/2 floor
        var outShape = (int[])input._shape.Clone();
        var output = new Tensor<T>(outShape);
        var fullD = full.GetDataArray();
        var outD = output.GetDataArray();
        for (int b = 0; b < leading; b++)
            Array.Copy(fullD, b * L + startOffset, outD, b * N, N);
        return output;
    }

    /// <summary>
    /// 2D FFT-based convolution with <c>same</c> padding, <c>stride = 1</c>,
    /// <c>dilation = 1</c>. Output shape matches input spatial dims.
    /// </summary>
    /// <param name="input">Input shape <c>[N, C_in, H, W]</c>.</param>
    /// <param name="weight">Weight shape <c>[C_out, C_in, Kh, Kw]</c>.</param>
    /// <param name="bias">Optional bias shape <c>[C_out]</c>.</param>
    public static Tensor<T> Conv2DSame<T>(Tensor<T> input, Tensor<T> weight, Tensor<T>? bias = null)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (input.Rank != 4) throw new ArgumentException("input must be rank-4 [N, C_in, H, W].", nameof(input));
        if (weight.Rank != 4) throw new ArgumentException("weight must be rank-4 [C_out, C_in, Kh, Kw].", nameof(weight));

        int N = input.Shape[0];
        int Cin = input.Shape[1];
        int H = input.Shape[2];
        int W = input.Shape[3];
        int Cout = weight.Shape[0];
        int CinW = weight.Shape[1];
        int Kh = weight.Shape[2];
        int Kw = weight.Shape[3];
        if (CinW != Cin) throw new ArgumentException($"weight in-channels ({CinW}) must match input channels ({Cin}).");

        // FFT spatial size: next dimensions that fit the full linear convolution.
        int Lh = H + Kh - 1;
        int Lw = W + Kw - 1;

        var ops = MathHelper.GetNumericOperations<T>();

        // Pre-transform all weight kernels: shape [Cout, Cin, Lh, 2·(Lw/2+1)] complex
        //    (RFft along last spatial dim; FFT along the second spatial dim.)
        int freqW = Lw / 2 + 1;
        var weightSpec = ComputeWeight2DSpectra(weight, Lh, Lw, freqW);

        // For each input [N, Cin] pad to [N, Cin, Lh, Lw], FFT, multiply+sum over Cin
        // per output channel, IFFT, crop to H×W.
        var output = new Tensor<T>(new[] { N, Cout, H, W });
        var outD = output.GetDataArray();
        var inD = input.GetDataArray();

        double[]? biasD = null;
        if (bias is not null)
        {
            if (bias.Rank != 1 || bias.Shape[0] != Cout)
                throw new ArgumentException($"bias must be 1D with length {Cout}.", nameof(bias));
            biasD = new double[Cout];
            var b = bias.GetDataArray();
            for (int i = 0; i < Cout; i++) biasD[i] = ops.ToDouble(b[i]);
        }

        // Batch loop parallelized.
        Parallel.For(0, N, n =>
        {
            // Transform input for this batch: pad each input channel to [Lh, Lw], 2D-RFft.
            var inputSpec = new double[Cin * Lh * 2 * freqW];
            for (int cin = 0; cin < Cin; cin++)
            {
                var spatial = new double[Lh * Lw];
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        spatial[y * Lw + x] = ops.ToDouble(inD[((n * Cin + cin) * H + y) * W + x]);
                var specSlice = Transform2DForward(spatial, Lh, Lw, freqW);
                Array.Copy(specSlice, 0, inputSpec, cin * Lh * 2 * freqW, Lh * 2 * freqW);
            }

            // For each output channel: sum over Cin of (inputSpec[cin] .* weightSpec[cout, cin]),
            // then IFFT2 and crop.
            for (int cout = 0; cout < Cout; cout++)
            {
                var accum = new double[Lh * 2 * freqW];
                for (int cin = 0; cin < Cin; cin++)
                {
                    int inBase = cin * Lh * 2 * freqW;
                    int wBase = (cout * Cin + cin) * Lh * 2 * freqW;
                    for (int i = 0; i < Lh; i++)
                    {
                        for (int k = 0; k < freqW; k++)
                        {
                            double xRe = inputSpec[inBase + i * 2 * freqW + 2 * k];
                            double xIm = inputSpec[inBase + i * 2 * freqW + 2 * k + 1];
                            double kRe = weightSpec[wBase + i * 2 * freqW + 2 * k];
                            double kIm = weightSpec[wBase + i * 2 * freqW + 2 * k + 1];
                            accum[i * 2 * freqW + 2 * k] += xRe * kRe - xIm * kIm;
                            accum[i * 2 * freqW + 2 * k + 1] += xRe * kIm + xIm * kRe;
                        }
                    }
                }
                // Inverse 2D RFFT: IFFT along rows (complex), then IRFft along cols.
                var spatial = Transform2DInverse(accum, Lh, Lw, freqW);
                // Crop to H×W with 'same' offset.
                int yOff = Kh / 2;
                int xOff = Kw / 2;
                double biasVal = biasD is null ? 0.0 : biasD[cout];
                for (int y = 0; y < H; y++)
                {
                    for (int x = 0; x < W; x++)
                    {
                        double v = spatial[(y + yOff) * Lw + (x + xOff)] + biasVal;
                        outD[((n * Cout + cout) * H + y) * W + x] = ops.FromDouble(v);
                    }
                }
            }
        });

        return output;
    }

    // ── Internal 2D transform helpers (direct double[] scratch) ─────────────
    private static double[] ComputeWeight2DSpectra<T>(Tensor<T> weight, int Lh, int Lw, int freqW)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        int Cout = weight.Shape[0];
        int Cin = weight.Shape[1];
        int Kh = weight.Shape[2];
        int Kw = weight.Shape[3];
        var ops = MathHelper.GetNumericOperations<T>();
        var wD = weight.GetDataArray();
        var result = new double[Cout * Cin * Lh * 2 * freqW];
        Parallel.For(0, Cout * Cin, idx =>
        {
            int cout = idx / Cin;
            int cin = idx % Cin;
            var spatial = new double[Lh * Lw];
            // Flip kernel along both spatial axes so the FFT product
            // realizes cross-correlation (ML conv) rather than pure math
            // convolution.
            for (int y = 0; y < Kh; y++)
                for (int x = 0; x < Kw; x++)
                    spatial[y * Lw + x] = ops.ToDouble(wD[((cout * Cin + cin) * Kh + (Kh - 1 - y)) * Kw + (Kw - 1 - x)]);
            var spec = Transform2DForward(spatial, Lh, Lw, freqW);
            Array.Copy(spec, 0, result, (cout * Cin + cin) * Lh * 2 * freqW, Lh * 2 * freqW);
        });
        return result;
    }

    // 2D RFFT: real [Lh, Lw] → packed complex [Lh, 2·freqW].
    private static double[] Transform2DForward(double[] spatial, int Lh, int Lw, int freqW)
    {
        // Row-wise RFft: result layout [Lh, 2·freqW].
        var rows = new double[Lh * 2 * freqW];
        for (int y = 0; y < Lh; y++)
        {
            var row = new double[2 * Lw];
            for (int x = 0; x < Lw; x++) row[2 * x] = spatial[y * Lw + x];
            FftKernels.Transform1D(row, Lw, inverse: false, FftNorm.Backward);
            // Keep first freqW complex bins (the RFft half).
            for (int k = 0; k < freqW; k++)
            {
                rows[y * 2 * freqW + 2 * k] = row[2 * k];
                rows[y * 2 * freqW + 2 * k + 1] = row[2 * k + 1];
            }
        }
        // Column-wise complex FFT along the first axis.
        for (int k = 0; k < freqW; k++)
        {
            var col = new double[2 * Lh];
            for (int y = 0; y < Lh; y++)
            {
                col[2 * y] = rows[y * 2 * freqW + 2 * k];
                col[2 * y + 1] = rows[y * 2 * freqW + 2 * k + 1];
            }
            FftKernels.Transform1D(col, Lh, inverse: false, FftNorm.Backward);
            for (int y = 0; y < Lh; y++)
            {
                rows[y * 2 * freqW + 2 * k] = col[2 * y];
                rows[y * 2 * freqW + 2 * k + 1] = col[2 * y + 1];
            }
        }
        return rows;
    }

    // 2D inverse RFFT: packed complex [Lh, 2·freqW] → real [Lh, Lw].
    private static double[] Transform2DInverse(double[] rows, int Lh, int Lw, int freqW)
    {
        // Column-wise inverse complex FFT first.
        for (int k = 0; k < freqW; k++)
        {
            var col = new double[2 * Lh];
            for (int y = 0; y < Lh; y++)
            {
                col[2 * y] = rows[y * 2 * freqW + 2 * k];
                col[2 * y + 1] = rows[y * 2 * freqW + 2 * k + 1];
            }
            FftKernels.Transform1D(col, Lh, inverse: true, FftNorm.Backward);
            for (int y = 0; y < Lh; y++)
            {
                rows[y * 2 * freqW + 2 * k] = col[2 * y];
                rows[y * 2 * freqW + 2 * k + 1] = col[2 * y + 1];
            }
        }
        // Row-wise inverse RFft.
        var spatial = new double[Lh * Lw];
        for (int y = 0; y < Lh; y++)
        {
            var buf = new double[2 * Lw];
            for (int k = 0; k < freqW; k++)
            {
                buf[2 * k] = rows[y * 2 * freqW + 2 * k];
                buf[2 * k + 1] = rows[y * 2 * freqW + 2 * k + 1];
            }
            // Hermitian-mirror for conjugate bins.
            for (int k = 1; k < freqW - (Lw % 2 == 0 ? 1 : 0); k++)
            {
                buf[2 * (Lw - k)] = buf[2 * k];
                buf[2 * (Lw - k) + 1] = -buf[2 * k + 1];
            }
            FftKernels.Transform1D(buf, Lw, inverse: true, FftNorm.Backward);
            for (int x = 0; x < Lw; x++) spatial[y * Lw + x] = buf[2 * x];
        }
        return spatial;
    }
}
