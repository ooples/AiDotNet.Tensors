// Copyright (c) AiDotNet. All rights reserved.
// Public FFT surface — mirrors torch.fft with C# naming conventions.
//
// Layout convention:
//   Complex tensors are stored with the LAST axis doubled (interleaved
//   real/imag). A length-N complex signal lives in a tensor whose final
//   axis is 2N, stored as [re₀ im₀ re₁ im₁ … re_{N−1} im_{N−1}]. This:
//     * matches the existing IEngine.RFFT / IRFFT / FFT convention
//     * matches how the GPU kernels already pass data
//     * avoids forcing Tensor<Complex<T>> on every caller
//
// For each op we compute by:
//   1. Reshaping input so the transform axis is the last axis
//   2. Iterating over the leading "batch" with Parallel.For
//   3. Copying each complex row to a double[2*N] scratch
//   4. Calling FftKernels.Transform1D
//   5. Copying back
// N-dimensional ops decompose into axis-separable 1D passes (in-place
// via transpose/permute to put each axis on the fastest stride).

using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

/// <summary>
/// Public FFT API. All complex I/O uses the "last axis doubled" interleaved
/// real/imag layout. See the <see cref="Fft"/> class-level remarks for the
/// convention and for how multi-dim variants are wired.
/// </summary>
public static class Fft
{
    // ═══════════════════════════════════════════════════════════════════════
    // 1D COMPLEX
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>Forward 1D complex FFT along the last axis.</summary>
    public static Tensor<T> Fft1<T>(Tensor<T> input, int? n = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var result = TransformComplex1D(input, n, norm, inverse: false);
        FftAutograd.RecordFft1(result, input, n ?? (input.Shape[input.Rank - 1] / 2), norm);
        return result;
    }

    /// <summary>Inverse 1D complex FFT along the last axis.</summary>
    public static Tensor<T> IFft1<T>(Tensor<T> input, int? n = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var result = TransformComplex1D(input, n, norm, inverse: true);
        FftAutograd.RecordIFft1(result, input, n ?? (input.Shape[input.Rank - 1] / 2), norm);
        return result;
    }

    /// <summary>Forward 2D complex FFT along the last two axes.</summary>
    public static Tensor<T> Fft2<T>(Tensor<T> input, int[]? s = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => TransformComplexND(input, s, axes: null, norm, inverse: false, axesCount: 2);

    /// <summary>Inverse 2D complex FFT along the last two axes.</summary>
    public static Tensor<T> IFft2<T>(Tensor<T> input, int[]? s = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => TransformComplexND(input, s, axes: null, norm, inverse: true, axesCount: 2);

    /// <summary>Forward N-D complex FFT along the specified (or trailing) axes.</summary>
    public static Tensor<T> FftN<T>(Tensor<T> input, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => TransformComplexND(input, s, axes, norm, inverse: false, axesCount: s?.Length ?? axes?.Length ?? input.Rank);

    /// <summary>Inverse N-D complex FFT along the specified (or trailing) axes.</summary>
    public static Tensor<T> IFftN<T>(Tensor<T> input, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => TransformComplexND(input, s, axes, norm, inverse: true, axesCount: s?.Length ?? axes?.Length ?? input.Rank);

    // ═══════════════════════════════════════════════════════════════════════
    // REAL FFT (RFFT / IRFFT)
    //   Input is real; output is length (N/2+1) complex (interleaved).
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// 1D real-to-complex FFT along the last axis. Output last-axis length is
    /// <c>2·(N/2 + 1)</c> (interleaved real/imag).
    /// </summary>
    public static Tensor<T> RFft<T>(Tensor<T> input, int? n = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        int realLen = n ?? input.Shape[input.Rank - 1];
        var result = RealTransform1D(input, n, norm, inverse: false);
        FftAutograd.RecordRFft(result, input, realLen, norm);
        return result;
    }

    /// <summary>
    /// 1D complex-to-real inverse FFT along the last axis.
    /// <paramref name="n"/> controls the output real length (default
    /// <c>2·(K−1)</c> where <c>K = last-axis length / 2</c>).
    /// </summary>
    public static Tensor<T> IRFft<T>(Tensor<T> input, int? n = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        int realLen = n ?? 2 * (input.Shape[input.Rank - 1] / 2 - 1);
        var result = RealTransform1D(input, n, norm, inverse: true);
        FftAutograd.RecordIRFft(result, input, realLen, norm);
        return result;
    }

    /// <summary>2D real FFT along the last two axes.</summary>
    public static Tensor<T> RFft2<T>(Tensor<T> input, int[]? s = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => RealTransformND(input, s, axes: null, norm, inverse: false, axesCount: 2);

    /// <summary>2D inverse real FFT along the last two axes.</summary>
    public static Tensor<T> IRFft2<T>(Tensor<T> input, int[]? s = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => RealTransformND(input, s, axes: null, norm, inverse: true, axesCount: 2);

    /// <summary>N-D real FFT along the specified (or trailing) axes.</summary>
    public static Tensor<T> RFftN<T>(Tensor<T> input, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => RealTransformND(input, s, axes, norm, inverse: false, axesCount: s?.Length ?? axes?.Length ?? input.Rank);

    /// <summary>N-D inverse real FFT along the specified (or trailing) axes.</summary>
    public static Tensor<T> IRFftN<T>(Tensor<T> input, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => RealTransformND(input, s, axes, norm, inverse: true, axesCount: s?.Length ?? axes?.Length ?? input.Rank);

    // ═══════════════════════════════════════════════════════════════════════
    // HERMITIAN FFT (HFFT / IHFFT)
    //   HFft:  complex-Hermitian input → real output (inverse of IHFft).
    //   IHFft: real input → complex-Hermitian output (inverse of HFft).
    // Relationships: HFft(X, n) = FFT(conj(X_extended), n) where X is
    // Hermitian-packed; IHFft(x, n) = conj(RFft(x, n)) / n with Backward norm.
    // Implementation: reuse RFft/IRFft machinery, flip conjugation and norm
    // sides — see the body of HFft/IHFft for the exact derivation.
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// 1D Hermitian FFT along the last axis: complex Hermitian-symmetric input
    /// (last axis of size 2·(N/2+1)) → real output of length <paramref name="n"/>.
    /// </summary>
    public static Tensor<T> HFft<T>(Tensor<T> input, int? n = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // HFft(X, n) is algebraically IRFFT(conj(X), n) under the convention
        // that the output is real. Equivalently: FFT of the full Hermitian-
        // extended complex signal, taking the real part.
        return RealTransform1D(ConjugateLastAxis(input), n, InvertNorm(norm), inverse: true);
    }

    /// <summary>
    /// 1D inverse Hermitian FFT along the last axis: real input → complex
    /// Hermitian-packed output (last axis length 2·(N/2+1)).
    /// </summary>
    public static Tensor<T> IHFft<T>(Tensor<T> input, int? n = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // IHFft(x, n) is algebraically conj(RFFT(x, n)) with the opposite
        // normalization side — derives from HFft(IHFft(x)) = x.
        var rfft = RealTransform1D(input, n, InvertNorm(norm), inverse: false);
        return ConjugateLastAxis(rfft);
    }

    /// <summary>2D Hermitian FFT along the last two axes.</summary>
    public static Tensor<T> HFft2<T>(Tensor<T> input, int[]? s = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => HermitianTransformND(input, s, axes: null, norm, inverse: false, axesCount: 2);

    /// <summary>2D inverse Hermitian FFT along the last two axes.</summary>
    public static Tensor<T> IHFft2<T>(Tensor<T> input, int[]? s = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => HermitianTransformND(input, s, axes: null, norm, inverse: true, axesCount: 2);

    /// <summary>N-D Hermitian FFT along the specified (or trailing) axes.</summary>
    public static Tensor<T> HFftN<T>(Tensor<T> input, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => HermitianTransformND(input, s, axes, norm, inverse: false, axesCount: s?.Length ?? axes?.Length ?? input.Rank);

    /// <summary>N-D inverse Hermitian FFT along the specified (or trailing) axes.</summary>
    public static Tensor<T> IHFftN<T>(Tensor<T> input, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => HermitianTransformND(input, s, axes, norm, inverse: true, axesCount: s?.Length ?? axes?.Length ?? input.Rank);

    // ═══════════════════════════════════════════════════════════════════════
    // FREQUENCY / SHIFT HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Returns the discrete Fourier transform sample frequencies for a length-
    /// <paramref name="n"/> signal sampled with spacing <paramref name="d"/>.
    /// Layout: <c>[0, 1, 2, …, n/2−1, −n/2, …, −1] / (d·n)</c> for even
    /// <paramref name="n"/>, slightly different for odd.
    /// </summary>
    public static Tensor<T> FftFreq<T>(int n, double d = 1.0)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (n <= 0) throw new ArgumentException("n must be positive.", nameof(n));
        var result = new Tensor<T>(new[] { n });
        var data = result.GetDataArray();
        double scale = 1.0 / (d * n);
        int split = (n + 1) / 2; // number of non-negative frequencies
        for (int i = 0; i < split; i++) data[i] = FromDouble<T>(i * scale);
        for (int i = split; i < n; i++) data[i] = FromDouble<T>((i - n) * scale);
        return result;
    }

    /// <summary>
    /// Non-negative sample frequencies for an <c>RFFT</c> of length
    /// <paramref name="n"/> sampled with spacing <paramref name="d"/>.
    /// Layout: <c>[0, 1, …, n/2] / (d·n)</c>. Length is <c>n/2 + 1</c>.
    /// </summary>
    public static Tensor<T> RFftFreq<T>(int n, double d = 1.0)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (n <= 0) throw new ArgumentException("n must be positive.", nameof(n));
        int m = n / 2 + 1;
        var result = new Tensor<T>(new[] { m });
        var data = result.GetDataArray();
        double scale = 1.0 / (d * n);
        for (int i = 0; i < m; i++) data[i] = FromDouble<T>(i * scale);
        return result;
    }

    /// <summary>
    /// <c>fftshift</c>: circular-shift the zero-frequency component to the
    /// center. Applied along the specified <paramref name="axes"/> (default:
    /// all axes). Inverse is <see cref="IFftShift"/>.
    /// </summary>
    public static Tensor<T> FftShift<T>(Tensor<T> input, int[]? axes = null)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => Shift(input, axes, inverse: false);

    /// <summary>Inverse of <see cref="FftShift"/>.</summary>
    public static Tensor<T> IFftShift<T>(Tensor<T> input, int[]? axes = null)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => Shift(input, axes, inverse: true);

    // ═══════════════════════════════════════════════════════════════════════
    // INTERNAL PLUMBING
    // ═══════════════════════════════════════════════════════════════════════

    // 1D complex FFT along the last axis.
    // Input last axis must be 2*N (interleaved). If `n` is provided, crop or
    // zero-pad the complex signal to length n before transforming.
    private static Tensor<T> TransformComplex1D<T>(Tensor<T> input, int? nRequested, FftNorm norm, bool inverse)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int rank = input.Rank;
        int last = input.Shape[rank - 1];
        if (last % 2 != 0) throw new ArgumentException("Complex FFT requires last axis length to be even (interleaved re/im pairs).");
        int nIn = last / 2;
        int n = nRequested ?? nIn;
        if (n <= 0) throw new ArgumentException("Transform length must be positive.", nameof(nRequested));

        // Output shape: same as input except last axis = 2*n.
        var outShape = (int[])input._shape.Clone();
        outShape[rank - 1] = 2 * n;
        var result = new Tensor<T>(outShape);

        int batch = 1;
        for (int i = 0; i < rank - 1; i++) batch *= input._shape[i];
        var inD = input.GetDataArray();
        var rD = result.GetDataArray();
        int inStride = 2 * nIn;
        int outStride = 2 * n;

        var ops = MathHelper.GetNumericOperations<T>();
        Parallel.For(0, batch, b =>
        {
            var buf = new double[2 * n];
            int copy = Math.Min(nIn, n);
            for (int i = 0; i < copy; i++)
            {
                buf[2 * i] = ops.ToDouble(inD[b * inStride + 2 * i]);
                buf[2 * i + 1] = ops.ToDouble(inD[b * inStride + 2 * i + 1]);
            }
            // Remaining entries already zero from array init.
            FftKernels.Transform1D(buf, n, inverse, norm);
            for (int i = 0; i < n; i++)
            {
                rD[b * outStride + 2 * i] = ops.FromDouble(buf[2 * i]);
                rD[b * outStride + 2 * i + 1] = ops.FromDouble(buf[2 * i + 1]);
            }
        });
        return result;
    }

    // 1D real FFT (forward) / inverse (complex → real).
    private static Tensor<T> RealTransform1D<T>(Tensor<T> input, int? nRequested, FftNorm norm, bool inverse)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int rank = input.Rank;
        int last = input.Shape[rank - 1];

        int n; // logical (real) transform length
        if (inverse)
        {
            // Input last axis is 2*K interleaved, K = N/2+1 → N = 2*(K-1) by default.
            if (last % 2 != 0) throw new ArgumentException("IRFFT input last axis must be even (interleaved re/im).");
            int kIn = last / 2;
            n = nRequested ?? 2 * (kIn - 1);
        }
        else
        {
            n = nRequested ?? last;
        }
        if (n <= 0) throw new ArgumentException("Transform length must be positive.", nameof(nRequested));

        int numFreqs = n / 2 + 1;
        var outShape = (int[])input._shape.Clone();
        outShape[rank - 1] = inverse ? n : 2 * numFreqs;
        var result = new Tensor<T>(outShape);

        int batch = 1;
        for (int i = 0; i < rank - 1; i++) batch *= input._shape[i];
        var inD = input.GetDataArray();
        var rD = result.GetDataArray();
        int inStride = last;
        int outStride = outShape[rank - 1];

        var ops = MathHelper.GetNumericOperations<T>();
        if (!inverse)
        {
            // Forward real FFT: pack real into complex, run full FFT, output first K bins.
            // (Packing-trick optimization comes later — this version is correct and clear.)
            Parallel.For(0, batch, b =>
            {
                var buf = new double[2 * n];
                int copyR = Math.Min(last, n);
                for (int i = 0; i < copyR; i++) buf[2 * i] = ops.ToDouble(inD[b * inStride + i]);
                FftKernels.Transform1D(buf, n, inverse: false, norm);
                for (int k = 0; k < numFreqs; k++)
                {
                    rD[b * outStride + 2 * k] = ops.FromDouble(buf[2 * k]);
                    rD[b * outStride + 2 * k + 1] = ops.FromDouble(buf[2 * k + 1]);
                }
            });
        }
        else
        {
            // Inverse real FFT: reconstruct full Hermitian-symmetric spectrum from
            // the positive-frequency half, run IFFT, take real part.
            int kIn = last / 2;
            Parallel.For(0, batch, b =>
            {
                var buf = new double[2 * n];
                int copyK = Math.Min(kIn, numFreqs);
                for (int k = 0; k < copyK; k++)
                {
                    buf[2 * k] = ops.ToDouble(inD[b * inStride + 2 * k]);
                    buf[2 * k + 1] = ops.ToDouble(inD[b * inStride + 2 * k + 1]);
                }
                // Mirror conjugate for k = 1..n-numFreqs (i.e., the negative half).
                for (int k = 1; k < numFreqs - (n % 2 == 0 ? 1 : 0); k++)
                {
                    buf[2 * (n - k)] = buf[2 * k];
                    buf[2 * (n - k) + 1] = -buf[2 * k + 1];
                }
                // For even n, bin numFreqs-1 is the Nyquist bin — real — no mirror.
                FftKernels.Transform1D(buf, n, inverse: true, norm);
                for (int i = 0; i < n; i++) rD[b * outStride + i] = ops.FromDouble(buf[2 * i]);
            });
        }
        return result;
    }

    // N-D complex FFT: apply 1D transform along each requested axis in turn.
    private static Tensor<T> TransformComplexND<T>(Tensor<T> input, int[]? s, int[]? axes, FftNorm norm, bool inverse, int axesCount)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        (int[] resolvedAxes, int[] sizes) = ResolveAxesAndSizes(input, s, axes, axesCount, complexLastAxis: true);
        var cur = input;
        // Apply along each axis. For the last axis we can use TransformComplex1D directly;
        // for others we permute the target axis to the last, transform, permute back.
        // To preserve the "last axis is interleaved real/imag" convention:
        //   - for axis == rank-1: transform 1D along last axis (interleaved)
        //   - for axis != rank-1: treat that axis as real; the complex dimension stays
        //     as the last axis. We permute the target axis to just before last, handle
        //     the interleaving explicitly in the 1D pass on a reshape.
        // For simplicity and correctness: always bring the target axis to position rank-1
        // *while treating the existing last axis (re/im) as a hidden pair dimension*.
        // That is implemented by interpreting the complex buffer as a (prefix, axis_len, 2)
        // tensor and applying 1D transform along axis_len.
        for (int ax = 0; ax < resolvedAxes.Length; ax++)
        {
            int axis = resolvedAxes[ax];
            int n = sizes[ax];
            cur = TransformComplexAlongAxis(cur, axis, n, norm, inverse);
        }
        return cur;
    }

    private static Tensor<T> TransformComplexAlongAxis<T>(Tensor<T> input, int axis, int n, FftNorm norm, bool inverse)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        int rank = input.Rank;
        if (axis < 0) axis += rank;
        // The last axis is interleaved real/imag. If axis is the last logical (complex) axis,
        // the tensor layout's last axis is 2*n_old and this is TransformComplex1D.
        int complexAxis = rank - 1;
        if (axis == complexAxis / 2) // conceptual check: we want the "logical" last axis
            // Actually we just dispatch by physical axis: target axis is a real axis EXCEPT when it's
            // the LAST logical axis; the tensor's last physical axis is always interleaved.
            { }

        // Physical layout: tensor has shape (...dims..., 2*N_complex_last). The "logical" axes are
        // (...dims..., N_complex_last). Each caller passes axes in logical coordinates.
        // We translate: logical axis == rank-1 means the physical last (interleaved) axis.
        // Otherwise the target axis is a real axis — we bring it to position rank-2 so the last two
        // axes are (target, 2), apply 1D FFT per complex row, permute back.
        int physicalRank = rank; // tensor rank
        int logicalRank = physicalRank; // same since we store complex as doubled last axis
        int targetPhysical = axis;

        if (targetPhysical == physicalRank - 1)
        {
            // Transform along the interleaved axis — but the "interleaved axis" contains 2*n doubles
            // representing n complex numbers. This is TransformComplex1D.
            return TransformComplex1D(input, n, norm, inverse);
        }

        // Otherwise permute target to rank-2 so the layout becomes (prefix..., target_len, 2*n_last).
        var perm = new int[rank];
        int idx = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i == targetPhysical || i == rank - 1) continue;
            perm[idx++] = i;
        }
        perm[idx++] = targetPhysical;
        perm[idx] = rank - 1;
        var permuted = input.Transpose(perm);
        // Now permuted's shape is (prefix..., targetLen, 2*n_last).
        // The last two axes form (targetLen, 2*n_last); we want to FFT along targetLen where each
        // "row" at a given last-axis offset is a 1D complex slice (striding every 2 elements for re/im).
        // Simplest: manually pack each complex 1D slice into a scratch buffer.
        var prm = permuted.Contiguous();
        int prmRank = prm.Rank;
        int innerComplex = prm.Shape[prmRank - 1] / 2; // n_last (number of complex points in last axis)
        int targetLen = prm.Shape[prmRank - 2];
        int prefix = 1;
        for (int i = 0; i < prmRank - 2; i++) prefix *= prm._shape[i];

        // Transform each (prefix, innerComplex) → length-n complex FFT along targetLen.
        var outShape = (int[])prm._shape.Clone();
        outShape[prmRank - 2] = n; // crop/pad
        var outTensor = new Tensor<T>(outShape);
        var inD = prm.GetDataArray();
        var outD = outTensor.GetDataArray();
        int inRowStride = 2 * innerComplex;
        int outRowStride = 2 * innerComplex;
        int inMatStride = targetLen * inRowStride;
        int outMatStride = n * outRowStride;
        var ops = MathHelper.GetNumericOperations<T>();

        Parallel.For(0, prefix * innerComplex, pi =>
        {
            int p = pi / innerComplex;
            int c = pi % innerComplex;
            var buf = new double[2 * n];
            int copy = Math.Min(targetLen, n);
            for (int t = 0; t < copy; t++)
            {
                int srcOff = p * inMatStride + t * inRowStride + 2 * c;
                buf[2 * t] = ops.ToDouble(inD[srcOff]);
                buf[2 * t + 1] = ops.ToDouble(inD[srcOff + 1]);
            }
            FftKernels.Transform1D(buf, n, inverse, norm);
            for (int t = 0; t < n; t++)
            {
                int dstOff = p * outMatStride + t * outRowStride + 2 * c;
                outD[dstOff] = ops.FromDouble(buf[2 * t]);
                outD[dstOff + 1] = ops.FromDouble(buf[2 * t + 1]);
            }
        });

        // Reverse the permutation.
        var invPerm = new int[rank];
        for (int i = 0; i < rank; i++) invPerm[perm[i]] = i;
        return outTensor.Transpose(invPerm).Contiguous();
    }

    // N-D real FFT: apply RFft/IRFft on the final listed axis, FftN on the rest.
    private static Tensor<T> RealTransformND<T>(Tensor<T> input, int[]? s, int[]? axes, FftNorm norm, bool inverse, int axesCount)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        // Convention: for RFftN the RFFT is taken along the LAST listed axis; the
        // remaining listed axes get complex FFT. For IRFftN the order is reversed:
        // complex FFT on the leading axes, then IRFFT on the last listed axis.
        (int[] resolvedAxes, int[] sizes) = ResolveAxesAndSizes(input, s, axes, axesCount, complexLastAxis: false);

        if (!inverse)
        {
            // Apply complex FFT on the leading axes (which are real at this point, but
            // we lift to complex by... wait — for the leading axes the input is real.)
            // We need to first RFFT along the LAST listed axis, which gives us a complex
            // tensor (last physical axis doubled), then apply complex FFT along the
            // remaining leading axes.
            int lastAxis = resolvedAxes[resolvedAxes.Length - 1];
            int lastSize = sizes[sizes.Length - 1];
            var rfftResult = RealTransformAlongAxis(input, lastAxis, lastSize, norm, inverse: false);
            // Now apply complex FFT along each of the remaining axes (in resolvedAxes[..-1]).
            var cur = rfftResult;
            for (int i = 0; i < resolvedAxes.Length - 1; i++)
            {
                cur = TransformComplexAlongAxis(cur, resolvedAxes[i], sizes[i], norm, inverse: false);
            }
            return cur;
        }
        else
        {
            // Apply IFFT along all leading axes, then IRFFT along the last listed axis.
            var cur = input;
            for (int i = 0; i < resolvedAxes.Length - 1; i++)
            {
                cur = TransformComplexAlongAxis(cur, resolvedAxes[i], sizes[i], norm, inverse: true);
            }
            int lastAxis = resolvedAxes[resolvedAxes.Length - 1];
            int lastSize = sizes[sizes.Length - 1];
            return RealTransformAlongAxis(cur, lastAxis, lastSize, norm, inverse: true);
        }
    }

    private static Tensor<T> RealTransformAlongAxis<T>(Tensor<T> input, int axis, int n, FftNorm norm, bool inverse)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        int rank = input.Rank;
        if (axis < 0) axis += rank;
        if (axis == rank - 1)
            return RealTransform1D(input, n, norm, inverse);

        // Permute target to last; transform; permute back.
        var perm = new int[rank];
        int idx = 0;
        for (int i = 0; i < rank; i++) if (i != axis) perm[idx++] = i;
        perm[idx] = axis;
        var permuted = input.Transpose(perm).Contiguous();
        var transformed = RealTransform1D(permuted, n, norm, inverse);
        var invPerm = new int[rank];
        for (int i = 0; i < rank; i++) invPerm[perm[i]] = i;
        return transformed.Transpose(invPerm).Contiguous();
    }

    // Hermitian (HFFT / IHFFT) N-D: thin wrapper that conjugates and inverts the norm.
    private static Tensor<T> HermitianTransformND<T>(Tensor<T> input, int[]? s, int[]? axes, FftNorm norm, bool inverse, int axesCount)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!inverse)
        {
            // HFftN(x) = IRFftN(conj(x), norm') — see the 1D derivation above.
            return RealTransformND(ConjugateLastAxis(input), s, axes, InvertNorm(norm), inverse: true, axesCount);
        }
        else
        {
            var rfft = RealTransformND(input, s, axes, InvertNorm(norm), inverse: false, axesCount);
            return ConjugateLastAxis(rfft);
        }
    }

    // ── Small helpers ──────────────────────────────────────────────────────

    private static (int[] axes, int[] sizes) ResolveAxesAndSizes<T>(
        Tensor<T> input, int[]? s, int[]? axes, int axesCount, bool complexLastAxis)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        int rank = input.Rank;
        int[] resolvedAxes = axes is null
            ? Enumerable.Range(rank - axesCount, axesCount).ToArray()
            : axes.Select(a => a < 0 ? a + rank : a).ToArray();
        int[] sizes;
        if (s is null)
        {
            sizes = new int[axesCount];
            for (int i = 0; i < axesCount; i++)
            {
                int ax = resolvedAxes[i];
                int dim = input.Shape[ax];
                // For the last (complex) axis in a complex tensor, the logical length is dim/2.
                sizes[i] = complexLastAxis && ax == rank - 1 ? dim / 2 : dim;
            }
        }
        else
        {
            if (s.Length != axesCount) throw new ArgumentException("s length must match axesCount.");
            sizes = s;
        }
        return (resolvedAxes, sizes);
    }

    private static Tensor<T> Shift<T>(Tensor<T> input, int[]? axes, bool inverse)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int rank = input.Rank;
        int[] ax = axes is null
            ? Enumerable.Range(0, rank).ToArray()
            : axes.Select(a => a < 0 ? a + rank : a).ToArray();

        var cur = input;
        foreach (int a in ax)
        {
            int dim = cur.Shape[a];
            // fftshift: shift by dim/2 (floor); ifftshift: shift by dim - dim/2.
            int shift = inverse ? dim - dim / 2 : dim / 2;
            cur = Roll(cur, a, shift);
        }
        return cur;
    }

    // Simple contiguous roll along a single axis, returning a new tensor.
    private static Tensor<T> Roll<T>(Tensor<T> input, int axis, int shift)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        int rank = input.Rank;
        int dim = input.Shape[axis];
        shift = ((shift % dim) + dim) % dim;
        if (shift == 0) return input;

        var src = input.Contiguous();
        var result = new Tensor<T>((int[])src._shape.Clone());
        var srcData = src.GetDataArray();
        var dstData = result.GetDataArray();

        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= src._shape[i];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++) inner *= src._shape[i];

        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < dim; i++)
            {
                int newI = (i + shift) % dim;
                int srcOff = (o * dim + i) * inner;
                int dstOff = (o * dim + newI) * inner;
                Array.Copy(srcData, srcOff, dstData, dstOff, inner);
            }
        }
        return result;
    }

    private static Tensor<T> ConjugateLastAxis<T>(Tensor<T> input)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var result = new Tensor<T>((int[])input._shape.Clone());
        var src = input.GetDataArray();
        var dst = result.GetDataArray();
        var ops = MathHelper.GetNumericOperations<T>();
        int last = input.Shape[input.Rank - 1];
        int batch = src.Length / last;
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < last; i += 2)
            {
                dst[b * last + i] = src[b * last + i];
                dst[b * last + i + 1] = ops.Negate(src[b * last + i + 1]);
            }
        }
        return result;
    }

    private static FftNorm InvertNorm(FftNorm norm) => norm switch
    {
        FftNorm.Backward => FftNorm.Forward,
        FftNorm.Forward => FftNorm.Backward,
        FftNorm.Ortho => FftNorm.Ortho,
        _ => throw new ArgumentOutOfRangeException(nameof(norm)),
    };

    private static T FromDouble<T>(double v)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => MathHelper.GetNumericOperations<T>().FromDouble(v);
}
