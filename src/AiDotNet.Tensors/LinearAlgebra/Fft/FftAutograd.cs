// Copyright (c) AiDotNet. All rights reserved.
// Autograd wiring for the Fft module.
//
// Derivation (complex FFT, interleaved real/imag layout):
//   Let M be the 2N×2N real matrix that implements y = FFT(x) on interleaved
//   real/imag pairs with the "backward" (unscaled forward) convention. Direct
//   computation of Mᵀ shows Mᵀ == IFFT_unscaled (the unnormalized inverse
//   DFT matrix). Therefore for a forward FFT with overall scale ε_fwd:
//
//     y = ε_fwd · M · x
//     dL/dx = ε_fwd · Mᵀ · dL/dy = ε_fwd · IFFT_unscaled(dL/dy)
//
//   Case by case, with N = transform length:
//     ε_fwd = 1       (Backward):  dL/dx = IFFT_unscaled       = N · IFFT_backward = IFFT_forward
//     ε_fwd = 1/N     (Forward):   dL/dx = (1/N) · IFFT_unscaled = IFFT_backward
//     ε_fwd = 1/√N    (Ortho):     dL/dx = (1/√N) · IFFT_unscaled = IFFT_ortho
//
//   So the backward always uses the DUAL norm:
//     Backward ↔ Forward (dual pair)
//     Forward  ↔ Backward
//     Ortho    ↔ Ortho   (self-dual, because unitary)
//
//   For RFft/IRFft the same dual-norm rule applies modulo factor-of-2 handling
//   for the non-edge bins (conjugate-symmetric packing), which IRFft's
//   Hermitian-expansion already takes care of correctly.
//
// See FftGradcheckTests for finite-difference validation across all three
// norms.

using System;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

/// <summary>
/// Tape recording helpers for the <see cref="Fft"/> module. Each public FFT
/// op calls one of these after computing its forward value, registering the
/// closed-form backward with <see cref="DifferentiableOps"/>.
/// </summary>
internal static class FftAutograd
{
    internal static void RecordFft1<T>(Tensor<T> output, Tensor<T> input, int n, FftNorm norm)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        DifferentiableOps.RecordUnary("Fft.Fft1", output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int savedN = (int)saved[0];
            var dualNorm = DualNorm((FftNorm)saved[1]);
            var gx = Fft.IFft1(gradOut, savedN, dualNorm);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { n, norm });
    }

    internal static void RecordIFft1<T>(Tensor<T> output, Tensor<T> input, int n, FftNorm norm)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        DifferentiableOps.RecordUnary("Fft.IFft1", output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int savedN = (int)saved[0];
            var dualNorm = DualNorm((FftNorm)saved[1]);
            var gx = Fft.Fft1(gradOut, savedN, dualNorm);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { n, norm });
    }

    internal static void RecordRFft<T>(Tensor<T> output, Tensor<T> input, int n, FftNorm norm)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        DifferentiableOps.RecordUnary("Fft.RFft", output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int savedN = (int)saved[0];
            var dualNorm = DualNorm((FftNorm)saved[1]);
            // RFft packs a length-N real input into K = N/2+1 complex bins by
            // using the Hermitian symmetry of the full spectrum. A unit change
            // in an INTERIOR bin of the packed output corresponds to a unit
            // change in TWO full-spectrum bins (the bin itself + its mirror),
            // so dL/d(packed_bin) enters the derivative with weight 1 but
            // the IRFft that reconstructs x doubles it via the mirror
            // expansion. We compensate by halving the interior bins of
            // gradOut before IRFft, then the result equals the true Jacobian.
            var scaled = HalveInteriorBins(gradOut, savedN);
            var gx = Fft.IRFft(scaled, savedN, dualNorm);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { n, norm });
    }

    internal static void RecordIRFft<T>(Tensor<T> output, Tensor<T> input, int n, FftNorm norm)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        DifferentiableOps.RecordUnary("Fft.IRFft", output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int savedN = (int)saved[0];
            var dualNorm = DualNorm((FftNorm)saved[1]);
            // Dual of the RFft rule: the IRFft Hermitian-expansion means a unit
            // change in an interior packed bin contributes to TWO full-spectrum
            // bins. After RFft'ing the real gradient we must double the interior
            // bins to get the true Jacobian w.r.t. the packed complex input.
            var rfft = Fft.RFft(gradOut, savedN, dualNorm);
            var gx = DoubleInteriorBins<T>(rfft, savedN);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { n, norm });
    }

    private static Tensor<T> HalveInteriorBins<T>(Tensor<T> packed, int n)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => ScaleInteriorBins(packed, n, scale: 0.5);

    private static Tensor<T> DoubleInteriorBins<T>(Tensor<T> packed, int n)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => ScaleInteriorBins(packed, n, scale: 2.0);

    /// <summary>
    /// Scale the interior (non-DC, non-Nyquist-when-N-even) bins of a packed
    /// RFft spectrum by <paramref name="scale"/>. Layout: last axis contains
    /// 2·(N/2 + 1) interleaved re/im doubles.
    /// </summary>
    private static Tensor<T> ScaleInteriorBins<T>(Tensor<T> packed, int n, double scale)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int K = n / 2 + 1;
        // Copy so we don't mutate the gradient tensor in place.
        var result = new Tensor<T>((int[])packed._shape.Clone());
        var src = packed.GetDataArray();
        var dst = result.GetDataArray();
        int last = packed.Shape[packed.Rank - 1];
        int batch = src.Length / last;
        bool evenN = n % 2 == 0;

        for (int b = 0; b < batch; b++)
        {
            int off = b * last;
            // DC bin (index 0): unchanged.
            dst[off] = src[off];
            dst[off + 1] = src[off + 1];
            // Interior bins k = 1 .. K-2 (or K-1 when N is odd) → scaled.
            int interiorEnd = evenN ? K - 1 : K;
            for (int k = 1; k < interiorEnd; k++)
            {
                dst[off + 2 * k] = ops.FromDouble(scale * ops.ToDouble(src[off + 2 * k]));
                dst[off + 2 * k + 1] = ops.FromDouble(scale * ops.ToDouble(src[off + 2 * k + 1]));
            }
            // Nyquist bin (only exists when N is even) at index K-1: unchanged.
            if (evenN && K >= 2)
            {
                dst[off + 2 * (K - 1)] = src[off + 2 * (K - 1)];
                dst[off + 2 * (K - 1) + 1] = src[off + 2 * (K - 1) + 1];
            }
        }
        return result;
    }

    // ── 2D / ND complex variants ───────────────────────────────────────────
    // Same rule as 1D: backward is the inverse transform under the dual norm,
    // applied along the same axes / sizes.

    internal static void RecordFftN<T>(Tensor<T> output, Tensor<T> input, int[]? s, int[]? axes, int axesCount, FftNorm norm, bool inverse)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        // Snapshot caller-owned s/axes arrays so a later caller mutation
        // doesn't change the backward pass parameters.
        int outRank = output.Rank;
        int[] savedAxes = axes is null ? DefaultTrailingAxes(outRank, axesCount) : (int[])axes.Clone();
        int[] savedS = s is null ? new int[] { } : (int[])s.Clone();
        string name = inverse ? "Fft.IFftN" : "Fft.FftN";
        DifferentiableOps.RecordUnary(name, output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int[] ss = (int[])saved[0];
            int[] axs = (int[])saved[1];
            var dualNorm = DualNorm((FftNorm)saved[2]);
            bool fwdWasInverse = (bool)saved[3];
            int[]? sArg = ss.Length == 0 ? null : ss;
            var gx = fwdWasInverse
                ? Fft.FftN(gradOut, sArg, axs, dualNorm)
                : Fft.IFftN(gradOut, sArg, axs, dualNorm);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { savedS, savedAxes, norm, inverse });
    }

    internal static void RecordRFftN<T>(Tensor<T> output, Tensor<T> input, int[]? s, int[]? axes, int axesCount, FftNorm norm, bool inverse, int lastAxisSize)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        int outRank = output.Rank;
        int[] savedAxes = axes is null ? DefaultTrailingAxes(outRank, axesCount) : (int[])axes.Clone();
        int[] savedS = s is null ? new int[] { } : (int[])s.Clone();
        string name = inverse ? "Fft.IRFftN" : "Fft.RFftN";
        DifferentiableOps.RecordUnary(name, output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int[] ss = (int[])saved[0];
            int[] axs = (int[])saved[1];
            var dualNorm = DualNorm((FftNorm)saved[2]);
            bool fwdWasInverse = (bool)saved[3];
            int lastSize = (int)saved[4];
            int[]? sArg = ss.Length == 0 ? null : ss;
            Tensor<T> gx;
            if (fwdWasInverse)
            {
                var rfft = Fft.RFftN(gradOut, sArg, axs, dualNorm);
                gx = DoubleInteriorLastAxis(rfft, lastSize);
            }
            else
            {
                var halved = HalveInteriorLastAxis(gradOut, lastSize);
                gx = Fft.IRFftN(halved, sArg, axs, dualNorm);
            }
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { savedS, savedAxes, norm, inverse, lastAxisSize });
    }

    private static int[] DefaultTrailingAxes(int rank, int count)
    {
        var arr = new int[count];
        for (int i = 0; i < count; i++) arr[i] = rank - count + i;
        return arr;
    }

    // ── HFft / IHFft family ────────────────────────────────────────────────
    // HFft(X) = IRFft(conj(X), n, dualNorm(norm))
    //   ⇒ dL/dX = conj(doubleInterior(RFft(gradOut, n, norm)))
    // IHFft(x) = conj(RFft(x, n, dualNorm(norm)))
    //   ⇒ dL/dx = IRFft(halveInterior(conj(gradOut)), n, norm)

    // N-D Hermitian variants — derivation extends the 1D rule axis-wise.
    //   HFftN(X) = IRFftN(conj(X), s, axes, dualNorm)
    //     ⇒ dL/dX = conj(doubleInterior(RFftN(gradOut, s, axes, norm)))
    //   IHFftN(x) = conj(RFftN(x, s, axes, dualNorm))
    //     ⇒ dL/dx = IRFftN(halveInterior(conj(gradOut)), s, axes, norm)

    internal static void RecordHFftN<T>(Tensor<T> output, Tensor<T> input, int[]? s, int[]? axes, int axesCount, FftNorm norm, int lastAxisSize)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        int inRank = input.Rank;
        int[] savedAxes = axes is null ? DefaultTrailingAxes(inRank, axesCount) : (int[])axes.Clone();
        int[] savedS = s is null ? new int[] { } : (int[])s.Clone();
        DifferentiableOps.RecordUnary("Fft.HFftN", output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int[] ss = (int[])saved[0];
            int[] axs = (int[])saved[1];
            var savedNorm = (FftNorm)saved[2];
            int lastSize = (int)saved[3];
            int[]? sArg = ss.Length == 0 ? null : ss;
            var rfft = Fft.RFftN(gradOut, sArg, axs, savedNorm);
            var doubled = DoubleInteriorLastAxis(rfft, lastSize);
            var gx = ConjugateLastAxis(doubled);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { savedS, savedAxes, norm, lastAxisSize });
    }

    internal static void RecordIHFftN<T>(Tensor<T> output, Tensor<T> input, int[]? s, int[]? axes, int axesCount, FftNorm norm, int lastAxisSize)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        int inRank = input.Rank;
        int[] savedAxes = axes is null ? DefaultTrailingAxes(inRank, axesCount) : (int[])axes.Clone();
        int[] savedS = s is null ? new int[] { } : (int[])s.Clone();
        DifferentiableOps.RecordUnary("Fft.IHFftN", output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int[] ss = (int[])saved[0];
            int[] axs = (int[])saved[1];
            var savedNorm = (FftNorm)saved[2];
            int lastSize = (int)saved[3];
            int[]? sArg = ss.Length == 0 ? null : ss;
            var conj = ConjugateLastAxis(gradOut);
            var halved = HalveInteriorLastAxis(conj, lastSize);
            var gx = Fft.IRFftN(halved, sArg, axs, savedNorm);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { savedS, savedAxes, norm, lastAxisSize });
    }

    internal static void RecordHFft<T>(Tensor<T> output, Tensor<T> input, int n, FftNorm norm)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        DifferentiableOps.RecordUnary("Fft.HFft", output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int savedN = (int)saved[0];
            var savedNorm = (FftNorm)saved[1];
            var rfft = Fft.RFft(gradOut, savedN, savedNorm);
            var doubled = DoubleInteriorBins<T>(rfft, savedN);
            var gx = ConjugateLastAxis(doubled);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { n, norm });
    }

    internal static void RecordIHFft<T>(Tensor<T> output, Tensor<T> input, int n, FftNorm norm)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        if (!DifferentiableOps.AnyTapeActive()) return;
        DifferentiableOps.RecordUnary("Fft.IHFft", output, input, static (gradOut, inputs, _, saved, engine, grads) =>
        {
            int savedN = (int)saved[0];
            var savedNorm = (FftNorm)saved[1];
            var conj = ConjugateLastAxis(gradOut);
            var halved = HalveInteriorBins<T>(conj, savedN);
            var gx = Fft.IRFft(halved, savedN, savedNorm);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
        }, new object[] { n, norm });
    }

    // ── ND versions of the scaling helpers used by RFftN backward ──────────
    private static Tensor<T> HalveInteriorLastAxis<T>(Tensor<T> packed, int lastAxisSize)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => HalveInteriorBins(packed, lastAxisSize); // same rule — scales along last axis

    private static Tensor<T> DoubleInteriorLastAxis<T>(Tensor<T> packed, int lastAxisSize)
        where T : unmanaged, IEquatable<T>, IComparable<T>
        => DoubleInteriorBins(packed, lastAxisSize);

    // Complex conjugate along the packed-complex last axis.
    private static Tensor<T> ConjugateLastAxis<T>(Tensor<T> packed)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>((int[])packed._shape.Clone());
        var src = packed.GetDataArray();
        var dst = result.GetDataArray();
        int last = packed.Shape[packed.Rank - 1];
        int batch = src.Length / last;
        for (int b = 0; b < batch; b++)
        {
            int off = b * last;
            for (int i = 0; i < last; i += 2)
            {
                dst[off + i] = src[off + i];
                dst[off + i + 1] = ops.Negate(src[off + i + 1]);
            }
        }
        return result;
    }

    private static FftNorm DualNorm(FftNorm norm) => norm switch
    {
        FftNorm.Backward => FftNorm.Forward,
        FftNorm.Forward => FftNorm.Backward,
        FftNorm.Ortho => FftNorm.Ortho,
        _ => throw new ArgumentOutOfRangeException(nameof(norm)),
    };
}
