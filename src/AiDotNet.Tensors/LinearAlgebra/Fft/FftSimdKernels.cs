// Copyright (c) AiDotNet. All rights reserved.
// AVX2 / AVX-512 radix-2 butterfly stages for the iterative Cooley-Tukey FFT.
//
// The scalar radix-2 in FftKernels.IterativeRadix2 iterates butterflies with:
//     w  ← 1
//     for k in 0..half:
//         t = w * odd;  even ± t
//         w *= wStep
//
// For each SIMD-wide slab of butterflies we vectorize the inner k-loop by
// pre-computing the twiddles w[0..vlen-1] and advancing w by vlen steps per
// iteration. A length-N stage has N/2 butterflies; with AVX2 we process 2
// complex (= 4 doubles) per vector, with AVX-512 we process 4 complex (= 8
// doubles). When "halfSize" is smaller than the vector lane count we fall
// back to scalar for that stage.
//
// Supported lanes:
//     AVX2:    2 complex (ymm registers, 4 doubles)
//     AVX-512: 4 complex (zmm registers, 8 doubles)
//     fallback: scalar (already in FftKernels.IterativeRadix2)

#if NET7_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

internal static class FftSimdKernels
{
    internal static bool Avx512Supported => Avx512F.IsSupported;
    internal static bool Avx2Supported => Avx2.IsSupported;

    /// <summary>
    /// Run one iterative radix-2 stage of size <paramref name="stageSize"/>
    /// using AVX-512 lanes (4 complex per vector). The buffer is interleaved
    /// complex (re/im doubles); stage loops over <c>n / stageSize</c>
    /// butterfly groups, each of <c>half = stageSize / 2</c> butterflies.
    /// Requires <paramref name="half"/> &gt;= 4.
    /// </summary>
    internal static unsafe void Radix2StageAvx512(
        double* buf, int n, int stageSize, bool inverse)
    {
        int half = stageSize >> 1;
        double sign = inverse ? 1.0 : -1.0;
        double theta = sign * 2.0 * Math.PI / stageSize;

        // Pre-compute the first 4 twiddle factors (for lanes 0..3) and the
        // step-by-4 twiddle so we can advance w by 4 lanes per iteration.
        var wReLane = Vector256.Create(
            Math.Cos(0 * theta),
            Math.Cos(1 * theta),
            Math.Cos(2 * theta),
            Math.Cos(3 * theta));
        var wImLane = Vector256.Create(
            Math.Sin(0 * theta),
            Math.Sin(1 * theta),
            Math.Sin(2 * theta),
            Math.Sin(3 * theta));

        double theta4 = 4.0 * theta;
        double step4Re = Math.Cos(theta4);
        double step4Im = Math.Sin(theta4);
        var step4ReV = Vector256.Create(step4Re);
        var step4ImV = Vector256.Create(step4Im);

        for (int start = 0; start < n; start += stageSize)
        {
            var wRe = wReLane;
            var wIm = wImLane;
            int k = 0;

            // Vectorize 4 butterflies at a time while we have room.
            for (; k + 4 <= half; k += 4)
            {
                int evenBase = 2 * (start + k);
                int oddBase = 2 * (start + k + half);

                // Gather even / odd lanes into ymm registers. Loads are
                // interleaved re/im so re = [buf[2i], buf[2(i+1)], ...] and
                // im = [buf[2i+1], ...]. Use shuffle to deinterleave.
                var evenReIm0 = Vector128.Create(buf[evenBase + 0], buf[evenBase + 1]);
                var evenReIm1 = Vector128.Create(buf[evenBase + 2], buf[evenBase + 3]);
                var evenReIm2 = Vector128.Create(buf[evenBase + 4], buf[evenBase + 5]);
                var evenReIm3 = Vector128.Create(buf[evenBase + 6], buf[evenBase + 7]);
                var oddReIm0 = Vector128.Create(buf[oddBase + 0], buf[oddBase + 1]);
                var oddReIm1 = Vector128.Create(buf[oddBase + 2], buf[oddBase + 3]);
                var oddReIm2 = Vector128.Create(buf[oddBase + 4], buf[oddBase + 5]);
                var oddReIm3 = Vector128.Create(buf[oddBase + 6], buf[oddBase + 7]);

                // Deinterleave into real/imag vectors.
                var evenRe = Vector256.Create(
                    evenReIm0.GetElement(0), evenReIm1.GetElement(0),
                    evenReIm2.GetElement(0), evenReIm3.GetElement(0));
                var evenIm = Vector256.Create(
                    evenReIm0.GetElement(1), evenReIm1.GetElement(1),
                    evenReIm2.GetElement(1), evenReIm3.GetElement(1));
                var oddRe = Vector256.Create(
                    oddReIm0.GetElement(0), oddReIm1.GetElement(0),
                    oddReIm2.GetElement(0), oddReIm3.GetElement(0));
                var oddIm = Vector256.Create(
                    oddReIm0.GetElement(1), oddReIm1.GetElement(1),
                    oddReIm2.GetElement(1), oddReIm3.GetElement(1));

                // t = w * odd
                var tRe = Avx.Subtract(Avx.Multiply(wRe, oddRe), Avx.Multiply(wIm, oddIm));
                var tIm = Avx.Add(Avx.Multiply(wRe, oddIm), Avx.Multiply(wIm, oddRe));

                // even ← even + t ; odd ← even - t
                var newEvenRe = Avx.Add(evenRe, tRe);
                var newEvenIm = Avx.Add(evenIm, tIm);
                var newOddRe = Avx.Subtract(evenRe, tRe);
                var newOddIm = Avx.Subtract(evenIm, tIm);

                // Re-interleave and store back.
                for (int lane = 0; lane < 4; lane++)
                {
                    buf[evenBase + 2 * lane] = newEvenRe.GetElement(lane);
                    buf[evenBase + 2 * lane + 1] = newEvenIm.GetElement(lane);
                    buf[oddBase + 2 * lane] = newOddRe.GetElement(lane);
                    buf[oddBase + 2 * lane + 1] = newOddIm.GetElement(lane);
                }

                // Advance w by 4 steps:  w ← w * step⁴
                var newWRe = Avx.Subtract(Avx.Multiply(wRe, step4ReV), Avx.Multiply(wIm, step4ImV));
                var newWIm = Avx.Add(Avx.Multiply(wRe, step4ImV), Avx.Multiply(wIm, step4ReV));
                wRe = newWRe;
                wIm = newWIm;
            }

            // Scalar tail.
            double wReScalar = wRe.GetElement(0);
            double wImScalar = wIm.GetElement(0);
            double wStepReScalar = Math.Cos(theta);
            double wStepImScalar = Math.Sin(theta);
            for (; k < half; k++)
            {
                int eIdx = 2 * (start + k);
                int oIdx = 2 * (start + k + half);
                double eRe = buf[eIdx], eIm = buf[eIdx + 1];
                double oRe = buf[oIdx], oIm = buf[oIdx + 1];
                double tRe = wReScalar * oRe - wImScalar * oIm;
                double tIm = wReScalar * oIm + wImScalar * oRe;
                buf[eIdx] = eRe + tRe;
                buf[eIdx + 1] = eIm + tIm;
                buf[oIdx] = eRe - tRe;
                buf[oIdx + 1] = eIm - tIm;
                double nwRe = wReScalar * wStepReScalar - wImScalar * wStepImScalar;
                double nwIm = wReScalar * wStepImScalar + wImScalar * wStepReScalar;
                wReScalar = nwRe;
                wImScalar = nwIm;
            }
        }
    }

    /// <summary>
    /// AVX2 version: 2 complex butterflies per vector (Vector128 of doubles).
    /// Requires <paramref name="half"/> &gt;= 2.
    /// </summary>
    internal static unsafe void Radix2StageAvx2(
        double* buf, int n, int stageSize, bool inverse)
    {
        int half = stageSize >> 1;
        double sign = inverse ? 1.0 : -1.0;
        double theta = sign * 2.0 * Math.PI / stageSize;

        // Pre-compute twiddles for lanes 0..1 and the step-by-2 advance.
        var wReLane = Vector128.Create(Math.Cos(0 * theta), Math.Cos(1 * theta));
        var wImLane = Vector128.Create(Math.Sin(0 * theta), Math.Sin(1 * theta));
        double theta2 = 2.0 * theta;
        var step2Re = Vector128.Create(Math.Cos(theta2));
        var step2Im = Vector128.Create(Math.Sin(theta2));

        for (int start = 0; start < n; start += stageSize)
        {
            var wRe = wReLane;
            var wIm = wImLane;
            int k = 0;
            for (; k + 2 <= half; k += 2)
            {
                int evenBase = 2 * (start + k);
                int oddBase = 2 * (start + k + half);

                var evenRe = Vector128.Create(buf[evenBase + 0], buf[evenBase + 2]);
                var evenIm = Vector128.Create(buf[evenBase + 1], buf[evenBase + 3]);
                var oddRe = Vector128.Create(buf[oddBase + 0], buf[oddBase + 2]);
                var oddIm = Vector128.Create(buf[oddBase + 1], buf[oddBase + 3]);

                var tRe = Sse2.Subtract(Sse2.Multiply(wRe, oddRe), Sse2.Multiply(wIm, oddIm));
                var tIm = Sse2.Add(Sse2.Multiply(wRe, oddIm), Sse2.Multiply(wIm, oddRe));

                var newEvenRe = Sse2.Add(evenRe, tRe);
                var newEvenIm = Sse2.Add(evenIm, tIm);
                var newOddRe = Sse2.Subtract(evenRe, tRe);
                var newOddIm = Sse2.Subtract(evenIm, tIm);

                buf[evenBase + 0] = newEvenRe.GetElement(0);
                buf[evenBase + 1] = newEvenIm.GetElement(0);
                buf[evenBase + 2] = newEvenRe.GetElement(1);
                buf[evenBase + 3] = newEvenIm.GetElement(1);
                buf[oddBase + 0] = newOddRe.GetElement(0);
                buf[oddBase + 1] = newOddIm.GetElement(0);
                buf[oddBase + 2] = newOddRe.GetElement(1);
                buf[oddBase + 3] = newOddIm.GetElement(1);

                // Advance w by 2 steps.
                var newWRe = Sse2.Subtract(Sse2.Multiply(wRe, step2Re), Sse2.Multiply(wIm, step2Im));
                var newWIm = Sse2.Add(Sse2.Multiply(wRe, step2Im), Sse2.Multiply(wIm, step2Re));
                wRe = newWRe;
                wIm = newWIm;
            }

            // Scalar tail.
            double wReScalar = wRe.GetElement(0);
            double wImScalar = wIm.GetElement(0);
            double wStepReScalar = Math.Cos(theta);
            double wStepImScalar = Math.Sin(theta);
            for (; k < half; k++)
            {
                int eIdx = 2 * (start + k);
                int oIdx = 2 * (start + k + half);
                double eRe = buf[eIdx], eIm = buf[eIdx + 1];
                double oRe = buf[oIdx], oIm = buf[oIdx + 1];
                double tRe = wReScalar * oRe - wImScalar * oIm;
                double tIm = wReScalar * oIm + wImScalar * oRe;
                buf[eIdx] = eRe + tRe;
                buf[eIdx + 1] = eIm + tIm;
                buf[oIdx] = eRe - tRe;
                buf[oIdx + 1] = eIm - tIm;
                double nwRe = wReScalar * wStepReScalar - wImScalar * wStepImScalar;
                double nwIm = wReScalar * wStepImScalar + wImScalar * wStepReScalar;
                wReScalar = nwRe;
                wImScalar = nwIm;
            }
        }
    }

    /// <summary>
    /// Dispatch a single radix-2 stage to the best available SIMD path.
    /// Returns <c>true</c> if a SIMD path ran, <c>false</c> if the caller
    /// must fall back to scalar (because neither path was applicable).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe bool TryRadix2Stage(
        double* buf, int n, int stageSize, bool inverse)
    {
        int half = stageSize >> 1;
        if (Avx512F.IsSupported && half >= 4)
        {
            Radix2StageAvx512(buf, n, stageSize, inverse);
            return true;
        }
        if (Avx2.IsSupported && half >= 2)
        {
            Radix2StageAvx2(buf, n, stageSize, inverse);
            return true;
        }
        return false;
    }
}
#endif
