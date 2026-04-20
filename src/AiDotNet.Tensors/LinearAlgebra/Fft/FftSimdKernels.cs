// Copyright (c) AiDotNet. All rights reserved.
// Vectorized radix-2 butterfly stages for the iterative Cooley-Tukey FFT,
// specialized per SIMD width. Each stage handles N/2 butterflies; we batch
// the inner `k` loop into SIMD groups of (vector-width / sizeof(double))
// butterflies at a time, pre-computing lane-0..V-1 twiddles and advancing
// `w` by V steps per iteration via a single complex multiply.
//
// Lane widths (doubles per vector):
//     Vector512  →  8 butterflies per iter (true AVX-512, NET 8+)
//     Vector256  →  4 butterflies per iter (AVX / AVX2)
//     Vector128  →  2 butterflies per iter (SSE2 baseline)
//
// The dispatcher TryRadix2Stage picks the widest applicable path based on
// `half = stageSize / 2` (vector width must not exceed `half`) and the
// runtime's ISA probes. Stages with half < 2 fall through to scalar.

#if NET7_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AiDotNet.Tensors.LinearAlgebra.Fft;

internal static class FftSimdKernels
{
#if NET8_0_OR_GREATER
    internal static bool Avx512Supported => Avx512F.IsSupported;
#else
    internal static bool Avx512Supported => false;
#endif
    internal static bool Avx2Supported => Avx2.IsSupported;
    internal static bool Sse2Supported => Sse2.IsSupported;

#if NET8_0_OR_GREATER
    /// <summary>
    /// AVX-512 radix-2 stage: 8 complex butterflies per vectorized inner
    /// iteration (Vector512 of doubles). Requires <c>half ≥ 8</c>.
    /// </summary>
    internal static unsafe void Radix2StageVector512(
        double* buf, int n, int stageSize, bool inverse)
    {
        int half = stageSize >> 1;
        double sign = inverse ? 1.0 : -1.0;
        double theta = sign * 2.0 * Math.PI / stageSize;

        // Lane twiddles for k = 0..7 and the step-by-8 advance factor.
        var wRe = Vector512.Create(
            Math.Cos(0 * theta), Math.Cos(1 * theta), Math.Cos(2 * theta), Math.Cos(3 * theta),
            Math.Cos(4 * theta), Math.Cos(5 * theta), Math.Cos(6 * theta), Math.Cos(7 * theta));
        var wIm = Vector512.Create(
            Math.Sin(0 * theta), Math.Sin(1 * theta), Math.Sin(2 * theta), Math.Sin(3 * theta),
            Math.Sin(4 * theta), Math.Sin(5 * theta), Math.Sin(6 * theta), Math.Sin(7 * theta));
        var step8Re = Vector512.Create(Math.Cos(8.0 * theta));
        var step8Im = Vector512.Create(Math.Sin(8.0 * theta));

        // Stack scratch for deinterleave/interleave and twiddle tail
        // (hoisted out of every loop to satisfy CA2014).
        double* evenReArr = stackalloc double[8];
        double* evenImArr = stackalloc double[8];
        double* oddReArr = stackalloc double[8];
        double* oddImArr = stackalloc double[8];
        double* resEvenRe = stackalloc double[8];
        double* resEvenIm = stackalloc double[8];
        double* resOddRe = stackalloc double[8];
        double* resOddIm = stackalloc double[8];
        double* wReTail = stackalloc double[8];
        double* wImTail = stackalloc double[8];

        for (int start = 0; start < n; start += stageSize)
        {
            var curWRe = wRe;
            var curWIm = wIm;
            int k = 0;
            for (; k + 8 <= half; k += 8)
            {
                int evenBase = 2 * (start + k);
                int oddBase = 2 * (start + k + half);

                for (int lane = 0; lane < 8; lane++)
                {
                    evenReArr[lane] = buf[evenBase + 2 * lane];
                    evenImArr[lane] = buf[evenBase + 2 * lane + 1];
                    oddReArr[lane] = buf[oddBase + 2 * lane];
                    oddImArr[lane] = buf[oddBase + 2 * lane + 1];
                }
                var evenRe = Vector512.Load(evenReArr);
                var evenIm = Vector512.Load(evenImArr);
                var oddRe = Vector512.Load(oddReArr);
                var oddIm = Vector512.Load(oddImArr);

                // t = w * odd
                var tRe = Avx512F.Subtract(Avx512F.Multiply(curWRe, oddRe), Avx512F.Multiply(curWIm, oddIm));
                var tIm = Avx512F.Add(Avx512F.Multiply(curWRe, oddIm), Avx512F.Multiply(curWIm, oddRe));

                var newEvenRe = Avx512F.Add(evenRe, tRe);
                var newEvenIm = Avx512F.Add(evenIm, tIm);
                var newOddRe = Avx512F.Subtract(evenRe, tRe);
                var newOddIm = Avx512F.Subtract(evenIm, tIm);

                newEvenRe.Store(resEvenRe);
                newEvenIm.Store(resEvenIm);
                newOddRe.Store(resOddRe);
                newOddIm.Store(resOddIm);
                for (int lane = 0; lane < 8; lane++)
                {
                    buf[evenBase + 2 * lane] = resEvenRe[lane];
                    buf[evenBase + 2 * lane + 1] = resEvenIm[lane];
                    buf[oddBase + 2 * lane] = resOddRe[lane];
                    buf[oddBase + 2 * lane + 1] = resOddIm[lane];
                }

                // w ← w * step⁸
                var nwRe = Avx512F.Subtract(Avx512F.Multiply(curWRe, step8Re), Avx512F.Multiply(curWIm, step8Im));
                var nwIm = Avx512F.Add(Avx512F.Multiply(curWRe, step8Im), Avx512F.Multiply(curWIm, step8Re));
                curWRe = nwRe;
                curWIm = nwIm;
            }

            // Scalar tail.
            curWRe.Store(wReTail);
            curWIm.Store(wImTail);
            double wReScalar = wReTail[0];
            double wImScalar = wImTail[0];
            double wStepRe = Math.Cos(theta);
            double wStepIm = Math.Sin(theta);
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
                double nwRe2 = wReScalar * wStepRe - wImScalar * wStepIm;
                double nwIm2 = wReScalar * wStepIm + wImScalar * wStepRe;
                wReScalar = nwRe2;
                wImScalar = nwIm2;
            }
        }
    }
#endif

    /// <summary>
    /// AVX/AVX2 radix-2 stage: 4 complex butterflies per vectorized inner
    /// iteration (Vector256 of doubles). Requires <c>half ≥ 4</c>.
    /// </summary>
    internal static unsafe void Radix2StageVector256(
        double* buf, int n, int stageSize, bool inverse)
    {
        int half = stageSize >> 1;
        double sign = inverse ? 1.0 : -1.0;
        double theta = sign * 2.0 * Math.PI / stageSize;

        var wReInit = Vector256.Create(
            Math.Cos(0 * theta), Math.Cos(1 * theta),
            Math.Cos(2 * theta), Math.Cos(3 * theta));
        var wImInit = Vector256.Create(
            Math.Sin(0 * theta), Math.Sin(1 * theta),
            Math.Sin(2 * theta), Math.Sin(3 * theta));
        var step4Re = Vector256.Create(Math.Cos(4.0 * theta));
        var step4Im = Vector256.Create(Math.Sin(4.0 * theta));

        for (int start = 0; start < n; start += stageSize)
        {
            var wRe = wReInit;
            var wIm = wImInit;
            int k = 0;
            for (; k + 4 <= half; k += 4)
            {
                int evenBase = 2 * (start + k);
                int oddBase = 2 * (start + k + half);

                var evenRe = Vector256.Create(
                    buf[evenBase + 0], buf[evenBase + 2], buf[evenBase + 4], buf[evenBase + 6]);
                var evenIm = Vector256.Create(
                    buf[evenBase + 1], buf[evenBase + 3], buf[evenBase + 5], buf[evenBase + 7]);
                var oddRe = Vector256.Create(
                    buf[oddBase + 0], buf[oddBase + 2], buf[oddBase + 4], buf[oddBase + 6]);
                var oddIm = Vector256.Create(
                    buf[oddBase + 1], buf[oddBase + 3], buf[oddBase + 5], buf[oddBase + 7]);

                var tRe = Avx.Subtract(Avx.Multiply(wRe, oddRe), Avx.Multiply(wIm, oddIm));
                var tIm = Avx.Add(Avx.Multiply(wRe, oddIm), Avx.Multiply(wIm, oddRe));

                var newEvenRe = Avx.Add(evenRe, tRe);
                var newEvenIm = Avx.Add(evenIm, tIm);
                var newOddRe = Avx.Subtract(evenRe, tRe);
                var newOddIm = Avx.Subtract(evenIm, tIm);

                for (int lane = 0; lane < 4; lane++)
                {
                    buf[evenBase + 2 * lane] = newEvenRe.GetElement(lane);
                    buf[evenBase + 2 * lane + 1] = newEvenIm.GetElement(lane);
                    buf[oddBase + 2 * lane] = newOddRe.GetElement(lane);
                    buf[oddBase + 2 * lane + 1] = newOddIm.GetElement(lane);
                }

                var newWRe = Avx.Subtract(Avx.Multiply(wRe, step4Re), Avx.Multiply(wIm, step4Im));
                var newWIm = Avx.Add(Avx.Multiply(wRe, step4Im), Avx.Multiply(wIm, step4Re));
                wRe = newWRe;
                wIm = newWIm;
            }

            double wReScalar = wRe.GetElement(0);
            double wImScalar = wIm.GetElement(0);
            double wStepReal = Math.Cos(theta);
            double wStepImag = Math.Sin(theta);
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
                double nwRe = wReScalar * wStepReal - wImScalar * wStepImag;
                double nwIm = wReScalar * wStepImag + wImScalar * wStepReal;
                wReScalar = nwRe;
                wImScalar = nwIm;
            }
        }
    }

    /// <summary>
    /// SSE2 radix-2 stage: 2 complex butterflies per vectorized inner
    /// iteration (Vector128 of doubles). Requires <c>half ≥ 2</c>.
    /// </summary>
    internal static unsafe void Radix2StageVector128(
        double* buf, int n, int stageSize, bool inverse)
    {
        int half = stageSize >> 1;
        double sign = inverse ? 1.0 : -1.0;
        double theta = sign * 2.0 * Math.PI / stageSize;

        var wReInit = Vector128.Create(Math.Cos(0 * theta), Math.Cos(1 * theta));
        var wImInit = Vector128.Create(Math.Sin(0 * theta), Math.Sin(1 * theta));
        var step2Re = Vector128.Create(Math.Cos(2.0 * theta));
        var step2Im = Vector128.Create(Math.Sin(2.0 * theta));

        for (int start = 0; start < n; start += stageSize)
        {
            var wRe = wReInit;
            var wIm = wImInit;
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

                var newWRe = Sse2.Subtract(Sse2.Multiply(wRe, step2Re), Sse2.Multiply(wIm, step2Im));
                var newWIm = Sse2.Add(Sse2.Multiply(wRe, step2Im), Sse2.Multiply(wIm, step2Re));
                wRe = newWRe;
                wIm = newWIm;
            }

            double wReScalar = wRe.GetElement(0);
            double wImScalar = wIm.GetElement(0);
            double wStepReal = Math.Cos(theta);
            double wStepImag = Math.Sin(theta);
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
                double nwRe = wReScalar * wStepReal - wImScalar * wStepImag;
                double nwIm = wReScalar * wStepImag + wImScalar * wStepReal;
                wReScalar = nwRe;
                wImScalar = nwIm;
            }
        }
    }

    /// <summary>
    /// Dispatch a single radix-2 stage to the widest SIMD path whose lane
    /// count fits into <c>half = stageSize/2</c>. Returns <c>true</c> if
    /// a SIMD path ran; <c>false</c> means the caller must run scalar.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe bool TryRadix2Stage(
        double* buf, int n, int stageSize, bool inverse)
    {
        int half = stageSize >> 1;
#if NET8_0_OR_GREATER
        if (Avx512F.IsSupported && half >= 8)
        {
            Radix2StageVector512(buf, n, stageSize, inverse);
            return true;
        }
#endif
        if (Avx.IsSupported && half >= 4)
        {
            Radix2StageVector256(buf, n, stageSize, inverse);
            return true;
        }
        if (Sse2.IsSupported && half >= 2)
        {
            Radix2StageVector128(buf, n, stageSize, inverse);
            return true;
        }
        return false;
    }
}
#endif
