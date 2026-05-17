using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX-512 tail-handling microkernels — variants of <see cref="Avx512Fp64_8x16"/>
/// and <see cref="Avx512Fp32_16x16"/> that accept <c>effectiveNr ≤ Nr</c> and use
/// <see cref="Vector512.ConditionalSelect"/> to write only the first
/// <c>effectiveNr</c> columns of each row. Used by the AVX-512 strategy when N
/// is not divisible by Nr.
/// </summary>
internal static class Avx512Tail
{
#if NET8_0_OR_GREATER
    public static bool IsSupported => Avx512F.IsSupported;

    /// <summary>
    /// Build a Vector512&lt;long&gt; mask where lanes 0..effectiveNr-1 are -1
    /// (all bits set) and the rest are 0. effectiveNr in [0, 8].
    /// </summary>
    private static Vector512<long> BuildMaskFp64(int effectiveNr)
    {
        return Vector512.Create(
            effectiveNr > 0 ? -1L : 0L,
            effectiveNr > 1 ? -1L : 0L,
            effectiveNr > 2 ? -1L : 0L,
            effectiveNr > 3 ? -1L : 0L,
            effectiveNr > 4 ? -1L : 0L,
            effectiveNr > 5 ? -1L : 0L,
            effectiveNr > 6 ? -1L : 0L,
            effectiveNr > 7 ? -1L : 0L);
    }

    /// <summary>Build a Vector512&lt;int&gt; mask for FP32; effectiveNr in [0, 16].</summary>
    private static Vector512<int> BuildMaskFp32(int effectiveNr)
    {
        return Vector512.Create(
            effectiveNr > 0  ? -1 : 0,
            effectiveNr > 1  ? -1 : 0,
            effectiveNr > 2  ? -1 : 0,
            effectiveNr > 3  ? -1 : 0,
            effectiveNr > 4  ? -1 : 0,
            effectiveNr > 5  ? -1 : 0,
            effectiveNr > 6  ? -1 : 0,
            effectiveNr > 7  ? -1 : 0,
            effectiveNr > 8  ? -1 : 0,
            effectiveNr > 9  ? -1 : 0,
            effectiveNr > 10 ? -1 : 0,
            effectiveNr > 11 ? -1 : 0,
            effectiveNr > 12 ? -1 : 0,
            effectiveNr > 13 ? -1 : 0,
            effectiveNr > 14 ? -1 : 0,
            effectiveNr > 15 ? -1 : 0);
    }

    /// <summary>
    /// AVX-512 FP64 8×N microkernel where N = effectiveNr ∈ [1, 16]. Reads packed-A
    /// in [Kc × Mr=8] vpanel, packed-B in [Kc × Nr=16]. Stores only the first
    /// effectiveNr cols of C via ConditionalSelect masking.
    /// </summary>
    public static unsafe void RunFp64_8xN(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc,
        int effectiveNr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Tail requires Avx512F.");
        if (effectiveNr < 1 || effectiveNr > 16)
            throw new ArgumentOutOfRangeException(nameof(effectiveNr));

        if (effectiveNr == 16)
        {
            Avx512Fp64_8x16.Run(packedA, packedB, c, ldc, kc);
            return;
        }

        // Split mask into lo (cols 0..7) and hi (cols 8..15) halves.
        int loCount = Math.Min(effectiveNr, 8);
        int hiCount = Math.Max(0, effectiveNr - 8);
        Vector512<double> maskLo = BuildMaskFp64(loCount).AsDouble();
        Vector512<double> maskHi = BuildMaskFp64(hiCount).AsDouble();

        fixed (double* cPtr = c)
        {
            // Load initial C lo and hi halves for all 8 rows.
            Vector512<double> orig0_lo = Avx512F.LoadVector512(cPtr + 0 * ldc + 0);
            Vector512<double> orig0_hi = Avx512F.LoadVector512(cPtr + 0 * ldc + 8);
            Vector512<double> orig1_lo = Avx512F.LoadVector512(cPtr + 1 * ldc + 0);
            Vector512<double> orig1_hi = Avx512F.LoadVector512(cPtr + 1 * ldc + 8);
            Vector512<double> orig2_lo = Avx512F.LoadVector512(cPtr + 2 * ldc + 0);
            Vector512<double> orig2_hi = Avx512F.LoadVector512(cPtr + 2 * ldc + 8);
            Vector512<double> orig3_lo = Avx512F.LoadVector512(cPtr + 3 * ldc + 0);
            Vector512<double> orig3_hi = Avx512F.LoadVector512(cPtr + 3 * ldc + 8);
            Vector512<double> orig4_lo = Avx512F.LoadVector512(cPtr + 4 * ldc + 0);
            Vector512<double> orig4_hi = Avx512F.LoadVector512(cPtr + 4 * ldc + 8);
            Vector512<double> orig5_lo = Avx512F.LoadVector512(cPtr + 5 * ldc + 0);
            Vector512<double> orig5_hi = Avx512F.LoadVector512(cPtr + 5 * ldc + 8);
            Vector512<double> orig6_lo = Avx512F.LoadVector512(cPtr + 6 * ldc + 0);
            Vector512<double> orig6_hi = Avx512F.LoadVector512(cPtr + 6 * ldc + 8);
            Vector512<double> orig7_lo = Avx512F.LoadVector512(cPtr + 7 * ldc + 0);
            Vector512<double> orig7_hi = Avx512F.LoadVector512(cPtr + 7 * ldc + 8);

            // Accumulators start from the loaded C values (read-modify-write).
            Vector512<double> acc0_lo = orig0_lo, acc0_hi = orig0_hi;
            Vector512<double> acc1_lo = orig1_lo, acc1_hi = orig1_hi;
            Vector512<double> acc2_lo = orig2_lo, acc2_hi = orig2_hi;
            Vector512<double> acc3_lo = orig3_lo, acc3_hi = orig3_hi;
            Vector512<double> acc4_lo = orig4_lo, acc4_hi = orig4_hi;
            Vector512<double> acc5_lo = orig5_lo, acc5_hi = orig5_hi;
            Vector512<double> acc6_lo = orig6_lo, acc6_hi = orig6_hi;
            Vector512<double> acc7_lo = orig7_lo, acc7_hi = orig7_hi;

            fixed (double* aPtr = packedA)
            fixed (double* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    Vector512<double> bRow_lo = Avx512F.LoadVector512(bPtr + k * 16 + 0);
                    Vector512<double> bRow_hi = Avx512F.LoadVector512(bPtr + k * 16 + 8);

                    Vector512<double> a0 = Vector512.Create(aPtr[k * 8 + 0]);
                    Vector512<double> a1 = Vector512.Create(aPtr[k * 8 + 1]);
                    Vector512<double> a2 = Vector512.Create(aPtr[k * 8 + 2]);
                    Vector512<double> a3 = Vector512.Create(aPtr[k * 8 + 3]);
                    Vector512<double> a4 = Vector512.Create(aPtr[k * 8 + 4]);
                    Vector512<double> a5 = Vector512.Create(aPtr[k * 8 + 5]);
                    Vector512<double> a6 = Vector512.Create(aPtr[k * 8 + 6]);
                    Vector512<double> a7 = Vector512.Create(aPtr[k * 8 + 7]);

                    acc0_lo = Avx512F.FusedMultiplyAdd(a0, bRow_lo, acc0_lo);
                    acc0_hi = Avx512F.FusedMultiplyAdd(a0, bRow_hi, acc0_hi);
                    acc1_lo = Avx512F.FusedMultiplyAdd(a1, bRow_lo, acc1_lo);
                    acc1_hi = Avx512F.FusedMultiplyAdd(a1, bRow_hi, acc1_hi);
                    acc2_lo = Avx512F.FusedMultiplyAdd(a2, bRow_lo, acc2_lo);
                    acc2_hi = Avx512F.FusedMultiplyAdd(a2, bRow_hi, acc2_hi);
                    acc3_lo = Avx512F.FusedMultiplyAdd(a3, bRow_lo, acc3_lo);
                    acc3_hi = Avx512F.FusedMultiplyAdd(a3, bRow_hi, acc3_hi);
                    acc4_lo = Avx512F.FusedMultiplyAdd(a4, bRow_lo, acc4_lo);
                    acc4_hi = Avx512F.FusedMultiplyAdd(a4, bRow_hi, acc4_hi);
                    acc5_lo = Avx512F.FusedMultiplyAdd(a5, bRow_lo, acc5_lo);
                    acc5_hi = Avx512F.FusedMultiplyAdd(a5, bRow_hi, acc5_hi);
                    acc6_lo = Avx512F.FusedMultiplyAdd(a6, bRow_lo, acc6_lo);
                    acc6_hi = Avx512F.FusedMultiplyAdd(a6, bRow_hi, acc6_hi);
                    acc7_lo = Avx512F.FusedMultiplyAdd(a7, bRow_lo, acc7_lo);
                    acc7_hi = Avx512F.FusedMultiplyAdd(a7, bRow_hi, acc7_hi);
                }
            }

            // Blend: ConditionalSelect(mask, newValue, original)
            // where mask lane is -1 (all bits set) → select newValue (acc)
            // where mask lane is  0                → select original (orig)
            Avx512F.Store(cPtr + 0 * ldc + 0, Vector512.ConditionalSelect(maskLo, acc0_lo, orig0_lo));
            Avx512F.Store(cPtr + 0 * ldc + 8, Vector512.ConditionalSelect(maskHi, acc0_hi, orig0_hi));
            Avx512F.Store(cPtr + 1 * ldc + 0, Vector512.ConditionalSelect(maskLo, acc1_lo, orig1_lo));
            Avx512F.Store(cPtr + 1 * ldc + 8, Vector512.ConditionalSelect(maskHi, acc1_hi, orig1_hi));
            Avx512F.Store(cPtr + 2 * ldc + 0, Vector512.ConditionalSelect(maskLo, acc2_lo, orig2_lo));
            Avx512F.Store(cPtr + 2 * ldc + 8, Vector512.ConditionalSelect(maskHi, acc2_hi, orig2_hi));
            Avx512F.Store(cPtr + 3 * ldc + 0, Vector512.ConditionalSelect(maskLo, acc3_lo, orig3_lo));
            Avx512F.Store(cPtr + 3 * ldc + 8, Vector512.ConditionalSelect(maskHi, acc3_hi, orig3_hi));
            Avx512F.Store(cPtr + 4 * ldc + 0, Vector512.ConditionalSelect(maskLo, acc4_lo, orig4_lo));
            Avx512F.Store(cPtr + 4 * ldc + 8, Vector512.ConditionalSelect(maskHi, acc4_hi, orig4_hi));
            Avx512F.Store(cPtr + 5 * ldc + 0, Vector512.ConditionalSelect(maskLo, acc5_lo, orig5_lo));
            Avx512F.Store(cPtr + 5 * ldc + 8, Vector512.ConditionalSelect(maskHi, acc5_hi, orig5_hi));
            Avx512F.Store(cPtr + 6 * ldc + 0, Vector512.ConditionalSelect(maskLo, acc6_lo, orig6_lo));
            Avx512F.Store(cPtr + 6 * ldc + 8, Vector512.ConditionalSelect(maskHi, acc6_hi, orig6_hi));
            Avx512F.Store(cPtr + 7 * ldc + 0, Vector512.ConditionalSelect(maskLo, acc7_lo, orig7_lo));
            Avx512F.Store(cPtr + 7 * ldc + 8, Vector512.ConditionalSelect(maskHi, acc7_hi, orig7_hi));
        }
    }

    /// <summary>
    /// AVX-512 FP32 16×N microkernel where N = effectiveNr ∈ [1, 16].
    /// </summary>
    public static unsafe void RunFp32_16xN(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc,
        int effectiveNr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Tail requires Avx512F.");
        if (effectiveNr < 1 || effectiveNr > 16)
            throw new ArgumentOutOfRangeException(nameof(effectiveNr));

        if (effectiveNr == 16)
        {
            Avx512Fp32_16x16.Run(packedA, packedB, c, ldc, kc);
            return;
        }

        Vector512<float> mask = BuildMaskFp32(effectiveNr).AsSingle();

        fixed (float* cPtr = c)
        {
            // 16 accumulators initialized from C.
            Vector512<float> orig0  = Avx512F.LoadVector512(cPtr + 0  * ldc);
            Vector512<float> orig1  = Avx512F.LoadVector512(cPtr + 1  * ldc);
            Vector512<float> orig2  = Avx512F.LoadVector512(cPtr + 2  * ldc);
            Vector512<float> orig3  = Avx512F.LoadVector512(cPtr + 3  * ldc);
            Vector512<float> orig4  = Avx512F.LoadVector512(cPtr + 4  * ldc);
            Vector512<float> orig5  = Avx512F.LoadVector512(cPtr + 5  * ldc);
            Vector512<float> orig6  = Avx512F.LoadVector512(cPtr + 6  * ldc);
            Vector512<float> orig7  = Avx512F.LoadVector512(cPtr + 7  * ldc);
            Vector512<float> orig8  = Avx512F.LoadVector512(cPtr + 8  * ldc);
            Vector512<float> orig9  = Avx512F.LoadVector512(cPtr + 9  * ldc);
            Vector512<float> orig10 = Avx512F.LoadVector512(cPtr + 10 * ldc);
            Vector512<float> orig11 = Avx512F.LoadVector512(cPtr + 11 * ldc);
            Vector512<float> orig12 = Avx512F.LoadVector512(cPtr + 12 * ldc);
            Vector512<float> orig13 = Avx512F.LoadVector512(cPtr + 13 * ldc);
            Vector512<float> orig14 = Avx512F.LoadVector512(cPtr + 14 * ldc);
            Vector512<float> orig15 = Avx512F.LoadVector512(cPtr + 15 * ldc);

            Vector512<float> acc0  = orig0;
            Vector512<float> acc1  = orig1;
            Vector512<float> acc2  = orig2;
            Vector512<float> acc3  = orig3;
            Vector512<float> acc4  = orig4;
            Vector512<float> acc5  = orig5;
            Vector512<float> acc6  = orig6;
            Vector512<float> acc7  = orig7;
            Vector512<float> acc8  = orig8;
            Vector512<float> acc9  = orig9;
            Vector512<float> acc10 = orig10;
            Vector512<float> acc11 = orig11;
            Vector512<float> acc12 = orig12;
            Vector512<float> acc13 = orig13;
            Vector512<float> acc14 = orig14;
            Vector512<float> acc15 = orig15;

            fixed (float* aPtr = packedA)
            fixed (float* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    Vector512<float> bRow = Avx512F.LoadVector512(bPtr + k * 16);

                    Vector512<float> a0  = Vector512.Create(aPtr[k * 16 + 0]);
                    Vector512<float> a1  = Vector512.Create(aPtr[k * 16 + 1]);
                    Vector512<float> a2  = Vector512.Create(aPtr[k * 16 + 2]);
                    Vector512<float> a3  = Vector512.Create(aPtr[k * 16 + 3]);
                    Vector512<float> a4  = Vector512.Create(aPtr[k * 16 + 4]);
                    Vector512<float> a5  = Vector512.Create(aPtr[k * 16 + 5]);
                    Vector512<float> a6  = Vector512.Create(aPtr[k * 16 + 6]);
                    Vector512<float> a7  = Vector512.Create(aPtr[k * 16 + 7]);
                    Vector512<float> a8  = Vector512.Create(aPtr[k * 16 + 8]);
                    Vector512<float> a9  = Vector512.Create(aPtr[k * 16 + 9]);
                    Vector512<float> a10 = Vector512.Create(aPtr[k * 16 + 10]);
                    Vector512<float> a11 = Vector512.Create(aPtr[k * 16 + 11]);
                    Vector512<float> a12 = Vector512.Create(aPtr[k * 16 + 12]);
                    Vector512<float> a13 = Vector512.Create(aPtr[k * 16 + 13]);
                    Vector512<float> a14 = Vector512.Create(aPtr[k * 16 + 14]);
                    Vector512<float> a15 = Vector512.Create(aPtr[k * 16 + 15]);

                    acc0  = Avx512F.FusedMultiplyAdd(a0,  bRow, acc0);
                    acc1  = Avx512F.FusedMultiplyAdd(a1,  bRow, acc1);
                    acc2  = Avx512F.FusedMultiplyAdd(a2,  bRow, acc2);
                    acc3  = Avx512F.FusedMultiplyAdd(a3,  bRow, acc3);
                    acc4  = Avx512F.FusedMultiplyAdd(a4,  bRow, acc4);
                    acc5  = Avx512F.FusedMultiplyAdd(a5,  bRow, acc5);
                    acc6  = Avx512F.FusedMultiplyAdd(a6,  bRow, acc6);
                    acc7  = Avx512F.FusedMultiplyAdd(a7,  bRow, acc7);
                    acc8  = Avx512F.FusedMultiplyAdd(a8,  bRow, acc8);
                    acc9  = Avx512F.FusedMultiplyAdd(a9,  bRow, acc9);
                    acc10 = Avx512F.FusedMultiplyAdd(a10, bRow, acc10);
                    acc11 = Avx512F.FusedMultiplyAdd(a11, bRow, acc11);
                    acc12 = Avx512F.FusedMultiplyAdd(a12, bRow, acc12);
                    acc13 = Avx512F.FusedMultiplyAdd(a13, bRow, acc13);
                    acc14 = Avx512F.FusedMultiplyAdd(a14, bRow, acc14);
                    acc15 = Avx512F.FusedMultiplyAdd(a15, bRow, acc15);
                }
            }

            Avx512F.Store(cPtr + 0  * ldc, Vector512.ConditionalSelect(mask, acc0,  orig0));
            Avx512F.Store(cPtr + 1  * ldc, Vector512.ConditionalSelect(mask, acc1,  orig1));
            Avx512F.Store(cPtr + 2  * ldc, Vector512.ConditionalSelect(mask, acc2,  orig2));
            Avx512F.Store(cPtr + 3  * ldc, Vector512.ConditionalSelect(mask, acc3,  orig3));
            Avx512F.Store(cPtr + 4  * ldc, Vector512.ConditionalSelect(mask, acc4,  orig4));
            Avx512F.Store(cPtr + 5  * ldc, Vector512.ConditionalSelect(mask, acc5,  orig5));
            Avx512F.Store(cPtr + 6  * ldc, Vector512.ConditionalSelect(mask, acc6,  orig6));
            Avx512F.Store(cPtr + 7  * ldc, Vector512.ConditionalSelect(mask, acc7,  orig7));
            Avx512F.Store(cPtr + 8  * ldc, Vector512.ConditionalSelect(mask, acc8,  orig8));
            Avx512F.Store(cPtr + 9  * ldc, Vector512.ConditionalSelect(mask, acc9,  orig9));
            Avx512F.Store(cPtr + 10 * ldc, Vector512.ConditionalSelect(mask, acc10, orig10));
            Avx512F.Store(cPtr + 11 * ldc, Vector512.ConditionalSelect(mask, acc11, orig11));
            Avx512F.Store(cPtr + 12 * ldc, Vector512.ConditionalSelect(mask, acc12, orig12));
            Avx512F.Store(cPtr + 13 * ldc, Vector512.ConditionalSelect(mask, acc13, orig13));
            Avx512F.Store(cPtr + 14 * ldc, Vector512.ConditionalSelect(mask, acc14, orig14));
            Avx512F.Store(cPtr + 15 * ldc, Vector512.ConditionalSelect(mask, acc15, orig15));
        }
    }
#else
    public static bool IsSupported => false;

    public static void RunFp64_8xN(
        ReadOnlySpan<double> packedA, ReadOnlySpan<double> packedB,
        Span<double> c, int ldc, int kc, int effectiveNr) =>
        throw new PlatformNotSupportedException("Avx512Tail requires net8.0+.");

    public static void RunFp32_16xN(
        ReadOnlySpan<float> packedA, ReadOnlySpan<float> packedB,
        Span<float> c, int ldc, int kc, int effectiveNr) =>
        throw new PlatformNotSupportedException("Avx512Tail requires net8.0+.");
#endif
}
