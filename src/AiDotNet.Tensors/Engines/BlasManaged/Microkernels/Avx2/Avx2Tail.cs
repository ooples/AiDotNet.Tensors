using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX2 tail-handling microkernels — variants of <see cref="Avx2Fp64_4x8"/>
/// and <see cref="Avx2Fp32_8x8"/> that accept an <c>effectiveNr</c> ≤ Nr and
/// use <see cref="Avx2.MaskStore(long*, Vector256{long}, Vector256{long})"/> (VPMASKMOVQ)
/// to write only the first <c>effectiveNr</c> columns of each row. Used by
/// the AVX2 strategy (Task C7) when N is not divisible by Nr.
///
/// <para>
/// The mask vectors are precomputed for each possible <c>effectiveNr</c>
/// value via <see cref="BuildMaskFp64"/> / <see cref="BuildMaskFp32"/>.
/// </para>
/// </summary>
internal static class Avx2Tail
{
#if NET5_0_OR_GREATER
    /// <summary>Runtime support gate.</summary>
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>
    /// Build a Vector256&lt;long&gt; mask where lanes 0..effectiveNr-1 have their
    /// high bit set (MaskStore will write those lanes) and the rest are zero
    /// (MaskStore skips). effectiveNr in [0, 4].
    /// </summary>
    private static Vector256<long> BuildMaskFp64(int effectiveNr)
    {
        // MaskStore checks the high bit of each lane. -1 (all ones) has high bit set.
        long v0 = effectiveNr > 0 ? -1L : 0L;
        long v1 = effectiveNr > 1 ? -1L : 0L;
        long v2 = effectiveNr > 2 ? -1L : 0L;
        long v3 = effectiveNr > 3 ? -1L : 0L;
        return Vector256.Create(v0, v1, v2, v3);
    }

    /// <summary>Build a Vector256&lt;int&gt; mask for FP32; effectiveNr in [0, 8].</summary>
    private static Vector256<int> BuildMaskFp32(int effectiveNr)
    {
        return Vector256.Create(
            effectiveNr > 0 ? -1 : 0,
            effectiveNr > 1 ? -1 : 0,
            effectiveNr > 2 ? -1 : 0,
            effectiveNr > 3 ? -1 : 0,
            effectiveNr > 4 ? -1 : 0,
            effectiveNr > 5 ? -1 : 0,
            effectiveNr > 6 ? -1 : 0,
            effectiveNr > 7 ? -1 : 0);
    }

    /// <summary>
    /// AVX2 FP64 4×N microkernel where N = effectiveNr ∈ [1, 8]. Reads packed-A
    /// in [Kc × Mr=4] vpanel, packed-B in [Kc × Nr=8] (full Nr columns even when
    /// effective is less — strategy must allocate the full Kc*Nr packed-B
    /// buffer and zero-fill the unused cols). Stores only the first
    /// effectiveNr cols of C via MaskStore.
    /// </summary>
    public static unsafe void RunFp64_4xN(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc,
        int effectiveNr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Tail requires Avx2 + Fma.");
        if (effectiveNr < 1 || effectiveNr > 8)
            throw new ArgumentOutOfRangeException(nameof(effectiveNr));

        // When effectiveNr == 8, just call the full microkernel — no masking needed.
        if (effectiveNr == 8)
        {
            Avx2Fp64_4x8.Run(packedA, packedB, c, ldc, kc);
            return;
        }

        // Same K-loop structure as Avx2Fp64_4x8 (load C, FMA over K, store C),
        // but the final store uses MaskStore for the lo or hi half depending on
        // which side has the partial columns.
        int loCount = Math.Min(effectiveNr, 4);
        int hiCount = Math.Max(0, effectiveNr - 4);
        Vector256<long> maskLo = BuildMaskFp64(loCount);
        Vector256<long> maskHi = BuildMaskFp64(hiCount);

        fixed (double* cPtr = c)
        {
            // Read initial C tile — masked load isn't strictly needed since we'll
            // mask the store, but reading partial tile via unmasked load is safe
            // because the strategy allocates the full Nr-wide row.
            Vector256<double> acc0_lo = Avx.LoadVector256(cPtr + 0 * ldc + 0);
            Vector256<double> acc0_hi = hiCount > 0 ? Avx.LoadVector256(cPtr + 0 * ldc + 4) : Vector256<double>.Zero;
            Vector256<double> acc1_lo = Avx.LoadVector256(cPtr + 1 * ldc + 0);
            Vector256<double> acc1_hi = hiCount > 0 ? Avx.LoadVector256(cPtr + 1 * ldc + 4) : Vector256<double>.Zero;
            Vector256<double> acc2_lo = Avx.LoadVector256(cPtr + 2 * ldc + 0);
            Vector256<double> acc2_hi = hiCount > 0 ? Avx.LoadVector256(cPtr + 2 * ldc + 4) : Vector256<double>.Zero;
            Vector256<double> acc3_lo = Avx.LoadVector256(cPtr + 3 * ldc + 0);
            Vector256<double> acc3_hi = hiCount > 0 ? Avx.LoadVector256(cPtr + 3 * ldc + 4) : Vector256<double>.Zero;

            fixed (double* aPtr = packedA)
            fixed (double* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    Vector256<double> bRow_lo = Avx.LoadVector256(bPtr + k * 8 + 0);
                    Vector256<double> bRow_hi = hiCount > 0 ? Avx.LoadVector256(bPtr + k * 8 + 4) : Vector256<double>.Zero;

                    Vector256<double> a0 = Vector256.Create(aPtr[k * 4 + 0]);
                    Vector256<double> a1 = Vector256.Create(aPtr[k * 4 + 1]);
                    Vector256<double> a2 = Vector256.Create(aPtr[k * 4 + 2]);
                    Vector256<double> a3 = Vector256.Create(aPtr[k * 4 + 3]);

                    acc0_lo = Fma.MultiplyAdd(a0, bRow_lo, acc0_lo);
                    acc1_lo = Fma.MultiplyAdd(a1, bRow_lo, acc1_lo);
                    acc2_lo = Fma.MultiplyAdd(a2, bRow_lo, acc2_lo);
                    acc3_lo = Fma.MultiplyAdd(a3, bRow_lo, acc3_lo);
                    if (hiCount > 0)
                    {
                        acc0_hi = Fma.MultiplyAdd(a0, bRow_hi, acc0_hi);
                        acc1_hi = Fma.MultiplyAdd(a1, bRow_hi, acc1_hi);
                        acc2_hi = Fma.MultiplyAdd(a2, bRow_hi, acc2_hi);
                        acc3_hi = Fma.MultiplyAdd(a3, bRow_hi, acc3_hi);
                    }
                }
            }

            // Masked stores. The lo half is always partially-or-fully written;
            // the hi half is only stored when hiCount > 0.
            // Avx2.MaskStore operates on integer types (VPMASKMOVQ for 64-bit lanes);
            // we reinterpret the double pointers as long* and the double vectors as
            // Vector256<long> via AsInt64() — the bit patterns are preserved exactly.
            Avx2.MaskStore((long*)(cPtr + 0 * ldc + 0), maskLo, acc0_lo.AsInt64());
            Avx2.MaskStore((long*)(cPtr + 1 * ldc + 0), maskLo, acc1_lo.AsInt64());
            Avx2.MaskStore((long*)(cPtr + 2 * ldc + 0), maskLo, acc2_lo.AsInt64());
            Avx2.MaskStore((long*)(cPtr + 3 * ldc + 0), maskLo, acc3_lo.AsInt64());
            if (hiCount > 0)
            {
                Avx2.MaskStore((long*)(cPtr + 0 * ldc + 4), maskHi, acc0_hi.AsInt64());
                Avx2.MaskStore((long*)(cPtr + 1 * ldc + 4), maskHi, acc1_hi.AsInt64());
                Avx2.MaskStore((long*)(cPtr + 2 * ldc + 4), maskHi, acc2_hi.AsInt64());
                Avx2.MaskStore((long*)(cPtr + 3 * ldc + 4), maskHi, acc3_hi.AsInt64());
            }
        }
    }

    /// <summary>
    /// AVX2 FP32 8×N microkernel where N = effectiveNr ∈ [1, 8].
    /// </summary>
    public static unsafe void RunFp32_8xN(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc,
        int effectiveNr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Tail requires Avx2 + Fma.");
        if (effectiveNr < 1 || effectiveNr > 8)
            throw new ArgumentOutOfRangeException(nameof(effectiveNr));

        if (effectiveNr == 8)
        {
            Avx2Fp32_8x8.Run(packedA, packedB, c, ldc, kc);
            return;
        }

        Vector256<int> mask = BuildMaskFp32(effectiveNr);

        fixed (float* cPtr = c)
        {
            Vector256<float> acc0 = Avx.LoadVector256(cPtr + 0 * ldc);
            Vector256<float> acc1 = Avx.LoadVector256(cPtr + 1 * ldc);
            Vector256<float> acc2 = Avx.LoadVector256(cPtr + 2 * ldc);
            Vector256<float> acc3 = Avx.LoadVector256(cPtr + 3 * ldc);
            Vector256<float> acc4 = Avx.LoadVector256(cPtr + 4 * ldc);
            Vector256<float> acc5 = Avx.LoadVector256(cPtr + 5 * ldc);
            Vector256<float> acc6 = Avx.LoadVector256(cPtr + 6 * ldc);
            Vector256<float> acc7 = Avx.LoadVector256(cPtr + 7 * ldc);

            fixed (float* aPtr = packedA)
            fixed (float* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    Vector256<float> bRow = Avx.LoadVector256(bPtr + k * 8);
                    Vector256<float> a0 = Vector256.Create(aPtr[k * 8 + 0]);
                    Vector256<float> a1 = Vector256.Create(aPtr[k * 8 + 1]);
                    Vector256<float> a2 = Vector256.Create(aPtr[k * 8 + 2]);
                    Vector256<float> a3 = Vector256.Create(aPtr[k * 8 + 3]);
                    Vector256<float> a4 = Vector256.Create(aPtr[k * 8 + 4]);
                    Vector256<float> a5 = Vector256.Create(aPtr[k * 8 + 5]);
                    Vector256<float> a6 = Vector256.Create(aPtr[k * 8 + 6]);
                    Vector256<float> a7 = Vector256.Create(aPtr[k * 8 + 7]);
                    acc0 = Fma.MultiplyAdd(a0, bRow, acc0);
                    acc1 = Fma.MultiplyAdd(a1, bRow, acc1);
                    acc2 = Fma.MultiplyAdd(a2, bRow, acc2);
                    acc3 = Fma.MultiplyAdd(a3, bRow, acc3);
                    acc4 = Fma.MultiplyAdd(a4, bRow, acc4);
                    acc5 = Fma.MultiplyAdd(a5, bRow, acc5);
                    acc6 = Fma.MultiplyAdd(a6, bRow, acc6);
                    acc7 = Fma.MultiplyAdd(a7, bRow, acc7);
                }
            }

            // Avx2.MaskStore for FP32: reinterpret float* as int* and float vectors
            // as Vector256<int> via AsInt32() — VPMASKMOVD stores 32-bit lanes conditionally.
            Avx2.MaskStore((int*)(cPtr + 0 * ldc), mask, acc0.AsInt32());
            Avx2.MaskStore((int*)(cPtr + 1 * ldc), mask, acc1.AsInt32());
            Avx2.MaskStore((int*)(cPtr + 2 * ldc), mask, acc2.AsInt32());
            Avx2.MaskStore((int*)(cPtr + 3 * ldc), mask, acc3.AsInt32());
            Avx2.MaskStore((int*)(cPtr + 4 * ldc), mask, acc4.AsInt32());
            Avx2.MaskStore((int*)(cPtr + 5 * ldc), mask, acc5.AsInt32());
            Avx2.MaskStore((int*)(cPtr + 6 * ldc), mask, acc6.AsInt32());
            Avx2.MaskStore((int*)(cPtr + 7 * ldc), mask, acc7.AsInt32());
        }
    }
#else
    public static bool IsSupported => false;
    public static void RunFp64_4xN(ReadOnlySpan<double> packedA, ReadOnlySpan<double> packedB, Span<double> c, int ldc, int kc, int effectiveNr) =>
        throw new PlatformNotSupportedException("Avx2Tail requires net5.0+.");
    public static void RunFp32_8xN(ReadOnlySpan<float> packedA, ReadOnlySpan<float> packedB, Span<float> c, int ldc, int kc, int effectiveNr) =>
        throw new PlatformNotSupportedException("Avx2Tail requires net5.0+.");
#endif
}
