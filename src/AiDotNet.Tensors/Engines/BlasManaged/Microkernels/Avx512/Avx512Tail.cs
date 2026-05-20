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

        // CodeRabbit #366 (Critical): the previous SIMD tail path here
        // unconditionally issued LoadVector512 / Store for the full 16-column
        // tile, which can OOB-read or OOB-write past the C row when
        // effectiveNr < 16 and the tile sits at the right edge of the matrix.
        // ConditionalSelect masks the values but not the underlying memory
        // access. Fall back to a safe scalar accumulation that only touches
        // columns [0, effectiveNr). This is the rare tail path — perf cost
        // is minimal vs the access-violation risk on an edge tile.
        // Future work: rewrite using Avx512F.MaskLoad / MaskStore for SIMD
        // throughput without the OOB hazard.
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < effectiveNr; j++)
            {
                double sum = c[i * ldc + j];
                for (int kk = 0; kk < kc; kk++)
                    sum += packedA[kk * 8 + i] * packedB[kk * 16 + j];
                c[i * ldc + j] = sum;
            }
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

        // CodeRabbit #366 (Critical): same OOB hazard as the FP64 path.
        // The prior body issued full-width LoadVector512 / Store calls that
        // touched columns past the valid C tile when effectiveNr < 16.
        // Safe scalar fallback for partial-N tiles; rewrite using
        // Avx512F.MaskLoad / MaskStore is tracked as follow-up.
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < effectiveNr; j++)
            {
                float sum = c[i * ldc + j];
                for (int kk = 0; kk < kc; kk++)
                    sum += packedA[kk * 16 + i] * packedB[kk * 16 + j];
                c[i * ldc + j] = sum;
            }
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
