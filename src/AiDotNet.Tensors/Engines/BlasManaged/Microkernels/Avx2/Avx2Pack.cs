using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX2 SIMD pack routines that produce bit-identical output to
/// <see cref="ScalarPack"/> but use Vector256-width loads/stores for higher
/// memory bandwidth on the transA=true path.
///
/// <para>
/// transA=true is the SIMD-friendly path: A is stored row-major [K, M], so
/// each k-step's Mr consecutive values are contiguous in memory and can be
/// copied with a single Vector256 load + store per (k, stripe) pair.
/// </para>
///
/// <para>
/// transA=false requires a 4×4 (FP64) or 8×8 (FP32) tile transpose to convert
/// row-major [M, K] reads into [Kc × Mr] vpanel writes. For Phase C this case
/// falls back to <see cref="ScalarPack.PackA{T}"/> — the scalar implementation
/// is correct and the optimization can land later without changing the
/// correctness contract.
/// </para>
/// </summary>
internal static class Avx2Pack
{
#if NET5_0_OR_GREATER
    /// <summary>Runtime support gate. True when AVX2 intrinsics are usable on the current process.</summary>
    public static bool IsSupported => Avx2.IsSupported;

    /// <summary>
    /// Pack a logical Mc-row × Kc-col panel of A into BLIS vpanel layout.
    /// Output is bit-identical to <see cref="ScalarPack.PackA{T}"/> for T=double.
    ///
    /// <para>
    /// When <paramref name="transA"/> is <c>true</c> and <paramref name="mr"/> is 4,
    /// each k-step reads exactly one <see cref="Vector256{T}"/> (4 doubles = 32 bytes)
    /// from the contiguous source row and writes it to the packed stripe. This is the
    /// SIMD-accelerated path.
    /// </para>
    ///
    /// <para>
    /// All other cases delegate to <see cref="ScalarPack.PackA{T}"/> for correctness.
    /// The transA=false transpose optimization is deferred to a later phase.
    /// </para>
    /// </summary>
    /// <param name="a">Source A buffer, stored row-major [K, M] when transA=true or [M, K] when transA=false.</param>
    /// <param name="lda">Leading dimension of A (number of columns in the stored layout).</param>
    /// <param name="transA">True if A is stored as A^T (logical [M, K] view from [K, M] memory).</param>
    /// <param name="packed">Destination vpanel buffer, length ≥ mc × kc.</param>
    /// <param name="mc">Rows of A to pack (must be exactly divisible by mr).</param>
    /// <param name="kc">Cols of A to pack (one Kc block).</param>
    /// <param name="mr">Microkernel row-tile width; must be 4 for the AVX2 FP64 SIMD path.</param>
    public static unsafe void PackA_Fp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        Span<double> packed, int mc, int kc, int mr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Pack requires AVX2.");

        // transA=false: full 4×4 transpose optimization deferred — delegate to scalar.
        if (!transA || mr != 4)
        {
            ScalarPack.PackA<double>(a, lda, transA, packed, mc, kc, mr);
            return;
        }

        // transA=true, mr=4 SIMD path:
        // A is stored row-major [K, M] with lda=M. For each stripe the Mr=4 source
        // values at a given k-step occupy consecutive addresses A[k, stripe*4 .. stripe*4+3],
        // which is exactly one Vector256<double> (4 doubles = 32 bytes). Load + store,
        // no arithmetic needed — bit-identical to scalar.
        int numStripes = mc / mr;
        fixed (double* aPtr = a)
        fixed (double* pPtr = packed)
        {
            for (int stripe = 0; stripe < numStripes; stripe++)
            {
                int srcCol = stripe * mr;
                double* packedStripe = pPtr + stripe * kc * mr;
                for (int k = 0; k < kc; k++)
                {
                    Vector256<double> row = Avx.LoadVector256(aPtr + k * lda + srcCol);
                    Avx.Store(packedStripe + k * mr, row);
                }
            }
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="PackA_Fp64"/>. The AVX2 FP32 microkernel uses Mr=8,
    /// which is exactly one <see cref="Vector256{T}"/> (8 floats = 32 bytes) wide.
    ///
    /// <para>
    /// When <paramref name="transA"/> is <c>true</c> and <paramref name="mr"/> is 8,
    /// each k-step reads one Vector256&lt;float&gt; from the source row and stores it
    /// into the packed stripe — bit-identical to <see cref="ScalarPack.PackA{T}"/>
    /// for T=float.
    /// </para>
    ///
    /// <para>
    /// All other cases delegate to <see cref="ScalarPack.PackA{T}"/> for correctness.
    /// </para>
    /// </summary>
    /// <param name="a">Source A buffer, stored row-major [K, M] when transA=true or [M, K] when transA=false.</param>
    /// <param name="lda">Leading dimension of A (number of columns in the stored layout).</param>
    /// <param name="transA">True if A is stored as A^T (logical [M, K] view from [K, M] memory).</param>
    /// <param name="packed">Destination vpanel buffer, length ≥ mc × kc.</param>
    /// <param name="mc">Rows of A to pack (must be exactly divisible by mr).</param>
    /// <param name="kc">Cols of A to pack (one Kc block).</param>
    /// <param name="mr">Microkernel row-tile width; must be 8 for the AVX2 FP32 SIMD path.</param>
    public static unsafe void PackA_Fp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        Span<float> packed, int mc, int kc, int mr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Pack requires AVX2.");

        // transA=false or mr != 8: delegate to scalar.
        if (!transA || mr != 8)
        {
            ScalarPack.PackA<float>(a, lda, transA, packed, mc, kc, mr);
            return;
        }

        // transA=true, mr=8 SIMD path:
        // A is stored row-major [K, M] with lda=M. For each stripe the Mr=8 source
        // values at a given k-step occupy consecutive addresses A[k, stripe*8 .. stripe*8+7],
        // which is exactly one Vector256<float> (8 floats = 32 bytes).
        int numStripes = mc / mr;
        fixed (float* aPtr = a)
        fixed (float* pPtr = packed)
        {
            for (int stripe = 0; stripe < numStripes; stripe++)
            {
                int srcCol = stripe * mr;
                float* packedStripe = pPtr + stripe * kc * mr;
                for (int k = 0; k < kc; k++)
                {
                    Vector256<float> row = Avx.LoadVector256(aPtr + k * lda + srcCol);
                    Avx.Store(packedStripe + k * mr, row);
                }
            }
        }
    }
#else
    /// <summary>Runtime support gate (always false on net471 — no Vector256&lt;T&gt; intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackA{T}"/> (no AVX2 available).
    /// </summary>
    public static void PackA_Fp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        Span<double> packed, int mc, int kc, int mr) =>
        ScalarPack.PackA<double>(a, lda, transA, packed, mc, kc, mr);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackA{T}"/> (no AVX2 available).
    /// </summary>
    public static void PackA_Fp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        Span<float> packed, int mc, int kc, int mr) =>
        ScalarPack.PackA<float>(a, lda, transA, packed, mc, kc, mr);
#endif
}
