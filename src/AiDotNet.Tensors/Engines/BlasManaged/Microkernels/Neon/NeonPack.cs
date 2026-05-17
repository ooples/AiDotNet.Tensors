using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// ARM64 Neon SIMD pack routines that produce bit-identical output to
/// <see cref="ScalarPack"/> but use Vector128-width loads/stores for higher
/// memory bandwidth on the transA=true path.
///
/// <para>
/// transA=true is the SIMD-friendly path: A is stored row-major [K, M], so
/// each k-step's Mr consecutive values are contiguous and can be copied via
/// Vector128 loads + stores. With Mr=4 (FP64) or Mr=8 (FP32), this requires
/// 2 Vector128 ops per k-step.
/// </para>
///
/// <para>
/// transA=false requires a 4×4 (FP64) or 8×8 (FP32) tile transpose to convert
/// row-major [M, K] reads into [Kc × Mr] vpanel writes. For Phase E this case
/// falls back to <see cref="ScalarPack.PackA{T}"/> — the scalar implementation
/// is correct and the optimization can land later without changing the
/// correctness contract.
/// </para>
/// </summary>
internal static class NeonPack
{
#if NET8_0_OR_GREATER
    /// <summary>Runtime support gate. True when ARM64 AdvSimd intrinsics are usable on the current process.</summary>
    public static bool IsSupported => AdvSimd.Arm64.IsSupported;

    /// <summary>
    /// Pack a logical Mc-row × Kc-col panel of A into BLIS vpanel layout.
    /// Output is bit-identical to <see cref="ScalarPack.PackA{T}"/> for T=double.
    ///
    /// <para>
    /// When <paramref name="transA"/> is <c>true</c> and <paramref name="mr"/> is 4,
    /// A is stored row-major [K, M]. Each k-step's Mr=4 source values at
    /// addresses A[k, stripe*4 .. stripe*4+3] are contiguous and span exactly
    /// two Vector128&lt;double&gt; (2 doubles each = 16 bytes each). Two loads
    /// and two stores per k-step — no arithmetic, bit-identical to scalar.
    /// </para>
    ///
    /// <para>
    /// transA=false and all other mr values delegate to <see cref="ScalarPack.PackA{T}"/>
    /// for correctness. The transA=false transpose optimization is deferred to a later phase.
    /// </para>
    /// </summary>
    /// <param name="a">Source A buffer, stored row-major [K, M] when transA=true or [M, K] when transA=false.</param>
    /// <param name="lda">Leading dimension of A (number of columns in the stored layout).</param>
    /// <param name="transA">True if A is stored as A^T (logical [M, K] view from [K, M] memory).</param>
    /// <param name="packed">Destination vpanel buffer, length ≥ mc × kc.</param>
    /// <param name="mc">Rows of A to pack (must be exactly divisible by mr).</param>
    /// <param name="kc">Cols of A to pack (one Kc block).</param>
    /// <param name="mr">Microkernel row-tile width; must be 4 for the Neon FP64 SIMD path.</param>
    public static unsafe void PackA_Fp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        Span<double> packed, int mc, int kc, int mr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("NeonPack requires ARM64 AdvSimd.");

        // transA=false: full 4×4 transpose optimization deferred — delegate to scalar.
        if (!transA || mr != 4)
        {
            ScalarPack.PackA<double>(a, lda, transA, packed, mc, kc, mr);
            return;
        }

        // transA=true, mr=4 SIMD path:
        // A is stored row-major [K, M] with lda=M. For each stripe the Mr=4 source
        // values at a given k-step occupy consecutive addresses A[k, stripe*4 .. stripe*4+3].
        // Vector128<double> holds 2 doubles (16 bytes), so 2 loads + 2 stores cover the
        // full Mr=4 width — bit-identical to scalar.
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
                    Vector128<double> lo = AdvSimd.LoadVector128(aPtr + k * lda + srcCol + 0);
                    Vector128<double> hi = AdvSimd.LoadVector128(aPtr + k * lda + srcCol + 2);
                    AdvSimd.Store(packedStripe + k * mr + 0, lo);
                    AdvSimd.Store(packedStripe + k * mr + 2, hi);
                }
            }
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="PackA_Fp64"/>. The Neon FP32 microkernel uses Mr=8,
    /// which requires two Vector128&lt;float&gt; (4 floats × 16 bytes each) per k-step.
    ///
    /// <para>
    /// When <paramref name="transA"/> is <c>true</c> and <paramref name="mr"/> is 8,
    /// A is stored row-major [K, M]. Each k-step's Mr=8 source values at
    /// addresses A[k, stripe*8 .. stripe*8+7] are contiguous and span exactly
    /// two Vector128&lt;float&gt;. Two loads and two stores per k-step — no arithmetic,
    /// bit-identical to <see cref="ScalarPack.PackA{T}"/> for T=float.
    /// </para>
    ///
    /// <para>
    /// transA=false and all other mr values delegate to <see cref="ScalarPack.PackA{T}"/>
    /// for correctness.
    /// </para>
    /// </summary>
    /// <param name="a">Source A buffer, stored row-major [K, M] when transA=true or [M, K] when transA=false.</param>
    /// <param name="lda">Leading dimension of A (number of columns in the stored layout).</param>
    /// <param name="transA">True if A is stored as A^T (logical [M, K] view from [K, M] memory).</param>
    /// <param name="packed">Destination vpanel buffer, length ≥ mc × kc.</param>
    /// <param name="mc">Rows of A to pack (must be exactly divisible by mr).</param>
    /// <param name="kc">Cols of A to pack (one Kc block).</param>
    /// <param name="mr">Microkernel row-tile width; must be 8 for the Neon FP32 SIMD path.</param>
    public static unsafe void PackA_Fp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        Span<float> packed, int mc, int kc, int mr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("NeonPack requires ARM64 AdvSimd.");

        // transA=false or mr != 8: delegate to scalar.
        if (!transA || mr != 8)
        {
            ScalarPack.PackA<float>(a, lda, transA, packed, mc, kc, mr);
            return;
        }

        // transA=true, mr=8 SIMD path:
        // A is stored row-major [K, M] with lda=M. For each stripe the Mr=8 source
        // values at a given k-step occupy consecutive addresses A[k, stripe*8 .. stripe*8+7].
        // Vector128<float> holds 4 floats (16 bytes), so 2 loads + 2 stores cover the
        // full Mr=8 width — bit-identical to scalar.
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
                    Vector128<float> lo = AdvSimd.LoadVector128(aPtr + k * lda + srcCol + 0);
                    Vector128<float> hi = AdvSimd.LoadVector128(aPtr + k * lda + srcCol + 4);
                    AdvSimd.Store(packedStripe + k * mr + 0, lo);
                    AdvSimd.Store(packedStripe + k * mr + 4, hi);
                }
            }
        }
    }
#else
    /// <summary>Runtime support gate (always false on net471 — no ARM64 AdvSimd intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackA{T}"/> (no ARM64 AdvSimd available).
    /// </summary>
    public static void PackA_Fp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        Span<double> packed, int mc, int kc, int mr) =>
        ScalarPack.PackA<double>(a, lda, transA, packed, mc, kc, mr);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackA{T}"/> (no ARM64 AdvSimd available).
    /// </summary>
    public static void PackA_Fp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        Span<float> packed, int mc, int kc, int mr) =>
        ScalarPack.PackA<float>(a, lda, transA, packed, mc, kc, mr);
#endif
}
