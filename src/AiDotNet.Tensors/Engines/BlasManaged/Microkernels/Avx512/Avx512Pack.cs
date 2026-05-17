using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX-512 SIMD pack routines that produce bit-identical output to
/// <see cref="ScalarPack"/> but use Vector512-width loads/stores for higher
/// memory bandwidth on the transA=true paths.
///
/// <para>
/// transA=true is the SIMD-friendly path: A is stored row-major [K, M], so
/// each k-step's Mr consecutive values are contiguous and can be copied with
/// a single Vector512 load + store per (k, stripe) pair when Mr matches the
/// vector's lane count (8 for FP64, 16 for FP32).
/// </para>
///
/// <para>
/// transA=false requires an 8×8 (FP64) or 16×16 (FP32) tile transpose using
/// <c>Avx512F.PermuteVar8x64</c> / shuffle intrinsics. For Phase D this case
/// falls back to <see cref="ScalarPack.PackA{T}"/> — the scalar implementation
/// is correct and the optimization can land later without changing the
/// correctness contract.
/// </para>
/// </summary>
internal static class Avx512Pack
{
#if NET8_0_OR_GREATER
    /// <summary>Runtime support gate. True when AVX-512F intrinsics are usable on the current process.</summary>
    public static bool IsSupported => Avx512F.IsSupported;

    /// <summary>
    /// Pack a logical Mc-row × Kc-col panel of A into BLIS vpanel layout.
    /// Output is bit-identical to <see cref="ScalarPack.PackA{T}"/> for T=double.
    ///
    /// <para>
    /// When <paramref name="transA"/> is <c>true</c> and <paramref name="mr"/> is 8,
    /// each k-step reads exactly one <see cref="Vector512{T}"/> (8 doubles = 64 bytes)
    /// from the contiguous source row and writes it to the packed stripe. This is the
    /// SIMD-accelerated path.
    /// </para>
    ///
    /// <para>
    /// All other cases delegate to <see cref="ScalarPack.PackA{T}"/> for correctness.
    /// The transA=false 8×8 transpose optimization is deferred to a later phase.
    /// </para>
    /// </summary>
    /// <param name="a">Source A buffer, stored row-major [K, M] when transA=true or [M, K] when transA=false.</param>
    /// <param name="lda">Leading dimension of A (number of columns in the stored layout).</param>
    /// <param name="transA">True if A is stored as A^T (logical [M, K] view from [K, M] memory).</param>
    /// <param name="packed">Destination vpanel buffer, length ≥ mc × kc.</param>
    /// <param name="mc">Rows of A to pack (must be exactly divisible by mr).</param>
    /// <param name="kc">Cols of A to pack (one Kc block).</param>
    /// <param name="mr">Microkernel row-tile width; must be 8 for the AVX-512 FP64 SIMD path.</param>
    public static unsafe void PackA_Fp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        Span<double> packed, int mc, int kc, int mr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Pack requires Avx512F.");

        if (!transA)
        {
            ScalarPack.PackA<double>(a, lda, transA, packed, mc, kc, mr);
            return;
        }

        if (mr == 8)
        {
            // transA=true, mr=8 SIMD path:
            // A is stored row-major [K, M] with lda=M. For each stripe the Mr=8 source
            // values at a given k-step occupy consecutive addresses A[k, stripe*8 .. stripe*8+7],
            // which is exactly one Vector512<double> (8 doubles = 64 bytes). Load + store,
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
                        Vector512<double> row = Avx512F.LoadVector512(aPtr + k * lda + srcCol);
                        Avx512F.Store(packedStripe + k * mr, row);
                    }
                }
            }
        }
        else
        {
            ScalarPack.PackA<double>(a, lda, transA, packed, mc, kc, mr);
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="PackA_Fp64"/>. The AVX-512 FP32 microkernel uses Mr=16,
    /// which is exactly one <see cref="Vector512{T}"/> (16 floats = 64 bytes) wide.
    ///
    /// <para>
    /// When <paramref name="transA"/> is <c>true</c> and <paramref name="mr"/> is 16,
    /// each k-step reads one Vector512&lt;float&gt; from the source row and stores it
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
    /// <param name="mr">Microkernel row-tile width; must be 16 for the AVX-512 FP32 SIMD path.</param>
    public static unsafe void PackA_Fp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        Span<float> packed, int mc, int kc, int mr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Pack requires Avx512F.");

        if (!transA)
        {
            ScalarPack.PackA<float>(a, lda, transA, packed, mc, kc, mr);
            return;
        }

        if (mr == 16)
        {
            // transA=true, mr=16 SIMD path:
            // A is stored row-major [K, M] with lda=M. For each stripe the Mr=16 source
            // values at a given k-step occupy consecutive addresses A[k, stripe*16 .. stripe*16+15],
            // which is exactly one Vector512<float> (16 floats = 64 bytes).
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
                        Vector512<float> row = Avx512F.LoadVector512(aPtr + k * lda + srcCol);
                        Avx512F.Store(packedStripe + k * mr, row);
                    }
                }
            }
        }
        else
        {
            ScalarPack.PackA<float>(a, lda, transA, packed, mc, kc, mr);
        }
    }

    /// <summary>
    /// Pack a logical Kc-row × Nc-col panel of B into BLIS stripe layout.
    /// Output is bit-identical to <see cref="ScalarPack.PackB{T}"/>.
    ///
    /// <para>
    /// When <paramref name="transB"/> is <c>false</c> and <paramref name="nr"/> is 16,
    /// each k-step's Nr=16 consecutive column values are split across TWO
    /// <see cref="Vector512{T}"/> loads (8+8=16 doubles = 128 bytes) and stored
    /// into the packed stripe. This is the SIMD-accelerated path.
    /// </para>
    ///
    /// <para>
    /// An additional SIMD path handles nr=8 (one 512-bit load per k-step).
    /// All other cases delegate to <see cref="ScalarPack.PackB{T}"/> for correctness.
    /// transB=true requires a gather pattern and is deferred to scalar.
    /// </para>
    /// </summary>
    public static unsafe void PackB_Fp64(
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> packed, int nc, int kc, int nr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Pack requires Avx512F.");

        if (transB)
        {
            ScalarPack.PackB<double>(b, ldb, transB, packed, nc, kc, nr);
            return;
        }

        if (nr == 16)
        {
            int numStripes = nc / nr;
            fixed (double* bPtr = b)
            fixed (double* pPtr = packed)
            {
                for (int stripe = 0; stripe < numStripes; stripe++)
                {
                    int srcCol = stripe * nr;
                    double* packedStripe = pPtr + stripe * kc * nr;
                    for (int k = 0; k < kc; k++)
                    {
                        Vector512<double> lo = Avx512F.LoadVector512(bPtr + k * ldb + srcCol);
                        Vector512<double> hi = Avx512F.LoadVector512(bPtr + k * ldb + srcCol + 8);
                        Avx512F.Store(packedStripe + k * nr, lo);
                        Avx512F.Store(packedStripe + k * nr + 8, hi);
                    }
                }
            }
        }
        else if (nr == 8)
        {
            // Secondary path for nr=8 (covers AVX2 microkernel width too if used here).
            int numStripes = nc / nr;
            fixed (double* bPtr = b)
            fixed (double* pPtr = packed)
            {
                for (int stripe = 0; stripe < numStripes; stripe++)
                {
                    int srcCol = stripe * nr;
                    double* packedStripe = pPtr + stripe * kc * nr;
                    for (int k = 0; k < kc; k++)
                    {
                        Vector512<double> row = Avx512F.LoadVector512(bPtr + k * ldb + srcCol);
                        Avx512F.Store(packedStripe + k * nr, row);
                    }
                }
            }
        }
        else
        {
            ScalarPack.PackB<double>(b, ldb, transB, packed, nc, kc, nr);
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="PackB_Fp64"/>. The AVX-512 FP32 microkernel uses Nr=16,
    /// which is exactly one <see cref="Vector512{T}"/> (16 floats = 64 bytes) wide.
    ///
    /// <para>
    /// When <paramref name="transB"/> is <c>false</c> and <paramref name="nr"/> is 16,
    /// each k-step reads one Vector512&lt;float&gt; from the source row and stores it
    /// into the packed stripe — bit-identical to <see cref="ScalarPack.PackB{T}"/>
    /// for T=float.
    /// </para>
    ///
    /// <para>
    /// All other cases delegate to <see cref="ScalarPack.PackB{T}"/> for correctness.
    /// </para>
    /// </summary>
    public static unsafe void PackB_Fp32(
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> packed, int nc, int kc, int nr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Pack requires Avx512F.");

        if (transB)
        {
            ScalarPack.PackB<float>(b, ldb, transB, packed, nc, kc, nr);
            return;
        }

        if (nr == 16)
        {
            int numStripes = nc / nr;
            fixed (float* bPtr = b)
            fixed (float* pPtr = packed)
            {
                for (int stripe = 0; stripe < numStripes; stripe++)
                {
                    int srcCol = stripe * nr;
                    float* packedStripe = pPtr + stripe * kc * nr;
                    for (int k = 0; k < kc; k++)
                    {
                        Vector512<float> row = Avx512F.LoadVector512(bPtr + k * ldb + srcCol);
                        Avx512F.Store(packedStripe + k * nr, row);
                    }
                }
            }
        }
        else
        {
            ScalarPack.PackB<float>(b, ldb, transB, packed, nc, kc, nr);
        }
    }
#else
    /// <summary>Runtime support gate (always false on net471 — no Vector512&lt;T&gt; intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackA{T}"/> (no AVX-512 available).
    /// </summary>
    public static void PackA_Fp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        Span<double> packed, int mc, int kc, int mr) =>
        ScalarPack.PackA<double>(a, lda, transA, packed, mc, kc, mr);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackA{T}"/> (no AVX-512 available).
    /// </summary>
    public static void PackA_Fp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        Span<float> packed, int mc, int kc, int mr) =>
        ScalarPack.PackA<float>(a, lda, transA, packed, mc, kc, mr);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackB{T}"/> (no AVX-512 available).
    /// </summary>
    public static void PackB_Fp64(
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> packed, int nc, int kc, int nr) =>
        ScalarPack.PackB<double>(b, ldb, transB, packed, nc, kc, nr);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackB{T}"/> (no AVX-512 available).
    /// </summary>
    public static void PackB_Fp32(
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> packed, int nc, int kc, int nr) =>
        ScalarPack.PackB<float>(b, ldb, transB, packed, nc, kc, nr);
#endif
}
