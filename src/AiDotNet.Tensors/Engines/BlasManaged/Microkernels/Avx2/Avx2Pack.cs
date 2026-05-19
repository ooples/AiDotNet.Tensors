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

    /// <summary>
    /// Pack a logical Kc-row × Nc-col panel of B into BLIS stripe layout.
    /// Output is bit-identical to <see cref="ScalarPack.PackB{T}"/> for T=double.
    ///
    /// <para>
    /// When <paramref name="transB"/> is <c>false</c> and <paramref name="nr"/> is 8,
    /// each k-step's Nr=8 consecutive source doubles span two Vector256&lt;double&gt;
    /// (4 doubles each), so two loads and two stores are used per (stripe, k) pair —
    /// no arithmetic, bit-identical to scalar.
    /// </para>
    ///
    /// <para>
    /// When <paramref name="transB"/> is <c>false</c> and <paramref name="nr"/> is 4,
    /// exactly one Vector256&lt;double&gt; load/store suffices per (stripe, k) pair.
    /// </para>
    ///
    /// <para>
    /// transB=true requires gathering Nr values from Nr different rows — not
    /// SIMD-friendly without vgatherdpd. This case delegates to
    /// <see cref="ScalarPack.PackB{T}"/>. SIMD gather optimization is deferred.
    /// </para>
    /// </summary>
    /// <param name="b">Source B buffer, stored row-major [K, N] when transB=false or [N, K] when transB=true.</param>
    /// <param name="ldb">Leading dimension of B (number of columns in the stored layout).</param>
    /// <param name="transB">True if B is stored as B^T (logical [K, N] view from [N, K] memory).</param>
    /// <param name="packed">Destination panel buffer, length ≥ nc × kc.</param>
    /// <param name="nc">Cols of B to pack (must be exactly divisible by nr).</param>
    /// <param name="kc">Rows of B to pack (one Kc block).</param>
    /// <param name="nr">Microkernel col-tile width; must be 8 or 4 for the AVX2 FP64 SIMD path.</param>
    public static unsafe void PackB_Fp64(
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> packed, int nc, int kc, int nr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Pack requires AVX2.");

        // transB=true requires a Nr-row strided gather — not SIMD-friendly without
        // vgatherdpd. Phase C falls back to scalar; SIMD gather optimization deferred.
        if (transB)
        {
            ScalarPack.PackB<double>(b, ldb, transB, packed, nc, kc, nr);
            return;
        }

        // transB=false SIMD path: B is row-major [K, N]. For each (stripe, k) pair,
        // copy Nr consecutive doubles starting at B[k, stripe*Nr] to packed.
        if (nr == 8)
        {
            // Nr=8: two Vector256<double> loads/stores per k-step (4 doubles each).
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
                        Vector256<double> lo = Avx.LoadVector256(bPtr + k * ldb + srcCol);
                        Vector256<double> hi = Avx.LoadVector256(bPtr + k * ldb + srcCol + 4);
                        Avx.Store(packedStripe + k * nr, lo);
                        Avx.Store(packedStripe + k * nr + 4, hi);
                    }
                }
            }
            // CodeRabbit #366: zero-pad the partial tail stripe so kernels
            // can safely read packedB[k*nr + col] for col in [0, effectiveNr).
            // ScalarPack.PackB does this for the full path; the SIMD path
            // dropped the tail silently before this guard.
            PackBTailFp64(b, ldb, packed, numStripes, nc - numStripes * nr, kc, nr);
        }
        else if (nr == 4)
        {
            // Nr=4: one Vector256<double> load/store per k-step.
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
                        Vector256<double> row = Avx.LoadVector256(bPtr + k * ldb + srcCol);
                        Avx.Store(packedStripe + k * nr, row);
                    }
                }
            }
            PackBTailFp64(b, ldb, packed, numStripes, nc - numStripes * nr, kc, nr);
        }
        else
        {
            ScalarPack.PackB<double>(b, ldb, transB, packed, nc, kc, nr);
        }
    }

    /// <summary>
    /// Zero-pads the final partial-N stripe for the AVX2 FP64 transB=false
    /// PackB paths. Mirrors the tail loop in <see cref="ScalarPack.PackB{T}"/>.
    /// </summary>
    private static void PackBTailFp64(
        ReadOnlySpan<double> b, int ldb, Span<double> packed,
        int numFullStripes, int tailCols, int kc, int nr)
    {
        if (tailCols <= 0) return;
        int tailPackedOff = numFullStripes * kc * nr;
        int tailBaseCol = numFullStripes * nr;
        for (int k = 0; k < kc; k++)
        {
            for (int col = 0; col < nr; col++)
            {
                int logicalCol = tailBaseCol + col;
                double value = col < tailCols ? b[k * ldb + logicalCol] : 0.0;
                packed[tailPackedOff + k * nr + col] = value;
            }
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="PackB_Fp64"/>. Nr=8 fits exactly in a
    /// Vector256&lt;float&gt; — single load/store per k-step.
    ///
    /// <para>
    /// When <paramref name="transB"/> is <c>false</c> and <paramref name="nr"/> is 8,
    /// each k-step's Nr=8 consecutive source floats fit in one Vector256&lt;float&gt;
    /// (32 bytes), so one load and one store suffice — bit-identical to scalar.
    /// </para>
    ///
    /// <para>
    /// transB=true and all other nr values delegate to <see cref="ScalarPack.PackB{T}"/>.
    /// </para>
    /// </summary>
    /// <param name="b">Source B buffer, stored row-major [K, N] when transB=false or [N, K] when transB=true.</param>
    /// <param name="ldb">Leading dimension of B (number of columns in the stored layout).</param>
    /// <param name="transB">True if B is stored as B^T (logical [K, N] view from [N, K] memory).</param>
    /// <param name="packed">Destination panel buffer, length ≥ nc × kc.</param>
    /// <param name="nc">Cols of B to pack (must be exactly divisible by nr).</param>
    /// <param name="kc">Rows of B to pack (one Kc block).</param>
    /// <param name="nr">Microkernel col-tile width; must be 8 for the AVX2 FP32 SIMD path.</param>
    public static unsafe void PackB_Fp32(
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> packed, int nc, int kc, int nr)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Pack requires AVX2.");

        if (transB)
        {
            // Sub-D10: SIMD transB=true PackB using AVX2 8x8 transpose.
            // B is stored [N, K] row-major; we need [Kc x Nr=8] packed.
            // For Nr=8, process the K dimension in groups of 8: load 8 rows
            // (one stripe of 8 N-values, contiguous K) and transpose to write
            // 8 packed K-rows, each holding 8 N-values.
            if (nr == 8)
            {
                PackBTransposedFp32(b, ldb, packed, nc, kc);
                return;
            }
            ScalarPack.PackB<float>(b, ldb, transB, packed, nc, kc, nr);
            return;
        }

        if (nr == 8)
        {
            // Nr=8: exactly one Vector256<float> (8 floats = 32 bytes) per k-step.
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
                        Vector256<float> row = Avx.LoadVector256(bPtr + k * ldb + srcCol);
                        Avx.Store(packedStripe + k * nr, row);
                    }
                }
            }
            // CodeRabbit #366: zero-pad the partial tail stripe — see
            // PackBTailFp64 for rationale.
            PackBTailFp32(b, ldb, packed, numStripes, nc - numStripes * nr, kc, nr);
        }
        else
        {
            ScalarPack.PackB<float>(b, ldb, transB, packed, nc, kc, nr);
        }
    }

    /// <summary>
    /// Sub-D10 SIMD path for transB=true PackB FP32 with Nr=8. B is stored
    /// [N, K] row-major; pack output is [Kc × Nr=8] row-major. For each stripe
    /// of 8 consecutive N values, load 8 source rows in K-groups of 8, do an
    /// 8×8 in-register transpose using AVX2 unpacklo/unpackhi + shuffle +
    /// permute2x128, then store 8 transposed rows to the packed buffer.
    /// Bit-identical output to <see cref="ScalarPack.PackB{T}"/> for FP32 transB.
    /// </summary>
    private static unsafe void PackBTransposedFp32(
        ReadOnlySpan<float> b, int ldb,
        Span<float> packed, int nc, int kc)
    {
        const int Nr = 8;
        int numFullStripes = nc / Nr;
        int kFull = (kc / 8) * 8;

        fixed (float* bPtr = b)
        fixed (float* pPtr = packed)
        {
            for (int stripe = 0; stripe < numFullStripes; stripe++)
            {
                int srcRow = stripe * Nr;
                float* packedStripe = pPtr + stripe * kc * Nr;

                for (int kBase = 0; kBase < kFull; kBase += 8)
                {
                    Vector256<float> r0 = Avx.LoadVector256(bPtr + (srcRow + 0) * ldb + kBase);
                    Vector256<float> r1 = Avx.LoadVector256(bPtr + (srcRow + 1) * ldb + kBase);
                    Vector256<float> r2 = Avx.LoadVector256(bPtr + (srcRow + 2) * ldb + kBase);
                    Vector256<float> r3 = Avx.LoadVector256(bPtr + (srcRow + 3) * ldb + kBase);
                    Vector256<float> r4 = Avx.LoadVector256(bPtr + (srcRow + 4) * ldb + kBase);
                    Vector256<float> r5 = Avx.LoadVector256(bPtr + (srcRow + 5) * ldb + kBase);
                    Vector256<float> r6 = Avx.LoadVector256(bPtr + (srcRow + 6) * ldb + kBase);
                    Vector256<float> r7 = Avx.LoadVector256(bPtr + (srcRow + 7) * ldb + kBase);

                    Vector256<float> t0 = Avx.UnpackLow(r0, r1);
                    Vector256<float> t1 = Avx.UnpackHigh(r0, r1);
                    Vector256<float> t2 = Avx.UnpackLow(r2, r3);
                    Vector256<float> t3 = Avx.UnpackHigh(r2, r3);
                    Vector256<float> t4 = Avx.UnpackLow(r4, r5);
                    Vector256<float> t5 = Avx.UnpackHigh(r4, r5);
                    Vector256<float> t6 = Avx.UnpackLow(r6, r7);
                    Vector256<float> t7 = Avx.UnpackHigh(r6, r7);

                    Vector256<float> s0 = Avx.Shuffle(t0, t2, 0x44);
                    Vector256<float> s1 = Avx.Shuffle(t0, t2, 0xEE);
                    Vector256<float> s2 = Avx.Shuffle(t1, t3, 0x44);
                    Vector256<float> s3 = Avx.Shuffle(t1, t3, 0xEE);
                    Vector256<float> s4 = Avx.Shuffle(t4, t6, 0x44);
                    Vector256<float> s5 = Avx.Shuffle(t4, t6, 0xEE);
                    Vector256<float> s6 = Avx.Shuffle(t5, t7, 0x44);
                    Vector256<float> s7 = Avx.Shuffle(t5, t7, 0xEE);

                    Vector256<float> p0 = Avx.Permute2x128(s0, s4, 0x20);
                    Vector256<float> p1 = Avx.Permute2x128(s1, s5, 0x20);
                    Vector256<float> p2 = Avx.Permute2x128(s2, s6, 0x20);
                    Vector256<float> p3 = Avx.Permute2x128(s3, s7, 0x20);
                    Vector256<float> p4 = Avx.Permute2x128(s0, s4, 0x31);
                    Vector256<float> p5 = Avx.Permute2x128(s1, s5, 0x31);
                    Vector256<float> p6 = Avx.Permute2x128(s2, s6, 0x31);
                    Vector256<float> p7 = Avx.Permute2x128(s3, s7, 0x31);

                    Avx.Store(packedStripe + (kBase + 0) * Nr, p0);
                    Avx.Store(packedStripe + (kBase + 1) * Nr, p1);
                    Avx.Store(packedStripe + (kBase + 2) * Nr, p2);
                    Avx.Store(packedStripe + (kBase + 3) * Nr, p3);
                    Avx.Store(packedStripe + (kBase + 4) * Nr, p4);
                    Avx.Store(packedStripe + (kBase + 5) * Nr, p5);
                    Avx.Store(packedStripe + (kBase + 6) * Nr, p6);
                    Avx.Store(packedStripe + (kBase + 7) * Nr, p7);
                }
                // K-tail: per-element fallback.
                for (int k = kFull; k < kc; k++)
                {
                    for (int col = 0; col < Nr; col++)
                    {
                        packedStripe[k * Nr + col] = bPtr[(srcRow + col) * ldb + k];
                    }
                }
            }
            // Partial-N tail stripe (nc % Nr != 0): zero-pad.
            int tailCols = nc - numFullStripes * Nr;
            if (tailCols > 0)
            {
                int tailStripe = numFullStripes;
                int tailPackedOff = tailStripe * kc * Nr;
                int tailBaseRow = tailStripe * Nr;
                for (int k = 0; k < kc; k++)
                {
                    for (int col = 0; col < Nr; col++)
                    {
                        float value = col < tailCols ? bPtr[(tailBaseRow + col) * ldb + k] : 0f;
                        pPtr[tailPackedOff + k * Nr + col] = value;
                    }
                }
            }
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="PackBTailFp64"/>.
    /// </summary>
    private static void PackBTailFp32(
        ReadOnlySpan<float> b, int ldb, Span<float> packed,
        int numFullStripes, int tailCols, int kc, int nr)
    {
        if (tailCols <= 0) return;
        int tailPackedOff = numFullStripes * kc * nr;
        int tailBaseCol = numFullStripes * nr;
        for (int k = 0; k < kc; k++)
        {
            for (int col = 0; col < nr; col++)
            {
                int logicalCol = tailBaseCol + col;
                float value = col < tailCols ? b[k * ldb + logicalCol] : 0f;
                packed[tailPackedOff + k * nr + col] = value;
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

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackB{T}"/> (no AVX2 available).
    /// </summary>
    public static void PackB_Fp64(
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> packed, int nc, int kc, int nr) =>
        ScalarPack.PackB<double>(b, ldb, transB, packed, nc, kc, nr);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarPack.PackB{T}"/> (no AVX2 available).
    /// </summary>
    public static void PackB_Fp32(
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> packed, int nc, int kc, int nr) =>
        ScalarPack.PackB<float>(b, ldb, transB, packed, nc, kc, nr);
#endif

    /// <summary>
    /// Sub-D5 (#372 follow-up): generic dispatcher matching <see cref="ScalarPack.PackA{T}"/>.
    /// Routes to the AVX2 SIMD path when supported AND dtype matches; falls back
    /// to <see cref="ScalarPack.PackA{T}"/> otherwise. Drop-in replacement.
    /// </summary>
    public static void PackA<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        Span<T> packed, int mc, int kc, int mr) where T : unmanaged
    {
        if (IsSupported)
        {
            if (typeof(T) == typeof(float))
            {
                PackA_Fp32(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(a),
                    lda, transA,
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packed),
                    mc, kc, mr);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                PackA_Fp64(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(a),
                    lda, transA,
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packed),
                    mc, kc, mr);
                return;
            }
        }
        ScalarPack.PackA<T>(a, lda, transA, packed, mc, kc, mr);
    }

    /// <summary>
    /// Sub-D5 (#372 follow-up): generic dispatcher matching <see cref="ScalarPack.PackB{T}"/>.
    /// </summary>
    public static void PackB<T>(
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> packed, int nc, int kc, int nr) where T : unmanaged
    {
        if (IsSupported)
        {
            if (typeof(T) == typeof(float))
            {
                PackB_Fp32(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(b),
                    ldb, transB,
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(packed),
                    nc, kc, nr);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                PackB_Fp64(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(b),
                    ldb, transB,
                    System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(packed),
                    nc, kc, nr);
                return;
            }
        }
        ScalarPack.PackB<T>(b, ldb, transB, packed, nc, kc, nr);
    }

    /// <summary>
    /// Sub-N (#404): pack a single (stripeStart … stripeStart+numStripes) range
    /// of B-stripes into the packed buffer at the matching stripe offset. Each
    /// stripe occupies <c>kc * nr</c> elements at <c>(stripeIdx - stripeStart) * kc * nr</c>
    /// when caller passes the packed-buffer slice rooted at <c>stripeStart * kc * nr</c>.
    /// Used by PackBothStrategy.RunParallelUnsafe to parallelize the pack-B
    /// step across nc-stripes — disjoint write regions, no synchronization.
    /// </summary>
    /// <param name="b">Source B with the SAME logical position as the full PackB call.</param>
    /// <param name="packedStripeSlice">Pre-sliced packed buffer rooted at <c>stripeStart * kc * nr</c>.</param>
    /// <param name="stripeStart">First stripe index to pack (inclusive).</param>
    /// <param name="numStripes">Number of stripes to pack.</param>
    /// <param name="totalNc">Original full <c>nc</c> for the call (for tail-stripe zero-pad).</param>
    public static void PackBStripeRange<T>(
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> packedStripeSlice,
        int stripeStart, int numStripes,
        int totalNc, int kc, int nr) where T : unmanaged
    {
        // Compute the source slice offset for this stripe range. PackB's source
        // walking starts at column 0 (transB=false) or row 0 (transB=true) of the
        // caller-supplied b. To pack only stripes [stripeStart, stripeStart+numStripes),
        // we advance the source by stripeStart*nr in N (transB=false: cols) or in K
        // (transB=true: rows-of-Bᵀ which are N).
        int srcOffset = transB ? stripeStart * nr * ldb : stripeStart * nr;
        var bSlice = b.Slice(srcOffset);
        // Effective Nc for this range is min(numStripes * nr, totalNc - stripeStart * nr)
        // so the last (possibly partial) stripe gets zero-padded by the underlying PackB.
        int effectiveNc = Math.Min(numStripes * nr, totalNc - stripeStart * nr);
        if (effectiveNc <= 0) return;
        PackB<T>(bSlice, ldb, transB, packedStripeSlice, effectiveNc, kc, nr);
    }
}
