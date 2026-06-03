using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX-512 FP32 microkernel: 8×32 output tile (8 rows × 32 cols) — the
/// higher-arithmetic-intensity #409 S.4 candidate for the load-bound 16×16
/// (<see cref="Avx512Fp32_16x16"/>).
///
/// <para>
/// <b>Why 8×32.</b> The 16×16 holds each C row in ONE <see cref="Vector512{T}"/>
/// (16 floats), so per K-step it issues 16 FMAs against 1 B-load + 16 A-broadcasts =
/// <c>16/17 ≈ 0.94 FMA/load</c> — load-port bound, the same pathology Sub-S (#409)
/// fixed for the AVX2 8×8 with the 6×16 tile. This 8×32 tile holds each row in TWO
/// Vector512 (lo = cols 0..15, hi = cols 16..31), so per K-step it issues 16 FMAs
/// against 2 B-loads + 8 A-broadcasts = <c>16/10 = 1.6 FMA/load</c> — the same
/// higher intensity the FP64 <see cref="Avx512Fp64_8x16"/> already runs at. Register
/// budget: 16 accumulators + 2 B + 1 transient A broadcast = 19 of 32 ZMM (ample
/// headroom for the JIT to issue the next step's loads ahead of the FMAs).
/// </para>
///
/// <para>
/// Reads packed-A in [Kc × Mr=8] vpanel layout and packed-B in [Kc × Nr=32] stripe
/// layout. Like the AVX2 6×16/6×8 wide tiles it is used in FAST (non-deterministic)
/// mode only — deterministic mode and the pre-pack path keep the validated 16×16 tile
/// whose layout/order their bit-exact invariants are pinned against. Gated by
/// <c>Avx512F.IsSupported</c> at the dispatcher; compiles only on net8.0+ (Vector512).
/// </para>
/// </summary>
internal static class Avx512Fp32_8x32
{
    /// <summary>Row-tile width (output rows per invocation).</summary>
    internal const int Mr = 8;
    /// <summary>Column-tile width (output cols per invocation).</summary>
    internal const int Nr = 32;

#if NET8_0_OR_GREATER
    /// <summary>Runtime support gate. True when AVX-512F intrinsics are usable.</summary>
    public static bool IsSupported => Avx512F.IsSupported;

    /// <summary>
    /// Prefetch lookahead, in K-steps. Each K-step reads one Mr-wide packed-A slice and
    /// one Nr-wide packed-B stripe (two 512-bit loads ⇒ two cache lines); prefetch A plus
    /// both B lines this many iterations ahead to hide L2→L1 latency behind the FMAs.
    /// </summary>
    private const int PrefetchDistance = 8;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile over kc K-steps.
    /// C is read-modify-write; caller zero-inits for a fresh result.
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=8] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr=32] row-major.</param>
    /// <param name="c">Output buffer; reads + writes C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static unsafe void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Fp32_8x32 requires Avx512F.");

        // 16 Vector512<float> accumulators (8 rows × 2 halves: lo = cols 0..15, hi = 16..31).
        fixed (float* cPtr = c)
        {
            Vector512<float> acc0_lo = Avx512F.LoadVector512(cPtr + 0 * ldc + 0);
            Vector512<float> acc0_hi = Avx512F.LoadVector512(cPtr + 0 * ldc + 16);
            Vector512<float> acc1_lo = Avx512F.LoadVector512(cPtr + 1 * ldc + 0);
            Vector512<float> acc1_hi = Avx512F.LoadVector512(cPtr + 1 * ldc + 16);
            Vector512<float> acc2_lo = Avx512F.LoadVector512(cPtr + 2 * ldc + 0);
            Vector512<float> acc2_hi = Avx512F.LoadVector512(cPtr + 2 * ldc + 16);
            Vector512<float> acc3_lo = Avx512F.LoadVector512(cPtr + 3 * ldc + 0);
            Vector512<float> acc3_hi = Avx512F.LoadVector512(cPtr + 3 * ldc + 16);
            Vector512<float> acc4_lo = Avx512F.LoadVector512(cPtr + 4 * ldc + 0);
            Vector512<float> acc4_hi = Avx512F.LoadVector512(cPtr + 4 * ldc + 16);
            Vector512<float> acc5_lo = Avx512F.LoadVector512(cPtr + 5 * ldc + 0);
            Vector512<float> acc5_hi = Avx512F.LoadVector512(cPtr + 5 * ldc + 16);
            Vector512<float> acc6_lo = Avx512F.LoadVector512(cPtr + 6 * ldc + 0);
            Vector512<float> acc6_hi = Avx512F.LoadVector512(cPtr + 6 * ldc + 16);
            Vector512<float> acc7_lo = Avx512F.LoadVector512(cPtr + 7 * ldc + 0);
            Vector512<float> acc7_hi = Avx512F.LoadVector512(cPtr + 7 * ldc + 16);

            fixed (float* aPtr = packedA)
            fixed (float* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    if (k + PrefetchDistance < kc)
                    {
                        Sse.Prefetch0(aPtr + (k + PrefetchDistance) * Mr);
                        Sse.Prefetch0(bPtr + (k + PrefetchDistance) * Nr + 0);
                        Sse.Prefetch0(bPtr + (k + PrefetchDistance) * Nr + 16);
                    }

                    // Two vector loads per K-step from packed-B (lo = cols 0..15, hi = 16..31).
                    Vector512<float> bRow_lo = Avx512F.LoadVector512(bPtr + k * Nr + 0);
                    Vector512<float> bRow_hi = Avx512F.LoadVector512(bPtr + k * Nr + 16);

                    Vector512<float> a0 = Vector512.Create(aPtr[k * Mr + 0]);
                    Vector512<float> a1 = Vector512.Create(aPtr[k * Mr + 1]);
                    Vector512<float> a2 = Vector512.Create(aPtr[k * Mr + 2]);
                    Vector512<float> a3 = Vector512.Create(aPtr[k * Mr + 3]);
                    Vector512<float> a4 = Vector512.Create(aPtr[k * Mr + 4]);
                    Vector512<float> a5 = Vector512.Create(aPtr[k * Mr + 5]);
                    Vector512<float> a6 = Vector512.Create(aPtr[k * Mr + 6]);
                    Vector512<float> a7 = Vector512.Create(aPtr[k * Mr + 7]);

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

            Avx512F.Store(cPtr + 0 * ldc + 0, acc0_lo);
            Avx512F.Store(cPtr + 0 * ldc + 16, acc0_hi);
            Avx512F.Store(cPtr + 1 * ldc + 0, acc1_lo);
            Avx512F.Store(cPtr + 1 * ldc + 16, acc1_hi);
            Avx512F.Store(cPtr + 2 * ldc + 0, acc2_lo);
            Avx512F.Store(cPtr + 2 * ldc + 16, acc2_hi);
            Avx512F.Store(cPtr + 3 * ldc + 0, acc3_lo);
            Avx512F.Store(cPtr + 3 * ldc + 16, acc3_hi);
            Avx512F.Store(cPtr + 4 * ldc + 0, acc4_lo);
            Avx512F.Store(cPtr + 4 * ldc + 16, acc4_hi);
            Avx512F.Store(cPtr + 5 * ldc + 0, acc5_lo);
            Avx512F.Store(cPtr + 5 * ldc + 16, acc5_hi);
            Avx512F.Store(cPtr + 6 * ldc + 0, acc6_lo);
            Avx512F.Store(cPtr + 6 * ldc + 16, acc6_hi);
            Avx512F.Store(cPtr + 7 * ldc + 0, acc7_lo);
            Avx512F.Store(cPtr + 7 * ldc + 16, acc7_hi);
        }
    }
#else
    /// <summary>Runtime support gate (false on net471 — no Vector512 intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>Throws on net471 — Vector512 unavailable. Dispatcher gates this.</summary>
    public static void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("Avx512Fp32_8x32 requires net8.0+ for Vector512<T>.");
#endif
}
