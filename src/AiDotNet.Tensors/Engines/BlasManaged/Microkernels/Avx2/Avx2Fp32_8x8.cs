using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX2 + FMA FP32 microkernel: 8×8 output tile (8 rows × 8 cols).
/// Reads packed-A in [Kc × Mr=8] vpanel layout and packed-B in [Kc × Nr=8]
/// layout. Each row of C is held in ONE <see cref="Vector256{T}"/> accumulator
/// (Vector256&lt;float&gt; has 8 lanes — exactly Nr wide), giving 8 total
/// accumulators across the K-loop.
///
/// <para>
/// Uses <see cref="Fma.MultiplyAdd"/> for fused multiply-add: 1 instruction
/// per row, 8 FMAs per K-step. The single-vector-per-row layout is more
/// register-efficient than C1's lo+hi split because FP32 has twice the lanes
/// per 256-bit register.
/// </para>
///
/// <para>
/// Gated by <c>Avx2.IsSupported &amp;&amp; Fma.IsSupported</c> at the dispatcher
/// (Task C7). On net471 the type still compiles but the kernel methods are
/// not reachable.
/// </para>
/// </summary>
internal static class Avx2Fp32_8x8
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    internal const int Mr = 8;
    /// <summary>The column-tile width of this microkernel (output cols per invocation).</summary>
    internal const int Nr = 8;

#if NET5_0_OR_GREATER
    /// <summary>Runtime support gate. True when AVX2 and FMA intrinsics are usable.</summary>
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile, summing over
    /// kc K-steps. C is read-modify-write; caller is responsible for zero-init
    /// if a fresh result is desired. When kc is 0 the kernel reads + writes C
    /// unchanged (no-op accumulation).
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=8] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr=8] row-major.</param>
    /// <param name="c">Output buffer; reads + writes C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    /// <summary>
    /// Sub-O (#405): software-prefetch distance in K-iterations. Each K-step
    /// consumes 32 bytes of A (8 floats) and 32 bytes of B (8 floats); a prefetch
    /// 8 iterations ahead pulls in the next 4 cache lines of each. Tuned for
    /// Zen-class L1 latency (~4 cycles) vs FMA latency (~4 cycles) — keeps the
    /// load front-end one issue ahead of the compute. PrefetchDistance > kc just
    /// becomes a no-op via the in-loop bounds check.
    /// </summary>
    private const int PrefetchDistance = 8;

    public static unsafe void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Fp32_8x8 requires Avx2 + Fma.");

        // 8 Vector256<float> accumulators — one per row. Each holds 8 floats = Nr cols.
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
                // Sub-S (#409): K-loop unrolled by 2 so two B-loads issue
                // back-to-back. The second load's L1-latency (~4 cycles on Zen)
                // overlaps with the first batch of 8 FMAs (~4 cycle each, but
                // 2 FMA ports → 4 FMAs/cycle issue rate). Halves branch
                // overhead too. Loop body produces 16 FMAs per iteration.
                int kc2 = kc & ~1;
                int k = 0;
                for (; k < kc2; k += 2)
                {
                    // Sub-O: prefetch 2 iterations ahead of the unroll-by-2 step.
                    if (k + PrefetchDistance < kc)
                    {
                        Sse.Prefetch0(aPtr + (k + PrefetchDistance) * Mr);
                        Sse.Prefetch0(bPtr + (k + PrefetchDistance) * Nr);
                    }

                    // Stage 1: load both B rows back-to-back so the second
                    // load can stream while stage-1 FMAs are issued.
                    Vector256<float> bRow0 = Avx.LoadVector256(bPtr + k * Nr);
                    Vector256<float> bRow1 = Avx.LoadVector256(bPtr + (k + 1) * Nr);

                    // Stage 1 A broadcasts + FMAs.
                    Vector256<float> a00 = Vector256.Create(aPtr[k * Mr + 0]);
                    Vector256<float> a01 = Vector256.Create(aPtr[k * Mr + 1]);
                    Vector256<float> a02 = Vector256.Create(aPtr[k * Mr + 2]);
                    Vector256<float> a03 = Vector256.Create(aPtr[k * Mr + 3]);
                    acc0 = Fma.MultiplyAdd(a00, bRow0, acc0);
                    acc1 = Fma.MultiplyAdd(a01, bRow0, acc1);
                    acc2 = Fma.MultiplyAdd(a02, bRow0, acc2);
                    acc3 = Fma.MultiplyAdd(a03, bRow0, acc3);

                    Vector256<float> a04 = Vector256.Create(aPtr[k * Mr + 4]);
                    Vector256<float> a05 = Vector256.Create(aPtr[k * Mr + 5]);
                    Vector256<float> a06 = Vector256.Create(aPtr[k * Mr + 6]);
                    Vector256<float> a07 = Vector256.Create(aPtr[k * Mr + 7]);
                    acc4 = Fma.MultiplyAdd(a04, bRow0, acc4);
                    acc5 = Fma.MultiplyAdd(a05, bRow0, acc5);
                    acc6 = Fma.MultiplyAdd(a06, bRow0, acc6);
                    acc7 = Fma.MultiplyAdd(a07, bRow0, acc7);

                    // Stage 2 A broadcasts + FMAs.
                    Vector256<float> a10 = Vector256.Create(aPtr[(k + 1) * Mr + 0]);
                    Vector256<float> a11 = Vector256.Create(aPtr[(k + 1) * Mr + 1]);
                    Vector256<float> a12 = Vector256.Create(aPtr[(k + 1) * Mr + 2]);
                    Vector256<float> a13 = Vector256.Create(aPtr[(k + 1) * Mr + 3]);
                    acc0 = Fma.MultiplyAdd(a10, bRow1, acc0);
                    acc1 = Fma.MultiplyAdd(a11, bRow1, acc1);
                    acc2 = Fma.MultiplyAdd(a12, bRow1, acc2);
                    acc3 = Fma.MultiplyAdd(a13, bRow1, acc3);

                    Vector256<float> a14 = Vector256.Create(aPtr[(k + 1) * Mr + 4]);
                    Vector256<float> a15 = Vector256.Create(aPtr[(k + 1) * Mr + 5]);
                    Vector256<float> a16 = Vector256.Create(aPtr[(k + 1) * Mr + 6]);
                    Vector256<float> a17 = Vector256.Create(aPtr[(k + 1) * Mr + 7]);
                    acc4 = Fma.MultiplyAdd(a14, bRow1, acc4);
                    acc5 = Fma.MultiplyAdd(a15, bRow1, acc5);
                    acc6 = Fma.MultiplyAdd(a16, bRow1, acc6);
                    acc7 = Fma.MultiplyAdd(a17, bRow1, acc7);
                }
                // K-tail when kc is odd: process the single remaining iter.
                if (k < kc)
                {
                    Vector256<float> bRow = Avx.LoadVector256(bPtr + k * Nr);
                    Vector256<float> a0 = Vector256.Create(aPtr[k * Mr + 0]);
                    Vector256<float> a1 = Vector256.Create(aPtr[k * Mr + 1]);
                    Vector256<float> a2 = Vector256.Create(aPtr[k * Mr + 2]);
                    Vector256<float> a3 = Vector256.Create(aPtr[k * Mr + 3]);
                    Vector256<float> a4 = Vector256.Create(aPtr[k * Mr + 4]);
                    Vector256<float> a5 = Vector256.Create(aPtr[k * Mr + 5]);
                    Vector256<float> a6 = Vector256.Create(aPtr[k * Mr + 6]);
                    Vector256<float> a7 = Vector256.Create(aPtr[k * Mr + 7]);
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

            Avx.Store(cPtr + 0 * ldc, acc0);
            Avx.Store(cPtr + 1 * ldc, acc1);
            Avx.Store(cPtr + 2 * ldc, acc2);
            Avx.Store(cPtr + 3 * ldc, acc3);
            Avx.Store(cPtr + 4 * ldc, acc4);
            Avx.Store(cPtr + 5 * ldc, acc5);
            Avx.Store(cPtr + 6 * ldc, acc6);
            Avx.Store(cPtr + 7 * ldc, acc7);
        }
    }

    /// <summary>
    /// Sub-D (#372) — strided-B variant for <see cref="PackAOnlyStrategy"/>. Same
    /// loop structure as <see cref="Run"/>, but B is read in caller-supplied stride
    /// <paramref name="ldb"/> instead of the packed stride <see cref="Nr"/>=8.
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=8] row-major.</param>
    /// <param name="b">Source B, read at stride ldb. Must have ≥ kc rows of ldb cols.</param>
    /// <param name="ldb">Leading dimension of B (caller's row stride).</param>
    /// <param name="c">Output buffer; reads + writes C[0..Mr, 0..Nr=8] tile.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static unsafe void RunStridedB(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> b,
        int ldb,
        Span<float> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Fp32_8x8 requires Avx2 + Fma.");

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
            fixed (float* bPtr = b)
            {
                for (int k = 0; k < kc; k++)
                {
                    // Sub-O (#405): prefetch A + B `PrefetchDistance` iters ahead.
                    if (k + PrefetchDistance < kc)
                    {
                        Sse.Prefetch0(aPtr + (k + PrefetchDistance) * Mr);
                        Sse.Prefetch0(bPtr + (k + PrefetchDistance) * ldb);
                    }

                    // ONE difference vs Run: stride is ldb (caller-supplied), not Nr.
                    Vector256<float> bRow = Avx.LoadVector256(bPtr + k * ldb);

                    Vector256<float> a0 = Vector256.Create(aPtr[k * Mr + 0]);
                    Vector256<float> a1 = Vector256.Create(aPtr[k * Mr + 1]);
                    Vector256<float> a2 = Vector256.Create(aPtr[k * Mr + 2]);
                    Vector256<float> a3 = Vector256.Create(aPtr[k * Mr + 3]);
                    Vector256<float> a4 = Vector256.Create(aPtr[k * Mr + 4]);
                    Vector256<float> a5 = Vector256.Create(aPtr[k * Mr + 5]);
                    Vector256<float> a6 = Vector256.Create(aPtr[k * Mr + 6]);
                    Vector256<float> a7 = Vector256.Create(aPtr[k * Mr + 7]);

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

            Avx.Store(cPtr + 0 * ldc, acc0);
            Avx.Store(cPtr + 1 * ldc, acc1);
            Avx.Store(cPtr + 2 * ldc, acc2);
            Avx.Store(cPtr + 3 * ldc, acc3);
            Avx.Store(cPtr + 4 * ldc, acc4);
            Avx.Store(cPtr + 5 * ldc, acc5);
            Avx.Store(cPtr + 6 * ldc, acc6);
            Avx.Store(cPtr + 7 * ldc, acc7);
        }
    }
#else
    /// <summary>Runtime support gate (false on net471 — no AVX2 intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this.</summary>
    public static void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("Avx2Fp32_8x8 requires net5.0+ for Vector256<T>.");

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this.</summary>
    public static void RunStridedB(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> b,
        int ldb,
        Span<float> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("Avx2Fp32_8x8 requires net5.0+ for Vector256<T>.");
#endif
}
