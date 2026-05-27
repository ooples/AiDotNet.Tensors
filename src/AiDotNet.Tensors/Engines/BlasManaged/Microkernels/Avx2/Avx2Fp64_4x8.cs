using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX2 + FMA FP64 microkernel: 4×8 output tile (4 rows × 8 cols).
/// Reads packed-A in [Kc × Mr=4] vpanel layout and packed-B in [Kc × Nr=8]
/// layout. Each row of C is held in 2 <see cref="Vector256{T}"/> accumulators
/// (lo + hi halves), giving 8 total accumulators across the K-loop.
///
/// <para>
/// Uses <see cref="Fma.MultiplyAdd"/> for fused multiply-add: 1 instruction
/// per (row, half), 8 FMAs per K-step. Inner loop is FMA-port bound on
/// Haswell+ x64.
/// </para>
///
/// <para>
/// Gated by <c>Avx2.IsSupported &amp;&amp; Fma.IsSupported</c> at the dispatcher
/// (Task C7). On net471 the type still compiles but the kernel methods are
/// not reachable because <see cref="Vector256{T}"/> doesn't exist there.
/// </para>
///
/// <para>
/// <b>Audit (Sub-S #409, Phase S.2 — measured via <c>--microkernel-gflops</c>).</b>
/// Microbench on a Ryzen 9 3950X (AVX2, no AVX-512): <b>~19 GFLOPS, ~44% of the
/// measured managed FP64 FMA ceiling (~44 GFLOPS at 8 chains)</b>.
/// <b>Correction (was previously mis-reported as "~85–95% of ceiling"):</b> the
/// old reading used a buggy 12-chain calibration that SPILLED and under-measured
/// the ceiling ~2× (FmaCeilingProbe: FP64 hits 46 GFLOPS at 8 chains but craters
/// to ~20 at 10–12 chains as the JIT spills past ~8 YMM accumulators). The kernel
/// is NOT near the ceiling — it has ~2.4× headroom, and OpenBLAS dgemm achieves
/// ~62 GFLOPS on this chip (it even exceeds the managed 8-chain FMA loop, so the
/// codegen ceiling can also be pushed). The gap is <i>kernel quality</i>.
/// </para>
///
/// <para>
/// <b>S.3 fix (prefetch removal) — landed.</b> JIT-disasm showed the hot loop ran
/// <b>3 prefetcht0 + 2 B-loads + 4 A-broadcasts = 9 load-port ops per iter</b>
/// feeding 8 FMAs. On Zen2 (2 loads/cyc) that's 4.5 cyc of load-port work vs 4
/// cyc of FMA, so the cache-resident K-block was <b>load-port bound</b> and the
/// prefetches (data already in L1 at the autotuned Kc) were pure overhead. The
/// loop no longer prefetches AND is explicitly unrolled by 4. Cumulative:
/// <b>~19 (baseline) → ~32 (prefetch removed) → ~35 GFLOPS (explicit 4-way
/// unroll) = 80% of the managed FMA ceiling, ~0.56× OpenBLAS (~62 GFLOPS)</b>.
/// (The hand-emitted machine-code 6×8 kernel later closed this gap to ~57 GFLOPS,
/// ~0.95× OpenBLAS — see MachineCodeFmaKernel.) A large kernel-dominated GEMM
/// (Square_2048 FP64) reached ~1.1× OpenBLAS end-to-end.
/// </para>
/// <para>
/// <b>Why the unroll is written out explicitly (not a <c>for</c> loop):</b>
/// JIT-disasm proved RyuJIT <i>re-rolls</i> a source-level inner unroll loop back
/// into a single-step loop (only 8 FMAs + a branch were emitted), so the
/// scheduler couldn't hoist the next steps' loads. Writing the 4 steps as
/// straight-line code gives it 8 B-loads + 16 broadcasts independent of the
/// accumulators to issue ahead of the 32 FMAs — that is what lifted 32 → 35.
/// </para>
/// <para>
/// <b>Remaining gap → JIT emission (the proper general fix).</b> The residual
/// ~20% to the managed ceiling is load-to-use latency. Pushing further by hand
/// means ever-more-verbose explicit unrolling per shape/precision, and manual
/// software-pipelining spills on RyuJIT (~18 stack spills, measured). The right
/// solution is the deferred Phase J2–J5 JIT emitter (<see cref="JittedKernelCache"/>):
/// emit shape-specialized straight-line unrolled kernels (constant Kc, no bounds
/// checks, no re-rollable loop) for arbitrary shapes — which the explicit-unroll
/// experiment here validates as the path to the ceiling. (<see cref="RunStridedB"/>
/// keeps its prefetch — its B is strided/non-packed, may not be L1-resident.)
/// </para>
/// </summary>
internal static class Avx2Fp64_4x8
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    internal const int Mr = 4;
    /// <summary>The column-tile width of this microkernel (output cols per invocation).</summary>
    internal const int Nr = 8;

#if NET5_0_OR_GREATER
    /// <summary>
    /// Runtime support gate. True when AVX2 and FMA intrinsics are usable on
    /// the current process.
    /// </summary>
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>
    /// Sub-O (#405): K-iteration prefetch distance. FP64 4×8 consumes 32 bytes
    /// of A (4 doubles) and 64 bytes of B (8 doubles) per K-step — one cache
    /// line of B per iter. Prefetching 8 iters ahead pulls in ~4 cache lines
    /// of A and 8 of B, hiding L2→L1 latency on the consume side.
    /// </summary>
    private const int PrefetchDistance = 8;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile, summing over
    /// kc K-steps. C is read-modify-write; caller is responsible for zero-init
    /// if a fresh result is desired. When kc is 0 the kernel reads + writes C
    /// unchanged (no-op accumulation).
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr] row-major.</param>
    /// <param name="c">Output buffer; reads + writes C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static unsafe void Run(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Fp64_4x8 requires Avx2 + Fma.");

        // Load initial C tile into 8 Vector256<double> accumulators.
        // Each row of C is split into 2 vectors: lo = cols 0..3, hi = cols 4..7.
        fixed (double* cPtr = c)
        {
            Vector256<double> acc0_lo = Avx.LoadVector256(cPtr + 0 * ldc + 0);
            Vector256<double> acc0_hi = Avx.LoadVector256(cPtr + 0 * ldc + 4);
            Vector256<double> acc1_lo = Avx.LoadVector256(cPtr + 1 * ldc + 0);
            Vector256<double> acc1_hi = Avx.LoadVector256(cPtr + 1 * ldc + 4);
            Vector256<double> acc2_lo = Avx.LoadVector256(cPtr + 2 * ldc + 0);
            Vector256<double> acc2_hi = Avx.LoadVector256(cPtr + 2 * ldc + 4);
            Vector256<double> acc3_lo = Avx.LoadVector256(cPtr + 3 * ldc + 0);
            Vector256<double> acc3_hi = Avx.LoadVector256(cPtr + 3 * ldc + 4);

            fixed (double* aPtr = packedA)
            fixed (double* bPtr = packedB)
            {
                // Sub-S (#409) S.3: prefetch removed (was load-port-bound for the
                // L1-resident K-block) AND K-loop unrolled by 4. With the loop
                // branch gone for 4 steps at a time, RyuJIT schedules a straight-
                // line block of 8 loads + 16 broadcasts + 32 FMAs — the independent
                // next-step loads issue ahead of the current FMAs, hiding the
                // load-to-use latency that the single-step loop couldn't (the
                // loop-carried branch blocked cross-iteration scheduling). The 8
                // accumulators are the only loop-carried values; per-step operands
                // are short-lived so they don't add register pressure.
                int kBase = 0;
                for (; kBase + 4 <= kc; kBase += 4)
                {
                    // Explicit 4-way unroll (NOT a loop — RyuJIT re-rolls inner
                    // loops). All 8 B-loads + 16 broadcasts are independent of the
                    // accumulators, so the scheduler can issue them ahead of the
                    // 32 FMAs and hide the load-to-use latency.
                    Vector256<double> b0lo = Avx.LoadVector256(bPtr + (kBase + 0) * Nr + 0);
                    Vector256<double> b0hi = Avx.LoadVector256(bPtr + (kBase + 0) * Nr + 4);
                    Vector256<double> b1lo = Avx.LoadVector256(bPtr + (kBase + 1) * Nr + 0);
                    Vector256<double> b1hi = Avx.LoadVector256(bPtr + (kBase + 1) * Nr + 4);
                    Vector256<double> b2lo = Avx.LoadVector256(bPtr + (kBase + 2) * Nr + 0);
                    Vector256<double> b2hi = Avx.LoadVector256(bPtr + (kBase + 2) * Nr + 4);
                    Vector256<double> b3lo = Avx.LoadVector256(bPtr + (kBase + 3) * Nr + 0);
                    Vector256<double> b3hi = Avx.LoadVector256(bPtr + (kBase + 3) * Nr + 4);

                    double* a0p = aPtr + (kBase + 0) * Mr;
                    double* a1p = aPtr + (kBase + 1) * Mr;
                    double* a2p = aPtr + (kBase + 2) * Mr;
                    double* a3p = aPtr + (kBase + 3) * Mr;

                    acc0_lo = Fma.MultiplyAdd(Vector256.Create(a0p[0]), b0lo, acc0_lo);
                    acc0_hi = Fma.MultiplyAdd(Vector256.Create(a0p[0]), b0hi, acc0_hi);
                    acc1_lo = Fma.MultiplyAdd(Vector256.Create(a0p[1]), b0lo, acc1_lo);
                    acc1_hi = Fma.MultiplyAdd(Vector256.Create(a0p[1]), b0hi, acc1_hi);
                    acc2_lo = Fma.MultiplyAdd(Vector256.Create(a0p[2]), b0lo, acc2_lo);
                    acc2_hi = Fma.MultiplyAdd(Vector256.Create(a0p[2]), b0hi, acc2_hi);
                    acc3_lo = Fma.MultiplyAdd(Vector256.Create(a0p[3]), b0lo, acc3_lo);
                    acc3_hi = Fma.MultiplyAdd(Vector256.Create(a0p[3]), b0hi, acc3_hi);

                    acc0_lo = Fma.MultiplyAdd(Vector256.Create(a1p[0]), b1lo, acc0_lo);
                    acc0_hi = Fma.MultiplyAdd(Vector256.Create(a1p[0]), b1hi, acc0_hi);
                    acc1_lo = Fma.MultiplyAdd(Vector256.Create(a1p[1]), b1lo, acc1_lo);
                    acc1_hi = Fma.MultiplyAdd(Vector256.Create(a1p[1]), b1hi, acc1_hi);
                    acc2_lo = Fma.MultiplyAdd(Vector256.Create(a1p[2]), b1lo, acc2_lo);
                    acc2_hi = Fma.MultiplyAdd(Vector256.Create(a1p[2]), b1hi, acc2_hi);
                    acc3_lo = Fma.MultiplyAdd(Vector256.Create(a1p[3]), b1lo, acc3_lo);
                    acc3_hi = Fma.MultiplyAdd(Vector256.Create(a1p[3]), b1hi, acc3_hi);

                    acc0_lo = Fma.MultiplyAdd(Vector256.Create(a2p[0]), b2lo, acc0_lo);
                    acc0_hi = Fma.MultiplyAdd(Vector256.Create(a2p[0]), b2hi, acc0_hi);
                    acc1_lo = Fma.MultiplyAdd(Vector256.Create(a2p[1]), b2lo, acc1_lo);
                    acc1_hi = Fma.MultiplyAdd(Vector256.Create(a2p[1]), b2hi, acc1_hi);
                    acc2_lo = Fma.MultiplyAdd(Vector256.Create(a2p[2]), b2lo, acc2_lo);
                    acc2_hi = Fma.MultiplyAdd(Vector256.Create(a2p[2]), b2hi, acc2_hi);
                    acc3_lo = Fma.MultiplyAdd(Vector256.Create(a2p[3]), b2lo, acc3_lo);
                    acc3_hi = Fma.MultiplyAdd(Vector256.Create(a2p[3]), b2hi, acc3_hi);

                    acc0_lo = Fma.MultiplyAdd(Vector256.Create(a3p[0]), b3lo, acc0_lo);
                    acc0_hi = Fma.MultiplyAdd(Vector256.Create(a3p[0]), b3hi, acc0_hi);
                    acc1_lo = Fma.MultiplyAdd(Vector256.Create(a3p[1]), b3lo, acc1_lo);
                    acc1_hi = Fma.MultiplyAdd(Vector256.Create(a3p[1]), b3hi, acc1_hi);
                    acc2_lo = Fma.MultiplyAdd(Vector256.Create(a3p[2]), b3lo, acc2_lo);
                    acc2_hi = Fma.MultiplyAdd(Vector256.Create(a3p[2]), b3hi, acc2_hi);
                    acc3_lo = Fma.MultiplyAdd(Vector256.Create(a3p[3]), b3lo, acc3_lo);
                    acc3_hi = Fma.MultiplyAdd(Vector256.Create(a3p[3]), b3hi, acc3_hi);
                }
                for (int k = kBase; k < kc; k++)
                {
                    Vector256<double> bLo = Avx.LoadVector256(bPtr + k * Nr + 0);
                    Vector256<double> bHi = Avx.LoadVector256(bPtr + k * Nr + 4);
                    Vector256<double> a0 = Vector256.Create(aPtr[k * Mr + 0]);
                    Vector256<double> a1 = Vector256.Create(aPtr[k * Mr + 1]);
                    Vector256<double> a2 = Vector256.Create(aPtr[k * Mr + 2]);
                    Vector256<double> a3 = Vector256.Create(aPtr[k * Mr + 3]);
                    acc0_lo = Fma.MultiplyAdd(a0, bLo, acc0_lo);
                    acc0_hi = Fma.MultiplyAdd(a0, bHi, acc0_hi);
                    acc1_lo = Fma.MultiplyAdd(a1, bLo, acc1_lo);
                    acc1_hi = Fma.MultiplyAdd(a1, bHi, acc1_hi);
                    acc2_lo = Fma.MultiplyAdd(a2, bLo, acc2_lo);
                    acc2_hi = Fma.MultiplyAdd(a2, bHi, acc2_hi);
                    acc3_lo = Fma.MultiplyAdd(a3, bLo, acc3_lo);
                    acc3_hi = Fma.MultiplyAdd(a3, bHi, acc3_hi);
                }
            }

            // Store accumulators back to C.
            Avx.Store(cPtr + 0 * ldc + 0, acc0_lo);
            Avx.Store(cPtr + 0 * ldc + 4, acc0_hi);
            Avx.Store(cPtr + 1 * ldc + 0, acc1_lo);
            Avx.Store(cPtr + 1 * ldc + 4, acc1_hi);
            Avx.Store(cPtr + 2 * ldc + 0, acc2_lo);
            Avx.Store(cPtr + 2 * ldc + 4, acc2_hi);
            Avx.Store(cPtr + 3 * ldc + 0, acc3_lo);
            Avx.Store(cPtr + 3 * ldc + 4, acc3_hi);
        }
    }

    /// <summary>
    /// Sub-D2 (#372 follow-up) — strided-B variant for <see cref="PackAOnlyStrategy"/>.
    /// Same loop structure as <see cref="Run"/>, but B is read in caller-supplied
    /// stride <paramref name="ldb"/> instead of packed stride <see cref="Nr"/>=8.
    /// </summary>
    public static unsafe void RunStridedB(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> b,
        int ldb,
        Span<double> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Fp64_4x8 requires Avx2 + Fma.");

        fixed (double* cPtr = c)
        {
            Vector256<double> acc0_lo = Avx.LoadVector256(cPtr + 0 * ldc + 0);
            Vector256<double> acc0_hi = Avx.LoadVector256(cPtr + 0 * ldc + 4);
            Vector256<double> acc1_lo = Avx.LoadVector256(cPtr + 1 * ldc + 0);
            Vector256<double> acc1_hi = Avx.LoadVector256(cPtr + 1 * ldc + 4);
            Vector256<double> acc2_lo = Avx.LoadVector256(cPtr + 2 * ldc + 0);
            Vector256<double> acc2_hi = Avx.LoadVector256(cPtr + 2 * ldc + 4);
            Vector256<double> acc3_lo = Avx.LoadVector256(cPtr + 3 * ldc + 0);
            Vector256<double> acc3_hi = Avx.LoadVector256(cPtr + 3 * ldc + 4);

            fixed (double* aPtr = packedA)
            fixed (double* bPtr = b)
            {
                for (int k = 0; k < kc; k++)
                {
                    // Sub-O (#405): prefetch A + strided-B PrefetchDistance iters ahead.
                    if (k + PrefetchDistance < kc)
                    {
                        Sse.Prefetch0(aPtr + (k + PrefetchDistance) * Mr);
                        Sse.Prefetch0(bPtr + (k + PrefetchDistance) * ldb);
                        Sse.Prefetch0(bPtr + (k + PrefetchDistance) * ldb + 4);
                    }

                    // Strided B: row at b[k*ldb], 8 doubles = 2 Vector256 halves.
                    Vector256<double> bRow_lo = Avx.LoadVector256(bPtr + k * ldb + 0);
                    Vector256<double> bRow_hi = Avx.LoadVector256(bPtr + k * ldb + 4);

                    Vector256<double> a0 = Vector256.Create(aPtr[k * Mr + 0]);
                    Vector256<double> a1 = Vector256.Create(aPtr[k * Mr + 1]);
                    Vector256<double> a2 = Vector256.Create(aPtr[k * Mr + 2]);
                    Vector256<double> a3 = Vector256.Create(aPtr[k * Mr + 3]);

                    acc0_lo = Fma.MultiplyAdd(a0, bRow_lo, acc0_lo);
                    acc0_hi = Fma.MultiplyAdd(a0, bRow_hi, acc0_hi);
                    acc1_lo = Fma.MultiplyAdd(a1, bRow_lo, acc1_lo);
                    acc1_hi = Fma.MultiplyAdd(a1, bRow_hi, acc1_hi);
                    acc2_lo = Fma.MultiplyAdd(a2, bRow_lo, acc2_lo);
                    acc2_hi = Fma.MultiplyAdd(a2, bRow_hi, acc2_hi);
                    acc3_lo = Fma.MultiplyAdd(a3, bRow_lo, acc3_lo);
                    acc3_hi = Fma.MultiplyAdd(a3, bRow_hi, acc3_hi);
                }
            }

            Avx.Store(cPtr + 0 * ldc + 0, acc0_lo);
            Avx.Store(cPtr + 0 * ldc + 4, acc0_hi);
            Avx.Store(cPtr + 1 * ldc + 0, acc1_lo);
            Avx.Store(cPtr + 1 * ldc + 4, acc1_hi);
            Avx.Store(cPtr + 2 * ldc + 0, acc2_lo);
            Avx.Store(cPtr + 2 * ldc + 4, acc2_hi);
            Avx.Store(cPtr + 3 * ldc + 0, acc3_lo);
            Avx.Store(cPtr + 3 * ldc + 4, acc3_hi);
        }
    }
#else
    /// <summary>Runtime support gate (false on net471 — no AVX2 intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this.</summary>
    public static void Run(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("Avx2Fp64_4x8 requires net5.0+ for Vector256<T>.");

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this.</summary>
    public static void RunStridedB(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> b,
        int ldb,
        Span<double> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("Avx2Fp64_4x8 requires net5.0+ for Vector256<T>.");
#endif
}
