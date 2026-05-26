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
/// loop above no longer prefetches: this lifted the kernel <b>~19 → ~32 GFLOPS
/// (44% → 72% of the managed FMA ceiling)</b>, and a large kernel-dominated GEMM
/// (Square_2048 FP64) reached <b>~1.1× OpenBLAS end-to-end</b> with no regression
/// on the large-Kc stress shape.
/// </para>
/// <para>
/// <b>Remaining gap (open).</b> The residual ~28% to the managed ceiling is
/// load-to-use latency (loads feed FMAs in the same iter). Software-pipelining
/// (preload next-iter B) would hide it but REGRESSED to ~15 GFLOPS — disasm
/// showed RyuJIT emits <b>18 stack spills</b>, refusing to sustain 8 accumulators
/// + double-buffered operands in 16 YMM. So that last step is bounded by RyuJIT
/// register allocation and needs an allocator-friendly shape or JIT-level work.
/// (<see cref="RunStridedB"/> keeps its prefetch — its B is the caller's strided,
/// non-packed matrix, which may not be L1-resident; evaluate separately.)
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
                for (int k = 0; k < kc; k++)
                {
                    // Sub-S (#409) S.3 experiment: prefetch removed. The JIT disasm
                    // showed 3 prefetcht0 + 2 B-loads + 4 A-broadcasts = 9 load-port
                    // ops per iter feeding only 8 FMAs — on Zen2 (2 loads/cyc) that's
                    // 4.5 cyc of load-port work vs 4 cyc of FMA, so the cache-resident
                    // K-block is load-port bound and the prefetches (data already in
                    // L1 at the autotuned Kc) are pure overhead stealing AGU slots.

                    // Load B row: 8 doubles = 2 Vector256 halves.
                    Vector256<double> bRow_lo = Avx.LoadVector256(bPtr + k * Nr + 0);
                    Vector256<double> bRow_hi = Avx.LoadVector256(bPtr + k * Nr + 4);

                    // Broadcast each A row scalar to a full vector, then FMA.
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
