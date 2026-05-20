using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX-512 FP64 microkernel: 8×16 output tile (8 rows × 16 cols).
/// Reads packed-A in [Kc × Mr=8] vpanel layout and packed-B in [Kc × Nr=16]
/// layout. Each row of C is held in 2 <see cref="Vector512{T}"/> accumulators
/// (lo + hi halves, since Vector512&lt;double&gt; has 8 lanes = Nr/2), giving
/// 16 total accumulators across the K-loop.
///
/// <para>
/// Uses <see cref="Avx512F.FusedMultiplyAdd"/> for fused multiply-add: 16 FMAs
/// per K-step. Inner loop is FMA-port bound on Skylake-X+ / Zen 4+ silicon at
/// ~256 GFLOPS per core.
/// </para>
///
/// <para>
/// <b>This is the microkernel that closes issue #358's L2-shape pathology.</b>
/// At M=4096, N=16, K=512, transA=true: a single call processes a 8×16 C tile
/// from 8 packed-A vpanel rows + the full 16-col packed-B stripe, performing
/// 8 · 16 · 512 · 2 = 131 K FLOPs per tile. With 4096/8 = 512 stripes, total
/// work is 67M FLOPs — completes in ≈0.5ms at AVX-512 peak.
/// </para>
///
/// <para>
/// Gated by <c>Avx512F.IsSupported</c> at the dispatcher. Compiles only on
/// net8.0+ (Vector512&lt;T&gt; was added in .NET 8). On net471 the type still
/// compiles but the kernel methods are not reachable.
/// </para>
/// </summary>
internal static class Avx512Fp64_8x16
{
    /// <summary>Row-tile width (output rows per invocation).</summary>
    internal const int Mr = 8;
    /// <summary>Column-tile width (output cols per invocation).</summary>
    internal const int Nr = 16;

#if NET8_0_OR_GREATER
    /// <summary>
    /// Runtime support gate. True when AVX-512F intrinsics are usable on the
    /// current process.
    /// </summary>
    public static bool IsSupported => Avx512F.IsSupported;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile, summing over
    /// kc K-steps. C is read-modify-write; caller is responsible for zero-init
    /// if a fresh result is desired. When kc is 0 the kernel reads + writes C
    /// unchanged (no-op accumulation).
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=8] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr=16] row-major.</param>
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
            throw new PlatformNotSupportedException("Avx512Fp64_8x16 requires Avx512F.");

        // 16 Vector512<double> accumulators (8 rows × 2 halves).
        fixed (double* cPtr = c)
        {
            Vector512<double> acc0_lo = Avx512F.LoadVector512(cPtr + 0 * ldc + 0);
            Vector512<double> acc0_hi = Avx512F.LoadVector512(cPtr + 0 * ldc + 8);
            Vector512<double> acc1_lo = Avx512F.LoadVector512(cPtr + 1 * ldc + 0);
            Vector512<double> acc1_hi = Avx512F.LoadVector512(cPtr + 1 * ldc + 8);
            Vector512<double> acc2_lo = Avx512F.LoadVector512(cPtr + 2 * ldc + 0);
            Vector512<double> acc2_hi = Avx512F.LoadVector512(cPtr + 2 * ldc + 8);
            Vector512<double> acc3_lo = Avx512F.LoadVector512(cPtr + 3 * ldc + 0);
            Vector512<double> acc3_hi = Avx512F.LoadVector512(cPtr + 3 * ldc + 8);
            Vector512<double> acc4_lo = Avx512F.LoadVector512(cPtr + 4 * ldc + 0);
            Vector512<double> acc4_hi = Avx512F.LoadVector512(cPtr + 4 * ldc + 8);
            Vector512<double> acc5_lo = Avx512F.LoadVector512(cPtr + 5 * ldc + 0);
            Vector512<double> acc5_hi = Avx512F.LoadVector512(cPtr + 5 * ldc + 8);
            Vector512<double> acc6_lo = Avx512F.LoadVector512(cPtr + 6 * ldc + 0);
            Vector512<double> acc6_hi = Avx512F.LoadVector512(cPtr + 6 * ldc + 8);
            Vector512<double> acc7_lo = Avx512F.LoadVector512(cPtr + 7 * ldc + 0);
            Vector512<double> acc7_hi = Avx512F.LoadVector512(cPtr + 7 * ldc + 8);

            fixed (double* aPtr = packedA)
            fixed (double* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    // Two vector loads per K-step from packed-B.
                    Vector512<double> bRow_lo = Avx512F.LoadVector512(bPtr + k * Nr + 0);
                    Vector512<double> bRow_hi = Avx512F.LoadVector512(bPtr + k * Nr + 8);

                    // Broadcast each of 8 packed-A row scalars and FMA into lo+hi halves.
                    Vector512<double> a0 = Vector512.Create(aPtr[k * Mr + 0]);
                    Vector512<double> a1 = Vector512.Create(aPtr[k * Mr + 1]);
                    Vector512<double> a2 = Vector512.Create(aPtr[k * Mr + 2]);
                    Vector512<double> a3 = Vector512.Create(aPtr[k * Mr + 3]);
                    Vector512<double> a4 = Vector512.Create(aPtr[k * Mr + 4]);
                    Vector512<double> a5 = Vector512.Create(aPtr[k * Mr + 5]);
                    Vector512<double> a6 = Vector512.Create(aPtr[k * Mr + 6]);
                    Vector512<double> a7 = Vector512.Create(aPtr[k * Mr + 7]);

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
            Avx512F.Store(cPtr + 0 * ldc + 8, acc0_hi);
            Avx512F.Store(cPtr + 1 * ldc + 0, acc1_lo);
            Avx512F.Store(cPtr + 1 * ldc + 8, acc1_hi);
            Avx512F.Store(cPtr + 2 * ldc + 0, acc2_lo);
            Avx512F.Store(cPtr + 2 * ldc + 8, acc2_hi);
            Avx512F.Store(cPtr + 3 * ldc + 0, acc3_lo);
            Avx512F.Store(cPtr + 3 * ldc + 8, acc3_hi);
            Avx512F.Store(cPtr + 4 * ldc + 0, acc4_lo);
            Avx512F.Store(cPtr + 4 * ldc + 8, acc4_hi);
            Avx512F.Store(cPtr + 5 * ldc + 0, acc5_lo);
            Avx512F.Store(cPtr + 5 * ldc + 8, acc5_hi);
            Avx512F.Store(cPtr + 6 * ldc + 0, acc6_lo);
            Avx512F.Store(cPtr + 6 * ldc + 8, acc6_hi);
            Avx512F.Store(cPtr + 7 * ldc + 0, acc7_lo);
            Avx512F.Store(cPtr + 7 * ldc + 8, acc7_hi);
        }
    }
#else
    /// <summary>Runtime support gate (false on net471 — no Vector512 intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>Throws on net471 — Vector512 unavailable. Dispatcher gates this.</summary>
    public static void Run(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("Avx512Fp64_8x16 requires net8.0+ for Vector512<T>.");
#endif
}
