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
