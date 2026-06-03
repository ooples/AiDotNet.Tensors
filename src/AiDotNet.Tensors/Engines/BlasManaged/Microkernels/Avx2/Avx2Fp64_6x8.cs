using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX2 + FMA FP64 microkernel: 6×8 output tile (6 rows × 8 cols).
///
/// <para>
/// <b>Why 6×8 and not the 4×8 sibling (<see cref="Avx2Fp64_4x8"/>).</b> Sub-S (#409)
/// Phase S.4: the <c>--microkernel-gflops</c> harness measured the 4×8 kernel at ~79%
/// of the managed FMA ceiling — better than the FP32 8×8 (load-port bound) thanks to its
/// unroll-4 load-hiding, but still short of the 89% the FP32 6×16 tile reaches. The cause
/// is the same arithmetic-intensity ceiling: the 4×8 issues 8 FMAs against 2 B-loads +
/// 4 A-broadcast-loads ≈ 1.33 FMA/load. This 6×8 tile is the FP64 analog of the FP32
/// 6×16 win — it holds 6 C rows (each in two <see cref="Vector256{T}"/> of 4 doubles =
/// 8 cols), so per K-step it issues 12 FMAs against 2 B-loads + 6 A-broadcast-loads =
/// 1.5 FMA/load, the higher intensity that lifts the load-bound kernel. Register budget:
/// 12 accumulators + 2 B vectors + 1 transient A broadcast = 15 of 16 YMM (no spill), so
/// the plain K-loop lets RyuJIT issue the next step's independent loads ahead of the
/// current FMAs without an explicit unroll.
/// </para>
///
/// <para>
/// Reads packed-A in [Kc × Mr=6] vpanel layout and packed-B in [Kc × Nr=8] stripe layout
/// (the same Nr=8 stripe the 4×8 kernel consumes — only the row tile widens, so packed-B
/// is unchanged). The 6×8 reduction order differs from the 4×8's unroll-4 order, so it is
/// gated to non-deterministic (Fast) mode at the dispatcher; deterministic mode and the
/// pre-pack path keep the 4×8 tile whose layout/order they are validated against. On
/// net471 the type compiles but the kernel methods throw (dispatcher gates them off).
/// </para>
/// </summary>
internal static class Avx2Fp64_6x8
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    internal const int Mr = 6;
    /// <summary>The column-tile width of this microkernel (output cols per invocation).</summary>
    internal const int Nr = 8;

#if NET5_0_OR_GREATER
    /// <summary>Runtime support gate. True when AVX2 and FMA intrinsics are usable.</summary>
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile over kc K-steps.
    /// C is read-modify-write; caller zero-inits for a fresh result.
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=6] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr=8] row-major.</param>
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
            throw new PlatformNotSupportedException("Avx2Fp64_6x8 requires Avx2 + Fma.");

        fixed (double* cPtr = c)
        {
            // 12 accumulators: row r occupies (lo = cols 0..3, hi = cols 4..7).
            Vector256<double> c0L = Avx.LoadVector256(cPtr + 0 * ldc), c0H = Avx.LoadVector256(cPtr + 0 * ldc + 4);
            Vector256<double> c1L = Avx.LoadVector256(cPtr + 1 * ldc), c1H = Avx.LoadVector256(cPtr + 1 * ldc + 4);
            Vector256<double> c2L = Avx.LoadVector256(cPtr + 2 * ldc), c2H = Avx.LoadVector256(cPtr + 2 * ldc + 4);
            Vector256<double> c3L = Avx.LoadVector256(cPtr + 3 * ldc), c3H = Avx.LoadVector256(cPtr + 3 * ldc + 4);
            Vector256<double> c4L = Avx.LoadVector256(cPtr + 4 * ldc), c4H = Avx.LoadVector256(cPtr + 4 * ldc + 4);
            Vector256<double> c5L = Avx.LoadVector256(cPtr + 5 * ldc), c5H = Avx.LoadVector256(cPtr + 5 * ldc + 4);

            fixed (double* aPtr = packedA)
            fixed (double* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    Vector256<double> bL = Avx.LoadVector256(bPtr + k * Nr);
                    Vector256<double> bH = Avx.LoadVector256(bPtr + k * Nr + 4);
                    int ao = k * Mr;

                    Vector256<double> a0 = Vector256.Create(aPtr[ao + 0]);
                    c0L = Fma.MultiplyAdd(a0, bL, c0L); c0H = Fma.MultiplyAdd(a0, bH, c0H);
                    Vector256<double> a1 = Vector256.Create(aPtr[ao + 1]);
                    c1L = Fma.MultiplyAdd(a1, bL, c1L); c1H = Fma.MultiplyAdd(a1, bH, c1H);
                    Vector256<double> a2 = Vector256.Create(aPtr[ao + 2]);
                    c2L = Fma.MultiplyAdd(a2, bL, c2L); c2H = Fma.MultiplyAdd(a2, bH, c2H);
                    Vector256<double> a3 = Vector256.Create(aPtr[ao + 3]);
                    c3L = Fma.MultiplyAdd(a3, bL, c3L); c3H = Fma.MultiplyAdd(a3, bH, c3H);
                    Vector256<double> a4 = Vector256.Create(aPtr[ao + 4]);
                    c4L = Fma.MultiplyAdd(a4, bL, c4L); c4H = Fma.MultiplyAdd(a4, bH, c4H);
                    Vector256<double> a5 = Vector256.Create(aPtr[ao + 5]);
                    c5L = Fma.MultiplyAdd(a5, bL, c5L); c5H = Fma.MultiplyAdd(a5, bH, c5H);
                }
            }

            Avx.Store(cPtr + 0 * ldc, c0L); Avx.Store(cPtr + 0 * ldc + 4, c0H);
            Avx.Store(cPtr + 1 * ldc, c1L); Avx.Store(cPtr + 1 * ldc + 4, c1H);
            Avx.Store(cPtr + 2 * ldc, c2L); Avx.Store(cPtr + 2 * ldc + 4, c2H);
            Avx.Store(cPtr + 3 * ldc, c3L); Avx.Store(cPtr + 3 * ldc + 4, c3H);
            Avx.Store(cPtr + 4 * ldc, c4L); Avx.Store(cPtr + 4 * ldc + 4, c4H);
            Avx.Store(cPtr + 5 * ldc, c5L); Avx.Store(cPtr + 5 * ldc + 4, c5H);
        }
    }

    /// <summary>
    /// Strided-B variant for <see cref="PackAOnlyStrategy"/>: B is read at caller stride
    /// <paramref name="ldb"/> instead of the packed stride <see cref="Nr"/>=8.
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=6] row-major.</param>
    /// <param name="b">Source B, read at stride ldb (≥ kc rows of ldb cols).</param>
    /// <param name="ldb">Leading dimension of B (caller's row stride).</param>
    /// <param name="c">Output buffer; reads + writes C[0..Mr, 0..Nr=8] tile.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static unsafe void RunStridedB(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> b,
        int ldb,
        Span<double> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Fp64_6x8 requires Avx2 + Fma.");

        fixed (double* cPtr = c)
        {
            Vector256<double> c0L = Avx.LoadVector256(cPtr + 0 * ldc), c0H = Avx.LoadVector256(cPtr + 0 * ldc + 4);
            Vector256<double> c1L = Avx.LoadVector256(cPtr + 1 * ldc), c1H = Avx.LoadVector256(cPtr + 1 * ldc + 4);
            Vector256<double> c2L = Avx.LoadVector256(cPtr + 2 * ldc), c2H = Avx.LoadVector256(cPtr + 2 * ldc + 4);
            Vector256<double> c3L = Avx.LoadVector256(cPtr + 3 * ldc), c3H = Avx.LoadVector256(cPtr + 3 * ldc + 4);
            Vector256<double> c4L = Avx.LoadVector256(cPtr + 4 * ldc), c4H = Avx.LoadVector256(cPtr + 4 * ldc + 4);
            Vector256<double> c5L = Avx.LoadVector256(cPtr + 5 * ldc), c5H = Avx.LoadVector256(cPtr + 5 * ldc + 4);

            fixed (double* aPtr = packedA)
            fixed (double* bPtr = b)
            {
                for (int k = 0; k < kc; k++)
                {
                    Vector256<double> bL = Avx.LoadVector256(bPtr + k * ldb);
                    Vector256<double> bH = Avx.LoadVector256(bPtr + k * ldb + 4);
                    int ao = k * Mr;

                    Vector256<double> a0 = Vector256.Create(aPtr[ao + 0]);
                    c0L = Fma.MultiplyAdd(a0, bL, c0L); c0H = Fma.MultiplyAdd(a0, bH, c0H);
                    Vector256<double> a1 = Vector256.Create(aPtr[ao + 1]);
                    c1L = Fma.MultiplyAdd(a1, bL, c1L); c1H = Fma.MultiplyAdd(a1, bH, c1H);
                    Vector256<double> a2 = Vector256.Create(aPtr[ao + 2]);
                    c2L = Fma.MultiplyAdd(a2, bL, c2L); c2H = Fma.MultiplyAdd(a2, bH, c2H);
                    Vector256<double> a3 = Vector256.Create(aPtr[ao + 3]);
                    c3L = Fma.MultiplyAdd(a3, bL, c3L); c3H = Fma.MultiplyAdd(a3, bH, c3H);
                    Vector256<double> a4 = Vector256.Create(aPtr[ao + 4]);
                    c4L = Fma.MultiplyAdd(a4, bL, c4L); c4H = Fma.MultiplyAdd(a4, bH, c4H);
                    Vector256<double> a5 = Vector256.Create(aPtr[ao + 5]);
                    c5L = Fma.MultiplyAdd(a5, bL, c5L); c5H = Fma.MultiplyAdd(a5, bH, c5H);
                }
            }

            Avx.Store(cPtr + 0 * ldc, c0L); Avx.Store(cPtr + 0 * ldc + 4, c0H);
            Avx.Store(cPtr + 1 * ldc, c1L); Avx.Store(cPtr + 1 * ldc + 4, c1H);
            Avx.Store(cPtr + 2 * ldc, c2L); Avx.Store(cPtr + 2 * ldc + 4, c2H);
            Avx.Store(cPtr + 3 * ldc, c3L); Avx.Store(cPtr + 3 * ldc + 4, c3H);
            Avx.Store(cPtr + 4 * ldc, c4L); Avx.Store(cPtr + 4 * ldc + 4, c4H);
            Avx.Store(cPtr + 5 * ldc, c5L); Avx.Store(cPtr + 5 * ldc + 4, c5H);
        }
    }
#else
    /// <summary>Runtime support gate (false on net471 — no AVX2 intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this.</summary>
    public static void Run(
        ReadOnlySpan<double> packedA, ReadOnlySpan<double> packedB, Span<double> c, int ldc, int kc)
        => throw new PlatformNotSupportedException("Avx2Fp64_6x8 requires AVX2 intrinsics (net5.0+).");

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this.</summary>
    public static void RunStridedB(
        ReadOnlySpan<double> packedA, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc, int kc)
        => throw new PlatformNotSupportedException("Avx2Fp64_6x8 requires AVX2 intrinsics (net5.0+).");
#endif
}
