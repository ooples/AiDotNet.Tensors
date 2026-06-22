using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX2 + FMA FP32 microkernel: 6×16 output tile (6 rows × 16 cols).
///
/// <para>
/// <b>Why 6×16 and not the 8×8 sibling (<see cref="Avx2Fp32_8x8"/>).</b> Sub-S (#409)
/// measured the 8×8 kernel stuck at ~59% of the managed FMA ceiling and proved (via the
/// <c>--microkernel-gflops</c> harness: deeper unroll regressed, prefetch already removed)
/// that it is <i>load-port bound</i>, not latency bound. Its arithmetic intensity is the
/// problem: per K-step it issues 8 FMAs against 1 B-load + 8 A-broadcast-loads ≈ 0.89
/// FMA/load. This 6×16 tile holds each C row in TWO <see cref="Vector256{T}"/> (16 cols),
/// so per K-step it issues 12 FMAs against 2 B-loads + 6 A-broadcast-loads = 1.5 FMA/load —
/// the higher intensity a load-bound kernel needs. Register budget: 12 accumulators + 2 B
/// vectors + 1 transient A broadcast = 15 of 16 YMM (no spill).
/// </para>
///
/// <para>
/// Reads packed-A in [Kc × Mr=6] vpanel layout and packed-B in [Kc × Nr=16] stripe layout.
/// Gated by <c>Avx2.IsSupported &amp;&amp; Fma.IsSupported</c> at the dispatcher. On net471
/// the type compiles but the kernel methods throw (dispatcher gates them off).
/// </para>
/// </summary>
internal static class Avx2Fp32_6x16
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    internal const int Mr = 6;
    /// <summary>The column-tile width of this microkernel (output cols per invocation).</summary>
    internal const int Nr = 16;

    // #653: software-prefetch distance (K-steps ahead) for RunPrefetch. Env-tunable for the
    // prefetch sweep; default 8 (~L2 latency at ~6 cyc/k). Read once at init.
    private static readonly int s_pfDist =
        int.TryParse(System.Environment.GetEnvironmentVariable("AIDOTNET_GEMM_PF_DIST"), out var d) && d > 0 ? d : 8;

#if NET5_0_OR_GREATER
    /// <summary>Runtime support gate. True when AVX2 and FMA intrinsics are usable.</summary>
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile over kc K-steps.
    /// C is read-modify-write; caller zero-inits for a fresh result.
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=6] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr=16] row-major.</param>
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
            throw new PlatformNotSupportedException("Avx2Fp32_6x16 requires Avx2 + Fma.");

        fixed (float* cPtr = c)
        {
            // 12 accumulators: row r occupies (lo = cols 0..8, hi = cols 8..16).
            Vector256<float> c0L = Avx.LoadVector256(cPtr + 0 * ldc), c0H = Avx.LoadVector256(cPtr + 0 * ldc + 8);
            Vector256<float> c1L = Avx.LoadVector256(cPtr + 1 * ldc), c1H = Avx.LoadVector256(cPtr + 1 * ldc + 8);
            Vector256<float> c2L = Avx.LoadVector256(cPtr + 2 * ldc), c2H = Avx.LoadVector256(cPtr + 2 * ldc + 8);
            Vector256<float> c3L = Avx.LoadVector256(cPtr + 3 * ldc), c3H = Avx.LoadVector256(cPtr + 3 * ldc + 8);
            Vector256<float> c4L = Avx.LoadVector256(cPtr + 4 * ldc), c4H = Avx.LoadVector256(cPtr + 4 * ldc + 8);
            Vector256<float> c5L = Avx.LoadVector256(cPtr + 5 * ldc), c5H = Avx.LoadVector256(cPtr + 5 * ldc + 8);

            fixed (float* aPtr = packedA)
            fixed (float* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    Vector256<float> bL = Avx.LoadVector256(bPtr + k * Nr);
                    Vector256<float> bH = Avx.LoadVector256(bPtr + k * Nr + 8);
                    int ao = k * Mr;

                    Vector256<float> a0 = Vector256.Create(aPtr[ao + 0]);
                    c0L = Fma.MultiplyAdd(a0, bL, c0L); c0H = Fma.MultiplyAdd(a0, bH, c0H);
                    Vector256<float> a1 = Vector256.Create(aPtr[ao + 1]);
                    c1L = Fma.MultiplyAdd(a1, bL, c1L); c1H = Fma.MultiplyAdd(a1, bH, c1H);
                    Vector256<float> a2 = Vector256.Create(aPtr[ao + 2]);
                    c2L = Fma.MultiplyAdd(a2, bL, c2L); c2H = Fma.MultiplyAdd(a2, bH, c2H);
                    Vector256<float> a3 = Vector256.Create(aPtr[ao + 3]);
                    c3L = Fma.MultiplyAdd(a3, bL, c3L); c3H = Fma.MultiplyAdd(a3, bH, c3H);
                    Vector256<float> a4 = Vector256.Create(aPtr[ao + 4]);
                    c4L = Fma.MultiplyAdd(a4, bL, c4L); c4H = Fma.MultiplyAdd(a4, bH, c4H);
                    Vector256<float> a5 = Vector256.Create(aPtr[ao + 5]);
                    c5L = Fma.MultiplyAdd(a5, bL, c5L); c5H = Fma.MultiplyAdd(a5, bH, c5H);
                }
            }

            Avx.Store(cPtr + 0 * ldc, c0L); Avx.Store(cPtr + 0 * ldc + 8, c0H);
            Avx.Store(cPtr + 1 * ldc, c1L); Avx.Store(cPtr + 1 * ldc + 8, c1H);
            Avx.Store(cPtr + 2 * ldc, c2L); Avx.Store(cPtr + 2 * ldc + 8, c2H);
            Avx.Store(cPtr + 3 * ldc, c3L); Avx.Store(cPtr + 3 * ldc + 8, c3H);
            Avx.Store(cPtr + 4 * ldc, c4L); Avx.Store(cPtr + 4 * ldc + 8, c4H);
            Avx.Store(cPtr + 5 * ldc, c5L); Avx.Store(cPtr + 5 * ldc + 8, c5H);
        }
    }

    /// <summary>
    /// #653: software-prefetching variant of <see cref="Run"/>. Identical math + FMA order
    /// (bit-identical) — it only adds PREFETCHT0 of the packed-A/B cache lines <c>s_pfDist</c>
    /// K-steps ahead to hide L2/L3 latency, which is the suspected residual per-core gap to
    /// native BLAS at large-N shapes (where packedB streams from L2/L3). Prefetch of an address
    /// past the panel end is fault-safe on x86 (a no-op), so no bounds guard is needed.
    /// </summary>
    public static unsafe void RunPrefetch(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Fp32_6x16 requires Avx2 + Fma.");

        int pfd = s_pfDist;
        fixed (float* cPtr = c)
        {
            Vector256<float> c0L = Avx.LoadVector256(cPtr + 0 * ldc), c0H = Avx.LoadVector256(cPtr + 0 * ldc + 8);
            Vector256<float> c1L = Avx.LoadVector256(cPtr + 1 * ldc), c1H = Avx.LoadVector256(cPtr + 1 * ldc + 8);
            Vector256<float> c2L = Avx.LoadVector256(cPtr + 2 * ldc), c2H = Avx.LoadVector256(cPtr + 2 * ldc + 8);
            Vector256<float> c3L = Avx.LoadVector256(cPtr + 3 * ldc), c3H = Avx.LoadVector256(cPtr + 3 * ldc + 8);
            Vector256<float> c4L = Avx.LoadVector256(cPtr + 4 * ldc), c4H = Avx.LoadVector256(cPtr + 4 * ldc + 8);
            Vector256<float> c5L = Avx.LoadVector256(cPtr + 5 * ldc), c5H = Avx.LoadVector256(cPtr + 5 * ldc + 8);

            fixed (float* aPtr = packedA)
            fixed (float* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    // Prefetch the B stripe (64B/k, one line) + A panel pfd steps ahead into L1.
                    Sse.Prefetch0(bPtr + (k + pfd) * Nr);
                    Sse.Prefetch0(aPtr + (k + pfd) * Mr);

                    Vector256<float> bL = Avx.LoadVector256(bPtr + k * Nr);
                    Vector256<float> bH = Avx.LoadVector256(bPtr + k * Nr + 8);
                    int ao = k * Mr;

                    Vector256<float> a0 = Vector256.Create(aPtr[ao + 0]);
                    c0L = Fma.MultiplyAdd(a0, bL, c0L); c0H = Fma.MultiplyAdd(a0, bH, c0H);
                    Vector256<float> a1 = Vector256.Create(aPtr[ao + 1]);
                    c1L = Fma.MultiplyAdd(a1, bL, c1L); c1H = Fma.MultiplyAdd(a1, bH, c1H);
                    Vector256<float> a2 = Vector256.Create(aPtr[ao + 2]);
                    c2L = Fma.MultiplyAdd(a2, bL, c2L); c2H = Fma.MultiplyAdd(a2, bH, c2H);
                    Vector256<float> a3 = Vector256.Create(aPtr[ao + 3]);
                    c3L = Fma.MultiplyAdd(a3, bL, c3L); c3H = Fma.MultiplyAdd(a3, bH, c3H);
                    Vector256<float> a4 = Vector256.Create(aPtr[ao + 4]);
                    c4L = Fma.MultiplyAdd(a4, bL, c4L); c4H = Fma.MultiplyAdd(a4, bH, c4H);
                    Vector256<float> a5 = Vector256.Create(aPtr[ao + 5]);
                    c5L = Fma.MultiplyAdd(a5, bL, c5L); c5H = Fma.MultiplyAdd(a5, bH, c5H);
                }
            }

            Avx.Store(cPtr + 0 * ldc, c0L); Avx.Store(cPtr + 0 * ldc + 8, c0H);
            Avx.Store(cPtr + 1 * ldc, c1L); Avx.Store(cPtr + 1 * ldc + 8, c1H);
            Avx.Store(cPtr + 2 * ldc, c2L); Avx.Store(cPtr + 2 * ldc + 8, c2H);
            Avx.Store(cPtr + 3 * ldc, c3L); Avx.Store(cPtr + 3 * ldc + 8, c3H);
            Avx.Store(cPtr + 4 * ldc, c4L); Avx.Store(cPtr + 4 * ldc + 8, c4H);
            Avx.Store(cPtr + 5 * ldc, c5L); Avx.Store(cPtr + 5 * ldc + 8, c5H);
        }
    }

    /// <summary>
    /// Strided-B variant for <see cref="PackAOnlyStrategy"/>: B is read at caller stride
    /// <paramref name="ldb"/> instead of the packed stride <see cref="Nr"/>=16.
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=6] row-major.</param>
    /// <param name="b">Source B, read at stride ldb (≥ kc rows of ldb cols).</param>
    /// <param name="ldb">Leading dimension of B (caller's row stride).</param>
    /// <param name="c">Output buffer; reads + writes C[0..Mr, 0..Nr=16] tile.</param>
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
            throw new PlatformNotSupportedException("Avx2Fp32_6x16 requires Avx2 + Fma.");

        fixed (float* cPtr = c)
        {
            Vector256<float> c0L = Avx.LoadVector256(cPtr + 0 * ldc), c0H = Avx.LoadVector256(cPtr + 0 * ldc + 8);
            Vector256<float> c1L = Avx.LoadVector256(cPtr + 1 * ldc), c1H = Avx.LoadVector256(cPtr + 1 * ldc + 8);
            Vector256<float> c2L = Avx.LoadVector256(cPtr + 2 * ldc), c2H = Avx.LoadVector256(cPtr + 2 * ldc + 8);
            Vector256<float> c3L = Avx.LoadVector256(cPtr + 3 * ldc), c3H = Avx.LoadVector256(cPtr + 3 * ldc + 8);
            Vector256<float> c4L = Avx.LoadVector256(cPtr + 4 * ldc), c4H = Avx.LoadVector256(cPtr + 4 * ldc + 8);
            Vector256<float> c5L = Avx.LoadVector256(cPtr + 5 * ldc), c5H = Avx.LoadVector256(cPtr + 5 * ldc + 8);

            fixed (float* aPtr = packedA)
            fixed (float* bPtr = b)
            {
                for (int k = 0; k < kc; k++)
                {
                    Vector256<float> bL = Avx.LoadVector256(bPtr + k * ldb);
                    Vector256<float> bH = Avx.LoadVector256(bPtr + k * ldb + 8);
                    int ao = k * Mr;

                    Vector256<float> a0 = Vector256.Create(aPtr[ao + 0]);
                    c0L = Fma.MultiplyAdd(a0, bL, c0L); c0H = Fma.MultiplyAdd(a0, bH, c0H);
                    Vector256<float> a1 = Vector256.Create(aPtr[ao + 1]);
                    c1L = Fma.MultiplyAdd(a1, bL, c1L); c1H = Fma.MultiplyAdd(a1, bH, c1H);
                    Vector256<float> a2 = Vector256.Create(aPtr[ao + 2]);
                    c2L = Fma.MultiplyAdd(a2, bL, c2L); c2H = Fma.MultiplyAdd(a2, bH, c2H);
                    Vector256<float> a3 = Vector256.Create(aPtr[ao + 3]);
                    c3L = Fma.MultiplyAdd(a3, bL, c3L); c3H = Fma.MultiplyAdd(a3, bH, c3H);
                    Vector256<float> a4 = Vector256.Create(aPtr[ao + 4]);
                    c4L = Fma.MultiplyAdd(a4, bL, c4L); c4H = Fma.MultiplyAdd(a4, bH, c4H);
                    Vector256<float> a5 = Vector256.Create(aPtr[ao + 5]);
                    c5L = Fma.MultiplyAdd(a5, bL, c5L); c5H = Fma.MultiplyAdd(a5, bH, c5H);
                }
            }

            Avx.Store(cPtr + 0 * ldc, c0L); Avx.Store(cPtr + 0 * ldc + 8, c0H);
            Avx.Store(cPtr + 1 * ldc, c1L); Avx.Store(cPtr + 1 * ldc + 8, c1H);
            Avx.Store(cPtr + 2 * ldc, c2L); Avx.Store(cPtr + 2 * ldc + 8, c2H);
            Avx.Store(cPtr + 3 * ldc, c3L); Avx.Store(cPtr + 3 * ldc + 8, c3H);
            Avx.Store(cPtr + 4 * ldc, c4L); Avx.Store(cPtr + 4 * ldc + 8, c4H);
            Avx.Store(cPtr + 5 * ldc, c5L); Avx.Store(cPtr + 5 * ldc + 8, c5H);
        }
    }
#else
    /// <summary>Runtime support gate (false on net471 — no AVX2 intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this.</summary>
    public static void Run(ReadOnlySpan<float> packedA, ReadOnlySpan<float> packedB, Span<float> c, int ldc, int kc) =>
        throw new PlatformNotSupportedException("Avx2Fp32_6x16 requires net5.0+ for Vector256<T>.");

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this.</summary>
    public static void RunStridedB(ReadOnlySpan<float> packedA, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int kc) =>
        throw new PlatformNotSupportedException("Avx2Fp32_6x16 requires net5.0+ for Vector256<T>.");

    /// <summary>Throws on net471 — AVX2 intrinsics unavailable. Dispatcher gates this (IsSupported=false).</summary>
    public static void RunPrefetch(ReadOnlySpan<float> packedA, ReadOnlySpan<float> packedB, Span<float> c, int ldc, int kc) =>
        throw new PlatformNotSupportedException("Avx2Fp32_6x16 requires net5.0+ for Vector256<T>.");
#endif
}
