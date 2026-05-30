using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX-512 FP32 microkernel: 16×16 output tile (16 rows × 16 cols).
/// Reads packed-A in [Kc × Mr=16] vpanel layout and packed-B in [Kc × Nr=16]
/// layout. Each row of C is held in ONE <see cref="Vector512{T}"/> accumulator
/// (Vector512&lt;float&gt; has 16 lanes = Nr exactly), giving 16 total
/// accumulators across the K-loop.
///
/// <para>
/// Uses <see cref="Avx512F.FusedMultiplyAdd"/> for fused multiply-add: 1 FMA
/// per row, 16 FMAs per K-step. The single-vector-per-row layout is more
/// register-efficient than the FP64 lo+hi split because FP32 has twice the
/// lanes per 512-bit register.
/// </para>
///
/// <para>
/// Gated by <see cref="Avx512F.IsSupported"/>. Compiles only on net8.0+
/// (Vector512&lt;T&gt; was added in .NET 8).
/// </para>
/// </summary>
internal static class Avx512Fp32_16x16
{
    /// <summary>Row-tile width (output rows per invocation).</summary>
    internal const int Mr = 16;
    /// <summary>Column-tile width (output cols per invocation).</summary>
    internal const int Nr = 16;

#if NET8_0_OR_GREATER
    /// <summary>Runtime support gate.</summary>
    public static bool IsSupported => Avx512F.IsSupported;

    /// <summary>
    /// Sub-O (#405): software-prefetch lookahead, in K-steps. Each K-step touches
    /// one Mr-wide packed-A vpanel slice and one Nr-wide packed-B stripe; issuing a
    /// <see cref="Sse.Prefetch0"/> this many iterations ahead hides the L2→L1 fetch
    /// latency behind the current step's FMAs (matches the AVX2 kernels). The guard
    /// <c>k + PrefetchDistance &lt; kc</c> avoids prefetching past the packed buffers.
    /// </summary>
    private const int PrefetchDistance = 8;

    /// <summary>
    /// Accumulate packedA · packedB into C[0..Mr, 0..Nr]. C is read-modify-write.
    /// </summary>
    public static unsafe void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Fp32_16x16 requires Avx512F.");

        // 16 Vector512<float> accumulators — one per row, each holding 16 floats = Nr cols.
        fixed (float* cPtr = c)
        {
            Vector512<float> acc0  = Avx512F.LoadVector512(cPtr + 0  * ldc);
            Vector512<float> acc1  = Avx512F.LoadVector512(cPtr + 1  * ldc);
            Vector512<float> acc2  = Avx512F.LoadVector512(cPtr + 2  * ldc);
            Vector512<float> acc3  = Avx512F.LoadVector512(cPtr + 3  * ldc);
            Vector512<float> acc4  = Avx512F.LoadVector512(cPtr + 4  * ldc);
            Vector512<float> acc5  = Avx512F.LoadVector512(cPtr + 5  * ldc);
            Vector512<float> acc6  = Avx512F.LoadVector512(cPtr + 6  * ldc);
            Vector512<float> acc7  = Avx512F.LoadVector512(cPtr + 7  * ldc);
            Vector512<float> acc8  = Avx512F.LoadVector512(cPtr + 8  * ldc);
            Vector512<float> acc9  = Avx512F.LoadVector512(cPtr + 9  * ldc);
            Vector512<float> acc10 = Avx512F.LoadVector512(cPtr + 10 * ldc);
            Vector512<float> acc11 = Avx512F.LoadVector512(cPtr + 11 * ldc);
            Vector512<float> acc12 = Avx512F.LoadVector512(cPtr + 12 * ldc);
            Vector512<float> acc13 = Avx512F.LoadVector512(cPtr + 13 * ldc);
            Vector512<float> acc14 = Avx512F.LoadVector512(cPtr + 14 * ldc);
            Vector512<float> acc15 = Avx512F.LoadVector512(cPtr + 15 * ldc);

            fixed (float* aPtr = packedA)
            fixed (float* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    // Sub-O (#405): prefetch the packed-A vpanel + packed-B stripe
                    // PrefetchDistance K-steps ahead so the next loads land in L1.
                    if (k + PrefetchDistance < kc)
                    {
                        Sse.Prefetch0(aPtr + (k + PrefetchDistance) * Mr);
                        Sse.Prefetch0(bPtr + (k + PrefetchDistance) * Nr);
                    }

                    Vector512<float> bRow = Avx512F.LoadVector512(bPtr + k * Nr);

                    Vector512<float> a0  = Vector512.Create(aPtr[k * Mr + 0]);
                    Vector512<float> a1  = Vector512.Create(aPtr[k * Mr + 1]);
                    Vector512<float> a2  = Vector512.Create(aPtr[k * Mr + 2]);
                    Vector512<float> a3  = Vector512.Create(aPtr[k * Mr + 3]);
                    Vector512<float> a4  = Vector512.Create(aPtr[k * Mr + 4]);
                    Vector512<float> a5  = Vector512.Create(aPtr[k * Mr + 5]);
                    Vector512<float> a6  = Vector512.Create(aPtr[k * Mr + 6]);
                    Vector512<float> a7  = Vector512.Create(aPtr[k * Mr + 7]);
                    Vector512<float> a8  = Vector512.Create(aPtr[k * Mr + 8]);
                    Vector512<float> a9  = Vector512.Create(aPtr[k * Mr + 9]);
                    Vector512<float> a10 = Vector512.Create(aPtr[k * Mr + 10]);
                    Vector512<float> a11 = Vector512.Create(aPtr[k * Mr + 11]);
                    Vector512<float> a12 = Vector512.Create(aPtr[k * Mr + 12]);
                    Vector512<float> a13 = Vector512.Create(aPtr[k * Mr + 13]);
                    Vector512<float> a14 = Vector512.Create(aPtr[k * Mr + 14]);
                    Vector512<float> a15 = Vector512.Create(aPtr[k * Mr + 15]);

                    acc0  = Avx512F.FusedMultiplyAdd(a0,  bRow, acc0);
                    acc1  = Avx512F.FusedMultiplyAdd(a1,  bRow, acc1);
                    acc2  = Avx512F.FusedMultiplyAdd(a2,  bRow, acc2);
                    acc3  = Avx512F.FusedMultiplyAdd(a3,  bRow, acc3);
                    acc4  = Avx512F.FusedMultiplyAdd(a4,  bRow, acc4);
                    acc5  = Avx512F.FusedMultiplyAdd(a5,  bRow, acc5);
                    acc6  = Avx512F.FusedMultiplyAdd(a6,  bRow, acc6);
                    acc7  = Avx512F.FusedMultiplyAdd(a7,  bRow, acc7);
                    acc8  = Avx512F.FusedMultiplyAdd(a8,  bRow, acc8);
                    acc9  = Avx512F.FusedMultiplyAdd(a9,  bRow, acc9);
                    acc10 = Avx512F.FusedMultiplyAdd(a10, bRow, acc10);
                    acc11 = Avx512F.FusedMultiplyAdd(a11, bRow, acc11);
                    acc12 = Avx512F.FusedMultiplyAdd(a12, bRow, acc12);
                    acc13 = Avx512F.FusedMultiplyAdd(a13, bRow, acc13);
                    acc14 = Avx512F.FusedMultiplyAdd(a14, bRow, acc14);
                    acc15 = Avx512F.FusedMultiplyAdd(a15, bRow, acc15);
                }
            }

            Avx512F.Store(cPtr + 0  * ldc, acc0);
            Avx512F.Store(cPtr + 1  * ldc, acc1);
            Avx512F.Store(cPtr + 2  * ldc, acc2);
            Avx512F.Store(cPtr + 3  * ldc, acc3);
            Avx512F.Store(cPtr + 4  * ldc, acc4);
            Avx512F.Store(cPtr + 5  * ldc, acc5);
            Avx512F.Store(cPtr + 6  * ldc, acc6);
            Avx512F.Store(cPtr + 7  * ldc, acc7);
            Avx512F.Store(cPtr + 8  * ldc, acc8);
            Avx512F.Store(cPtr + 9  * ldc, acc9);
            Avx512F.Store(cPtr + 10 * ldc, acc10);
            Avx512F.Store(cPtr + 11 * ldc, acc11);
            Avx512F.Store(cPtr + 12 * ldc, acc12);
            Avx512F.Store(cPtr + 13 * ldc, acc13);
            Avx512F.Store(cPtr + 14 * ldc, acc14);
            Avx512F.Store(cPtr + 15 * ldc, acc15);
        }
    }
#else
    public static bool IsSupported => false;

    public static void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("Avx512Fp32_16x16 requires net8.0+ for Vector512<T>.");
#endif
}
