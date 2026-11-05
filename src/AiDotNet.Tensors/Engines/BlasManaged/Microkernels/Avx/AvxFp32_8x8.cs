using System;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX (no AVX2, no FMA) FP32 microkernel: 8x8 output tile, the layout of <see cref="Avx2Fp32_8x8"/> with each fused
/// multiply-add split into a multiply and an add.
/// </summary>
/// <remarks>
/// <para>
/// For x64 hosts with AVX but not AVX2, and for any host run with <c>DOTNET_EnableAVX2=0</c>, which turns off AVX2
/// and FMA but leaves AVX's 256-bit float multiply and add. Without this tier those hosts ran every FP32 GEMM on the
/// 4x4 tile at a quarter of the width: VGG-16's AVX2-off training step on four threads was over twice the AVX2
/// step and timed out AiDotNet's Cross-ISA nightly (ooples/AiDotNet#2388).
/// </para>
/// <para>
/// Each lane computes <c>c_ij = c_ij + a_i * b_j</c> in FP32, rounding the product and then the sum, over k in order.
/// That is the arithmetic of the scalar FP32 N-tail in <see cref="PackBothStrategy"/> and of
/// <see cref="AvxStreaming.RunFp32"/>, so an element computes to the same bits whichever of the three it lands in, as
/// deterministic mode requires. An FMA would round once and break that.
/// </para>
/// <para>
/// Broadcasts go through <see cref="Avx.BroadcastScalarToVector256(float*)"/> rather than
/// <see cref="Vector256.Create(float)"/>: with AVX2 off, <see cref="Vector256.IsHardwareAccelerated"/> is false and
/// the general helpers are not guaranteed to lower to the AVX instruction.
/// </para>
/// </remarks>
internal static class AvxFp32_8x8
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    internal const int Mr = 8;
    /// <summary>The column-tile width of this microkernel (output cols per invocation).</summary>
    internal const int Nr = 8;

#if NET5_0_OR_GREATER
    /// <summary>True on hosts with AVX whose AVX2/FMA tier is unavailable; that tier is preferred where present.</summary>
    public static bool IsSupported => Avx.IsSupported && !(Avx2.IsSupported && Fma.IsSupported);

    /// <summary>Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile over <paramref name="kc"/> K-steps.</summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc x Mr] row-major.</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc x Nr] row-major.</param>
    /// <param name="c">Output buffer; the kernel reads and writes the C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C, at least <see cref="Nr"/>.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    [MethodImpl(Hot)]
    public static unsafe void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("AvxFp32_8x8 requires AVX.");

        fixed (float* cPtr = c)
        fixed (float* aPtr = packedA)
        fixed (float* bPtr = packedB)
        {
            Accumulate(aPtr, bPtr, Nr, cPtr, ldc, kc);
        }
    }

    /// <summary>
    /// Variant of <see cref="Run"/> reading B from caller memory with stride <paramref name="ldb"/> (B is [K, N]
    /// row-major, so a k-row's eight columns are contiguous).
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc x Mr] row-major.</param>
    /// <param name="b">B slice positioned at the (pc, jc) corner of the current panel.</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="c">Output buffer; reads and writes the C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    [MethodImpl(Hot)]
    public static unsafe void RunStridedB(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> b,
        int ldb,
        Span<float> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("AvxFp32_8x8 requires AVX.");

        fixed (float* cPtr = c)
        fixed (float* aPtr = packedA)
        fixed (float* bPtr = b)
        {
            Accumulate(aPtr, bPtr, ldb, cPtr, ldc, kc);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Accumulate(float* aPtr, float* bPtr, int ldb, float* cPtr, int ldc, int kc)
    {
        Vector256<float> acc0 = Avx.LoadVector256(cPtr + (0 * ldc));
        Vector256<float> acc1 = Avx.LoadVector256(cPtr + (1 * ldc));
        Vector256<float> acc2 = Avx.LoadVector256(cPtr + (2 * ldc));
        Vector256<float> acc3 = Avx.LoadVector256(cPtr + (3 * ldc));
        Vector256<float> acc4 = Avx.LoadVector256(cPtr + (4 * ldc));
        Vector256<float> acc5 = Avx.LoadVector256(cPtr + (5 * ldc));
        Vector256<float> acc6 = Avx.LoadVector256(cPtr + (6 * ldc));
        Vector256<float> acc7 = Avx.LoadVector256(cPtr + (7 * ldc));

        for (int k = 0; k < kc; k++)
        {
            Vector256<float> bRow = Avx.LoadVector256(bPtr + (k * ldb));
            float* aK = aPtr + (k * Mr);
            // No `a == 0` early-out: 0 * Infinity must give NaN and signed zeros must survive, as in the scalar tail.
            acc0 = Avx.Add(acc0, Avx.Multiply(Avx.BroadcastScalarToVector256(aK + 0), bRow));
            acc1 = Avx.Add(acc1, Avx.Multiply(Avx.BroadcastScalarToVector256(aK + 1), bRow));
            acc2 = Avx.Add(acc2, Avx.Multiply(Avx.BroadcastScalarToVector256(aK + 2), bRow));
            acc3 = Avx.Add(acc3, Avx.Multiply(Avx.BroadcastScalarToVector256(aK + 3), bRow));
            acc4 = Avx.Add(acc4, Avx.Multiply(Avx.BroadcastScalarToVector256(aK + 4), bRow));
            acc5 = Avx.Add(acc5, Avx.Multiply(Avx.BroadcastScalarToVector256(aK + 5), bRow));
            acc6 = Avx.Add(acc6, Avx.Multiply(Avx.BroadcastScalarToVector256(aK + 6), bRow));
            acc7 = Avx.Add(acc7, Avx.Multiply(Avx.BroadcastScalarToVector256(aK + 7), bRow));
        }

        Avx.Store(cPtr + (0 * ldc), acc0);
        Avx.Store(cPtr + (1 * ldc), acc1);
        Avx.Store(cPtr + (2 * ldc), acc2);
        Avx.Store(cPtr + (3 * ldc), acc3);
        Avx.Store(cPtr + (4 * ldc), acc4);
        Avx.Store(cPtr + (5 * ldc), acc5);
        Avx.Store(cPtr + (6 * ldc), acc6);
        Avx.Store(cPtr + (7 * ldc), acc7);
    }
#else
    /// <summary>False on targets without hardware intrinsics.</summary>
    public static bool IsSupported => false;

    /// <summary>Unreachable when <see cref="IsSupported"/> is false.</summary>
    public static void Run(ReadOnlySpan<float> packedA, ReadOnlySpan<float> packedB, Span<float> c, int ldc, int kc)
        => throw new PlatformNotSupportedException("AvxFp32_8x8 requires net5.0+.");

    /// <summary>Unreachable when <see cref="IsSupported"/> is false.</summary>
    public static void RunStridedB(
        ReadOnlySpan<float> packedA, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int kc)
        => throw new PlatformNotSupportedException("AvxFp32_8x8 requires net5.0+.");
#endif
}
