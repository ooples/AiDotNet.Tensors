using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// 128-bit FP32 4x4 packed microkernel: a SIMD sibling of <see cref="ScalarFp32_4x4"/>, bit-identical to it, for
/// x64 hosts without AVX2 (SSE2 through SSE4.2, or AVX2 switched off with <c>DOTNET_EnableAVX2=0</c>).
/// </summary>
/// <remarks>
/// <para>
/// Without AVX2 every FP32 GEMM fell to the scalar 4x4 kernel. Its 16 double accumulators plus four A and four B
/// operands need 24 registers against x64's 16, so the K-loop spilled: a 512-cube SGEMM ran at 16 GFLOP/s against
/// 511 with AVX2 on the same machine, and VGG-16's 3x3 convolutions at 38-46 GFLOP/s against 206-647. That gap is
/// what timed out VGG on the Cross-ISA nightly's AVX2-off leg (ooples/AiDotNet#2388).
/// </para>
/// <para><b>Why this is bit-exact.</b> The scalar kernel widens each FP32 operand to double, so its product is exact
/// (24 + 24 significand bits fit in 53) and only the add rounds. Here each C row is two
/// <see cref="Vector128{T}"/> of double, <c>[c_i0, c_i1]</c> and <c>[c_i2, c_i3]</c>, and each k-step adds
/// <c>broadcast(a_i) * widen(b)</c>: lane j performs exactly <c>c_ij += (double)a_i * (double)b_j</c>, over the same k
/// values in the same order, with no horizontal reduction. The multiply and the add stay separate; .NET never
/// contracts them into an FMA, which would round once instead of twice. The FP32 rounding happens once, at the
/// write-back, as in the scalar kernel.
/// </para>
/// <para>
/// ADDED ALONGSIDE, NEVER REPLACING, as with <see cref="PortableFp64_4x4"/>: the scalar kernel stays the reference
/// and the net471 path. Because the tile, the packing and the accumulation are unchanged, the M/N tails, the
/// streaming kernel and the deterministic-mode contract see the same bits as before.
/// </para>
/// </remarks>
internal static class PortableFp32_4x4
{
    /// <summary>Microkernel row-tile width, matching <see cref="ScalarFp32_4x4"/>.</summary>
    public const int Mr = 4;
    /// <summary>Microkernel column-tile width, matching <see cref="ScalarFp32_4x4"/>.</summary>
    public const int Nr = 4;

#if NET8_0_OR_GREATER
    /// <summary>True when 128-bit vectors are hardware-accelerated (every x64 and ARM64 host).</summary>
    public static bool IsSupported => Vector128.IsHardwareAccelerated;

    /// <summary>
    /// Compute <c>C[0..Mr, 0..Nr] += packedA · packedB</c>. Bit-identical to <see cref="ScalarFp32_4x4.Run"/>.
    /// </summary>
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
        fixed (float* aBase = &MemoryMarshal.GetReference(packedA))
        fixed (float* bBase = &MemoryMarshal.GetReference(packedB))
        fixed (float* cBase = &MemoryMarshal.GetReference(c))
        {
            Load(cBase, ldc, out var c0l, out var c0h, out var c1l, out var c1h,
                out var c2l, out var c2h, out var c3l, out var c3h);
            for (int k = 0; k < kc; k++)
            {
                var b = Vector128.Load(bBase + (k * Nr));
                Step(aBase + (k * Mr), Vector128.WidenLower(b), Vector128.WidenUpper(b),
                    ref c0l, ref c0h, ref c1l, ref c1h, ref c2l, ref c2h, ref c3l, ref c3h);
            }
            Store(cBase, ldc, c0l, c0h, c1l, c1h, c2l, c2h, c3l, c3h);
        }
    }

    /// <summary>
    /// Variant of <see cref="Run"/> reading B from caller memory with stride <paramref name="ldb"/>. Bit-identical to
    /// <see cref="ScalarFp32_4x4.RunStridedB"/>. B is [K, N] row-major, so a k-row's four columns are contiguous.
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
        fixed (float* aBase = &MemoryMarshal.GetReference(packedA))
        fixed (float* bBase = &MemoryMarshal.GetReference(b))
        fixed (float* cBase = &MemoryMarshal.GetReference(c))
        {
            Load(cBase, ldc, out var c0l, out var c0h, out var c1l, out var c1h,
                out var c2l, out var c2h, out var c3l, out var c3h);
            for (int k = 0; k < kc; k++)
            {
                var bRow = Vector128.Load(bBase + (k * ldb));
                Step(aBase + (k * Mr), Vector128.WidenLower(bRow), Vector128.WidenUpper(bRow),
                    ref c0l, ref c0h, ref c1l, ref c1h, ref c2l, ref c2h, ref c3l, ref c3h);
            }
            Store(cBase, ldc, c0l, c0h, c1l, c1h, c2l, c2h, c3l, c3h);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Step(float* aK, Vector128<double> bLo, Vector128<double> bHi,
        ref Vector128<double> c0l, ref Vector128<double> c0h, ref Vector128<double> c1l, ref Vector128<double> c1h,
        ref Vector128<double> c2l, ref Vector128<double> c2h, ref Vector128<double> c3l, ref Vector128<double> c3h)
    {
        // No `a == 0` early-out: the scalar kernel propagates 0 * Infinity => NaN and signed zeros, and so must this.
        var a0 = Vector128.Create((double)aK[0]);
        var a1 = Vector128.Create((double)aK[1]);
        var a2 = Vector128.Create((double)aK[2]);
        var a3 = Vector128.Create((double)aK[3]);
        c0l += a0 * bLo; c0h += a0 * bHi;
        c1l += a1 * bLo; c1h += a1 * bHi;
        c2l += a2 * bLo; c2h += a2 * bHi;
        c3l += a3 * bLo; c3h += a3 * bHi;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Load(float* c, int ldc,
        out Vector128<double> c0l, out Vector128<double> c0h, out Vector128<double> c1l, out Vector128<double> c1h,
        out Vector128<double> c2l, out Vector128<double> c2h, out Vector128<double> c3l, out Vector128<double> c3h)
    {
        var r0 = Vector128.Load(c);
        var r1 = Vector128.Load(c + ldc);
        var r2 = Vector128.Load(c + (2 * ldc));
        var r3 = Vector128.Load(c + (3 * ldc));
        c0l = Vector128.WidenLower(r0); c0h = Vector128.WidenUpper(r0);
        c1l = Vector128.WidenLower(r1); c1h = Vector128.WidenUpper(r1);
        c2l = Vector128.WidenLower(r2); c2h = Vector128.WidenUpper(r2);
        c3l = Vector128.WidenLower(r3); c3h = Vector128.WidenUpper(r3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(float* c, int ldc,
        Vector128<double> c0l, Vector128<double> c0h, Vector128<double> c1l, Vector128<double> c1h,
        Vector128<double> c2l, Vector128<double> c2h, Vector128<double> c3l, Vector128<double> c3h)
    {
        // Narrow rounds each double to the nearest float, as the scalar kernel's (float) cast does.
        Vector128.Narrow(c0l, c0h).Store(c);
        Vector128.Narrow(c1l, c1h).Store(c + ldc);
        Vector128.Narrow(c2l, c2h).Store(c + (2 * ldc));
        Vector128.Narrow(c3l, c3h).Store(c + (3 * ldc));
    }
#else
    /// <summary>False on targets without <c>System.Runtime.Intrinsics</c>; the scalar kernel runs there.</summary>
    public static bool IsSupported => false;

    /// <summary>Unreachable when <see cref="IsSupported"/> is false.</summary>
    public static void Run(ReadOnlySpan<float> packedA, ReadOnlySpan<float> packedB, Span<float> c, int ldc, int kc)
        => throw new PlatformNotSupportedException();

    /// <summary>Unreachable when <see cref="IsSupported"/> is false.</summary>
    public static void RunStridedB(
        ReadOnlySpan<float> packedA, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int kc)
        => throw new PlatformNotSupportedException();
#endif
}
