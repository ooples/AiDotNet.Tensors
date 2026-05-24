using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// ARM64 Neon FP64 microkernel: 4×4 output tile (4 rows × 4 cols).
/// Reads packed-A in [Kc × Mr=4] vpanel layout and packed-B in [Kc × Nr=4]
/// layout. Each row of C is held in 2 <see cref="Vector128{T}"/> accumulators
/// (since Vector128&lt;double&gt; has 2 lanes, Nr=4 → 2 vectors per row),
/// giving 8 total accumulators across the K-loop.
///
/// <para>
/// Uses <see cref="AdvSimd.Arm64.FusedMultiplyAdd"/> for fused multiply-add.
/// Note Neon FMA argument order: <c>FusedMultiplyAdd(addend, left, right)</c>
/// returns <c>addend + left * right</c>.
/// </para>
///
/// <para>
/// Gated by <c>AdvSimd.Arm64.IsSupported</c> at the dispatcher. Compiles
/// only on net8.0+ (ARM64 FP64 intrinsics are .NET 8+). Tests on non-ARM64
/// hosts skip via the <see cref="IsSupported"/> early-return.
/// </para>
/// </summary>
internal static class NeonFp64_4x4
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    internal const int Mr = 4;
    /// <summary>The column-tile width of this microkernel (output cols per invocation).</summary>
    internal const int Nr = 4;

#if NET8_0_OR_GREATER
    /// <summary>Runtime support gate. True only on ARM64 with AdvSimd.</summary>
    public static bool IsSupported => AdvSimd.Arm64.IsSupported;

    /// <summary>
    /// Accumulate packedA · packedB into the C[0..Mr, 0..Nr] tile, summing over
    /// kc K-steps. C is read-modify-write; caller is responsible for zero-init
    /// if a fresh result is desired. When kc is 0 the kernel reads + writes C
    /// unchanged (no-op accumulation).
    /// </summary>
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=4] row-major (Mr-contiguous within each k-slice).</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr=4] row-major.</param>
    /// <param name="c">Output buffer; the kernel reads + writes the C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C (cols of the full C matrix, ≥ Nr).</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static unsafe void Run(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("NeonFp64_4x4 requires ARM64 AdvSimd.");

        fixed (double* cPtr = c)
        {
            // 8 Vector128<double> accumulators — 4 rows × 2 lo/hi halves (each half has 2 doubles).
            Vector128<double> acc0_lo = AdvSimd.LoadVector128(cPtr + 0 * ldc + 0);
            Vector128<double> acc0_hi = AdvSimd.LoadVector128(cPtr + 0 * ldc + 2);
            Vector128<double> acc1_lo = AdvSimd.LoadVector128(cPtr + 1 * ldc + 0);
            Vector128<double> acc1_hi = AdvSimd.LoadVector128(cPtr + 1 * ldc + 2);
            Vector128<double> acc2_lo = AdvSimd.LoadVector128(cPtr + 2 * ldc + 0);
            Vector128<double> acc2_hi = AdvSimd.LoadVector128(cPtr + 2 * ldc + 2);
            Vector128<double> acc3_lo = AdvSimd.LoadVector128(cPtr + 3 * ldc + 0);
            Vector128<double> acc3_hi = AdvSimd.LoadVector128(cPtr + 3 * ldc + 2);

            fixed (double* aPtr = packedA)
            fixed (double* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    // 2 vector loads from packedB — lo (cols 0..1) and hi (cols 2..3).
                    Vector128<double> bRow_lo = AdvSimd.LoadVector128(bPtr + k * Nr + 0);
                    Vector128<double> bRow_hi = AdvSimd.LoadVector128(bPtr + k * Nr + 2);

                    // Broadcast each of 4 A row scalars and FMA into row's lo+hi halves.
                    Vector128<double> a0 = Vector128.Create(aPtr[k * Mr + 0]);
                    Vector128<double> a1 = Vector128.Create(aPtr[k * Mr + 1]);
                    Vector128<double> a2 = Vector128.Create(aPtr[k * Mr + 2]);
                    Vector128<double> a3 = Vector128.Create(aPtr[k * Mr + 3]);

                    // AdvSimd.Arm64.FusedMultiplyAdd(addend, left, right) = addend + left * right
                    acc0_lo = AdvSimd.Arm64.FusedMultiplyAdd(acc0_lo, a0, bRow_lo);
                    acc0_hi = AdvSimd.Arm64.FusedMultiplyAdd(acc0_hi, a0, bRow_hi);
                    acc1_lo = AdvSimd.Arm64.FusedMultiplyAdd(acc1_lo, a1, bRow_lo);
                    acc1_hi = AdvSimd.Arm64.FusedMultiplyAdd(acc1_hi, a1, bRow_hi);
                    acc2_lo = AdvSimd.Arm64.FusedMultiplyAdd(acc2_lo, a2, bRow_lo);
                    acc2_hi = AdvSimd.Arm64.FusedMultiplyAdd(acc2_hi, a2, bRow_hi);
                    acc3_lo = AdvSimd.Arm64.FusedMultiplyAdd(acc3_lo, a3, bRow_lo);
                    acc3_hi = AdvSimd.Arm64.FusedMultiplyAdd(acc3_hi, a3, bRow_hi);
                }
            }

            AdvSimd.Store(cPtr + 0 * ldc + 0, acc0_lo);
            AdvSimd.Store(cPtr + 0 * ldc + 2, acc0_hi);
            AdvSimd.Store(cPtr + 1 * ldc + 0, acc1_lo);
            AdvSimd.Store(cPtr + 1 * ldc + 2, acc1_hi);
            AdvSimd.Store(cPtr + 2 * ldc + 0, acc2_lo);
            AdvSimd.Store(cPtr + 2 * ldc + 2, acc2_hi);
            AdvSimd.Store(cPtr + 3 * ldc + 0, acc3_lo);
            AdvSimd.Store(cPtr + 3 * ldc + 2, acc3_hi);
        }
    }
#else
    /// <summary>Runtime support gate. Always false on net471 (ARM64 FP64 intrinsics require net8.0+).</summary>
    public static bool IsSupported => false;

    /// <summary>
    /// Not supported on net471. Throws <see cref="PlatformNotSupportedException"/>.
    /// </summary>
    public static void Run(
        ReadOnlySpan<double> packedA,
        ReadOnlySpan<double> packedB,
        Span<double> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("NeonFp64_4x4 requires net8.0+ ARM64.");
#endif
}
