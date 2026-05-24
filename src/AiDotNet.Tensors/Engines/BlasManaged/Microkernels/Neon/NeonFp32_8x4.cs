using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// ARM64 Neon FP32 microkernel: 8×4 output tile (8 rows × 4 cols).
/// Reads packed-A in [Kc × Mr=8] vpanel layout and packed-B in [Kc × Nr=4]
/// layout. Each row of C is held in ONE <see cref="Vector128{T}"/> accumulator
/// (Vector128&lt;float&gt; has 4 lanes = Nr exactly), giving 8 total
/// accumulators across the K-loop.
///
/// <para>
/// Uses <see cref="AdvSimd.FusedMultiplyAdd"/>: 1 FMA per row, 8 FMAs per
/// K-step. FP32 FMA lives on the base <see cref="AdvSimd"/> class (not
/// <see cref="AdvSimd.Arm64"/>, which is FP64-only). The single-vector-per-row
/// layout is more register-efficient than the FP64 lo+hi split because FP32
/// has twice the lanes per 128-bit register.
/// </para>
///
/// <para>
/// Gated by <see cref="AdvSimd.Arm64.IsSupported"/>. Compiles only on net8.0+
/// (ARM64 intrinsics). Tests on non-ARM64 hosts skip via the
/// <see cref="IsSupported"/> early-return.
/// </para>
/// </summary>
internal static class NeonFp32_8x4
{
    /// <summary>The row-tile width of this microkernel (output rows per invocation).</summary>
    internal const int Mr = 8;
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
    /// <param name="packedA">Packed-A vpanel, layout [Kc × Mr=8] row-major (Mr-contiguous within each k-slice).</param>
    /// <param name="packedB">Packed-B stripe, layout [Kc × Nr=4] row-major.</param>
    /// <param name="c">Output buffer; the kernel reads + writes the C[0..Mr, 0..Nr] tile.</param>
    /// <param name="ldc">Leading dimension of C (cols of the full C matrix, ≥ Nr).</param>
    /// <param name="kc">Number of K-steps to accumulate.</param>
    public static unsafe void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("NeonFp32_8x4 requires ARM64 AdvSimd.");

        fixed (float* cPtr = c)
        {
            // 8 Vector128<float> accumulators — one per row, each holding 4 floats = Nr.
            Vector128<float> acc0 = AdvSimd.LoadVector128(cPtr + 0 * ldc);
            Vector128<float> acc1 = AdvSimd.LoadVector128(cPtr + 1 * ldc);
            Vector128<float> acc2 = AdvSimd.LoadVector128(cPtr + 2 * ldc);
            Vector128<float> acc3 = AdvSimd.LoadVector128(cPtr + 3 * ldc);
            Vector128<float> acc4 = AdvSimd.LoadVector128(cPtr + 4 * ldc);
            Vector128<float> acc5 = AdvSimd.LoadVector128(cPtr + 5 * ldc);
            Vector128<float> acc6 = AdvSimd.LoadVector128(cPtr + 6 * ldc);
            Vector128<float> acc7 = AdvSimd.LoadVector128(cPtr + 7 * ldc);

            fixed (float* aPtr = packedA)
            fixed (float* bPtr = packedB)
            {
                for (int k = 0; k < kc; k++)
                {
                    // 1 vector load from packedB (4 floats = Nr).
                    Vector128<float> bRow = AdvSimd.LoadVector128(bPtr + k * Nr);

                    // Broadcast each of 8 A row scalars into a Vector128<float>.
                    Vector128<float> a0 = Vector128.Create(aPtr[k * Mr + 0]);
                    Vector128<float> a1 = Vector128.Create(aPtr[k * Mr + 1]);
                    Vector128<float> a2 = Vector128.Create(aPtr[k * Mr + 2]);
                    Vector128<float> a3 = Vector128.Create(aPtr[k * Mr + 3]);
                    Vector128<float> a4 = Vector128.Create(aPtr[k * Mr + 4]);
                    Vector128<float> a5 = Vector128.Create(aPtr[k * Mr + 5]);
                    Vector128<float> a6 = Vector128.Create(aPtr[k * Mr + 6]);
                    Vector128<float> a7 = Vector128.Create(aPtr[k * Mr + 7]);

                    // AdvSimd.FusedMultiplyAdd(addend, left, right) = addend + left * right
                    // Note: FP32 FMA lives on AdvSimd (base), not AdvSimd.Arm64 (which is FP64-only).
                    acc0 = AdvSimd.FusedMultiplyAdd(acc0, a0, bRow);
                    acc1 = AdvSimd.FusedMultiplyAdd(acc1, a1, bRow);
                    acc2 = AdvSimd.FusedMultiplyAdd(acc2, a2, bRow);
                    acc3 = AdvSimd.FusedMultiplyAdd(acc3, a3, bRow);
                    acc4 = AdvSimd.FusedMultiplyAdd(acc4, a4, bRow);
                    acc5 = AdvSimd.FusedMultiplyAdd(acc5, a5, bRow);
                    acc6 = AdvSimd.FusedMultiplyAdd(acc6, a6, bRow);
                    acc7 = AdvSimd.FusedMultiplyAdd(acc7, a7, bRow);
                }
            }

            AdvSimd.Store(cPtr + 0 * ldc, acc0);
            AdvSimd.Store(cPtr + 1 * ldc, acc1);
            AdvSimd.Store(cPtr + 2 * ldc, acc2);
            AdvSimd.Store(cPtr + 3 * ldc, acc3);
            AdvSimd.Store(cPtr + 4 * ldc, acc4);
            AdvSimd.Store(cPtr + 5 * ldc, acc5);
            AdvSimd.Store(cPtr + 6 * ldc, acc6);
            AdvSimd.Store(cPtr + 7 * ldc, acc7);
        }
    }
#else
    /// <summary>Runtime support gate. Always false on net471 (ARM64 intrinsics require net8.0+).</summary>
    public static bool IsSupported => false;

    /// <summary>
    /// Not supported on net471. Throws <see cref="PlatformNotSupportedException"/>.
    /// </summary>
    public static void Run(
        ReadOnlySpan<float> packedA,
        ReadOnlySpan<float> packedB,
        Span<float> c,
        int ldc,
        int kc) =>
        throw new PlatformNotSupportedException("NeonFp32_8x4 requires net8.0+ ARM64.");
#endif
}
