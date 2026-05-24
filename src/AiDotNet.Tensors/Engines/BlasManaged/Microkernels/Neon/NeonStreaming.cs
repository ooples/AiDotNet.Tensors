using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// ARM64 Neon streaming microkernel for FP64 and FP32. Reads A and B directly
/// in caller-supplied strides — no pack step. Used by the Neon streaming
/// strategy when K is small (typically &lt; 32).
///
/// <para>
/// SIMD-accelerates NN and TN cases (B contiguous along N). For NT and TT
/// (B strided), falls back to <see cref="ScalarStreaming"/>.
/// </para>
///
/// <para>
/// FMA API note: FP32 uses base <see cref="AdvSimd.FusedMultiplyAdd"/>; FP64
/// uses <see cref="AdvSimd.Arm64.FusedMultiplyAdd"/>. Both have semantics
/// <c>FusedMultiplyAdd(addend, left, right) = addend + left * right</c>.
/// </para>
/// </summary>
internal static class NeonStreaming
{
#if NET8_0_OR_GREATER
    /// <summary>Runtime support gate. True when ARM64 AdvSimd intrinsics are usable on the current process.</summary>
    public static bool IsSupported => AdvSimd.Arm64.IsSupported;

    /// <summary>
    /// Compute C += op(A) · op(B) directly without packing. C is read-modify-write.
    /// FP64: 2-wide Vector128&lt;double&gt; blocks with scalar tail for n % 2.
    /// transB=true falls back to <see cref="ScalarStreaming.RunFp64"/>.
    /// </summary>
    public static unsafe void RunFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("NeonStreaming requires ARM64 AdvSimd.");

        // transB=true: B is strided across N rows. Fall back to scalar — gather
        // would require non-contiguous loads that cancel the SIMD benefit.
        if (transB)
        {
            ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            return;
        }

        // NN and TN: B is contiguous along N. Process the C output in 2-wide
        // column blocks. For each output row i and each col-block jb (2 cols),
        // accumulate over k using Vector128<double> loads from B.
        int nBlocks = n / 2;

        fixed (double* aPtr = a)
        fixed (double* bPtr = b)
        fixed (double* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                // Process 2-col blocks via SIMD.
                for (int jb = 0; jb < nBlocks; jb++)
                {
                    int jStart = jb * 2;
                    Vector128<double> acc = AdvSimd.LoadVector128(cPtr + i * ldc + jStart);
                    for (int kk = 0; kk < k; kk++)
                    {
                        double aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector128<double> aVec = Vector128.Create(aval);
                        Vector128<double> bVec = AdvSimd.LoadVector128(bPtr + kk * ldb + jStart);
                        acc = AdvSimd.Arm64.FusedMultiplyAdd(acc, aVec, bVec);
                    }
                    AdvSimd.Store(cPtr + i * ldc + jStart, acc);
                }
                // Scalar tail for n % 2.
                for (int j = nBlocks * 2; j < n; j++)
                {
                    double sum = cPtr[i * ldc + j];
                    for (int kk = 0; kk < k; kk++)
                    {
                        double aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        sum += aval * bPtr[kk * ldb + j];
                    }
                    cPtr[i * ldc + j] = sum;
                }
            }
        }
    }

    /// <summary>
    /// FP32 mirror of <see cref="RunFp64"/>. Uses Vector128&lt;float&gt; (4 lanes)
    /// and processes 4-col blocks per inner loop iteration.
    /// transB=true falls back to <see cref="ScalarStreaming.RunFp32"/>.
    /// </summary>
    public static unsafe void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("NeonStreaming requires ARM64 AdvSimd.");

        // transB=true: strided B rows — fall back to scalar.
        if (transB)
        {
            ScalarStreaming.RunFp32(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            return;
        }

        // FP32: 4-wide Vector128<float> blocks.
        int nBlocks = n / 4;

        fixed (float* aPtr = a)
        fixed (float* bPtr = b)
        fixed (float* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                // Process 4-col blocks via SIMD.
                for (int jb = 0; jb < nBlocks; jb++)
                {
                    int jStart = jb * 4;
                    Vector128<float> acc = AdvSimd.LoadVector128(cPtr + i * ldc + jStart);
                    for (int kk = 0; kk < k; kk++)
                    {
                        float aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector128<float> aVec = Vector128.Create(aval);
                        Vector128<float> bVec = AdvSimd.LoadVector128(bPtr + kk * ldb + jStart);
                        acc = AdvSimd.FusedMultiplyAdd(acc, aVec, bVec);
                    }
                    AdvSimd.Store(cPtr + i * ldc + jStart, acc);
                }
                // Scalar tail for n % 4.
                for (int j = nBlocks * 4; j < n; j++)
                {
                    float sum = cPtr[i * ldc + j];
                    for (int kk = 0; kk < k; kk++)
                    {
                        float aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        sum += aval * bPtr[kk * ldb + j];
                    }
                    cPtr[i * ldc + j] = sum;
                }
            }
        }
    }
#else
    /// <summary>Runtime support gate (always false on net471 — no Vector128&lt;T&gt; ARM intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarStreaming.RunFp64"/> (no Neon available).
    /// </summary>
    public static void RunFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k) =>
        ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarStreaming.RunFp32"/> (no Neon available).
    /// </summary>
    public static void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k) =>
        ScalarStreaming.RunFp32(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
#endif
}
