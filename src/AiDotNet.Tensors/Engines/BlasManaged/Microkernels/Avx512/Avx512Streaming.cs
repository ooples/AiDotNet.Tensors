using System;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX-512 streaming microkernel for FP64 and FP32. Reads A and B directly in
/// caller-supplied strides — no pack step. Used by the AVX-512 streaming
/// strategy when K is small (typically &lt; 32).
///
/// <para>
/// SIMD-accelerates the NN and TN cases (B contiguous along N). For NT and TT
/// (B strided), falls back to <see cref="ScalarStreaming"/>.
/// </para>
/// </summary>
internal static class Avx512Streaming
{
#if NET8_0_OR_GREATER
    /// <summary>Runtime support gate.</summary>
    public static bool IsSupported => Avx512F.IsSupported;

    /// <summary>
    /// Compute C += op(A) · op(B) directly without packing. C is read-modify-write.
    /// FP64: 8-wide blocks via Vector512&lt;double&gt;.
    /// </summary>
    public static unsafe void RunFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Streaming requires Avx512F.");

        // transB=true: B is strided across N rows. Fall back to scalar — strided
        // gathers are not worth the AVX-512 complexity at this scale.
        if (transB)
        {
            ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            return;
        }

        // NN and TN: B is contiguous along N. Process the C output in 8-wide
        // column blocks (Vector512<double> has 8 lanes = fp64 register width / 8B).
        int nBlocks = n / 8;

        fixed (double* aPtr = a)
        fixed (double* bPtr = b)
        fixed (double* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                // SIMD path: 8-col blocks.
                for (int jb = 0; jb < nBlocks; jb++)
                {
                    int jStart = jb * 8;
                    Vector512<double> acc = Avx512F.LoadVector512(cPtr + i * ldc + jStart);
                    for (int kk = 0; kk < k; kk++)
                    {
                        double aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector512<double> aVec = Vector512.Create(aval);
                        Vector512<double> bVec = Avx512F.LoadVector512(bPtr + kk * ldb + jStart);
                        acc = Avx512F.FusedMultiplyAdd(aVec, bVec, acc);
                    }
                    Avx512F.Store(cPtr + i * ldc + jStart, acc);
                }
                // Scalar tail for n % 8.
                for (int j = nBlocks * 8; j < n; j++)
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
    /// FP32 mirror of <see cref="RunFp64"/>. Uses Vector512&lt;float&gt; (16 lanes)
    /// and processes 16-col blocks per inner loop iteration.
    /// </summary>
    public static unsafe void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx512Streaming requires Avx512F.");

        if (transB)
        {
            ScalarStreaming.RunFp32(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            return;
        }

        // FP32: 16-wide blocks via Vector512<float>.
        int nBlocks = n / 16;

        fixed (float* aPtr = a)
        fixed (float* bPtr = b)
        fixed (float* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                // SIMD path: 16-col blocks.
                for (int jb = 0; jb < nBlocks; jb++)
                {
                    int jStart = jb * 16;
                    Vector512<float> acc = Avx512F.LoadVector512(cPtr + i * ldc + jStart);
                    for (int kk = 0; kk < k; kk++)
                    {
                        float aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector512<float> aVec = Vector512.Create(aval);
                        Vector512<float> bVec = Avx512F.LoadVector512(bPtr + kk * ldb + jStart);
                        acc = Avx512F.FusedMultiplyAdd(aVec, bVec, acc);
                    }
                    Avx512F.Store(cPtr + i * ldc + jStart, acc);
                }
                // Scalar tail for n % 16.
                for (int j = nBlocks * 16; j < n; j++)
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
    /// <summary>Runtime support gate (always false on net471 — no Vector512&lt;T&gt; intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarStreaming.RunFp64"/> (no AVX-512 available).
    /// </summary>
    public static void RunFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k) =>
        ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarStreaming.RunFp32"/> (no AVX-512 available).
    /// </summary>
    public static void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k) =>
        ScalarStreaming.RunFp32(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
#endif
}
