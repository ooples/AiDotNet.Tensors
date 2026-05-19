using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX2 streaming microkernel for FP64 and FP32. Reads A and B directly in
/// caller-supplied strides — no pack step. Used by the AVX2 streaming
/// strategy when K is small (typically &lt; 32).
///
/// <para>
/// SIMD-accelerates the NN and TN cases (B is contiguous along N). For NT
/// and TT (B is strided), falls back to <see cref="ScalarStreaming"/> — the
/// strided gather is not worth the AVX2 complexity at this scale.
/// </para>
/// </summary>
internal static class Avx2Streaming
{
#if NET5_0_OR_GREATER
    /// <summary>Runtime support gate.</summary>
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>
    /// Compute C += op(A) · op(B) directly without packing. C is read-modify-write.
    /// </summary>
    public static unsafe void RunFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Streaming requires Avx2 + Fma.");

        // transB=true: B is strided across N rows. Fall back to scalar — gather
        // would require vgatherqpd which is slow on most µarchs.
        if (transB)
        {
            ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            return;
        }

        // NN and TN: B is contiguous along N. We process the C output in 4-wide
        // column blocks. For each output row i and each col-block jb (4 cols),
        // accumulate over k using Vector256<double> loads from B.
        int nBlocks = n / 4;

        fixed (double* aPtr = a)
        fixed (double* bPtr = b)
        fixed (double* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                // Process 4-col blocks via SIMD.
                for (int jb = 0; jb < nBlocks; jb++)
                {
                    int jStart = jb * 4;
                    Vector256<double> acc = Avx.LoadVector256(cPtr + i * ldc + jStart);
                    for (int kk = 0; kk < k; kk++)
                    {
                        double aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector256<double> aVec = Vector256.Create(aval);
                        Vector256<double> bVec = Avx.LoadVector256(bPtr + kk * ldb + jStart);
                        acc = Fma.MultiplyAdd(aVec, bVec, acc);
                    }
                    Avx.Store(cPtr + i * ldc + jStart, acc);
                }
                // Scalar tail for n % 4.
                for (int j = nBlocks * 4; j < n; j++)
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
    /// FP32 mirror of <see cref="RunFp64"/>. Uses Vector256&lt;float&gt; (8 lanes).
    ///
    /// <para>
    /// <b>Sub-D6 (#372 follow-up):</b> processes 4 col-blocks (32 cols) per inner
    /// iteration using 4 accumulators in parallel. The FMA latency (4-5 cycles on
    /// Zen3) is hidden by the independent accumulators — peak throughput becomes
    /// ~1 FMA/cycle instead of 1 FMA per 4 cycles. Per-iter K-loop cost is the
    /// same (4 FMAs + 4 B-loads + 1 broadcast), but pipelined.
    /// </para>
    /// </summary>
    public static unsafe void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("Avx2Streaming requires Avx2 + Fma.");

        if (transB)
        {
            ScalarStreaming.RunFp32(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
            return;
        }

        // Process 64 cols per outer iter using 8 Vector256<float> accumulators.
        // Each iter advances j by 64. With 8 accumulators, the FMA chain has
        // enough independent ops to fully pipeline on Zen3 (FMA throughput 1
        // per cycle once pipelined, latency 4-5 cycles → 5-acc minimum to hide).
        int nBig = (n / 64) * 64;       // largest multiple of 64 ≤ n
        int nBig32 = (n / 32) * 32;     // largest multiple of 32 ≤ n
        int nBlocks = n / 8;             // 8-col blocks for the trailing tail
        int nBig64_blocks = nBig / 8;    // covered by the 8-acc loop
        int nBig32_blocks = nBig32 / 8;  // covered by either 8-acc or 4-acc

        fixed (float* aPtr = a)
        fixed (float* bPtr = b)
        fixed (float* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                // ── 8-acc SIMD path: 64 cols per iter ──────────────────────────
                for (int jStart = 0; jStart < nBig; jStart += 64)
                {
                    Vector256<float> acc0 = Avx.LoadVector256(cPtr + i * ldc + jStart + 0);
                    Vector256<float> acc1 = Avx.LoadVector256(cPtr + i * ldc + jStart + 8);
                    Vector256<float> acc2 = Avx.LoadVector256(cPtr + i * ldc + jStart + 16);
                    Vector256<float> acc3 = Avx.LoadVector256(cPtr + i * ldc + jStart + 24);
                    Vector256<float> acc4 = Avx.LoadVector256(cPtr + i * ldc + jStart + 32);
                    Vector256<float> acc5 = Avx.LoadVector256(cPtr + i * ldc + jStart + 40);
                    Vector256<float> acc6 = Avx.LoadVector256(cPtr + i * ldc + jStart + 48);
                    Vector256<float> acc7 = Avx.LoadVector256(cPtr + i * ldc + jStart + 56);
                    for (int kk = 0; kk < k; kk++)
                    {
                        float aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector256<float> aVec = Vector256.Create(aval);
                        float* bRow = bPtr + kk * ldb + jStart;
                        acc0 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 0), acc0);
                        acc1 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 8), acc1);
                        acc2 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 16), acc2);
                        acc3 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 24), acc3);
                        acc4 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 32), acc4);
                        acc5 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 40), acc5);
                        acc6 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 48), acc6);
                        acc7 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 56), acc7);
                    }
                    Avx.Store(cPtr + i * ldc + jStart + 0, acc0);
                    Avx.Store(cPtr + i * ldc + jStart + 8, acc1);
                    Avx.Store(cPtr + i * ldc + jStart + 16, acc2);
                    Avx.Store(cPtr + i * ldc + jStart + 24, acc3);
                    Avx.Store(cPtr + i * ldc + jStart + 32, acc4);
                    Avx.Store(cPtr + i * ldc + jStart + 40, acc5);
                    Avx.Store(cPtr + i * ldc + jStart + 48, acc6);
                    Avx.Store(cPtr + i * ldc + jStart + 56, acc7);
                }
                // ── 4-acc SIMD path: 32 cols per iter for [nBig, nBig32) ──────
                for (int jStart = nBig; jStart < nBig32; jStart += 32)
                {
                    Vector256<float> acc0 = Avx.LoadVector256(cPtr + i * ldc + jStart + 0);
                    Vector256<float> acc1 = Avx.LoadVector256(cPtr + i * ldc + jStart + 8);
                    Vector256<float> acc2 = Avx.LoadVector256(cPtr + i * ldc + jStart + 16);
                    Vector256<float> acc3 = Avx.LoadVector256(cPtr + i * ldc + jStart + 24);
                    for (int kk = 0; kk < k; kk++)
                    {
                        float aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector256<float> aVec = Vector256.Create(aval);
                        float* bRow = bPtr + kk * ldb + jStart;
                        acc0 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 0), acc0);
                        acc1 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 8), acc1);
                        acc2 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 16), acc2);
                        acc3 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 24), acc3);
                    }
                    Avx.Store(cPtr + i * ldc + jStart + 0, acc0);
                    Avx.Store(cPtr + i * ldc + jStart + 8, acc1);
                    Avx.Store(cPtr + i * ldc + jStart + 16, acc2);
                    Avx.Store(cPtr + i * ldc + jStart + 24, acc3);
                }
                // ── 1-acc SIMD path: remaining 8-col blocks ──────
                for (int jb = nBig32_blocks; jb < nBlocks; jb++)
                {
                    int jStart = jb * 8;
                    Vector256<float> acc = Avx.LoadVector256(cPtr + i * ldc + jStart);
                    for (int kk = 0; kk < k; kk++)
                    {
                        float aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector256<float> aVec = Vector256.Create(aval);
                        Vector256<float> bVec = Avx.LoadVector256(bPtr + kk * ldb + jStart);
                        acc = Fma.MultiplyAdd(aVec, bVec, acc);
                    }
                    Avx.Store(cPtr + i * ldc + jStart, acc);
                }
                // ── Scalar n tail (n % 8) ──────────────────────────────────────
                for (int j = nBlocks * 8; j < n; j++)
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
    /// <summary>Runtime support gate (always false on net471 — no Vector256&lt;T&gt; intrinsics).</summary>
    public static bool IsSupported => false;

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarStreaming.RunFp64"/> (no AVX2 available).
    /// </summary>
    public static void RunFp64(
        ReadOnlySpan<double> a, int lda, bool transA,
        ReadOnlySpan<double> b, int ldb, bool transB,
        Span<double> c, int ldc,
        int m, int n, int k) =>
        ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);

    /// <summary>
    /// net471 stub: delegates to <see cref="ScalarStreaming.RunFp32"/> (no AVX2 available).
    /// </summary>
    public static void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k) =>
        ScalarStreaming.RunFp32(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
#endif
}
