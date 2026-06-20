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

        if (transB)
        {
            // TT: A strided along K — keep scalar (rare).
            if (transA)
            {
                ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
                return;
            }
            // NT (#639): contiguous-K dot product, vectorized. See the FP32 path.
            RunFp64Nt(a, lda, b, ldb, c, ldc, m, n, k);
            return;
        }

        // Sub-D8 — multi-accumulator FP64 streaming. Two SIMD tiers:
        //   4-acc loop: 16 cols per iter (4 Vector256<double> accumulators)
        //   1-acc loop: 4-col tail
        // Then scalar tail for n % 4.
        // FP64 vec width = 4 doubles, so 4-acc covers 16 cols which is the sweet
        // spot for Zen3 FMA pipelining (needs 5+ acc, but 4 + B-load chain is close).
        int nBig = (n / 16) * 16;     // largest multiple of 16 ≤ n
        int nBlocks = n / 4;           // 4-col blocks for the tail
        int nBig4_blocks = nBig / 4;   // covered by the 4-acc loop

        fixed (double* aPtr = a)
        fixed (double* bPtr = b)
        fixed (double* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                // ── 4-acc SIMD path: 16 cols per iter ──────────────────────────
                for (int jStart = 0; jStart < nBig; jStart += 16)
                {
                    Vector256<double> acc0 = Avx.LoadVector256(cPtr + i * ldc + jStart + 0);
                    Vector256<double> acc1 = Avx.LoadVector256(cPtr + i * ldc + jStart + 4);
                    Vector256<double> acc2 = Avx.LoadVector256(cPtr + i * ldc + jStart + 8);
                    Vector256<double> acc3 = Avx.LoadVector256(cPtr + i * ldc + jStart + 12);
                    for (int kk = 0; kk < k; kk++)
                    {
                        double aval = transA ? aPtr[kk * lda + i] : aPtr[i * lda + kk];
                        Vector256<double> aVec = Vector256.Create(aval);
                        double* bRow = bPtr + kk * ldb + jStart;
                        acc0 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 0), acc0);
                        acc1 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 4), acc1);
                        acc2 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 8), acc2);
                        acc3 = Fma.MultiplyAdd(aVec, Avx.LoadVector256(bRow + 12), acc3);
                    }
                    Avx.Store(cPtr + i * ldc + jStart + 0, acc0);
                    Avx.Store(cPtr + i * ldc + jStart + 4, acc1);
                    Avx.Store(cPtr + i * ldc + jStart + 8, acc2);
                    Avx.Store(cPtr + i * ldc + jStart + 12, acc3);
                }
                // ── 1-acc SIMD path: remaining 4-col blocks ──
                for (int jb = nBig4_blocks; jb < nBlocks; jb++)
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
            // TT (A also transposed): A is strided along K, so the contiguous-K
            // dot product below doesn't apply — keep the scalar reference (rare).
            if (transA)
            {
                ScalarStreaming.RunFp32(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
                return;
            }
            // NT (#639): C[i,j] += dot(A[i,:K], B[j,:K]). With transB=true, B is [N,K]
            // row-major so B's row j is contiguous along K — same as A's row i. This is
            // the conv-backward dX = dY·Wᵀ shape, which previously fell to the SCALAR
            // kernel for the WHOLE GEMM (no SIMD at all — the top compiled-plan compute
            // frame on an AVX2 box). Vectorize it as an 8-wide FMA dot product, 4 output
            // columns per inner pass so one A-row load feeds four B-row FMAs.
            RunFp32Nt(a, lda, b, ldb, c, ldc, m, n, k);
            return;
        }

        // Sub-D9: for M=1 (gemv pattern, memory-bound), use 4-acc only.
        // The 8-acc inner loop adds register pressure that the compiler spills
        // on M=1 shapes (M=1 means no outer row reuse, so register pressure
        // is the dominant cost vs FMA throughput). 4-acc was the sweet spot
        // in the D6 baseline for these shapes (ResNet50_fc, MobileNetV2_fc).
        int nBig = m == 1 ? 0 : (n / 64) * 64;       // M=1: skip 8-acc loop entirely
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
    /// <summary>
    /// #639 NT microkernel (transB=true, transA=false): C[i,j] += Σ_k A[i,k]·B[j,k].
    /// Both A row i and B row j are contiguous along K, so each output is a vectorized
    /// 8-wide FMA dot product. Processes 4 output columns per pass so a single A-row
    /// load feeds four independent accumulators (hides FMA latency, amortizes the load).
    /// C is read-modify-write (caller zeroed it on the first streaming call).
    /// </summary>
    private static unsafe void RunFp32Nt(
        ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        int kVec = (k / 8) * 8;
        fixed (float* aPtr = a)
        fixed (float* bPtr = b)
        fixed (float* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                float* aRow = aPtr + i * lda;
                float* cRow = cPtr + i * ldc;
                int j = 0;
                for (; j + 4 <= n; j += 4)
                {
                    float* b0 = bPtr + (j + 0) * ldb;
                    float* b1 = bPtr + (j + 1) * ldb;
                    float* b2 = bPtr + (j + 2) * ldb;
                    float* b3 = bPtr + (j + 3) * ldb;
                    Vector256<float> acc0 = Vector256<float>.Zero;
                    Vector256<float> acc1 = Vector256<float>.Zero;
                    Vector256<float> acc2 = Vector256<float>.Zero;
                    Vector256<float> acc3 = Vector256<float>.Zero;
                    for (int kk = 0; kk < kVec; kk += 8)
                    {
                        Vector256<float> av = Avx.LoadVector256(aRow + kk);
                        acc0 = Fma.MultiplyAdd(av, Avx.LoadVector256(b0 + kk), acc0);
                        acc1 = Fma.MultiplyAdd(av, Avx.LoadVector256(b1 + kk), acc1);
                        acc2 = Fma.MultiplyAdd(av, Avx.LoadVector256(b2 + kk), acc2);
                        acc3 = Fma.MultiplyAdd(av, Avx.LoadVector256(b3 + kk), acc3);
                    }
                    float s0 = HSum256(acc0), s1 = HSum256(acc1), s2 = HSum256(acc2), s3 = HSum256(acc3);
                    for (int kk = kVec; kk < k; kk++)
                    {
                        float av = aRow[kk];
                        s0 += av * b0[kk]; s1 += av * b1[kk]; s2 += av * b2[kk]; s3 += av * b3[kk];
                    }
                    cRow[j + 0] += s0; cRow[j + 1] += s1; cRow[j + 2] += s2; cRow[j + 3] += s3;
                }
                for (; j < n; j++)
                {
                    float* bj = bPtr + j * ldb;
                    Vector256<float> acc = Vector256<float>.Zero;
                    for (int kk = 0; kk < kVec; kk += 8)
                        acc = Fma.MultiplyAdd(Avx.LoadVector256(aRow + kk), Avx.LoadVector256(bj + kk), acc);
                    float s = HSum256(acc);
                    for (int kk = kVec; kk < k; kk++) s += aRow[kk] * bj[kk];
                    cRow[j] += s;
                }
            }
        }
    }

    /// <summary>FP64 mirror of <see cref="RunFp32Nt"/> (Vector256&lt;double&gt;, 4 lanes).</summary>
    private static unsafe void RunFp64Nt(
        ReadOnlySpan<double> a, int lda,
        ReadOnlySpan<double> b, int ldb,
        Span<double> c, int ldc,
        int m, int n, int k)
    {
        int kVec = (k / 4) * 4;
        fixed (double* aPtr = a)
        fixed (double* bPtr = b)
        fixed (double* cPtr = c)
        {
            for (int i = 0; i < m; i++)
            {
                double* aRow = aPtr + i * lda;
                double* cRow = cPtr + i * ldc;
                int j = 0;
                for (; j + 4 <= n; j += 4)
                {
                    double* b0 = bPtr + (j + 0) * ldb;
                    double* b1 = bPtr + (j + 1) * ldb;
                    double* b2 = bPtr + (j + 2) * ldb;
                    double* b3 = bPtr + (j + 3) * ldb;
                    Vector256<double> acc0 = Vector256<double>.Zero;
                    Vector256<double> acc1 = Vector256<double>.Zero;
                    Vector256<double> acc2 = Vector256<double>.Zero;
                    Vector256<double> acc3 = Vector256<double>.Zero;
                    for (int kk = 0; kk < kVec; kk += 4)
                    {
                        Vector256<double> av = Avx.LoadVector256(aRow + kk);
                        acc0 = Fma.MultiplyAdd(av, Avx.LoadVector256(b0 + kk), acc0);
                        acc1 = Fma.MultiplyAdd(av, Avx.LoadVector256(b1 + kk), acc1);
                        acc2 = Fma.MultiplyAdd(av, Avx.LoadVector256(b2 + kk), acc2);
                        acc3 = Fma.MultiplyAdd(av, Avx.LoadVector256(b3 + kk), acc3);
                    }
                    double s0 = HSum256(acc0), s1 = HSum256(acc1), s2 = HSum256(acc2), s3 = HSum256(acc3);
                    for (int kk = kVec; kk < k; kk++)
                    {
                        double av = aRow[kk];
                        s0 += av * b0[kk]; s1 += av * b1[kk]; s2 += av * b2[kk]; s3 += av * b3[kk];
                    }
                    cRow[j + 0] += s0; cRow[j + 1] += s1; cRow[j + 2] += s2; cRow[j + 3] += s3;
                }
                for (; j < n; j++)
                {
                    double* bj = bPtr + j * ldb;
                    Vector256<double> acc = Vector256<double>.Zero;
                    for (int kk = 0; kk < kVec; kk += 4)
                        acc = Fma.MultiplyAdd(Avx.LoadVector256(aRow + kk), Avx.LoadVector256(bj + kk), acc);
                    double s = HSum256(acc);
                    for (int kk = kVec; kk < k; kk++) s += aRow[kk] * bj[kk];
                    cRow[j] += s;
                }
            }
        }
    }

    /// <summary>Horizontal sum of an 8-lane FP32 vector.</summary>
    private static float HSum256(Vector256<float> v)
    {
        Vector128<float> s = Sse.Add(v.GetLower(), v.GetUpper());   // 4 partials
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));                    // 2 partials
        s = Sse.AddScalar(s, Sse.Shuffle(s, s, 0x55));              // + lane 1
        return s.ToScalar();
    }

    /// <summary>Horizontal sum of a 4-lane FP64 vector.</summary>
    private static double HSum256(Vector256<double> v)
    {
        Vector128<double> s = Sse2.Add(v.GetLower(), v.GetUpper());  // 2 partials
        s = Sse2.AddScalar(s, Sse2.UnpackHigh(s, s));               // + lane 1
        return s.ToScalar();
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
