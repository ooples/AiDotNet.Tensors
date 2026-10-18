using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

#if NETCOREAPP
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-R (#408): dedicated GEMV (matrix × vector) kernels for the M=1, N=1,
/// and K=1 cases. The full GEMM dispatcher's per-call overhead (~10-30 µs of
/// strategy selection + autotune lookup + microkernel-tile picking + epilogue
/// chain) dominates the actual compute for tiny vector ops. Routing M=1/N=1/K=1
/// here avoids that overhead entirely.
///
/// <para>
/// Three specialised cases:
/// </para>
/// <list type="bullet">
///   <item><b>M=1</b> (row × matrix): <c>c[0, j] = Σ_k a[0, k] · b[k, j]</c>.
///     Reads a once per K-iter; B sequentially. Used by LSTM/GRU cell-state ops.</item>
///   <item><b>N=1</b> (matrix × col): <c>c[i, 0] = Σ_k a[i, k] · b[k, 0]</c>.
///     Reads b once per K-iter; A sequentially. Used by embedding lookups,
///     attention scoring against a single query.</item>
///   <item><b>K=1</b> (outer product): <c>c[i, j] = a[i, 0] · b[0, j]</c>.
///     No reduction; pure outer product. Used by gating / residual paths.</item>
/// </list>
/// </summary>
internal static class GemvKernel
{
    /// <summary>
    /// Returns true when (m, n, k) qualifies for the GEMV fast path. Caller
    /// (<see cref="BlasManaged.Gemm{T}"/>) checks this at entry and routes here
    /// instead of the full strategy dispatcher.
    /// </summary>
    public static bool QualifiesFor(int m, int n, int k)
    {
        return m == 1 || n == 1 || k == 1;
    }

    /// <summary>
    /// Compute C = op(A) · op(B) for the GEMV cases. Caller must have already
    /// cleared C (matches <see cref="BlasManaged.Gemm{T}"/>'s contract).
    /// </summary>
    public static void Run<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n, int k) where T : unmanaged
    {
        if (k == 1)
        {
            // Outer product — no reduction, just one multiply per output cell.
            RunOuterProduct<T>(a, lda, transA, b, ldb, transB, c, ldc, m, n);
            return;
        }
        if (m == 1)
        {
            // Row × matrix: c[0, j] = Σ_k a[k or 0]·b[k, j]
            // After applying op(A): the effective row of A has K elements.
            //   transA=false → a is [1, K] row-major, stride 1
            //   transA=true  → a is [K, 1], a[k*lda + 0]
            RunRowTimesMatrix<T>(a, lda, transA, b, ldb, transB, c, ldc, n, k);
            return;
        }
        if (n == 1)
        {
            // Matrix × col: c[i, 0] = Σ_k a[i, k]·b[k or 0]
            RunMatrixTimesCol<T>(a, lda, transA, b, ldb, transB, c, ldc, m, k);
            return;
        }
        throw new ArgumentException($"GemvKernel.Run requires m==1 or n==1 or k==1; got ({m}, {n}, {k}).");
    }

    [MethodImpl(Hot)]
    private static unsafe void RunOuterProduct<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n) where T : unmanaged
    {
        // K=1 ⇒ c[i, j] = a_i · b_j (pure outer product, no reduction).
        //   a_i: transA=false → af[i * lda]; transA=true → af[i]  (stride below)
        //   b_j: transB=false → bf[j] (contiguous row); transB=true → bf[j * ldb]
        if (typeof(T) == typeof(float))
        {
            var af = MemoryMarshal.Cast<T, float>(a);
            var bf = MemoryMarshal.Cast<T, float>(b);
            var cf = MemoryMarshal.Cast<T, float>(c);
            int aStride = transA ? 1 : lda;

#if NETCOREAPP
            // Sub-R (#408) AVX2 outer-product fast path: broadcast the scalar
            // a_i across a Vector256 and multiply by 8 contiguous b-values per
            // iter, storing to the contiguous c-row. No reduction, so a plain
            // Avx.Multiply (no FMA) is enough. Requires transB=false (b is a
            // contiguous [1, N] row); transB=true / tiny N / non-AVX2 fall
            // through to the scalar loop below.
            if (!transB && Avx.IsSupported && n >= 8)
            {
                int nFull = n & ~7;
                fixed (float* aPtr = af)
                fixed (float* bPtr = bf)
                fixed (float* cPtr = cf)
                {
                    for (int i = 0; i < m; i++)
                    {
                        float ai = aPtr[i * aStride];
                        var av = Vector256.Create(ai);
                        float* cRow = cPtr + (long)i * ldc;
                        for (int j = 0; j < nFull; j += 8)
                        {
                            var bv = Avx.LoadVector256(bPtr + j);
                            Avx.Store(cRow + j, Avx.Multiply(av, bv));
                        }
                        for (int j = nFull; j < n; j++)
                            cRow[j] = ai * bPtr[j];
                    }
                }
                return;
            }
#endif
            for (int i = 0; i < m; i++)
            {
                float ai = af[i * aStride];
                for (int j = 0; j < n; j++)
                {
                    float bj = transB ? bf[j * ldb] : bf[j];
                    cf[i * ldc + j] = ai * bj;
                }
            }
            return;
        }
        if (typeof(T) == typeof(double))
        {
            var ad = MemoryMarshal.Cast<T, double>(a);
            var bd = MemoryMarshal.Cast<T, double>(b);
            var cd = MemoryMarshal.Cast<T, double>(c);
            int aStride = transA ? 1 : lda;

#if NETCOREAPP
            // AVX2 FP64 mirror — 4 doubles per Vector256<double>.
            if (!transB && Avx.IsSupported && n >= 4)
            {
                int nFull = n & ~3;
                fixed (double* aPtr = ad)
                fixed (double* bPtr = bd)
                fixed (double* cPtr = cd)
                {
                    for (int i = 0; i < m; i++)
                    {
                        double ai = aPtr[i * aStride];
                        var av = Vector256.Create(ai);
                        double* cRow = cPtr + (long)i * ldc;
                        for (int j = 0; j < nFull; j += 4)
                        {
                            var bv = Avx.LoadVector256(bPtr + j);
                            Avx.Store(cRow + j, Avx.Multiply(av, bv));
                        }
                        for (int j = nFull; j < n; j++)
                            cRow[j] = ai * bPtr[j];
                    }
                }
                return;
            }
#endif
            for (int i = 0; i < m; i++)
            {
                double ai = ad[i * aStride];
                for (int j = 0; j < n; j++)
                {
                    double bj = transB ? bd[j * ldb] : bd[j];
                    cd[i * ldc + j] = ai * bj;
                }
            }
            return;
        }
        throw new NotSupportedException($"GemvKernel does not support T={typeof(T).Name}.");
    }

#if NETCOREAPP
    // Row x matrix keeps eight accumulators across the whole K loop, so each B row contributes one contiguous run
    // (64 floats / 32 doubles: four cache lines). The earlier kernel kept one accumulator per 8 columns and walked B
    // down all K rows for each strip, reading half of every cache line it touched, on one thread. Column blocks run
    // in parallel above RowTimesMatrixParallelWork; each output is still one thread's ascending-K FMA chain (the tails
    // keep the earlier vector/scalar forms), so results are bit-identical to the serial kernel at any thread count.
    // On 1x4096x4096 and 1x11008x4096 FP32 (Llama-7B single-token inference) the strip kernel ran ~5.4x OpenBLAS.
    private const int RowBlockFloat = 64, RowBlockDouble = 32;
    private const long RowTimesMatrixParallelWork = 1L << 20;

    private static unsafe void RowTimesMatrixAvx2(float* a, int aStride, float* b, int ldb, float* c, int n, int k)
    {
        int blocks = (n + RowBlockFloat - 1) / RowBlockFloat;
        if ((long)n * k < RowTimesMatrixParallelWork || blocks == 1)
        {
            RowTimesMatrixColumns(a, aStride, b, ldb, c, 0, n, k);
            return;
        }
        IntPtr ap = (IntPtr)a, bp = (IntPtr)b, cp = (IntPtr)c;
        Helpers.CpuParallelSettings.ParallelForOrSerial(0, blocks, (long)n * k, blk =>
        {
            int j0 = blk * RowBlockFloat;
            RowTimesMatrixColumns((float*)ap, aStride, (float*)bp, ldb, (float*)cp, j0, Math.Min(n, j0 + RowBlockFloat), k);
        }, deterministicSafe: true);
    }

    [MethodImpl(Hot)]
    private static unsafe void RowTimesMatrixColumns(float* a, int aStride, float* b, int ldb, float* c, int j0, int j1, int k)
    {
        int j = j0;
        for (; j + RowBlockFloat <= j1; j += RowBlockFloat)
        {
            Vector256<float> c0 = default, c1 = default, c2 = default, c3 = default,
                c4 = default, c5 = default, c6 = default, c7 = default;
            float* bj = b + j;
            for (int kk = 0; kk < k; kk++)
            {
                var av = Vector256.Create(a[(long)kk * aStride]);
                float* br = bj + (long)kk * ldb;
                c0 = Fma.MultiplyAdd(av, Avx.LoadVector256(br), c0);
                c1 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 8), c1);
                c2 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 16), c2);
                c3 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 24), c3);
                c4 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 32), c4);
                c5 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 40), c5);
                c6 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 48), c6);
                c7 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 56), c7);
            }
            float* cj = c + j;
            Avx.Store(cj, c0); Avx.Store(cj + 8, c1); Avx.Store(cj + 16, c2); Avx.Store(cj + 24, c3);
            Avx.Store(cj + 32, c4); Avx.Store(cj + 40, c5); Avx.Store(cj + 48, c6); Avx.Store(cj + 56, c7);
        }
        for (; j + 8 <= j1; j += 8)
        {
            var acc = Vector256<float>.Zero;
            for (int kk = 0; kk < k; kk++)
                acc = Fma.MultiplyAdd(Vector256.Create(a[(long)kk * aStride]), Avx.LoadVector256(b + (long)kk * ldb + j), acc);
            Avx.Store(c + j, acc);
        }
        for (; j < j1; j++)
        {
            float sum = 0;
            for (int kk = 0; kk < k; kk++) sum += a[(long)kk * aStride] * b[(long)kk * ldb + j];
            c[j] = sum;
        }
    }

    private static unsafe void RowTimesMatrixAvx2(double* a, int aStride, double* b, int ldb, double* c, int n, int k)
    {
        int blocks = (n + RowBlockDouble - 1) / RowBlockDouble;
        if ((long)n * k < RowTimesMatrixParallelWork || blocks == 1)
        {
            RowTimesMatrixColumns(a, aStride, b, ldb, c, 0, n, k);
            return;
        }
        IntPtr ap = (IntPtr)a, bp = (IntPtr)b, cp = (IntPtr)c;
        Helpers.CpuParallelSettings.ParallelForOrSerial(0, blocks, (long)n * k, blk =>
        {
            int j0 = blk * RowBlockDouble;
            RowTimesMatrixColumns((double*)ap, aStride, (double*)bp, ldb, (double*)cp, j0, Math.Min(n, j0 + RowBlockDouble), k);
        }, deterministicSafe: true);
    }

    [MethodImpl(Hot)]
    private static unsafe void RowTimesMatrixColumns(double* a, int aStride, double* b, int ldb, double* c, int j0, int j1, int k)
    {
        int j = j0;
        for (; j + RowBlockDouble <= j1; j += RowBlockDouble)
        {
            Vector256<double> c0 = default, c1 = default, c2 = default, c3 = default,
                c4 = default, c5 = default, c6 = default, c7 = default;
            double* bj = b + j;
            for (int kk = 0; kk < k; kk++)
            {
                var av = Vector256.Create(a[(long)kk * aStride]);
                double* br = bj + (long)kk * ldb;
                c0 = Fma.MultiplyAdd(av, Avx.LoadVector256(br), c0);
                c1 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 4), c1);
                c2 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 8), c2);
                c3 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 12), c3);
                c4 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 16), c4);
                c5 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 20), c5);
                c6 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 24), c6);
                c7 = Fma.MultiplyAdd(av, Avx.LoadVector256(br + 28), c7);
            }
            double* cj = c + j;
            Avx.Store(cj, c0); Avx.Store(cj + 4, c1); Avx.Store(cj + 8, c2); Avx.Store(cj + 12, c3);
            Avx.Store(cj + 16, c4); Avx.Store(cj + 20, c5); Avx.Store(cj + 24, c6); Avx.Store(cj + 28, c7);
        }
        for (; j + 4 <= j1; j += 4)
        {
            var acc = Vector256<double>.Zero;
            for (int kk = 0; kk < k; kk++)
                acc = Fma.MultiplyAdd(Vector256.Create(a[(long)kk * aStride]), Avx.LoadVector256(b + (long)kk * ldb + j), acc);
            Avx.Store(c + j, acc);
        }
        for (; j < j1; j++)
        {
            double sum = 0;
            for (int kk = 0; kk < k; kk++) sum += a[(long)kk * aStride] * b[(long)kk * ldb + j];
            c[j] = sum;
        }
    }
#endif

    [MethodImpl(Hot)]
    private static unsafe void RunRowTimesMatrix<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int n, int k) where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            var af = MemoryMarshal.Cast<T, float>(a);
            var bf = MemoryMarshal.Cast<T, float>(b);
            var cf = MemoryMarshal.Cast<T, float>(c);
            // a is logically [1, K]: transA=false → af[kk]; transA=true → af[kk * lda]
            int aStride = transA ? lda : 1;

#if NETCOREAPP
            // AVX2 fast path for transB=false (B is [K, N] row-major).
            if (!transB && Avx2.IsSupported && Fma.IsSupported && n >= 8)
            {
                fixed (float* aPtr = af)
                fixed (float* bPtr = bf)
                fixed (float* cPtr = cf)
                    RowTimesMatrixAvx2(aPtr, aStride, bPtr, ldb, cPtr, n, k);
                return;
            }
#endif
            // Scalar fallback (also handles transB=true and tiny N).
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                int bColStride = transB ? ldb : 1;
                int bRowStride = transB ? 1 : ldb;
                int bColStart = transB ? j * bColStride : j;
                for (int kk = 0; kk < k; kk++)
                {
                    sum += (double)af[kk * aStride] * bf[kk * bRowStride + bColStart];
                }
                cf[j] = (float)sum;
            }
            return;
        }
        if (typeof(T) == typeof(double))
        {
            var ad = MemoryMarshal.Cast<T, double>(a);
            var bd = MemoryMarshal.Cast<T, double>(b);
            var cd = MemoryMarshal.Cast<T, double>(c);
            int aStride = transA ? lda : 1;

#if NETCOREAPP
            if (!transB && Avx2.IsSupported && Fma.IsSupported && n >= 4)
            {
                fixed (double* aPtr = ad)
                fixed (double* bPtr = bd)
                fixed (double* cPtr = cd)
                    RowTimesMatrixAvx2(aPtr, aStride, bPtr, ldb, cPtr, n, k);
                return;
            }
#endif
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                int bColStride = transB ? ldb : 1;
                int bRowStride = transB ? 1 : ldb;
                int bColStart = transB ? j * bColStride : j;
                for (int kk = 0; kk < k; kk++)
                {
                    sum += ad[kk * aStride] * bd[kk * bRowStride + bColStart];
                }
                cd[j] = sum;
            }
            return;
        }
        throw new NotSupportedException($"GemvKernel does not support T={typeof(T).Name}.");
    }

    [MethodImpl(Hot)]
    private static unsafe void RunMatrixTimesCol<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int k) where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            var af = MemoryMarshal.Cast<T, float>(a);
            var bf = MemoryMarshal.Cast<T, float>(b);
            var cf = MemoryMarshal.Cast<T, float>(c);
            int bStride = transB ? 1 : ldb;

#if NETCOREAPP
            // Sub-R (#408) AVX2 fast path: transA=false (A-row contiguous,
            // dominant case for embedding lookups, attention-against-single-
            // query, and any "matrix · column-vector" workload). Per-row
            // dot product with FMA into a Vector256<float> accumulator,
            // horizontal reduce at end. Compared to the scalar fallback
            // below (~1 op/cycle), this delivers ~8× speedup on long-K
            // shapes (e.g. K=512 dropped from ~600 ns/row to ~80 ns/row
            // on Skylake-class hosts in micro-bench).
            //
            // B must be contiguous as a K-vector: either transB=true
            // (ldb is the n-stride, irrelevant when n=1) OR transB=false
            // with ldb=1 (B is already a [K, 1] column vector — the
            // caller's contract for n=1 GEMV). The bStride==1 check
            // captures both cases.
            bool aRowContiguous = !transA;
            bool bIsContiguousVector = bStride == 1;
            if (aRowContiguous && bIsContiguousVector
                && Avx2.IsSupported && Fma.IsSupported && k >= 8)
            {
                int kFull = k & ~7;
                fixed (float* aPtr = af)
                fixed (float* bPtr = bf)
                fixed (float* cPtr = cf)
                {
                    for (int i = 0; i < m; i++)
                    {
                        var acc = Vector256<float>.Zero;
                        float* aRow = aPtr + (long)i * lda;
                        for (int kk = 0; kk < kFull; kk += 8)
                        {
                            var av = Avx.LoadVector256(aRow + kk);
                            var bv = Avx.LoadVector256(bPtr + kk);
                            acc = Fma.MultiplyAdd(av, bv, acc);
                        }
                        // Horizontal reduce the 8-element AVX2 vector to scalar.
                        var hi128 = Avx.ExtractVector128(acc, 1);
                        var lo128 = acc.GetLower();
                        var sum128 = Sse.Add(hi128, lo128);
                        // HorizontalAdd pairs adjacent lanes: [a0+a1, a2+a3, a0+a1, a2+a3]
                        sum128 = Sse3.HorizontalAdd(sum128, sum128);
                        sum128 = Sse3.HorizontalAdd(sum128, sum128);
                        double sum = sum128.ToScalar();
                        // K-tail scalar (k % 8 != 0).
                        for (int kk = kFull; kk < k; kk++)
                            sum += (double)aRow[kk] * bPtr[kk];
                        cPtr[(long)i * ldc] = (float)sum;
                    }
                }
                return;
            }
#endif

            // Scalar fallback — handles transA=true, non-contiguous B
            // (transB=false with ldb>1), tiny K (k<8), and non-AVX2 hosts.
            // b is logically [K, 1]: transB=false → bf[kk * ldb]; transB=true → bf[kk].
            // Each output cell c[i, 0] is a dot product of a's row i with b's column:
            //   transA=false → a row i starts at af[i * lda];
            //   transA=true  → a row i (logical) is column i in storage, af[kk * lda + i].
            for (int i = 0; i < m; i++)
            {
                double sum = 0;
                if (transA)
                {
                    for (int kk = 0; kk < k; kk++)
                        sum += (double)af[kk * lda + i] * bf[kk * bStride];
                }
                else
                {
                    var aRow = af.Slice(i * lda, k);
                    if (transB)
                    {
                        // b stride = 1, contiguous K-vec
                        for (int kk = 0; kk < k; kk++) sum += (double)aRow[kk] * bf[kk];
                    }
                    else
                    {
                        for (int kk = 0; kk < k; kk++) sum += (double)aRow[kk] * bf[kk * ldb];
                    }
                }
                cf[i * ldc] = (float)sum;
            }
            return;
        }
        if (typeof(T) == typeof(double))
        {
            var ad = MemoryMarshal.Cast<T, double>(a);
            var bd = MemoryMarshal.Cast<T, double>(b);
            var cd = MemoryMarshal.Cast<T, double>(c);
            int bStride = transB ? 1 : ldb;

#if NETCOREAPP
            // AVX2 FP64 mirror — 4 doubles per Vector256<double>.
            bool aRowContiguous = !transA;
            bool bIsContiguousVector = bStride == 1;
            if (aRowContiguous && bIsContiguousVector
                && Avx2.IsSupported && Fma.IsSupported && k >= 4)
            {
                int kFull = k & ~3;
                fixed (double* aPtr = ad)
                fixed (double* bPtr = bd)
                fixed (double* cPtr = cd)
                {
                    for (int i = 0; i < m; i++)
                    {
                        var acc = Vector256<double>.Zero;
                        double* aRow = aPtr + (long)i * lda;
                        for (int kk = 0; kk < kFull; kk += 4)
                        {
                            var av = Avx.LoadVector256(aRow + kk);
                            var bv = Avx.LoadVector256(bPtr + kk);
                            acc = Fma.MultiplyAdd(av, bv, acc);
                        }
                        // Horizontal reduce 4-lane Vector256<double>.
                        var hi128 = Avx.ExtractVector128(acc, 1);
                        var lo128 = acc.GetLower();
                        var sum128 = Sse2.Add(hi128, lo128);
                        // Horizontal-add 2-lane SSE2 register.
                        sum128 = Sse3.HorizontalAdd(sum128, sum128);
                        double sum = sum128.ToScalar();
                        for (int kk = kFull; kk < k; kk++)
                            sum += aRow[kk] * bPtr[kk];
                        cPtr[(long)i * ldc] = sum;
                    }
                }
                return;
            }
#endif

            for (int i = 0; i < m; i++)
            {
                double sum = 0;
                if (transA)
                {
                    for (int kk = 0; kk < k; kk++)
                        sum += ad[kk * lda + i] * bd[kk * bStride];
                }
                else
                {
                    var aRow = ad.Slice(i * lda, k);
                    if (transB)
                    {
                        for (int kk = 0; kk < k; kk++) sum += aRow[kk] * bd[kk];
                    }
                    else
                    {
                        for (int kk = 0; kk < k; kk++) sum += aRow[kk] * bd[kk * ldb];
                    }
                }
                cd[i * ldc] = sum;
            }
            return;
        }
        throw new NotSupportedException($"GemvKernel does not support T={typeof(T).Name}.");
    }
}
