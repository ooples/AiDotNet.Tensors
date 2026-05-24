using System;
using System.Runtime.InteropServices;

#if !NET471
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

    private static void RunOuterProduct<T>(
        ReadOnlySpan<T> a, int lda, bool transA,
        ReadOnlySpan<T> b, int ldb, bool transB,
        Span<T> c, int ldc,
        int m, int n) where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            var af = MemoryMarshal.Cast<T, float>(a);
            var bf = MemoryMarshal.Cast<T, float>(b);
            var cf = MemoryMarshal.Cast<T, float>(c);
            for (int i = 0; i < m; i++)
            {
                float ai = transA ? af[i] : af[i * lda];  // K=1 ⇒ stride collapses
                for (int j = 0; j < n; j++)
                {
                    float bj = transB ? bf[j * ldb] : bf[j];  // K=1 ⇒ same
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
            for (int i = 0; i < m; i++)
            {
                double ai = transA ? ad[i] : ad[i * lda];
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

#if !NET471
            // AVX2 fast path for transB=false (B is [K, N] row-major).
            if (!transB && Avx2.IsSupported && Fma.IsSupported && n >= 8)
            {
                int nFull = n & ~7;
                fixed (float* aPtr = af)
                fixed (float* bPtr = bf)
                fixed (float* cPtr = cf)
                {
                    Span<Vector256<float>> accs = stackalloc Vector256<float>[1];
                    for (int j = 0; j < nFull; j += 8)
                    {
                        var acc = Vector256<float>.Zero;
                        for (int kk = 0; kk < k; kk++)
                        {
                            var av = Vector256.Create(aPtr[kk * aStride]);
                            var bv = Avx.LoadVector256(bPtr + kk * ldb + j);
                            acc = Fma.MultiplyAdd(av, bv, acc);
                        }
                        Avx.Store(cPtr + j, acc);
                    }
                    // N-tail scalar.
                    for (int j = nFull; j < n; j++)
                    {
                        float sum = 0;
                        for (int kk = 0; kk < k; kk++) sum += aPtr[kk * aStride] * bPtr[kk * ldb + j];
                        cPtr[j] = sum;
                    }
                }
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

#if !NET471
            if (!transB && Avx2.IsSupported && Fma.IsSupported && n >= 4)
            {
                int nFull = n & ~3;
                fixed (double* aPtr = ad)
                fixed (double* bPtr = bd)
                fixed (double* cPtr = cd)
                {
                    for (int j = 0; j < nFull; j += 4)
                    {
                        var acc = Vector256<double>.Zero;
                        for (int kk = 0; kk < k; kk++)
                        {
                            var av = Vector256.Create(aPtr[kk * aStride]);
                            var bv = Avx.LoadVector256(bPtr + kk * ldb + j);
                            acc = Fma.MultiplyAdd(av, bv, acc);
                        }
                        Avx.Store(cPtr + j, acc);
                    }
                    for (int j = nFull; j < n; j++)
                    {
                        double sum = 0;
                        for (int kk = 0; kk < k; kk++) sum += aPtr[kk * aStride] * bPtr[kk * ldb + j];
                        cPtr[j] = sum;
                    }
                }
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
            // b is logically [K, 1]: transB=false → bf[kk * ldb]; transB=true → bf[kk]
            int bStride = transB ? 1 : ldb;
            // Each output cell c[i, 0] is a dot product of a's row i with b's column.
            //   transA=false: a row i starts at af[i * lda]
            //   transA=true:  a row i (logical) is column i in storage — af[kk * lda + i]
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
