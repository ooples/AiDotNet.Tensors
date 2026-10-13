using System;
using System.Buffers;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines.BlasManaged;

public static partial class BlasManaged
{
    /// <summary>
    /// Symmetric matrix multiply. Side.Left: C = α·A·B + β·C (A is m×m symmetric).
    /// Side.Right: C = α·B·A + β·C (A is n×n symmetric). A is stored in the
    /// <paramref name="uplo"/> triangle. Drop-in for cblas_ssymm/cblas_dsymm.
    /// </summary>
    public static void Symm<T>(
        Side side, Uplo uplo,
        int m, int n, T alpha,
        ReadOnlySpan<T> a, int lda,
        ReadOnlySpan<T> b, int ldb, T beta,
        Span<T> c, int ldc,
        in BlasOptions<T> options = default) where T : unmanaged
    {
        if (m < 0) throw new ArgumentOutOfRangeException(nameof(m), "m must be non-negative.");
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "n must be non-negative.");
        if (m == 0 || n == 0) return;
        if (side is not Side.Left and not Side.Right)
            throw new ArgumentOutOfRangeException(nameof(side));
        if (uplo is not Uplo.Lower and not Uplo.Upper)
            throw new ArgumentOutOfRangeException(nameof(uplo));

        var ops = MathHelper.GetNumericOperations<T>();
        int s = side == Side.Left ? m : n;   // dimension of square symmetric A
        if (lda < s) throw new ArgumentOutOfRangeException(nameof(lda), "lda must be >= the symmetric matrix dimension.");
        if (ldb < n) throw new ArgumentOutOfRangeException(nameof(ldb), "ldb must be >= n.");
        if (ldc < n) throw new ArgumentOutOfRangeException(nameof(ldc), "ldc must be >= n.");
        if (a.Length < (long)(s - 1) * lda + s)
            throw new ArgumentException("Span 'a' is too short for the symmetric matrix storage.", nameof(a));
        if (b.Length < (long)(m - 1) * ldb + n)
            throw new ArgumentException("Span 'b' is too short.", nameof(b));
        if (c.Length < (long)(m - 1) * ldc + n)
            throw new ArgumentException("Span 'c' is too short.", nameof(c));

        // Materialize full symmetric A (s×s) from the uplo triangle (mirror).
        // #517: the product reads the stored triangle and its mirror. A narrow panel (few columns of B on the left, few
        // rows on the right) against a matrix too large for cache reads the stored triangle once, in place; otherwise the
        // matrix is materialized (tiled, into pooled memory) and one GEMM runs.
        T[] result = ArrayPool<T>.Shared.Rent(m * n);
        T[]? full = null;
        try
        {
            var resultSpan = result.AsSpan(0, m * n);
            int panel = side == Side.Left ? n : m;
            bool direct = panel <= SymmDirectMaxPanel && (long)s * s * Unsafe.SizeOf<T>() >= SymmDirectMinMatrixBytes;
            if (direct && typeof(T) == typeof(double))
                SymmDirect(side, uplo, s, panel, MemoryMarshal.Cast<T, double>(a), lda, MemoryMarshal.Cast<T, double>(b), ldb,
                    MemoryMarshal.Cast<T, double>(resultSpan), n);
            else if (direct && typeof(T) == typeof(float))
                SymmDirect(side, uplo, s, panel, MemoryMarshal.Cast<T, float>(a), lda, MemoryMarshal.Cast<T, float>(b), ldb,
                    MemoryMarshal.Cast<T, float>(resultSpan), n);
            else
            {
                full = ArrayPool<T>.Shared.Rent(s * s);
                var fullSpan = full.AsSpan(0, s * s);
                MirrorTriangle(uplo, s, a, lda, fullSpan);
                // result = A·B (Left, m×n) or B·A (Right, m×n) via the existing GEMM core.
                var gemmOpts = new BlasOptions<T> { NumThreads = options.NumThreads, Mode = options.Mode };
                if (side == Side.Left)
                    Gemm<T>(fullSpan, s, false, b, ldb, false, resultSpan, n, m, n, s, gemmOpts);
                else
                    Gemm<T>(b, ldb, false, fullSpan, s, false, resultSpan, n, m, n, s, gemmOpts);
            }
            Epilogue(m, n, alpha, resultSpan, beta, c, ldc, ops);
        }
        finally
        {
            if (full is not null) ArrayPool<T>.Shared.Return(full);
            ArrayPool<T>.Shared.Return(result);
        }
    }

    private static void Epilogue<T>(int m, int n, T alpha, Span<T> result, T beta, Span<T> c, int ldc, INumericOperations<T> ops)
        where T : unmanaged
    {
        // C = α·result + β·C. float/double take a typed, JIT-vectorizable per-row AXPBY;
        // the generic INumericOperations path is per-element interface-dispatched + boxed.
        if (typeof(T) == typeof(float))
            SymmEpilogueFloat(m, n, (float)(object)alpha, MemoryMarshal.Cast<T, float>(result),
                (float)(object)beta, MemoryMarshal.Cast<T, float>(c), ldc);
        else if (typeof(T) == typeof(double))
            SymmEpilogueDouble(m, n, (double)(object)alpha, MemoryMarshal.Cast<T, double>(result),
                (double)(object)beta, MemoryMarshal.Cast<T, double>(c), ldc);
        else
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    int ci = i * ldc + j;
                    c[ci] = ops.Add(ops.Multiply(alpha, result[i * n + j]), ops.Multiply(beta, c[ci]));
                }
    }

    /// <summary>
    /// Widest panel (columns of B on the left, rows of B on the right) that reads the stored triangle in place instead of
    /// materializing the matrix. Measured on a 16-core desktop at s = 1024: the direct form was 1.4-3.6x faster at 8 or
    /// fewer columns and slower at 16, since it does s^2 * panel work without GEMM's blocking.
    /// </summary>
    internal const int SymmDirectMaxPanel = 8;

    /// <summary>
    /// Smallest symmetric matrix, in bytes, for which the direct form pays: one that fits in cache materializes cheaply
    /// (at s = 256 doubles, 512 KB, the two forms tied or the GEMM won), while an 8 MB one cost the copy 2-4 ms.
    /// </summary>
    internal const long SymmDirectMinMatrixBytes = 1L << 20;

    /// <summary>The full symmetric matrix from its stored triangle, copied in 32 x 32 tiles so the mirrored reads stay in cache.</summary>
    private static void MirrorTriangle<T>(Uplo uplo, int s, ReadOnlySpan<T> a, int lda, Span<T> full) where T : unmanaged
    {
        const int Tile = 32;
        for (int ib = 0; ib < s; ib += Tile)
            for (int jb = 0; jb < s; jb += Tile)
            {
                int iEnd = Math.Min(ib + Tile, s), jEnd = Math.Min(jb + Tile, s);
                for (int i = ib; i < iEnd; i++)
                    for (int j = jb; j < jEnd; j++)
                    {
                        bool stored = uplo == Uplo.Lower ? j <= i : j >= i;
                        full[i * s + j] = stored ? a[i * lda + j] : a[j * lda + i];
                    }
            }
    }

    /// <summary>
    /// result = A·B (left, s x panel) or B·A (right, panel x s) reading each stored element of A once, in place. An
    /// off-diagonal A[i, k] = A[k, i] = v adds v·B[k, :] to row i and v·B[i, :] to row k on the left; on the right it adds
    /// v·B[r, i] to result[r, k] and v·B[r, k] to result[r, i]. Single-threaded, so bit-identical at any thread count.
    /// </summary>
    private static void SymmDirect(Side side, Uplo uplo, int s, int panel, ReadOnlySpan<double> a, int lda,
        ReadOnlySpan<double> b, int ldb, Span<double> result, int ldr)
    {
        result.Slice(0, (side == Side.Left ? s : panel) * ldr).Clear();
        for (int i = 0; i < s; i++)
        {
            int kStart = uplo == Uplo.Lower ? 0 : i + 1, kEnd = uplo == Uplo.Lower ? i : s;
            double d = a[i * lda + i];
            if (side == Side.Left)
            {
                var outI = result.Slice(i * ldr, panel);
                var inI = b.Slice(i * ldb, panel);
                for (int k = kStart; k < kEnd; k++)
                {
                    double v = a[i * lda + k];
                    var inK = b.Slice(k * ldb, panel);
                    var outK = result.Slice(k * ldr, panel);
                    int j = 0;
#if NET5_0_OR_GREATER
                    if (System.Numerics.Vector.IsHardwareAccelerated && panel >= System.Numerics.Vector<double>.Count)
                    {
                        var vv = new System.Numerics.Vector<double>(v);
                        int w = System.Numerics.Vector<double>.Count;
                        for (; j <= panel - w; j += w)
                        {
                            (new System.Numerics.Vector<double>(outI.Slice(j)) + vv * new System.Numerics.Vector<double>(inK.Slice(j))).CopyTo(outI.Slice(j));
                            (new System.Numerics.Vector<double>(outK.Slice(j)) + vv * new System.Numerics.Vector<double>(inI.Slice(j))).CopyTo(outK.Slice(j));
                        }
                    }
#endif
                    for (; j < panel; j++) { outI[j] += v * inK[j]; outK[j] += v * inI[j]; }
                }
                for (int j = 0; j < panel; j++) outI[j] += d * inI[j];
            }
            else
            {
                for (int r = 0; r < panel; r++)
                {
                    var outR = result.Slice(r * ldr, s);
                    var inR = b.Slice(r * ldb, s);
                    double bi = inR[i], acc = 0;
                    for (int k = kStart; k < kEnd; k++)
                    {
                        double v = a[i * lda + k];
                        outR[k] += v * bi;
                        acc += v * inR[k];
                    }
                    outR[i] += acc + d * bi;
                }
            }
        }
    }

    private static void SymmDirect(Side side, Uplo uplo, int s, int panel, ReadOnlySpan<float> a, int lda,
        ReadOnlySpan<float> b, int ldb, Span<float> result, int ldr)
    {
        result.Slice(0, (side == Side.Left ? s : panel) * ldr).Clear();
        for (int i = 0; i < s; i++)
        {
            int kStart = uplo == Uplo.Lower ? 0 : i + 1, kEnd = uplo == Uplo.Lower ? i : s;
            float d = a[i * lda + i];
            if (side == Side.Left)
            {
                var outI = result.Slice(i * ldr, panel);
                var inI = b.Slice(i * ldb, panel);
                for (int k = kStart; k < kEnd; k++)
                {
                    float v = a[i * lda + k];
                    var inK = b.Slice(k * ldb, panel);
                    var outK = result.Slice(k * ldr, panel);
                    int j = 0;
#if NET5_0_OR_GREATER
                    if (System.Numerics.Vector.IsHardwareAccelerated && panel >= System.Numerics.Vector<float>.Count)
                    {
                        var vv = new System.Numerics.Vector<float>(v);
                        int w = System.Numerics.Vector<float>.Count;
                        for (; j <= panel - w; j += w)
                        {
                            (new System.Numerics.Vector<float>(outI.Slice(j)) + vv * new System.Numerics.Vector<float>(inK.Slice(j))).CopyTo(outI.Slice(j));
                            (new System.Numerics.Vector<float>(outK.Slice(j)) + vv * new System.Numerics.Vector<float>(inI.Slice(j))).CopyTo(outK.Slice(j));
                        }
                    }
#endif
                    for (; j < panel; j++) { outI[j] += v * inK[j]; outK[j] += v * inI[j]; }
                }
                for (int j = 0; j < panel; j++) outI[j] += d * inI[j];
            }
            else
            {
                for (int r = 0; r < panel; r++)
                {
                    var outR = result.Slice(r * ldr, s);
                    var inR = b.Slice(r * ldb, s);
                    float bi = inR[i], acc = 0;
                    for (int k = kStart; k < kEnd; k++)
                    {
                        float v = a[i * lda + k];
                        outR[k] += v * bi;
                        acc += v * inR[k];
                    }
                    outR[i] += acc + d * bi;
                }
            }
        }
    }

    [MethodImpl(Hot)]
    private static void SymmEpilogueFloat(
        int m, int n, float alpha, ReadOnlySpan<float> result, float beta, Span<float> c, int ldc)
    {
        for (int i = 0; i < m; i++)
        {
            var r = result.Slice(i * n, n);
            var cc = c.Slice(i * ldc, n);
            for (int j = 0; j < n; j++)
                cc[j] = alpha * r[j] + beta * cc[j];
        }
    }

    [MethodImpl(Hot)]
    private static void SymmEpilogueDouble(
        int m, int n, double alpha, ReadOnlySpan<double> result, double beta, Span<double> c, int ldc)
    {
        for (int i = 0; i < m; i++)
        {
            var r = result.Slice(i * n, n);
            var cc = c.Slice(i * ldc, n);
            for (int j = 0; j < n; j++)
                cc[j] = alpha * r[j] + beta * cc[j];
        }
    }
}
