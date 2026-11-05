using System;
using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// AVX (no AVX2, no FMA) FP32 streaming kernel, the streaming twin of <see cref="AvxFp32_8x8"/>:
/// <c>C[m, n] += A[m, k] · B[k, n]</c> without packing.
/// </summary>
/// <remarks>
/// Every element is computed as <c>c = c + a * b</c> in FP32, product rounded then sum rounded, over k in ascending
/// order: eight columns per <see cref="Vector256{T}"/> and the remainder, and every transposed layout, one element at a
/// time with the same two roundings. That is <see cref="AvxFp32_8x8"/>'s per-lane arithmetic, so a row computes to
/// the same bits whether the GEMM puts it in a microkernel tile or here, which is what deterministic mode promises
/// when the parallel split moves the interior/tail boundary.
/// </remarks>
internal static class AvxStreaming
{
    /// <summary>Columns per vector step.</summary>
    internal const int Fp32ColumnTileWidth = 8;

#if NET5_0_OR_GREATER
    /// <summary>Same gate as <see cref="AvxFp32_8x8"/>, whose bits this kernel matches.</summary>
    public static bool IsSupported => AvxFp32_8x8.IsSupported;

    /// <summary>FP32 streaming GEMM; see the type remarks for the arithmetic contract.</summary>
    /// <param name="a">A, [m, k] or, when <paramref name="transA"/>, [k, m].</param>
    /// <param name="lda">Leading dimension of A.</param>
    /// <param name="transA">Whether A is stored transposed.</param>
    /// <param name="b">B, [k, n] or, when <paramref name="transB"/>, [n, k].</param>
    /// <param name="ldb">Leading dimension of B.</param>
    /// <param name="transB">Whether B is stored transposed.</param>
    /// <param name="c">C, [m, n], read and written.</param>
    /// <param name="ldc">Leading dimension of C.</param>
    /// <param name="m">Rows of C.</param>
    /// <param name="n">Columns of C.</param>
    /// <param name="k">Shared dimension.</param>
    [MethodImpl(Hot)]
    public static unsafe void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException("AvxStreaming requires AVX.");

        fixed (float* aPtr = a)
        fixed (float* bPtr = b)
        fixed (float* cPtr = c)
        {
            // With B transposed its columns are ldb apart, so there is nothing contiguous to load: element by element.
            int vectorCols = transB ? 0 : (n / Fp32ColumnTileWidth) * Fp32ColumnTileWidth;
            for (int i = 0; i < m; i++)
            {
                float* cRow = cPtr + (i * ldc);
                for (int j = 0; j < vectorCols; j += Fp32ColumnTileWidth)
                {
                    Vector256<float> acc = Avx.LoadVector256(cRow + j);
                    for (int kk = 0; kk < k; kk++)
                    {
                        float* aElement = transA ? aPtr + (kk * lda) + i : aPtr + (i * lda) + kk;
                        acc = Avx.Add(acc, Avx.Multiply(Avx.BroadcastScalarToVector256(aElement),
                            Avx.LoadVector256(bPtr + (kk * ldb) + j)));
                    }
                    Avx.Store(cRow + j, acc);
                }
                for (int j = vectorCols; j < n; j++)
                {
                    float sum = cRow[j];
                    for (int kk = 0; kk < k; kk++)
                    {
                        float aval = transA ? aPtr[(kk * lda) + i] : aPtr[(i * lda) + kk];
                        float bval = transB ? bPtr[(j * ldb) + kk] : bPtr[(kk * ldb) + j];
                        sum += aval * bval;
                    }
                    cRow[j] = sum;
                }
            }
        }
    }
#else
    /// <summary>False on targets without hardware intrinsics.</summary>
    public static bool IsSupported => false;

    /// <summary>Unreachable when <see cref="IsSupported"/> is false.</summary>
    public static void RunFp32(
        ReadOnlySpan<float> a, int lda, bool transA,
        ReadOnlySpan<float> b, int ldb, bool transB,
        Span<float> c, int ldc,
        int m, int n, int k)
        => throw new PlatformNotSupportedException("AvxStreaming requires net5.0+.");
#endif
}
