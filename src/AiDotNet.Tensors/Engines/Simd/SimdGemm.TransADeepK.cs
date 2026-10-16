// Copyright (c) AiDotNet. All rights reserved.
//
// C = Aᵀ·B for a deep reduction and a small output: a layer's weight gradient dW[in, out] = Xᵀ·dY summed over every
// (batch, position) row.
//
// The small-K panel route split such a GEMM over row chunks, transposed each chunk of A with a scalar strided loop,
// ran the panel into a partial per chunk and summed the partials. Here the rows are split into fixed slices small
// enough to stay in one core's L2 (both operands' rows of a slice), and each slice runs every 6x16 tile of C over its
// rows in registers: row r of A holds a tile's 6 input values contiguously (A is stored [k, m]), so they are broadcast
// straight from memory, and row r of B supplies its 16 outputs -- nothing is transposed. The slices' partials are then
// summed in slice order. The slice count is a function of k alone, and every partial element is one fused multiply-add
// chain over the slice's rows in order (the scalar edges use MathF.FusedMultiplyAdd, the same rounding), so the result
// does not depend on the thread count.

#if NET5_0_OR_GREATER
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>Shallowest reduction routed to <see cref="TrySgemmTransADeepK"/>: enough rows for several L2-sized
    /// slices.</summary>
    internal const int TransADeepKMinK = 512;

    /// <summary>Largest m and n routed to <see cref="TrySgemmTransADeepK"/>: every slice keeps a whole [m, n] partial,
    /// and past this the output alone fills the pool and the packed routes take over.</summary>
    internal const int TransADeepKMaxOutput = 512;

    // Rows per slice: 128 rows of [64 + 128] floats is 96 KB, inside a core's L2 alongside its partial.
    private const int TransADeepKSliceRows = 128;
    private const int TransADeepKMaxSlices = 32;

    /// <summary>
    /// <c>C[m,n] = Aᵀ·B</c> (overwrite) with A stored [k, m] (row stride <paramref name="lda"/>), B [k, n] (row stride
    /// <paramref name="ldb"/>) and C [m, n] contiguous, for k &gt;= <see cref="TransADeepKMinK"/> and m, n up to
    /// <see cref="TransADeepKMaxOutput"/>. Returns false, having written nothing, outside that band or without AVX2+FMA.
    /// </summary>
    [MethodImpl(Hot)]
    internal static unsafe bool TrySgemmTransADeepK(float[] a, int lda, float[] b, int ldb, float[] c, int m, int k, int n)
    {
        if (!Fma.IsSupported || k < TransADeepKMinK || m < 1 || n < 8
            || m > TransADeepKMaxOutput || n > TransADeepKMaxOutput || lda < m || ldb < n) return false;
        if ((long)(k - 1) * lda + m > a.Length || (long)(k - 1) * ldb + n > b.Length || (long)m * n > c.Length)
            throw new ArgumentException("An operand is shorter than the GEMM it describes.");

        int slices = Math.Min(TransADeepKMaxSlices, (k + TransADeepKSliceRows - 1) / TransADeepKSliceRows);
        int rowsPerSlice = (k + slices - 1) / slices;
        slices = (k + rowsPerSlice - 1) / rowsPerSlice;
        int outLen = m * n;
        var pool = ArrayPool<float>.Shared;
        var partials = pool.Rent(slices * outLen);
        try
        {
            fixed (float* pa = a, pb = b, pc = c, pp = partials)
            {
                IntPtr ia = (IntPtr)pa, ib = (IntPtr)pb, ic = (IntPtr)pc, ip = (IntPtr)pp;
                CpuParallelSettings.ParallelForOrSerial(0, slices, (long)m * n * k, [MethodImpl(Hot)] (s) =>
                {
                    int r0 = s * rowsPerSlice, r1 = Math.Min(k, r0 + rowsPerSlice);
                    float* part = (float*)ip + (long)s * outLen;
                    for (int j0 = 0; j0 < n; j0 += 16)
                        for (int i0 = 0; i0 < m; i0 += 6)
                            TransADeepKTile((float*)ia, lda, (float*)ib, ldb, part, n, r0, r1,
                                i0, Math.Min(6, m - i0), j0, Math.Min(16, n - j0));
                }, deterministicSafe: true);

                // c[i, :] = partial[0][i, :] + partial[1][i, :] + ... in slice order, rows across the pool.
                CpuParallelSettings.ParallelForOrSerial(0, m, (long)m * n * slices, [MethodImpl(Hot)] (i) =>
                {
                    float* dst = (float*)ic + (long)i * n;
                    float* src = (float*)ip + (long)i * n;
                    int j = 0;
                    for (; j + 8 <= n; j += 8)
                    {
                        var acc = Avx.LoadVector256(src + j);
                        for (int s = 1; s < slices; s++) acc = Avx.Add(acc, Avx.LoadVector256(src + (long)s * outLen + j));
                        Avx.Store(dst + j, acc);
                    }
                    for (; j < n; j++)
                    {
                        float acc = src[j];
                        for (int s = 1; s < slices; s++) acc += src[(long)s * outLen + j];
                        dst[j] = acc;
                    }
                }, deterministicSafe: true);
            }
        }
        finally { pool.Return(partials); }
        return true;
    }

    /// <summary>One tile of a slice's partial: rows [i0, i0+mr) by columns [j0, j0+nr), each a fused multiply-add
    /// chain over r in [r0, r1).</summary>
    [MethodImpl(Hot)]
    private static unsafe void TransADeepKTile(float* a, int lda, float* b, int ldb, float* c, int ldc, int r0, int r1,
        int i0, int mr, int j0, int nr)
    {
        if (mr == 6 && nr == 16)
        {
            Vector256<float> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default,
                c30 = default, c31 = default, c40 = default, c41 = default, c50 = default, c51 = default;
            float* ar = a + (long)r0 * lda + i0;
            float* br = b + (long)r0 * ldb + j0;
            for (int r = r0; r < r1; r++, ar += lda, br += ldb)
            {
                var b0 = Avx.LoadVector256(br);
                var b1 = Avx.LoadVector256(br + 8);
                var x = Vector256.Create(ar[0]); c00 = Fma.MultiplyAdd(x, b0, c00); c01 = Fma.MultiplyAdd(x, b1, c01);
                x = Vector256.Create(ar[1]); c10 = Fma.MultiplyAdd(x, b0, c10); c11 = Fma.MultiplyAdd(x, b1, c11);
                x = Vector256.Create(ar[2]); c20 = Fma.MultiplyAdd(x, b0, c20); c21 = Fma.MultiplyAdd(x, b1, c21);
                x = Vector256.Create(ar[3]); c30 = Fma.MultiplyAdd(x, b0, c30); c31 = Fma.MultiplyAdd(x, b1, c31);
                x = Vector256.Create(ar[4]); c40 = Fma.MultiplyAdd(x, b0, c40); c41 = Fma.MultiplyAdd(x, b1, c41);
                x = Vector256.Create(ar[5]); c50 = Fma.MultiplyAdd(x, b0, c50); c51 = Fma.MultiplyAdd(x, b1, c51);
            }
            float* o = c + (long)i0 * ldc + j0;
            Avx.Store(o, c00); Avx.Store(o + 8, c01); o += ldc;
            Avx.Store(o, c10); Avx.Store(o + 8, c11); o += ldc;
            Avx.Store(o, c20); Avx.Store(o + 8, c21); o += ldc;
            Avx.Store(o, c30); Avx.Store(o + 8, c31); o += ldc;
            Avx.Store(o, c40); Avx.Store(o + 8, c41); o += ldc;
            Avx.Store(o, c50); Avx.Store(o + 8, c51);
            return;
        }

        // Edge tile: one output row at a time, 8-wide vectors then scalars, the same FMA chain per element.
        for (int ii = 0; ii < mr; ii++)
        {
            float* arow = a + (long)r0 * lda + i0 + ii;
            float* o = c + (long)(i0 + ii) * ldc + j0;
            int jj = 0;
            for (; jj + 8 <= nr; jj += 8)
            {
                Vector256<float> acc = default;
                float* ar = arow, br = b + (long)r0 * ldb + j0 + jj;
                for (int r = r0; r < r1; r++, ar += lda, br += ldb)
                    acc = Fma.MultiplyAdd(Vector256.Create(*ar), Avx.LoadVector256(br), acc);
                Avx.Store(o + jj, acc);
            }
            for (; jj < nr; jj++)
            {
                float acc = 0f;
                float* ar = arow, br = b + (long)r0 * ldb + j0 + jj;
                for (int r = r0; r < r1; r++, ar += lda, br += ldb)
                    acc = MathF.FusedMultiplyAdd(*ar, *br, acc);
                o[jj] = acc;
            }
        }
    }
}
#endif
