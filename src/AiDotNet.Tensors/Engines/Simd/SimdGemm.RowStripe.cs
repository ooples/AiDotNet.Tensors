// Copyright (c) AiDotNet. All rights reserved.
//
// Unpacked small-row GEMM with a fused bias + ReLU epilogue (#436).
//
// A stripe of a few rows times a row-major weight, read in place. A whole-network dispatch (MlpForward's row stripes)
// runs every layer of a few rows on one worker; the packed kernels would repack the full weight for every stripe and
// every call (a 784x512 weight is 400K floats per pack), which cost more than the arithmetic. This kernel streams B's
// rows straight from cache instead: a 4-row x 16-column register tile, one broadcast of A per row and two loads of B
// per k step. Each output element is summed by one thread in k order, so the result does not depend on how rows are
// split across threads.

using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines.Simd;

internal static partial class SimdGemm
{
    /// <summary>True when <see cref="RowStripeGemmBiasAct"/> runs its vector kernel (AVX2 and FMA).</summary>
#if NET5_0_OR_GREATER
    internal static bool RowStripeKernelSupported => Avx2.IsSupported && Fma.IsSupported;
#else
    internal static bool RowStripeKernelSupported => false;
#endif

    /// <summary>
    /// <c>C[m, n] = act(A[m, k] · B[k, n] + bias)</c>, row-major with leading dimensions <paramref name="lda"/>,
    /// <paramref name="ldb"/> and <paramref name="ldc"/>. <paramref name="bias"/> may be null; <paramref name="relu"/>
    /// applies max(0, ·). B is read in place (no packing), so the call suits a few rows against a weight that stays in
    /// cache across calls. Callers check <see cref="RowStripeKernelSupported"/>.
    /// </summary>
    [MethodImpl(Hot)]
    internal static unsafe void RowStripeGemmBiasAct(
        float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int k, int n, float* bias, bool relu)
    {
#if NET5_0_OR_GREATER
        int r = 0;
        for (; r + 4 <= m; r += 4)
            RowStripeRows4(a + (long)r * lda, lda, b, ldb, c + (long)r * ldc, ldc, k, n, bias, relu);
        for (; r < m; r++)
            RowStripeRow1(a + (long)r * lda, b, ldb, c + (long)r * ldc, k, n, bias, relu);
#else
        throw new PlatformNotSupportedException("RowStripeGemmBiasAct needs AVX2 and FMA.");
#endif
    }

#if NET5_0_OR_GREATER
    [MethodImpl(Hot)]
    private static unsafe void RowStripeRows4(
        float* a, int lda, float* b, int ldb, float* c, int ldc, int k, int n, float* bias, bool relu)
    {
        float* a0 = a, a1 = a + lda, a2 = a + 2 * lda, a3 = a + 3 * lda;
        var zero = Vector256<float>.Zero;
        int j = 0;
        for (; j + 16 <= n; j += 16)
        {
            Vector256<float> c00 = zero, c01 = zero, c10 = zero, c11 = zero;
            Vector256<float> c20 = zero, c21 = zero, c30 = zero, c31 = zero;
            float* bp = b + j;
            for (int p = 0; p < k; p++, bp += ldb)
            {
                var b0 = Avx.LoadVector256(bp);
                var b1 = Avx.LoadVector256(bp + 8);
                var x = Vector256.Create(a0[p]);
                c00 = Fma.MultiplyAdd(x, b0, c00); c01 = Fma.MultiplyAdd(x, b1, c01);
                x = Vector256.Create(a1[p]);
                c10 = Fma.MultiplyAdd(x, b0, c10); c11 = Fma.MultiplyAdd(x, b1, c11);
                x = Vector256.Create(a2[p]);
                c20 = Fma.MultiplyAdd(x, b0, c20); c21 = Fma.MultiplyAdd(x, b1, c21);
                x = Vector256.Create(a3[p]);
                c30 = Fma.MultiplyAdd(x, b0, c30); c31 = Fma.MultiplyAdd(x, b1, c31);
            }
            if (bias is not null)
            {
                var bv0 = Avx.LoadVector256(bias + j);
                var bv1 = Avx.LoadVector256(bias + j + 8);
                c00 = Avx.Add(c00, bv0); c01 = Avx.Add(c01, bv1); c10 = Avx.Add(c10, bv0); c11 = Avx.Add(c11, bv1);
                c20 = Avx.Add(c20, bv0); c21 = Avx.Add(c21, bv1); c30 = Avx.Add(c30, bv0); c31 = Avx.Add(c31, bv1);
            }
            if (relu)
            {
                c00 = Avx.Max(c00, zero); c01 = Avx.Max(c01, zero); c10 = Avx.Max(c10, zero); c11 = Avx.Max(c11, zero);
                c20 = Avx.Max(c20, zero); c21 = Avx.Max(c21, zero); c30 = Avx.Max(c30, zero); c31 = Avx.Max(c31, zero);
            }
            Avx.Store(c + j, c00); Avx.Store(c + j + 8, c01);
            Avx.Store(c + ldc + j, c10); Avx.Store(c + ldc + j + 8, c11);
            Avx.Store(c + 2 * ldc + j, c20); Avx.Store(c + 2 * ldc + j + 8, c21);
            Avx.Store(c + 3 * ldc + j, c30); Avx.Store(c + 3 * ldc + j + 8, c31);
        }
        for (; j + 8 <= n; j += 8)
        {
            Vector256<float> c0 = zero, c1 = zero, c2 = zero, c3 = zero;
            float* bp = b + j;
            for (int p = 0; p < k; p++, bp += ldb)
            {
                var b0 = Avx.LoadVector256(bp);
                c0 = Fma.MultiplyAdd(Vector256.Create(a0[p]), b0, c0);
                c1 = Fma.MultiplyAdd(Vector256.Create(a1[p]), b0, c1);
                c2 = Fma.MultiplyAdd(Vector256.Create(a2[p]), b0, c2);
                c3 = Fma.MultiplyAdd(Vector256.Create(a3[p]), b0, c3);
            }
            if (bias is not null)
            {
                var bv = Avx.LoadVector256(bias + j);
                c0 = Avx.Add(c0, bv); c1 = Avx.Add(c1, bv); c2 = Avx.Add(c2, bv); c3 = Avx.Add(c3, bv);
            }
            if (relu)
            {
                c0 = Avx.Max(c0, zero); c1 = Avx.Max(c1, zero); c2 = Avx.Max(c2, zero); c3 = Avx.Max(c3, zero);
            }
            Avx.Store(c + j, c0); Avx.Store(c + ldc + j, c1); Avx.Store(c + 2 * ldc + j, c2); Avx.Store(c + 3 * ldc + j, c3);
        }
        if (j < n)
        {
            RowStripeScalarTail(a0, b, ldb, c, k, j, n, bias, relu);
            RowStripeScalarTail(a1, b, ldb, c + ldc, k, j, n, bias, relu);
            RowStripeScalarTail(a2, b, ldb, c + 2 * ldc, k, j, n, bias, relu);
            RowStripeScalarTail(a3, b, ldb, c + 3 * ldc, k, j, n, bias, relu);
        }
    }

    [MethodImpl(Hot)]
    private static unsafe void RowStripeRow1(float* a, float* b, int ldb, float* c, int k, int n, float* bias, bool relu)
    {
        var zero = Vector256<float>.Zero;
        int j = 0;
        for (; j + 16 <= n; j += 16)
        {
            Vector256<float> c0 = zero, c1 = zero;
            float* bp = b + j;
            for (int p = 0; p < k; p++, bp += ldb)
            {
                var x = Vector256.Create(a[p]);
                c0 = Fma.MultiplyAdd(x, Avx.LoadVector256(bp), c0);
                c1 = Fma.MultiplyAdd(x, Avx.LoadVector256(bp + 8), c1);
            }
            if (bias is not null)
            {
                c0 = Avx.Add(c0, Avx.LoadVector256(bias + j));
                c1 = Avx.Add(c1, Avx.LoadVector256(bias + j + 8));
            }
            if (relu) { c0 = Avx.Max(c0, zero); c1 = Avx.Max(c1, zero); }
            Avx.Store(c + j, c0); Avx.Store(c + j + 8, c1);
        }
        for (; j + 8 <= n; j += 8)
        {
            var c0 = zero;
            float* bp = b + j;
            for (int p = 0; p < k; p++, bp += ldb)
                c0 = Fma.MultiplyAdd(Vector256.Create(a[p]), Avx.LoadVector256(bp), c0);
            if (bias is not null) c0 = Avx.Add(c0, Avx.LoadVector256(bias + j));
            if (relu) c0 = Avx.Max(c0, zero);
            Avx.Store(c + j, c0);
        }
        if (j < n) RowStripeScalarTail(a, b, ldb, c, k, j, n, bias, relu);
    }

    // Columns [j0, n) of one row, scalar: the same k-order sum as the vector lanes.
    [MethodImpl(Hot)]
    private static unsafe void RowStripeScalarTail(
        float* a, float* b, int ldb, float* c, int k, int j0, int n, float* bias, bool relu)
    {
        for (int j = j0; j < n; j++)
        {
            float sum = 0f;
            float* bp = b + j;
            for (int p = 0; p < k; p++, bp += ldb)
                sum = MathF.FusedMultiplyAdd(a[p], *bp, sum);
            if (bias is not null) sum += bias[j];
            c[j] = relu && sum < 0f ? 0f : sum;
        }
    }
#endif
}
