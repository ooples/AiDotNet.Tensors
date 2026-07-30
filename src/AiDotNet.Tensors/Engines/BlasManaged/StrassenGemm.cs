#if NET5_0_OR_GREATER
using System;
using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// One-level Strassen for large square FP32 GEMM (#23): 7 sub-multiplies of n/2 blocks instead of 8,
/// trading ~12.5% of the n^3 multiply-FLOPs for ~18 n^2/4 add/sub passes. This attacks the all-core AVX2
/// FMA-throttle ceiling (the binding constraint per the #19 bf16 finding — the GEMM is FMA-bound, not
/// operand-bandwidth-bound) which standard GEMM and MKL both hit. The 7 sub-products reuse the validated
/// <see cref="GotoGemmFp32.RunParallel"/>. Slightly worse accuracy than standard (extra cancellation in the
/// sums) — opt-in, large-even-n only. Whether it actually wins depends on whether the 12.5% FMA saving
/// beats the extra ~0.75 n^2 pack work of 7 sub-GEMMs + the add passes (measured via --ab-strassen).
/// </summary>
internal static class StrassenGemm
{
    /// <summary>dst[h×h, contiguous] = x ± y, where x,y are h×h blocks with row-stride ld (e.g. quadrants).</summary>
    private static unsafe void Combine2(float* x, float* y, int ld, float* dst, int h, bool sub)
    {
        for (int i = 0; i < h; i++)
        {
            float* xr = x + (long)i * ld, yr = y + (long)i * ld, dr = dst + (long)i * h;
            int j = 0;
            if (Avx.IsSupported)
            {
                if (sub) for (; j + 8 <= h; j += 8) Avx.Store(dr + j, Avx.Subtract(Avx.LoadVector256(xr + j), Avx.LoadVector256(yr + j)));
                else for (; j + 8 <= h; j += 8) Avx.Store(dr + j, Avx.Add(Avx.LoadVector256(xr + j), Avx.LoadVector256(yr + j)));
            }
            for (; j < h; j++) dr[j] = sub ? xr[j] - yr[j] : xr[j] + yr[j];
        }
    }

    /// <summary>Write a C quadrant (row-stride ldc) = sum of up to four contiguous h×h M-temps with signs.</summary>
    private static unsafe void WriteQuad(float* c, int ldc, int h,
        float* t0, int s0, float* t1, int s1, float* t2, int s2, float* t3, int s3)
    {
        for (int i = 0; i < h; i++)
        {
            long ro = (long)i * h; float* cr = c + (long)i * ldc;
            float* a0 = t0 + ro, a1 = t1 + ro;
            float* a2 = t2 != null ? t2 + ro : null, a3 = t3 != null ? t3 + ro : null;
            int j = 0;
            if (Avx.IsSupported)
            {
                var vs0 = Vector256.Create((float)s0); var vs1 = Vector256.Create((float)s1);
                var vs2 = Vector256.Create((float)s2); var vs3 = Vector256.Create((float)s3);
                for (; j + 8 <= h; j += 8)
                {
                    var acc = Avx.Add(Avx.Multiply(Avx.LoadVector256(a0 + j), vs0), Avx.Multiply(Avx.LoadVector256(a1 + j), vs1));
                    if (a2 != null) acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(a2 + j), vs2));
                    if (a3 != null) acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(a3 + j), vs3));
                    Avx.Store(cr + j, acc);
                }
            }
            for (; j < h; j++)
            {
                float v = s0 * a0[j] + s1 * a1[j];
                if (a2 != null) v += s2 * a2[j];
                if (a3 != null) v += s3 * a3[j];
                cr[j] = v;
            }
        }
    }

    /// <summary>True if a one-level Strassen is applicable (square, even, large enough to amortize the adds).</summary>
    internal static bool Applies(int m, int n, int k) => m == n && n == k && (n & 1) == 0 && n >= 2048;

    /// <summary>C[n×n] = A·B via one-level Strassen. Caller guarantees <see cref="Applies"/>.</summary>
    internal static unsafe void RunSquare(float* a, int lda, float* b, int ldb, float* c, int ldc, int n)
    {
        int h = n / 2;
        long h2 = (long)h * h;
        float[] pool = ArrayPool<float>.Shared.Rent(checked((int)(h2 * 9)));
        try
        {
            fixed (float* p = pool)
            {
                float* sa = p, sb = p + h2, m1 = p + 2 * h2, m2 = p + 3 * h2, m3 = p + 4 * h2,
                       m4 = p + 5 * h2, m5 = p + 6 * h2, m6 = p + 7 * h2, m7 = p + 8 * h2;
                float* A11 = a, A12 = a + h, A21 = a + (long)h * lda, A22 = a + (long)h * lda + h;
                float* B11 = b, B12 = b + h, B21 = b + (long)h * ldb, B22 = b + (long)h * ldb + h;
                float* C11 = c, C12 = c + h, C21 = c + (long)h * ldc, C22 = c + (long)h * ldc + h;
                var (mc, nc, kc) = GotoGemmFp32.ChooseParallelBlocks(h, h);

                // M1 = (A11+A22)(B11+B22)
                Combine2(A11, A22, lda, sa, h, false); Combine2(B11, B22, ldb, sb, h, false);
                GotoGemmFp32.RunParallel(sa, h, sb, h, m1, h, h, h, h, mc, nc, kc);
                // M2 = (A21+A22) B11
                Combine2(A21, A22, lda, sa, h, false);
                GotoGemmFp32.RunParallel(sa, h, B11, ldb, m2, h, h, h, h, mc, nc, kc);
                // M3 = A11 (B12-B22)
                Combine2(B12, B22, ldb, sb, h, true);
                GotoGemmFp32.RunParallel(A11, lda, sb, h, m3, h, h, h, h, mc, nc, kc);
                // M4 = A22 (B21-B11)
                Combine2(B21, B11, ldb, sb, h, true);
                GotoGemmFp32.RunParallel(A22, lda, sb, h, m4, h, h, h, h, mc, nc, kc);
                // M5 = (A11+A12) B22
                Combine2(A11, A12, lda, sa, h, false);
                GotoGemmFp32.RunParallel(sa, h, B22, ldb, m5, h, h, h, h, mc, nc, kc);
                // M6 = (A21-A11)(B11+B12)
                Combine2(A21, A11, lda, sa, h, true); Combine2(B11, B12, ldb, sb, h, false);
                GotoGemmFp32.RunParallel(sa, h, sb, h, m6, h, h, h, h, mc, nc, kc);
                // M7 = (A12-A22)(B21+B22)
                Combine2(A12, A22, lda, sa, h, true); Combine2(B21, B22, ldb, sb, h, false);
                GotoGemmFp32.RunParallel(sa, h, sb, h, m7, h, h, h, h, mc, nc, kc);

                // C11 = M1+M4-M5+M7 ; C12 = M3+M5 ; C21 = M2+M4 ; C22 = M1-M2+M3+M6
                WriteQuad(C11, ldc, h, m1, 1, m4, 1, m5, -1, m7, 1);
                WriteQuad(C12, ldc, h, m3, 1, m5, 1, null, 0, null, 0);
                WriteQuad(C21, ldc, h, m2, 1, m4, 1, null, 0, null, 0);
                WriteQuad(C22, ldc, h, m1, 1, m2, -1, m3, 1, m6, 1);
            }
        }
        finally { ArrayPool<float>.Shared.Return(pool, clearArray: true); } // clear: holds operand sums/sub-products (tensor data)
    }
}
#endif
