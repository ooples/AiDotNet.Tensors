// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET9_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace AiDotNet.Tensors.Engines.Simd;

// The blocked passes' transform-domain GEMM on the 512-bit registers: eight rows x 32 columns per register tile (sixteen
// accumulators; per depth step eight broadcasts and two loads), where the 256-bit tile holds three rows x 32 columns in
// twelve. Every output still sums its depth terms one fused multiply-add at a time in depth order, so the results are bit
// for bit those of the 256-bit tile.
internal static unsafe partial class WinogradF43
{
    /// <summary>A/B switch: AIDOTNET_WINOGRAD_GEMM512=0 keeps the blocked GEMM on the 256-bit tile.</summary>
    internal static bool s_blockGemm512 = Environment.GetEnvironmentVariable("AIDOTNET_WINOGRAD_GEMM512") != "0";

    private static bool UseBlockGemm512 => s_blockGemm512 && SimdGemm.UseVector512Direct;

    private const int Gemm512Rows = 8;

    // c [rows x cols] (=, or += when accumulate) = a [rows x depth] b [depth x cols]; as BlockGemm.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void BlockGemm512(float* a, int aRow, int aStep, float* b, float* c, int rows, int depth, int cols,
        bool accumulate)
    {
        for (int i = 0; i < rows; i += Gemm512Rows)
        {
            int live = Math.Min(Gemm512Rows, rows - i);
            // Rows past `live` alias row 0: computed (and read, when accumulating) but never stored.
            float* a0 = a + (long)i * aRow;
            float* a1 = live > 1 ? a0 + aRow : a0, a2 = live > 2 ? a0 + 2L * aRow : a0, a3 = live > 3 ? a0 + 3L * aRow : a0;
            float* a4 = live > 4 ? a0 + 4L * aRow : a0, a5 = live > 5 ? a0 + 5L * aRow : a0;
            float* a6 = live > 6 ? a0 + 6L * aRow : a0, a7 = live > 7 ? a0 + 7L * aRow : a0;
            float* c0 = c + (long)i * cols;
            float* c1 = live > 1 ? c0 + cols : c0, c2 = live > 2 ? c0 + 2L * cols : c0, c3 = live > 3 ? c0 + 3L * cols : c0;
            float* c4 = live > 4 ? c0 + 4L * cols : c0, c5 = live > 5 ? c0 + 5L * cols : c0;
            float* c6 = live > 6 ? c0 + 6L * cols : c0, c7 = live > 7 ? c0 + 7L * cols : c0;
            int j = 0;
            for (; j + 32 <= cols; j += 32)
            {
                Vector512<float> s00 = default, s01 = default, s10 = default, s11 = default;
                Vector512<float> s20 = default, s21 = default, s30 = default, s31 = default;
                Vector512<float> s40 = default, s41 = default, s50 = default, s51 = default;
                Vector512<float> s60 = default, s61 = default, s70 = default, s71 = default;
                if (accumulate)
                {
                    s00 = Vector512.Load(c0 + j); s01 = Vector512.Load(c0 + j + 16);
                    s10 = Vector512.Load(c1 + j); s11 = Vector512.Load(c1 + j + 16);
                    s20 = Vector512.Load(c2 + j); s21 = Vector512.Load(c2 + j + 16);
                    s30 = Vector512.Load(c3 + j); s31 = Vector512.Load(c3 + j + 16);
                    s40 = Vector512.Load(c4 + j); s41 = Vector512.Load(c4 + j + 16);
                    s50 = Vector512.Load(c5 + j); s51 = Vector512.Load(c5 + j + 16);
                    s60 = Vector512.Load(c6 + j); s61 = Vector512.Load(c6 + j + 16);
                    s70 = Vector512.Load(c7 + j); s71 = Vector512.Load(c7 + j + 16);
                }
                float* bk = b + j;
                long step = 0;
                for (int k = 0; k < depth; k++, bk += cols, step += aStep)
                {
                    var b0 = Vector512.Load(bk); var b1 = Vector512.Load(bk + 16);
                    var g = Vector512.Create(a0[step]);
                    s00 = Vector512.FusedMultiplyAdd(g, b0, s00); s01 = Vector512.FusedMultiplyAdd(g, b1, s01);
                    g = Vector512.Create(a1[step]);
                    s10 = Vector512.FusedMultiplyAdd(g, b0, s10); s11 = Vector512.FusedMultiplyAdd(g, b1, s11);
                    g = Vector512.Create(a2[step]);
                    s20 = Vector512.FusedMultiplyAdd(g, b0, s20); s21 = Vector512.FusedMultiplyAdd(g, b1, s21);
                    g = Vector512.Create(a3[step]);
                    s30 = Vector512.FusedMultiplyAdd(g, b0, s30); s31 = Vector512.FusedMultiplyAdd(g, b1, s31);
                    g = Vector512.Create(a4[step]);
                    s40 = Vector512.FusedMultiplyAdd(g, b0, s40); s41 = Vector512.FusedMultiplyAdd(g, b1, s41);
                    g = Vector512.Create(a5[step]);
                    s50 = Vector512.FusedMultiplyAdd(g, b0, s50); s51 = Vector512.FusedMultiplyAdd(g, b1, s51);
                    g = Vector512.Create(a6[step]);
                    s60 = Vector512.FusedMultiplyAdd(g, b0, s60); s61 = Vector512.FusedMultiplyAdd(g, b1, s61);
                    g = Vector512.Create(a7[step]);
                    s70 = Vector512.FusedMultiplyAdd(g, b0, s70); s71 = Vector512.FusedMultiplyAdd(g, b1, s71);
                }
                s00.Store(c0 + j); s01.Store(c0 + j + 16);
                if (live > 1) { s10.Store(c1 + j); s11.Store(c1 + j + 16); }
                if (live > 2) { s20.Store(c2 + j); s21.Store(c2 + j + 16); }
                if (live > 3) { s30.Store(c3 + j); s31.Store(c3 + j + 16); }
                if (live > 4) { s40.Store(c4 + j); s41.Store(c4 + j + 16); }
                if (live > 5) { s50.Store(c5 + j); s51.Store(c5 + j + 16); }
                if (live > 6) { s60.Store(c6 + j); s61.Store(c6 + j + 16); }
                if (live > 7) { s70.Store(c7 + j); s71.Store(c7 + j + 16); }
            }
            // Column tails: sixteen, then eight.
            for (; j + 16 <= cols; j += 16)
            {
                Vector512<float> s0 = default, s1 = default, s2 = default, s3 = default;
                Vector512<float> s4 = default, s5 = default, s6 = default, s7 = default;
                if (accumulate)
                {
                    s0 = Vector512.Load(c0 + j); s1 = Vector512.Load(c1 + j); s2 = Vector512.Load(c2 + j); s3 = Vector512.Load(c3 + j);
                    s4 = Vector512.Load(c4 + j); s5 = Vector512.Load(c5 + j); s6 = Vector512.Load(c6 + j); s7 = Vector512.Load(c7 + j);
                }
                float* bk = b + j;
                long step = 0;
                for (int k = 0; k < depth; k++, bk += cols, step += aStep)
                {
                    var bv = Vector512.Load(bk);
                    s0 = Vector512.FusedMultiplyAdd(Vector512.Create(a0[step]), bv, s0);
                    s1 = Vector512.FusedMultiplyAdd(Vector512.Create(a1[step]), bv, s1);
                    s2 = Vector512.FusedMultiplyAdd(Vector512.Create(a2[step]), bv, s2);
                    s3 = Vector512.FusedMultiplyAdd(Vector512.Create(a3[step]), bv, s3);
                    s4 = Vector512.FusedMultiplyAdd(Vector512.Create(a4[step]), bv, s4);
                    s5 = Vector512.FusedMultiplyAdd(Vector512.Create(a5[step]), bv, s5);
                    s6 = Vector512.FusedMultiplyAdd(Vector512.Create(a6[step]), bv, s6);
                    s7 = Vector512.FusedMultiplyAdd(Vector512.Create(a7[step]), bv, s7);
                }
                s0.Store(c0 + j);
                if (live > 1) s1.Store(c1 + j);
                if (live > 2) s2.Store(c2 + j);
                if (live > 3) s3.Store(c3 + j);
                if (live > 4) s4.Store(c4 + j);
                if (live > 5) s5.Store(c5 + j);
                if (live > 6) s6.Store(c6 + j);
                if (live > 7) s7.Store(c7 + j);
            }
            for (; j < cols; j += 8)
            {
                Vector256<float> s0 = default, s1 = default, s2 = default, s3 = default;
                Vector256<float> s4 = default, s5 = default, s6 = default, s7 = default;
                if (accumulate)
                {
                    s0 = Vector256.Load(c0 + j); s1 = Vector256.Load(c1 + j); s2 = Vector256.Load(c2 + j); s3 = Vector256.Load(c3 + j);
                    s4 = Vector256.Load(c4 + j); s5 = Vector256.Load(c5 + j); s6 = Vector256.Load(c6 + j); s7 = Vector256.Load(c7 + j);
                }
                float* bk = b + j;
                long step = 0;
                for (int k = 0; k < depth; k++, bk += cols, step += aStep)
                {
                    var bv = Vector256.Load(bk);
                    s0 = Vector256.FusedMultiplyAdd(Vector256.Create(a0[step]), bv, s0);
                    s1 = Vector256.FusedMultiplyAdd(Vector256.Create(a1[step]), bv, s1);
                    s2 = Vector256.FusedMultiplyAdd(Vector256.Create(a2[step]), bv, s2);
                    s3 = Vector256.FusedMultiplyAdd(Vector256.Create(a3[step]), bv, s3);
                    s4 = Vector256.FusedMultiplyAdd(Vector256.Create(a4[step]), bv, s4);
                    s5 = Vector256.FusedMultiplyAdd(Vector256.Create(a5[step]), bv, s5);
                    s6 = Vector256.FusedMultiplyAdd(Vector256.Create(a6[step]), bv, s6);
                    s7 = Vector256.FusedMultiplyAdd(Vector256.Create(a7[step]), bv, s7);
                }
                s0.Store(c0 + j);
                if (live > 1) s1.Store(c1 + j);
                if (live > 2) s2.Store(c2 + j);
                if (live > 3) s3.Store(c3 + j);
                if (live > 4) s4.Store(c4 + j);
                if (live > 5) s5.Store(c5 + j);
                if (live > 6) s6.Store(c6 + j);
                if (live > 7) s7.Store(c7 + j);
            }
        }
    }
}
#endif
