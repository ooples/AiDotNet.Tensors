// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

// The few-patch form of Run, for wide layers on small planes (ResNet-18's 512 -> 512 on 4x4 planes: one tile per image,
// eight patches at batch 8). Run transforms the weights into U [36][C][K] before its 36 GEMMs: at 512 -> 512 that is 37.7
// MB written and read back for GEMMs eight rows tall, so F(4x4) ran 2.5x slower than the direct kernel. Here each job
// owns 16 output channels (32 on 512-bit hosts) and transforms their weights a slab of input channels at a time
// (FewInputBlock), which the 36 multiplies use at once; U never exists whole. Per (input channel, eight output channels)
// the transform costs about 80 vector operations against 36 x patches multiply-adds, so it pays where patches are few and
// channels many.
internal static unsafe partial class WinogradF43
{
    /// <summary>Output channels per few-patch job (two vectors).</summary>
    private const int FewKBlock = 16;

    /// <summary>
    /// Input channels per transformed-weight slab (a multiple of 8): 16 for the 512-bit tiles and 32 for the 256-bit
    /// ones at eight patches (one patch group), twice that past it. A deeper slab spends fewer accumulator loads and
    /// stores per multiply-add, but at eight patches each slab is used once, and past L1 its loads cost more than they
    /// save. In-process sweep at four threads (8, 16, 32 and 64 input channels per 512-bit slab, twice that per 256-bit
    /// one), ms, ResNet-18 at batch 8: 512 -> 512 on 4x4 planes (8 patches), forward, best at 16 on EPYC 9V45 (1.15 vs
    /// 1.28 at 8 and 1.64 at 32), 9V74 and Xeon 6973P, and at 32 on the 256-bit EPYC 7763 (3.16-3.30 vs 3.26-3.51 at 16);
    /// 256 -> 256 on 8x8 (32 patches) best at 32 / 64 on 9V74 and 6973P (0.99 vs 1.06 at 16 on 6973P) and at 64 / 128 on
    /// the 7763 (1.84-1.95 vs 1.91-2.00 at 64).
    /// </summary>
    private static int FewInputBlock(bool wide, int paddedPatches)
        => (wide ? 16 : 32) * (paddedPatches <= FewPatchGroup ? 1 : 2);

    /// <summary>Patches per multiply group (the microkernel's rows); the patch count is padded up to it.</summary>
    private const int FewPatchGroup = 8;

    /// <summary>Whether the few-patch form runs 512-bit tiles where Vector512 is accelerated (AIDOTNET_WINOGRAD_FEW512=0
    /// keeps 256-bit).</summary>
    internal static bool s_fewPatches512 = Environment.GetEnvironmentVariable("AIDOTNET_WINOGRAD_FEW512") != "0";

#if NET9_0_OR_GREATER
    private static bool UseFewPatches512 => s_fewPatches512 && (Vector512.IsHardwareAccelerated || s_forceFewPatches512ForTest);
#else
    private static bool UseFewPatches512 => false;
#endif

    /// <summary>Test hook: run the 512-bit few-patch tiles in software Vector512 on hosts without AVX-512.</summary>
    internal static bool s_forceFewPatches512ForTest;

    /// <summary>
    /// Most patches the few-patch form takes (AIDOTNET_WINOGRAD_FEW_MAX_PATCHES overrides). In-process A/B at four
    /// threads, ms, forward / input gradient, against the route each shape took before and PyTorch 2.x on the same host:
    /// ResNet-18 512 -> 512 on 4x4 planes at batch 8 (8 patches, before: direct) EPYC 9V45 1.15 / 1.19 vs 1.65 / 1.63
    /// (PyTorch 1.44 / 1.15), Xeon 6973P 1.91 / 1.99 vs 2.35 / 2.59 (2.07 / 2.34), EPYC 9V74 2.18 / 2.31 vs 3.35 / 3.47
    /// (2.29 / 2.25), EPYC 7763 3.16-3.30 / 3.25-3.36 vs 4.19 / 4.30 (3.96 / 3.77); 256 -> 256 on 8x8 at batch 8 (32
    /// patches, before: direct or batch-wide F(4x4)) 9V45 0.71 / 0.71 vs 1.29 / 1.29 (1.20 / 1.21), 6973P 0.99 / 1.05 vs
    /// 1.84 / 1.85 (1.77 / 1.90), 9V74 1.23-1.25 / 1.25 vs 2.08-2.14 / 2.05-2.09 (2.35 / 2.37), 7763 1.91 / 1.92 vs
    /// 2.44-2.51 / 2.54-2.55 (3.38 / 3.16).
    /// </summary>
    internal static int s_fewMaxPatches =
        int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_WINOGRAD_FEW_MAX_PATCHES"), out int fewMax) && fewMax >= 0 ? fewMax : 32;

    /// <summary>Fewest channels on either side the few-patch form takes: narrower, the weight transform is not repaid.</summary>
    internal const int FewMinChannels = 128;

    /// <summary>Whether the forward or input gradient runs in the few-patch form: wide layers (no narrower than
    /// <see cref="FewMinChannels"/>, output channels a multiple of 16) with at most <see cref="s_fewMaxPatches"/> patches,
    /// on planes of at least 4x4.</summary>
    internal static bool UsesFewPatches(int inChannels, int outChannels, int height, int width, long patches)
        => patches <= s_fewMaxPatches && height >= Tile && width >= Tile
           && Math.Min(inChannels, outChannels) >= FewMinChannels && outChannels % FewKBlock == 0 && !UseBlocked(inChannels, outChannels);

    // Run's few-patch form: V [36][P][cIn] for the batch, then per job of 16 output channels the weight slabs, the
    // multiplies into a job-local M [36][P8][16] (P8 = P rounded up to 8) and the output transform.
    private static void RunFewPatches(float[] x, int xOff, float[] w, int wOff, float[] y, int yOff, bool accumulate,
        int batch, int cIn, int height, int width, int cOut, bool inputGradient)
    {
        int tilesH = (height + Tile - 1) / Tile, tilesW = (width + Tile - 1) / Tile;
        int tiles = tilesH * tilesW, patches = batch * tiles;
        int paddedPatches = (patches + FewPatchGroup - 1) / FewPatchGroup * FewPatchGroup;
        float[] v = System.Buffers.ArrayPool<float>.Shared.Rent(Points * patches * cIn);
        try
        {
            TransformInput(x, xOff, v, batch, cIn, height, width, tilesH, tilesW);
            int outW = tilesW * Tile, hw = height * width;
            // 512-bit hosts: jobs of 32 output channels (two zmm).
            bool wide = UseFewPatches512 && cOut % (2 * FewKBlock) == 0;
            int kb = wide ? 2 * FewKBlock : FewKBlock, cb = FewInputBlock(wide, paddedPatches), groups = kb / 8;
            int mFloats = Points * paddedPatches * kb;
            int slabFloats = Points * cb * kb, stageFloats = 9 * cb * kb;
            int bandFloats = tilesH * Tile * outW * 8;
            CpuParallelSettings.ParallelForOrSerial(0, cOut / kb, (long)Points * patches * cIn * cOut * 2,
                [MethodImpl(MethodImplOptions.AggressiveOptimization)] (job) =>
            {
                int k0 = job * kb;
                float[] mb = ThreadBuffer(ref t_blockM, mFloats);
                float[] tb = ThreadBuffer(ref t_blockTile, slabFloats + stageFloats + Math.Max(bandFloats + 64, 0));
                fixed (float* vp = v, wp = w, yp = y, mp = mb, tp = tb)
                {
                    float* slab = tp, stage = tp + slabFloats, band = stage + stageFloats;
                    var g = stackalloc Vector256<float>[9];
                    var t = stackalloc Vector256<float>[36];
                    var uu = stackalloc Vector256<float>[36];
                    var rows = stackalloc float*[FewPatchGroup];
                    long vStride = (long)patches * cIn;
                    for (int c0 = 0; c0 < cIn; c0 += cb)
                    {
                        int cCount = Math.Min(cb, cIn - c0);
                        StageWeights(wp + wOff, stage, k0, groups, c0, cCount, cb, cIn, cOut, inputGradient);
                        // U slab [36][cCount][kb]: G g G^T per (input channel, eight or sixteen output channels).
#if NET9_0_OR_GREATER
                        if (wide)
                            TransformSlab512(stage, slab, cCount, cb, kb);
                        else
#endif
                        for (int c = 0; c < cCount; c++)
                            for (int grp = 0; grp < groups; grp++)
                            {
                                float* gs = stage + (grp * cb * 9 + c * 9) * 8;
                                for (int e = 0; e < 9; e++) g[e] = Avx.LoadVector256(gs + e * 8);
                                for (int j = 0; j < 3; j++) TransformKernelColumn(g[j], g[3 + j], g[6 + j], t, j, 3);
                                for (int i = 0; i < 6; i++) TransformKernelColumn(t[i * 3], t[i * 3 + 1], t[i * 3 + 2], uu, i * 6, 1);
                                for (int xi = 0; xi < Points; xi++) Avx.Store(slab + (xi * cb + c) * kb + grp * 8, uu[xi]);
                            }
                        for (int xi = 0; xi < Points; xi++)
                            for (int p0 = 0; p0 < paddedPatches; p0 += FewPatchGroup)
                            {
                                for (int r = 0; r < FewPatchGroup; r++)
                                    rows[r] = vp + xi * vStride + (long)Math.Min(p0 + r, patches - 1) * cIn + c0;
                                float* mBlock = mp + ((long)xi * paddedPatches + p0) * kb;
#if NET9_0_OR_GREATER
                                if (wide) MultiplyFew512(slab + xi * cb * kb, rows, cCount, mBlock, accumulate: c0 != 0);
                                else
#endif
                                MultiplyFew(slab + xi * cb * kb, rows, cCount, mBlock, accumulate: c0 != 0);
                            }
                    }
                    // Output transform: y = A^T m A per patch, eight output channels at a time through a whole-image band.
                    var d = stackalloc Vector256<float>[36];
                    var r16 = stackalloc Vector256<float>[16];
                    float* tmp = band + bandFloats;
                    for (int n = 0; n < batch; n++)
                        for (int grp = 0; grp < groups; grp++)
                        {
                            for (int th = 0; th < tilesH; th++)
                                for (int tw = 0; tw < tilesW; tw++)
                                {
                                    int p = n * tiles + th * tilesW + tw;
                                    LoadPoints(mp + (long)p * kb + grp * 8, (long)paddedPatches * kb, d);
                                    for (int j = 0; j < 6; j++) OutputTransform(d, j, 6, t, j, 6);
                                    for (int a = 0; a < 4; a++) OutputTransform(t, a * 6, 1, r16, a * 4, 1);
                                    StoreTile(r16, 4, 4, band + (th * Tile * outW + tw * Tile) * 8, outW * 8);
                                }
                            StoreOutputRows(band, outW, yp + yOff + ((long)n * cOut + k0 + grp * 8) * hw, 0, height, width, hw,
                                accumulate, tmp);
                        }
                }
            }, deterministicSafe: true);
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(v);
        }
    }

    // stage [groups][cb][9][8]: the 3x3 taps of input channels c0..c0+cCount for output channels k0..k0+8 groups, eight
    // output channels per vector. Forward w is [cOut][cIn][3][3]: an output channel's taps for the block are 9 cCount
    // consecutive floats, so 8x8 transposes move them. Input gradient: w is [cIn][cOut][3][3] read turned 180 degrees.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StageWeights(float* w, float* stage, int k0, int groups, int c0, int cCount, int cb, int cIn, int cOut,
        bool inputGradient)
    {
        for (int grp = 0; grp < groups; grp++)
        {
            float* dst = stage + grp * cb * 9 * 8;
            int kh = k0 + grp * 8;
            if (!inputGradient)
            {
                float* src = w + (long)kh * cIn * 9 + (long)c0 * 9;
                for (int chunk = 0; chunk < cCount * 9 / 8; chunk++)
                    Transpose8x8(src + chunk * 8, (long)cIn * 9, dst + chunk * 64, 8);
            }
            else
            {
                // Eight output channels' taps are 72 consecutive floats [8][9]: taps 0..7 transpose into rows 8..1
                // (written upwards: tap e lands at row 8 - e), tap 8 into row 0.
                for (int c = 0; c < cCount; c++)
                {
                    float* src = w + (long)(c0 + c) * cOut * 9 + (long)kh * 9;
                    float* rows = dst + c * 72;
                    Transpose8x8(src, 9, rows + 64, -8);
                    for (int l = 0; l < 8; l++) rows[l] = src[l * 9 + 8];
                }
            }
        }
    }

#if NET9_0_OR_GREATER
    // U slab [36][cCount][32] from stage [4][cb][9][8]: sixteen output channels per Vector512 (two stage groups).
    private static void TransformSlab512(float* stage, float* slab, int cCount, int cb, int kb)
    {
        var g = stackalloc Vector512<float>[9];
        var t = stackalloc Vector512<float>[18];
        var uu = stackalloc Vector512<float>[36];
        for (int c = 0; c < cCount; c++)
            for (int pair = 0; pair < kb / 16; pair++)
            {
                float* lo = stage + (2 * pair * cb * 9 + c * 9) * 8, hi = lo + cb * 9 * 8;
                for (int e = 0; e < 9; e++)
                    g[e] = Vector512.Create(Vector256.Load(lo + e * 8), Vector256.Load(hi + e * 8));
                for (int j = 0; j < 3; j++) TransformKernelColumn512(g[j], g[3 + j], g[6 + j], t, j, 3);
                for (int i = 0; i < 6; i++) TransformKernelColumn512(t[i * 3], t[i * 3 + 1], t[i * 3 + 2], uu, i * 6, 1);
                for (int xi = 0; xi < Points; xi++) uu[xi].Store(slab + (xi * cb + c) * kb + pair * 16);
            }
    }

    // TransformKernelColumn on sixteen lanes (the same operations in the same order).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void TransformKernelColumn512(Vector512<float> a, Vector512<float> b, Vector512<float> c,
        Vector512<float>* dst, int offset, int step)
    {
        var ac = a + c;
        dst[offset] = a * Vector512.Create(0.25f);
        dst[offset + step] = (ac + b) * Vector512.Create(-1f / 6f);
        dst[offset + 2 * step] = (ac - b) * Vector512.Create(-1f / 6f);
        var even = Vector512.FusedMultiplyAdd(a, Vector512.Create(1f / 24f), c * Vector512.Create(1f / 6f));
        var odd = b * Vector512.Create(1f / 12f);
        dst[offset + 3 * step] = even + odd;
        dst[offset + 4 * step] = even - odd;
        dst[offset + 5 * step] = c;
    }

    // m [8 patches][32] (=, or += when accumulate) = sum over c of rows[p][c] u[c][32]: 16 zmm accumulators.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void MultiplyFew512(float* u, float** rows, int depth, float* m, bool accumulate)
    {
        Vector512<float> a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;
        if (accumulate)
        {
            a0 = Vector512.Load(m); b0 = Vector512.Load(m + 16);
            a1 = Vector512.Load(m + 32); b1 = Vector512.Load(m + 48);
            a2 = Vector512.Load(m + 64); b2 = Vector512.Load(m + 80);
            a3 = Vector512.Load(m + 96); b3 = Vector512.Load(m + 112);
            a4 = Vector512.Load(m + 128); b4 = Vector512.Load(m + 144);
            a5 = Vector512.Load(m + 160); b5 = Vector512.Load(m + 176);
            a6 = Vector512.Load(m + 192); b6 = Vector512.Load(m + 208);
            a7 = Vector512.Load(m + 224); b7 = Vector512.Load(m + 240);
        }
        else
        {
            a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = Vector512<float>.Zero;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = Vector512<float>.Zero;
        }
        float* r0 = rows[0], r1 = rows[1], r2 = rows[2], r3 = rows[3], r4 = rows[4], r5 = rows[5], r6 = rows[6], r7 = rows[7];
        for (int c = 0; c < depth; c++)
        {
            var lo = Vector512.Load(u + c * 32);
            var hi = Vector512.Load(u + c * 32 + 16);
            var s = Vector512.Create(r0[c]); a0 = Vector512.FusedMultiplyAdd(lo, s, a0); b0 = Vector512.FusedMultiplyAdd(hi, s, b0);
            s = Vector512.Create(r1[c]); a1 = Vector512.FusedMultiplyAdd(lo, s, a1); b1 = Vector512.FusedMultiplyAdd(hi, s, b1);
            s = Vector512.Create(r2[c]); a2 = Vector512.FusedMultiplyAdd(lo, s, a2); b2 = Vector512.FusedMultiplyAdd(hi, s, b2);
            s = Vector512.Create(r3[c]); a3 = Vector512.FusedMultiplyAdd(lo, s, a3); b3 = Vector512.FusedMultiplyAdd(hi, s, b3);
            s = Vector512.Create(r4[c]); a4 = Vector512.FusedMultiplyAdd(lo, s, a4); b4 = Vector512.FusedMultiplyAdd(hi, s, b4);
            s = Vector512.Create(r5[c]); a5 = Vector512.FusedMultiplyAdd(lo, s, a5); b5 = Vector512.FusedMultiplyAdd(hi, s, b5);
            s = Vector512.Create(r6[c]); a6 = Vector512.FusedMultiplyAdd(lo, s, a6); b6 = Vector512.FusedMultiplyAdd(hi, s, b6);
            s = Vector512.Create(r7[c]); a7 = Vector512.FusedMultiplyAdd(lo, s, a7); b7 = Vector512.FusedMultiplyAdd(hi, s, b7);
        }
        a0.Store(m); b0.Store(m + 16); a1.Store(m + 32); b1.Store(m + 48);
        a2.Store(m + 64); b2.Store(m + 80); a3.Store(m + 96); b3.Store(m + 112);
        a4.Store(m + 128); b4.Store(m + 144); a5.Store(m + 160); b5.Store(m + 176);
        a6.Store(m + 192); b6.Store(m + 208); a7.Store(m + 224); b7.Store(m + 240);
    }
#endif

    // m [8 patches][16] (=, or += when accumulate) = sum over c of rows[p][c] u[c][16], u = one point's U slab.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void MultiplyFew(float* u, float** rows, int depth, float* m, bool accumulate)
    {
        Vector256<float> a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;
        if (accumulate)
        {
            a0 = Avx.LoadVector256(m); b0 = Avx.LoadVector256(m + 8);
            a1 = Avx.LoadVector256(m + 16); b1 = Avx.LoadVector256(m + 24);
            a2 = Avx.LoadVector256(m + 32); b2 = Avx.LoadVector256(m + 40);
            a3 = Avx.LoadVector256(m + 48); b3 = Avx.LoadVector256(m + 56);
            a4 = Avx.LoadVector256(m + 64); b4 = Avx.LoadVector256(m + 72);
            a5 = Avx.LoadVector256(m + 80); b5 = Avx.LoadVector256(m + 88);
            a6 = Avx.LoadVector256(m + 96); b6 = Avx.LoadVector256(m + 104);
            a7 = Avx.LoadVector256(m + 112); b7 = Avx.LoadVector256(m + 120);
        }
        else
        {
            a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = Vector256<float>.Zero;
            b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = Vector256<float>.Zero;
        }
        float* r0 = rows[0], r1 = rows[1], r2 = rows[2], r3 = rows[3], r4 = rows[4], r5 = rows[5], r6 = rows[6], r7 = rows[7];
        for (int c = 0; c < depth; c++)
        {
            var lo = Avx.LoadVector256(u + c * FewKBlock);
            var hi = Avx.LoadVector256(u + c * FewKBlock + 8);
            var s = Avx.BroadcastScalarToVector256(r0 + c); a0 = Fma.MultiplyAdd(lo, s, a0); b0 = Fma.MultiplyAdd(hi, s, b0);
            s = Avx.BroadcastScalarToVector256(r1 + c); a1 = Fma.MultiplyAdd(lo, s, a1); b1 = Fma.MultiplyAdd(hi, s, b1);
            s = Avx.BroadcastScalarToVector256(r2 + c); a2 = Fma.MultiplyAdd(lo, s, a2); b2 = Fma.MultiplyAdd(hi, s, b2);
            s = Avx.BroadcastScalarToVector256(r3 + c); a3 = Fma.MultiplyAdd(lo, s, a3); b3 = Fma.MultiplyAdd(hi, s, b3);
            s = Avx.BroadcastScalarToVector256(r4 + c); a4 = Fma.MultiplyAdd(lo, s, a4); b4 = Fma.MultiplyAdd(hi, s, b4);
            s = Avx.BroadcastScalarToVector256(r5 + c); a5 = Fma.MultiplyAdd(lo, s, a5); b5 = Fma.MultiplyAdd(hi, s, b5);
            s = Avx.BroadcastScalarToVector256(r6 + c); a6 = Fma.MultiplyAdd(lo, s, a6); b6 = Fma.MultiplyAdd(hi, s, b6);
            s = Avx.BroadcastScalarToVector256(r7 + c); a7 = Fma.MultiplyAdd(lo, s, a7); b7 = Fma.MultiplyAdd(hi, s, b7);
        }
        Avx.Store(m, a0); Avx.Store(m + 8, b0); Avx.Store(m + 16, a1); Avx.Store(m + 24, b1);
        Avx.Store(m + 32, a2); Avx.Store(m + 40, b2); Avx.Store(m + 48, a3); Avx.Store(m + 56, b3);
        Avx.Store(m + 64, a4); Avx.Store(m + 72, b4); Avx.Store(m + 80, a5); Avx.Store(m + 88, b5);
        Avx.Store(m + 96, a6); Avx.Store(m + 104, b6); Avx.Store(m + 112, a7); Avx.Store(m + 120, b7);
    }
}
#endif
