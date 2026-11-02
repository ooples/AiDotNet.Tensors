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
// owns sixteen output channels and transforms their weights sixteen input channels at a time into an L1 slab, which the
// 36 multiplies use at once; U never exists whole. Per (input channel, eight output channels) the transform costs about
// 80 vector operations against 36 x patches multiply-adds, so it pays where patches are few and channels many.
internal static unsafe partial class WinogradF43
{
    /// <summary>Output channels per few-patch job (two vectors).</summary>
    private const int FewKBlock = 16;

    /// <summary>Input channels per transformed-weight slab: 36 x 16 x 16 floats (36 KB) stay in L1.</summary>
    private const int FewCBlock = 16;

    /// <summary>Patches per multiply group (the microkernel's rows); the patch count is padded up to it.</summary>
    private const int FewPatchGroup = 8;

    /// <summary>Most patches the few-patch form takes.</summary>
    internal static int s_fewMaxPatches =
        int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_WINOGRAD_FEW_MAX_PATCHES"), out int fewMax) && fewMax >= 0 ? fewMax : 8;

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
            int mFloats = Points * paddedPatches * FewKBlock;
            int slabFloats = Points * FewCBlock * FewKBlock, stageFloats = 9 * FewCBlock * FewKBlock;
            int bandFloats = tilesH * Tile * outW * 8;
            CpuParallelSettings.ParallelForOrSerial(0, cOut / FewKBlock, (long)Points * patches * cIn * cOut * 2,
                [MethodImpl(MethodImplOptions.AggressiveOptimization)] (job) =>
            {
                int k0 = job * FewKBlock;
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
                    for (int c0 = 0; c0 < cIn; c0 += FewCBlock)
                    {
                        int cCount = Math.Min(FewCBlock, cIn - c0);
                        StageWeights(wp + wOff, stage, k0, c0, cCount, cIn, cOut, inputGradient);
                        // U slab [36][cCount][16]: G g G^T per (input channel, eight output channels).
                        for (int c = 0; c < cCount; c++)
                            for (int half = 0; half < 2; half++)
                            {
                                float* gs = stage + (half * FewCBlock * 9 + c * 9) * 8;
                                for (int e = 0; e < 9; e++) g[e] = Avx.LoadVector256(gs + e * 8);
                                for (int j = 0; j < 3; j++) TransformKernelColumn(g[j], g[3 + j], g[6 + j], t, j, 3);
                                for (int i = 0; i < 6; i++) TransformKernelColumn(t[i * 3], t[i * 3 + 1], t[i * 3 + 2], uu, i * 6, 1);
                                for (int xi = 0; xi < Points; xi++) Avx.Store(slab + (xi * FewCBlock + c) * FewKBlock + half * 8, uu[xi]);
                            }
                        for (int xi = 0; xi < Points; xi++)
                            for (int p0 = 0; p0 < paddedPatches; p0 += FewPatchGroup)
                            {
                                for (int r = 0; r < FewPatchGroup; r++)
                                    rows[r] = vp + xi * vStride + (long)Math.Min(p0 + r, patches - 1) * cIn + c0;
                                MultiplyFew(slab + xi * FewCBlock * FewKBlock, rows, cCount,
                                    mp + ((long)xi * paddedPatches + p0) * FewKBlock, accumulate: c0 != 0);
                            }
                    }
                    // Output transform: y = A^T m A per patch, eight output channels at a time through a whole-image band.
                    var d = stackalloc Vector256<float>[36];
                    var r16 = stackalloc Vector256<float>[16];
                    float* tmp = band + bandFloats;
                    for (int n = 0; n < batch; n++)
                        for (int half = 0; half < 2; half++)
                        {
                            for (int th = 0; th < tilesH; th++)
                                for (int tw = 0; tw < tilesW; tw++)
                                {
                                    int p = n * tiles + th * tilesW + tw;
                                    LoadPoints(mp + (long)p * FewKBlock + half * 8, (long)paddedPatches * FewKBlock, d);
                                    for (int j = 0; j < 6; j++) OutputTransform(d, j, 6, t, j, 6);
                                    for (int a = 0; a < 4; a++) OutputTransform(t, a * 6, 1, r16, a * 4, 1);
                                    StoreTile(r16, 4, 4, band + (th * Tile * outW + tw * Tile) * 8, outW * 8);
                                }
                            StoreOutputRows(band, outW, yp + yOff + ((long)n * cOut + k0 + half * 8) * hw, 0, height, width, hw,
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

    // stage [2][cCount][9][8]: the 3x3 taps of input channels c0..c0+cCount for output channels k0..k0+15, eight output
    // channels per vector. Forward w is [cOut][cIn][3][3]: an output channel's taps for the block are 9 cCount
    // consecutive floats, so 8x8 transposes move them. Input gradient: w is [cIn][cOut][3][3] read turned 180 degrees.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StageWeights(float* w, float* stage, int k0, int c0, int cCount, int cIn, int cOut, bool inputGradient)
    {
        for (int half = 0; half < 2; half++)
        {
            float* dst = stage + half * FewCBlock * 9 * 8;
            int kh = k0 + half * 8;
            if (!inputGradient)
            {
                float* src = w + (long)kh * cIn * 9 + (long)c0 * 9;
                for (int chunk = 0; chunk < cCount * 9 / 8; chunk++)
                    Transpose8x8(src + chunk * 8, (long)cIn * 9, dst + chunk * 64, 8);
            }
            else
            {
                for (int c = 0; c < cCount; c++)
                {
                    float* src = w + (long)(c0 + c) * cOut * 9 + (long)kh * 9;
                    for (int e = 0; e < 9; e++)
                        for (int l = 0; l < 8; l++) dst[(c * 9 + e) * 8 + l] = src[l * 9 + 8 - e];
                }
            }
        }
    }

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
