// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

// The patch-blocked form of Run, for narrow layers. Run transforms every patch of the batch into V [36][P][C], multiplies
// into M [36][P][K] and transforms back: at 32 -> 64 channels on 14x14 planes (batch 64) V and M are 14 MB, streamed
// through memory between the passes, which made F(4x4) slower than the direct kernel. Here one job takes a band of an
// image's tile rows (about 32 patches): its V and M blocks (36 x 32 x C and 36 x 32 x K floats) stay in the core's L2
// from the input transform through the 36 multiplies to the output transform.
internal static unsafe partial class WinogradF43
{
    /// <summary>Patches per job of <see cref="RunBlocked"/> (whole tile rows, at least one).</summary>
    private const int BlockPatches = 32;

    // AIDOTNET_WINOGRAD_BLOCKED: 1 runs every shape blocked, 0 none (narrow layers then keep the direct kernel).
    private static readonly string? s_blockedEnv = Environment.GetEnvironmentVariable("AIDOTNET_WINOGRAD_BLOCKED");

    /// <summary>Route every shape through <see cref="RunBlocked"/> (tests and A/B; AIDOTNET_WINOGRAD_BLOCKED=1).</summary>
    internal static bool s_forceBlockedForTest = s_blockedEnv == "1";

    /// <summary>Whether narrow layers run blocked (AIDOTNET_WINOGRAD_BLOCKED=0 turns it off).</summary>
    internal static bool s_blockedNarrow = s_blockedEnv != "0";

    /// <summary>Fewest input and output channels the batch-wide passes take; narrower layers run blocked.</summary>
    private const int MinUnblockedChannels = 64;

    /// <summary>Fewest input and output channels the blocked form takes (fewer: the transforms outweigh the saving).</summary>
    internal const int MinBlockedChannels = 16;

    [ThreadStatic] private static float[]? t_blockV;
    [ThreadStatic] private static float[]? t_blockM;
    [ThreadStatic] private static float[]? t_blockTile;

    private static float[] ThreadBuffer(ref float[]? slot, int length)
    {
        var s = slot;
        if (s is null || s.Length < length) slot = s = new float[length];
        return s;
    }

    private static bool UseBlocked(int cIn, int cOut)
        => s_forceBlockedForTest || (s_blockedNarrow && (cIn < MinUnblockedChannels || cOut < MinUnblockedChannels));

    /// <summary>y [N, cOut, H, W] (=, or +=) from x [N, cIn, H, W] through the transformed weights u [36][cIn][cOut].</summary>
    private static void RunBlocked(float[] x, int xOff, float[] u, float[] y, int yOff, bool accumulate,
        int batch, int cIn, int height, int width, int cOut)
    {
        int tilesH = (height + Tile - 1) / Tile, tilesW = (width + Tile - 1) / Tile;
        int bandRows = Math.Max(1, Math.Min(tilesH, BlockPatches / tilesW));
        int bands = (tilesH + bandRows - 1) / bandRows;
        int maxPatches = bandRows * tilesW;
        int padW = tilesW * Tile + 2, outW = tilesW * Tile;
        int inTileFloats = (bandRows * Tile + 2) * padW * 8, outTileFloats = bandRows * Tile * outW * 8;
        int hw = height * width;
        CpuParallelSettings.ParallelForOrSerial(0, batch * bands, (long)batch * hw * cIn * cOut * 4,
            [MethodImpl(MethodImplOptions.AggressiveOptimization)] (job) =>
        {
            int n = job / bands, band = job - n * bands;
            int r0 = band * bandRows, r1 = Math.Min(tilesH, r0 + bandRows);
            int patches = (r1 - r0) * tilesW;
            float[] vb = ThreadBuffer(ref t_blockV, Points * maxPatches * cIn);
            float[] mb = ThreadBuffer(ref t_blockM, Points * maxPatches * cOut);
            float[] tb = ThreadBuffer(ref t_blockTile, Math.Max(inTileFloats, outTileFloats + 64));
            fixed (float* xp = x, up = u, yp = y, vp = vb, mp = mb, tp = tb)
            {
                var d = stackalloc Vector256<float>[36];
                var t = stackalloc Vector256<float>[36];
                // Input transform: V [36][patches][cIn], eight channels at a time through a padded band of the image.
                long vStride = (long)patches * cIn;
                int firstRow = r0 * Tile - 1;                       // input row of the band's padded row 0
                int bandPadRows = (r1 - r0) * Tile + 2;
                for (int cb = 0; cb < cIn / 8; cb++)
                {
                    new Span<float>(tp, bandPadRows * padW * 8).Clear();
                    float* planes = xp + xOff + ((long)n * cIn + cb * 8) * hw;
                    for (int pr = 0; pr < bandPadRows; pr++)
                    {
                        int h = firstRow + pr;
                        if (h < 0 || h >= height) continue;
                        float* src = planes + h * width;
                        float* dst = tp + (pr * padW + 1) * 8;
                        int q = 0;
                        for (; q + 8 <= width; q += 8) Transpose8x8(src + q, hw, dst + q * 8, 8);
                        for (; q < width; q++)
                            for (int l = 0; l < 8; l++) dst[q * 8 + l] = src[(long)l * hw + q];
                    }
                    for (int th = 0; th < r1 - r0; th++)
                        for (int tw = 0; tw < tilesW; tw++)
                        {
                            LoadTile(tp + (th * Tile * padW + tw * Tile) * 8, padW * 8, 6, 6, d);
                            for (int j = 0; j < 6; j++) InputTransform(d, j, 6, t, j, 6);
                            for (int i = 0; i < 6; i++) InputTransform(t, i * 6, 1, d, i * 6, 1);
                            StorePoints(d, vp + (long)(th * tilesW + tw) * cIn + cb * 8, vStride);
                        }
                }
                // M_xi [patches x cOut] = V_xi [patches x cIn] U_xi [cIn x cOut].
                for (int xi = 0; xi < Points; xi++)
                    BlockGemm(vp + xi * vStride, up + (long)xi * cIn * cOut, mp + (long)xi * patches * cOut,
                        patches, cIn, cOut);
                // Output transform: y = A^T m A per patch, eight output channels at a time through an output band.
                long mStride = (long)patches * cOut;
                var r = stackalloc Vector256<float>[16];
                float* tmp = tp + outTileFloats;
                int rowLo = r0 * Tile, rowHi = Math.Min(height, r1 * Tile);
                for (int kb = 0; kb < cOut / 8; kb++)
                {
                    for (int th = 0; th < r1 - r0; th++)
                        for (int tw = 0; tw < tilesW; tw++)
                        {
                            LoadPoints(mp + (long)(th * tilesW + tw) * cOut + kb * 8, mStride, d);
                            for (int j = 0; j < 6; j++) OutputTransform(d, j, 6, t, j, 6);
                            for (int a = 0; a < 4; a++) OutputTransform(t, a * 6, 1, r, a * 4, 1);
                            StoreTile(r, 4, 4, tp + (th * Tile * outW + tw * Tile) * 8, outW * 8);
                        }
                    float* planes = yp + yOff + ((long)n * cOut + kb * 8) * hw;
                    for (int h = rowLo; h < rowHi; h++)
                    {
                        float* src = tp + (h - rowLo) * outW * 8;
                        float* dstRow = planes + h * width;
                        int q = 0;
                        if (accumulate)
                        {
                            for (; q + 8 <= width; q += 8)
                            {
                                Transpose8x8(src + q * 8, 8, tmp, 8);
                                for (int l = 0; l < 8; l++)
                                {
                                    float* row = dstRow + (long)l * hw + q;
                                    Avx.Store(row, Avx.Add(Avx.LoadVector256(row), Avx.LoadVector256(tmp + l * 8)));
                                }
                            }
                            for (; q < width; q++)
                                for (int l = 0; l < 8; l++) dstRow[(long)l * hw + q] += src[q * 8 + l];
                        }
                        else
                        {
                            for (; q + 8 <= width; q += 8) Transpose8x8(src + q * 8, 8, dstRow + q, hw);
                            for (; q < width; q++)
                                for (int l = 0; l < 8; l++) dstRow[(long)l * hw + q] = src[q * 8 + l];
                        }
                    }
                }
            }
        }, deterministicSafe: true);
    }

    // c [rows x cols] = a [rows x depth] b [depth x cols], row-major, cols a multiple of eight: three rows x 32 columns
    // per register tile (twelve accumulators; per depth step three broadcasts and four loads), eight-column tiles past
    // the last multiple of 32.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void BlockGemm(float* a, float* b, float* c, int rows, int depth, int cols)
    {
        for (int i = 0; i < rows; i += 3)
        {
            int live = Math.Min(3, rows - i);
            float* a0 = a + (long)i * depth;
            float* a1 = live > 1 ? a0 + depth : a0;
            float* a2 = live > 2 ? a0 + 2 * depth : a0;
            float* c0 = c + (long)i * cols;
            int j = 0;
            for (; j + 32 <= cols; j += 32)
            {
                Vector256<float> s00 = Vector256<float>.Zero, s01 = s00, s02 = s00, s03 = s00;
                Vector256<float> s10 = s00, s11 = s00, s12 = s00, s13 = s00;
                Vector256<float> s20 = s00, s21 = s00, s22 = s00, s23 = s00;
                float* bk = b + j;
                for (int k = 0; k < depth; k++, bk += cols)
                {
                    var b0 = Avx.LoadVector256(bk); var b1 = Avx.LoadVector256(bk + 8);
                    var b2 = Avx.LoadVector256(bk + 16); var b3 = Avx.LoadVector256(bk + 24);
                    var g = Vector256.Create(a0[k]);
                    s00 = Fma.MultiplyAdd(g, b0, s00); s01 = Fma.MultiplyAdd(g, b1, s01);
                    s02 = Fma.MultiplyAdd(g, b2, s02); s03 = Fma.MultiplyAdd(g, b3, s03);
                    g = Vector256.Create(a1[k]);
                    s10 = Fma.MultiplyAdd(g, b0, s10); s11 = Fma.MultiplyAdd(g, b1, s11);
                    s12 = Fma.MultiplyAdd(g, b2, s12); s13 = Fma.MultiplyAdd(g, b3, s13);
                    g = Vector256.Create(a2[k]);
                    s20 = Fma.MultiplyAdd(g, b0, s20); s21 = Fma.MultiplyAdd(g, b1, s21);
                    s22 = Fma.MultiplyAdd(g, b2, s22); s23 = Fma.MultiplyAdd(g, b3, s23);
                }
                Avx.Store(c0 + j, s00); Avx.Store(c0 + j + 8, s01); Avx.Store(c0 + j + 16, s02); Avx.Store(c0 + j + 24, s03);
                if (live > 1)
                {
                    float* c1 = c0 + cols;
                    Avx.Store(c1 + j, s10); Avx.Store(c1 + j + 8, s11); Avx.Store(c1 + j + 16, s12); Avx.Store(c1 + j + 24, s13);
                }
                if (live > 2)
                {
                    float* c2 = c0 + 2 * cols;
                    Avx.Store(c2 + j, s20); Avx.Store(c2 + j + 8, s21); Avx.Store(c2 + j + 16, s22); Avx.Store(c2 + j + 24, s23);
                }
            }
            for (; j < cols; j += 8)
            {
                Vector256<float> s0 = Vector256<float>.Zero, s1 = s0, s2 = s0;
                float* bk = b + j;
                for (int k = 0; k < depth; k++, bk += cols)
                {
                    var bv = Avx.LoadVector256(bk);
                    s0 = Fma.MultiplyAdd(Vector256.Create(a0[k]), bv, s0);
                    s1 = Fma.MultiplyAdd(Vector256.Create(a1[k]), bv, s1);
                    s2 = Fma.MultiplyAdd(Vector256.Create(a2[k]), bv, s2);
                }
                Avx.Store(c0 + j, s0);
                if (live > 1) Avx.Store(c0 + cols + j, s1);
                if (live > 2) Avx.Store(c0 + 2 * cols + j, s2);
            }
        }
    }
}
