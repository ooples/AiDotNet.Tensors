// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
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

    /// <summary>Channel bound of the batch-wide passes: with narrow blocking on, a layer with fewer than this many channels
    /// on either side, or no more than this many on both (64 -> 64), runs blocked; wider layers run batch-wide.</summary>
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

    internal static bool s_old = Environment.GetEnvironmentVariable("AIDOTNET_WINOGRAD_OLD") == "1";

    internal static bool UseBlocked(int cIn, int cOut)
        => s_forceBlockedForTest || (s_blockedNarrow && (Math.Min(cIn, cOut) < MinUnblockedChannels
                                                        || Math.Max(cIn, cOut) <= MinUnblockedChannels));

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
                // Input transform: V [36][patches][cIn].
                long vStride = (long)patches * cIn;
                TransformInputBand(xp + xOff, vp, n, r0, r1, tilesW, cIn, height, width, tp, d, t);
                // M_xi [patches x cOut] = V_xi [patches x cIn] U_xi [cIn x cOut].
                for (int xi = 0; xi < Points; xi++)
                    BlockGemm(vp + xi * vStride, cIn, 1, up + (long)xi * cIn * cOut, mp + (long)xi * patches * cOut,
                        patches, cIn, cOut, accumulate: false);
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
                    StoreOutputRows(tp, outW, yp + yOff + ((long)n * cOut + kb * 8) * hw, rowLo, rowHi, width, hw, accumulate, tmp);
                }
            }
        }, deterministicSafe: true);
    }

    // Rows rowLo..rowHi of eight output planes (planes, hw floats apart) from a band tp of [rows][outW][8 channels]
    // (row 0 = rowLo), written or added (accumulate); tmp holds 64 floats.
    private static void StoreOutputRows(float* tp, int outW, float* planes, int rowLo, int rowHi, int width, int hw,
        bool accumulate, float* tmp)
    {
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

    // V [36][patches][cIn] for tile rows r0..r1 of image n: eight channels at a time through a padded band of the image
    // (tp holds (r1 - r0) * 4 + 2 rows of tilesW * 4 + 2 eight-channel columns).
    private static void TransformInputBand(float* x, float* v, int n, int r0, int r1, int tilesW, int cIn, int height,
        int width, float* tp, Vector256<float>* d, Vector256<float>* t)
    {
        int padW = tilesW * Tile + 2, hw = height * width;
        long vStride = (long)(r1 - r0) * tilesW * cIn;
        int firstRow = r0 * Tile - 1;                       // input row of the band's padded row 0
        int bandPadRows = (r1 - r0) * Tile + 2;
        for (int cb = 0; cb < cIn / 8; cb++)
        {
            new Span<float>(tp, bandPadRows * padW * 8).Clear();
            float* planes = x + ((long)n * cIn + cb * 8) * hw;
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
                    StorePoints(d, v + (long)(th * tilesW + tw) * cIn + cb * 8, vStride);
                }
        }
    }

    // D [36][patches][cOut] = A t A^T for tile rows r0..r1 of image n, t the 4x4 output-gradient tiles (zero past the
    // plane); tp holds (r1 - r0) * 4 rows of tilesW * 4 eight-channel columns.
    private static void TransformOutputGradientBand(float* dy, float* dPoints, int n, int r0, int r1, int tilesW, int cOut,
        int height, int width, float* tp, Vector256<float>* g, Vector256<float>* t, Vector256<float>* r)
    {
        int outW = tilesW * Tile, hw = height * width;
        long dStride = (long)(r1 - r0) * tilesW * cOut;
        int rowLo = r0 * Tile, rowHi = Math.Min(height, r1 * Tile);
        int bandFloats = (r1 - r0) * Tile * outW * 8;
        for (int kb = 0; kb < cOut / 8; kb++)
        {
            new Span<float>(tp, bandFloats).Clear();
            float* planes = dy + ((long)n * cOut + kb * 8) * hw;
            for (int h = rowLo; h < rowHi; h++)
            {
                float* src = planes + h * width;
                float* dst = tp + (h - rowLo) * outW * 8;
                int q = 0;
                for (; q + 8 <= width; q += 8) Transpose8x8(src + q, hw, dst + q * 8, 8);
                for (; q < width; q++)
                    for (int l = 0; l < 8; l++) dst[q * 8 + l] = src[(long)l * hw + q];
            }
            for (int th = 0; th < r1 - r0; th++)
                for (int tw = 0; tw < tilesW; tw++)
                {
                    LoadTile(tp + (th * Tile * outW + tw * Tile) * 8, outW * 8, 4, 4, g);
                    for (int b = 0; b < 4; b++) GradientTransform(g, b, 4, t, b, 4);
                    for (int i = 0; i < 6; i++) GradientTransform(t, i * 4, 1, r, i * 6, 1);
                    StorePoints(r, dPoints + (long)(th * tilesW + tw) * cOut + kb * 8, dStride);
                }
        }
    }

    /// <summary>Partial weight gradients of <see cref="BackwardKernelBlocked"/>: a fixed count (not the thread count), so
    /// every host sums the same partials in the same order.</summary>
    private const int WeightGradientSplits = 8;

    // The patch-blocked form of BackwardKernel, for narrow layers: the batch-wide passes hold V and D for every patch
    // (36 x P x (C + K) floats, 9.4 MB at 64 -> 64 on 32x32 planes, batch 8) and stream them through memory. Here each
    // of a fixed number of splits takes a run of bands (about 32 patches), transforms the band's V and D into L2-sized
    // blocks and adds D_xi^T V_xi into its own M; the weight transform sums the splits' M in order.
    private static void BackwardKernelBlocked(float[] x, int xOff, float[] dy, int dyOff, float[] dw, int dwOff,
        bool accumulate, int batch, int cIn, int height, int width, int cOut)
    {
        int tilesH = (height + Tile - 1) / Tile, tilesW = (width + Tile - 1) / Tile;
        int bandRows = Math.Max(1, Math.Min(tilesH, BlockPatches / tilesW));
        int bands = (tilesH + bandRows - 1) / bandRows;
        int totalBands = batch * bands;
        int splits = Math.Min(totalBands, WeightGradientSplits);
        int maxPatches = bandRows * tilesW;
        int padW = tilesW * Tile + 2;
        int tileFloats = Math.Max((bandRows * Tile + 2) * padW * 8, bandRows * Tile * tilesW * Tile * 8);
        long partialFloats = (long)Points * cIn * cOut;
        float[] m = ArrayPool<float>.Shared.Rent((int)(splits * partialFloats));
        try
        {
            CpuParallelSettings.ParallelForOrSerial(0, splits, (long)batch * height * width * cIn * cOut * 4,
                [MethodImpl(MethodImplOptions.AggressiveOptimization)] (split) =>
            {
                int first = (int)((long)split * totalBands / splits), last = (int)((long)(split + 1) * totalBands / splits);
                float[] vb = ThreadBuffer(ref t_blockV, Points * maxPatches * cIn);
                float[] db = ThreadBuffer(ref t_blockM, Points * maxPatches * cOut);
                float[] tb = ThreadBuffer(ref t_blockTile, tileFloats);
                fixed (float* xp = x, gp = dy, mp = m, vp = vb, dp = db, tp = tb)
                {
                    var d = stackalloc Vector256<float>[36];
                    var t = stackalloc Vector256<float>[36];
                    var r = stackalloc Vector256<float>[36];
                    float* mSplit = mp + split * partialFloats;
                    for (int job = first; job < last; job++)
                    {
                        int n = job / bands, band = job - n * bands;
                        int r0 = band * bandRows, r1 = Math.Min(tilesH, r0 + bandRows);
                        int patches = (r1 - r0) * tilesW;
                        TransformInputBand(xp + xOff, vp, n, r0, r1, tilesW, cIn, height, width, tp, d, t);
                        TransformOutputGradientBand(gp + dyOff, dp, n, r0, r1, tilesW, cOut, height, width, tp, d, t, r);
                        // M_xi [cOut x cIn] (+)= D_xi^T [cOut x patches] V_xi [patches x cIn]: D_xi's row o is column o
                        // of its [patches x cOut] block.
                        long vStride = (long)patches * cIn, dStride = (long)patches * cOut;
                        for (int xi = 0; xi < Points; xi++)
                            BlockGemm(dp + xi * dStride, 1, cOut, vp + xi * vStride, mSplit + (long)xi * cOut * cIn,
                                cOut, patches, cIn, accumulate: job != first);
                    }
                }
            }, deterministicSafe: true);
            TransformWeightGradient(m, splits, partialFloats, dw, dwOff, accumulate, cIn, cOut);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(m);
        }
    }

    // c [rows x cols] (=, or += when accumulate) = a [rows x depth] b [depth x cols], b and c row-major, cols a multiple
    // of eight; a's element (i, k) at a[i * aRow + k * aStep]. Three rows x 32 columns per register tile (twelve
    // accumulators; per depth step three broadcasts and four loads), eight-column tiles past the last multiple of 32.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void BlockGemm(float* a, int aRow, int aStep, float* b, float* c, int rows, int depth, int cols,
        bool accumulate)
    {
#if NET9_0_OR_GREATER
        if (UseBlockGemm512)
        {
            BlockGemm512(a, aRow, aStep, b, c, rows, depth, cols, accumulate);
            return;
        }
#endif
        for (int i = 0; i < rows; i += 3)
        {
            int live = Math.Min(3, rows - i);
            float* a0 = a + (long)i * aRow;
            float* a1 = live > 1 ? a0 + aRow : a0;
            float* a2 = live > 2 ? a0 + 2L * aRow : a0;
            float* c0 = c + (long)i * cols;
            // Rows past `live` alias row 0: read (when accumulating) but never stored.
            float* c1 = live > 1 ? c0 + cols : c0;
            float* c2 = live > 2 ? c0 + 2L * cols : c0;
            int j = 0;
            for (; j + 32 <= cols; j += 32)
            {
                Vector256<float> s00 = Vector256<float>.Zero, s01 = s00, s02 = s00, s03 = s00;
                Vector256<float> s10 = s00, s11 = s00, s12 = s00, s13 = s00;
                Vector256<float> s20 = s00, s21 = s00, s22 = s00, s23 = s00;
                if (accumulate)
                {
                    s00 = Avx.LoadVector256(c0 + j); s01 = Avx.LoadVector256(c0 + j + 8);
                    s02 = Avx.LoadVector256(c0 + j + 16); s03 = Avx.LoadVector256(c0 + j + 24);
                    s10 = Avx.LoadVector256(c1 + j); s11 = Avx.LoadVector256(c1 + j + 8);
                    s12 = Avx.LoadVector256(c1 + j + 16); s13 = Avx.LoadVector256(c1 + j + 24);
                    s20 = Avx.LoadVector256(c2 + j); s21 = Avx.LoadVector256(c2 + j + 8);
                    s22 = Avx.LoadVector256(c2 + j + 16); s23 = Avx.LoadVector256(c2 + j + 24);
                }
                float* bk = b + j;
                float* p0 = a0, p1 = a1, p2 = a2;
                for (int k = 0; k < depth; k++, bk += cols, p0 += aStep, p1 += aStep, p2 += aStep)
                {
                    var b0 = Avx.LoadVector256(bk); var b1 = Avx.LoadVector256(bk + 8);
                    var b2 = Avx.LoadVector256(bk + 16); var b3 = Avx.LoadVector256(bk + 24);
                    var g = Vector256.Create(*p0);
                    s00 = Fma.MultiplyAdd(g, b0, s00); s01 = Fma.MultiplyAdd(g, b1, s01);
                    s02 = Fma.MultiplyAdd(g, b2, s02); s03 = Fma.MultiplyAdd(g, b3, s03);
                    g = Vector256.Create(*p1);
                    s10 = Fma.MultiplyAdd(g, b0, s10); s11 = Fma.MultiplyAdd(g, b1, s11);
                    s12 = Fma.MultiplyAdd(g, b2, s12); s13 = Fma.MultiplyAdd(g, b3, s13);
                    g = Vector256.Create(*p2);
                    s20 = Fma.MultiplyAdd(g, b0, s20); s21 = Fma.MultiplyAdd(g, b1, s21);
                    s22 = Fma.MultiplyAdd(g, b2, s22); s23 = Fma.MultiplyAdd(g, b3, s23);
                }
                Avx.Store(c0 + j, s00); Avx.Store(c0 + j + 8, s01); Avx.Store(c0 + j + 16, s02); Avx.Store(c0 + j + 24, s03);
                if (live > 1)
                {
                    Avx.Store(c1 + j, s10); Avx.Store(c1 + j + 8, s11); Avx.Store(c1 + j + 16, s12); Avx.Store(c1 + j + 24, s13);
                }
                if (live > 2)
                {
                    Avx.Store(c2 + j, s20); Avx.Store(c2 + j + 8, s21); Avx.Store(c2 + j + 16, s22); Avx.Store(c2 + j + 24, s23);
                }
            }
            for (; j < cols; j += 8)
            {
                Vector256<float> s0 = Vector256<float>.Zero, s1 = s0, s2 = s0;
                if (accumulate)
                {
                    s0 = Avx.LoadVector256(c0 + j); s1 = Avx.LoadVector256(c1 + j); s2 = Avx.LoadVector256(c2 + j);
                }
                float* bk = b + j;
                float* p0 = a0, p1 = a1, p2 = a2;
                for (int k = 0; k < depth; k++, bk += cols, p0 += aStep, p1 += aStep, p2 += aStep)
                {
                    var bv = Avx.LoadVector256(bk);
                    s0 = Fma.MultiplyAdd(Vector256.Create(*p0), bv, s0);
                    s1 = Fma.MultiplyAdd(Vector256.Create(*p1), bv, s1);
                    s2 = Fma.MultiplyAdd(Vector256.Create(*p2), bv, s2);
                }
                Avx.Store(c0 + j, s0);
                if (live > 1) Avx.Store(c1 + j, s1);
                if (live > 2) Avx.Store(c2 + j, s2);
            }
        }
    }
}
#endif
