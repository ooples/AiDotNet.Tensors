// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

// Small planes across channels: on 7x7 to 28x28 maps a row is one or two vectors, mostly tail, so the width kernels spent
// their time on copies and scalar tails (b16 960 channels at 7x7, forward 0.67 ms against PyTorch's 0.13). Here a vector
// holds one position of eight channels: each block of eight planes is transposed (8x8 AVX transposes) into a zero-
// bordered [(h + 2), (w + 2), 8] tile, every output is nine FMAs with per-channel weight vectors and no tails, and the
// result is transposed back.
internal static unsafe partial class DepthwiseConv3x3
{
    /// <summary>Widest input plane the channel-blocked kernels take (A/B knob); wider planes run the width kernels.</summary>
    internal static int s_channelBlockMaxWidth = 28;

    /// <summary>Widest plane on which the stride-1 forward and input gradient take the channel blocks: at 28x28 the
    /// width kernels were faster there (b16 192 channels: 0.43 ms against 0.65-0.77), at 14x14 slower (0.27-0.36 against
    /// 0.18-0.20).</summary>
    internal static int s_channelBlockMaxWidthStride1 = 16;

    private static bool UseChannelBlocks(int channels, int wd)
        => wd <= s_channelBlockMaxWidth && (channels & 7) == 0 && Avx2.IsSupported && Fma.IsSupported;

    private static bool UseChannelBlocksStride1Product(int channels, int wd)
        => UseChannelBlocks(channels, wd) && wd <= s_channelBlockMaxWidthStride1;

    [ThreadStatic] private static float[]? t_tile;
    [ThreadStatic] private static float[]? t_tileOut;

    private static float[] Tile(ref float[]? slot, int length)
    {
        var s = slot;
        if (s is null || s.Length < length) slot = s = new float[length];
        return s;
    }

    /// <summary>Forward, or (flip) the stride-1 input gradient over dy, on blocks of eight channels.</summary>
    private static void ForwardChannelBlocks(float[] x, float[] w, float[] y, int batch, int channels, int h, int wd,
        int stride, bool flip)
    {
        int oH = (h - 1) / stride + 1, oW = (wd - 1) / stride + 1;
        int blocks = channels / 8, units = batch * blocks;
        int group = GroupSize(oH * oW * 8), tasks = (units + group - 1) / group;
        CpuParallelSettings.ParallelForOrSerial(0, tasks, (long)batch * channels * oH * oW * 9, t =>
        {
            int tw = wd + 2, tileLen = (h + 2) * tw * 8;
            var tile = Tile(ref t_tile, tileLen);
            var outTile = Tile(ref t_tileOut, (oH * oW + h * wd) * 8 + 64);
            float* k = stackalloc float[72];
            fixed (float* px = x, pw = w, py = y, tp = tile, op = outTile)
            {
                new Span<float>(tp, tileLen).Clear();
                for (int u = t * group, end = Math.Min(units, u + group); u < end; u++)
                {
                    int b = u / blocks, c0 = (u % blocks) * 8;
                    LoadWeights(pw + c0 * 9, k, flip);
                    PackPadded(px + ((long)b * channels + c0) * h * wd, h, wd, tp, op + oH * oW * 8);
                    ConvTile(tp, tw, stride, k, op, oH, oW);
                    Unpack(op, oH * oW, py + ((long)b * channels + c0) * oH * oW);
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>
    /// The stride-2 input gradient on blocks of eight channels, by output parity: with G the output gradient (zero past
    /// its last row and column), dx[2a][2b] = G[a][b] w11; dx[2a][2b+1] = G[a][b+1] w10 + G[a][b] w12;
    /// dx[2a+1][2b] = G[a+1][b] w01 + G[a][b] w21; dx[2a+1][2b+1] = G[a+1][b+1] w00 + G[a+1][b] w02 + G[a][b+1] w20
    /// + G[a][b] w22. Nine FMAs per four outputs (spreading dy onto the input grid took 36).
    /// </summary>
    private static void BackwardInputStride2ChannelBlocks(float[] dy, float[] w, float[] dx, int batch, int channels,
        int h, int wd)
    {
        int oH = (h - 1) / 2 + 1, oW = (wd - 1) / 2 + 1;
        int blocks = channels / 8, units = batch * blocks;
        int group = GroupSize(h * wd * 8), tasks = (units + group - 1) / group;
        CpuParallelSettings.ParallelForOrSerial(0, tasks, (long)batch * channels * h * wd * 9, t =>
        {
            int gw = oW + 1, tileLen = (oH + 1) * gw * 8;
            var tile = Tile(ref t_tile, tileLen);
            var outTile = Tile(ref t_tileOut, (h * wd + oH * oW) * 8 + 64);
            float* k = stackalloc float[72];
            fixed (float* pdy = dy, pw = w, pdx = dx, tp = tile, op = outTile)
            {
                // The last row and column stay zero: every unit writes only the interior.
                new Span<float>(tp, tileLen).Clear();
                float* packed = op + h * wd * 8;
                for (int u = t * group, end = Math.Min(units, u + group); u < end; u++)
                {
                    int b = u / blocks, c0 = (u % blocks) * 8;
                    LoadWeights(pw + c0 * 9, k, flip: false);
                    Pack(pdy + ((long)b * channels + c0) * oH * oW, oH * oW, packed);
                    for (int i = 0; i < oH; i++)
                        for (int j = 0; j < oW; j++)
                            Avx.Store(tp + (i * gw + j) * 8, Avx.LoadVector256(packed + (i * oW + j) * 8));
                    var w00 = Avx.LoadVector256(k); var w01 = Avx.LoadVector256(k + 8); var w02 = Avx.LoadVector256(k + 16);
                    var w10 = Avx.LoadVector256(k + 24); var w11 = Avx.LoadVector256(k + 32); var w12 = Avx.LoadVector256(k + 40);
                    var w20 = Avx.LoadVector256(k + 48); var w21 = Avx.LoadVector256(k + 56); var w22 = Avx.LoadVector256(k + 64);
                    for (int a = 0; a < oH; a++)
                    {
                        float* g0 = tp + a * gw * 8, g1 = g0 + gw * 8;
                        float* even = op + (2 * a) * wd * 8;
                        float* odd = even + wd * 8;
                        bool oddRow = 2 * a + 1 < h;
                        for (int bb = 0; bb < oW; bb++)
                        {
                            var gab = Avx.LoadVector256(g0 + bb * 8);
                            var gab1 = Avx.LoadVector256(g0 + bb * 8 + 8);
                            var ga1b = Avx.LoadVector256(g1 + bb * 8);
                            var ga1b1 = Avx.LoadVector256(g1 + bb * 8 + 8);
                            bool oddCol = 2 * bb + 1 < wd;
                            Avx.Store(even + (2 * bb) * 8, Avx.Multiply(gab, w11));
                            if (oddCol)
                                Avx.Store(even + (2 * bb + 1) * 8, Fma.MultiplyAdd(gab, w12, Avx.Multiply(gab1, w10)));
                            if (oddRow)
                            {
                                Avx.Store(odd + (2 * bb) * 8, Fma.MultiplyAdd(gab, w21, Avx.Multiply(ga1b, w01)));
                                if (oddCol)
                                    Avx.Store(odd + (2 * bb + 1) * 8, Fma.MultiplyAdd(gab, w22, Fma.MultiplyAdd(gab1, w20,
                                        Fma.MultiplyAdd(ga1b, w02, Avx.Multiply(ga1b1, w00)))));
                            }
                        }
                    }
                    Unpack(op, h * wd, pdx + ((long)b * channels + c0) * h * wd);
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>The weight gradient on blocks of eight channels, nine vector accumulators per block over the batch.</summary>
    private static void BackwardKernelChannelBlocks(float[] dy, float[] x, float[] dw, int batch, int channels, int h,
        int wd, int stride)
    {
        int oH = (h - 1) / stride + 1, oW = (wd - 1) / stride + 1;
        int blocks = channels / 8;
        CpuParallelSettings.ParallelForOrSerial(0, blocks, (long)batch * channels * oH * oW * 9, blk =>
        {
            int tw = wd + 2, tileLen = (h + 2) * tw * 8;
            var tile = Tile(ref t_tile, tileLen);
            var outTile = Tile(ref t_tileOut, h * wd * 8 + 64);
            int c0 = blk * 8;
            Vector256<float> a0 = Vector256<float>.Zero, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0;
            var gTile = Tile(ref t_scratch, oH * oW * 8 + 64);
            fixed (float* px = x, pdy = dy, tp = tile, gp = gTile, sp = outTile)
            {
                new Span<float>(tp, tileLen).Clear();
                for (int b = 0; b < batch; b++)
                {
                    PackPadded(px + ((long)b * channels + c0) * h * wd, h, wd, tp, sp);
                    Pack(pdy + ((long)b * channels + c0) * oH * oW, oH * oW, gp);
                    for (int i = 0; i < oH; i++)
                    {
                        float* r0 = tp + (i * stride) * tw * 8, r1 = r0 + tw * 8, r2 = r1 + tw * 8;
                        for (int j = 0; j < oW; j++)
                        {
                            var g = Avx.LoadVector256(gp + (i * oW + j) * 8);
                            int c = j * stride * 8;
                            a0 = Fma.MultiplyAdd(g, Avx.LoadVector256(r0 + c), a0);
                            a1 = Fma.MultiplyAdd(g, Avx.LoadVector256(r0 + c + 8), a1);
                            a2 = Fma.MultiplyAdd(g, Avx.LoadVector256(r0 + c + 16), a2);
                            a3 = Fma.MultiplyAdd(g, Avx.LoadVector256(r1 + c), a3);
                            a4 = Fma.MultiplyAdd(g, Avx.LoadVector256(r1 + c + 8), a4);
                            a5 = Fma.MultiplyAdd(g, Avx.LoadVector256(r1 + c + 16), a5);
                            a6 = Fma.MultiplyAdd(g, Avx.LoadVector256(r2 + c), a6);
                            a7 = Fma.MultiplyAdd(g, Avx.LoadVector256(r2 + c + 8), a7);
                            a8 = Fma.MultiplyAdd(g, Avx.LoadVector256(r2 + c + 16), a8);
                        }
                    }
                }
            }
            // Lane l of accumulator k is dw[c0 + l][k].
            float* acc = stackalloc float[72];
            Avx.Store(acc, a0); Avx.Store(acc + 8, a1); Avx.Store(acc + 16, a2); Avx.Store(acc + 24, a3);
            Avx.Store(acc + 32, a4); Avx.Store(acc + 40, a5); Avx.Store(acc + 48, a6); Avx.Store(acc + 56, a7);
            Avx.Store(acc + 64, a8);
            for (int l = 0; l < 8; l++)
                for (int kk = 0; kk < 9; kk++)
                    dw[(c0 + l) * 9 + kk] = acc[kk * 8 + l];
        }, deterministicSafe: true);
    }

    // k[tap * 8 + l] = w[(c0 + l) * 9 + tap], the taps turned 180 degrees when flip.
    private static void LoadWeights(float* w, float* k, bool flip)
    {
        for (int l = 0; l < 8; l++)
            for (int tap = 0; tap < 9; tap++)
                k[tap * 8 + l] = w[l * 9 + (flip ? 8 - tap : tap)];
    }

    /// <summary>Eight planes [hw] (plane stride hw) into d [hw, 8].</summary>
    private static void Pack(float* s, int hw, float* d)
    {
        int q = 0;
        for (; q + 8 <= hw; q += 8) Transpose8x8(s + q, hw, d + q * 8, 8);
        for (; q < hw; q++)
            for (int l = 0; l < 8; l++) d[q * 8 + l] = s[(long)l * hw + q];
    }

    /// <summary>Eight planes [h, wd] into the interior of the zero-bordered tile [(h + 2), (wd + 2), 8], through
    /// scratch [h * wd, 8].</summary>
    private static void PackPadded(float* s, int h, int wd, float* tile, float* scratch)
    {
        Pack(s, h * wd, scratch);
        int tw = wd + 2;
        for (int i = 0; i < h; i++)
        {
            float* src = scratch + i * wd * 8, dst = tile + ((i + 1) * tw + 1) * 8;
            for (int j = 0; j < wd; j++) Avx.Store(dst + j * 8, Avx.LoadVector256(src + j * 8));
        }
    }

    /// <summary>o [hw, 8] back to eight planes [hw] (plane stride hw).</summary>
    private static void Unpack(float* o, int hw, float* d)
    {
        int q = 0;
        for (; q + 8 <= hw; q += 8) Transpose8x8(o + q * 8, 8, d + q, hw);
        for (; q < hw; q++)
            for (int l = 0; l < 8; l++) d[(long)l * hw + q] = o[q * 8 + l];
    }

    /// <summary>o [oH * oW, 8] = the 3x3 product (padding 1, the stride) over the tile, k [9, 8]. Separate multiply and
    /// add in (kh, kw) order, as the generic loop sums, so the forward equals it bit for bit (border taps add a zero).</summary>
    private static void ConvTile(float* tile, int tw, int stride, float* k, float* o, int oH, int oW)
    {
        var k0 = Avx.LoadVector256(k); var k1 = Avx.LoadVector256(k + 8); var k2 = Avx.LoadVector256(k + 16);
        var k3 = Avx.LoadVector256(k + 24); var k4 = Avx.LoadVector256(k + 32); var k5 = Avx.LoadVector256(k + 40);
        var k6 = Avx.LoadVector256(k + 48); var k7 = Avx.LoadVector256(k + 56); var k8 = Avx.LoadVector256(k + 64);
        for (int i = 0; i < oH; i++)
        {
            float* r0 = tile + i * stride * tw * 8, r1 = r0 + tw * 8, r2 = r1 + tw * 8;
            float* orow = o + i * oW * 8;
            for (int j = 0; j < oW; j++)
            {
                int c = j * stride * 8;
                var acc = Avx.Multiply(Avx.LoadVector256(r0 + c), k0);
                acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(r0 + c + 8), k1));
                acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(r0 + c + 16), k2));
                acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(r1 + c), k3));
                acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(r1 + c + 8), k4));
                acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(r1 + c + 16), k5));
                acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(r2 + c), k6));
                acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(r2 + c + 8), k7));
                acc = Avx.Add(acc, Avx.Multiply(Avx.LoadVector256(r2 + c + 16), k8));
                Avx.Store(orow + j * 8, acc);
            }
        }
    }

    // d[j * dStride + c] = s[c * sStride + j] for an 8x8 block.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose8x8(float* s, long sStride, float* d, long dStride)
    {
        var r0 = Avx.LoadVector256(s); var r1 = Avx.LoadVector256(s + sStride);
        var r2 = Avx.LoadVector256(s + 2 * sStride); var r3 = Avx.LoadVector256(s + 3 * sStride);
        var r4 = Avx.LoadVector256(s + 4 * sStride); var r5 = Avx.LoadVector256(s + 5 * sStride);
        var r6 = Avx.LoadVector256(s + 6 * sStride); var r7 = Avx.LoadVector256(s + 7 * sStride);
        var t0 = Avx.UnpackLow(r0, r1); var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3); var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5); var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7); var t7 = Avx.UnpackHigh(r6, r7);
        var u0 = Avx.Shuffle(t0, t2, 0x44); var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44); var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44); var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44); var u7 = Avx.Shuffle(t5, t7, 0xEE);
        Avx.Store(d, Avx.Permute2x128(u0, u4, 0x20));
        Avx.Store(d + dStride, Avx.Permute2x128(u1, u5, 0x20));
        Avx.Store(d + 2 * dStride, Avx.Permute2x128(u2, u6, 0x20));
        Avx.Store(d + 3 * dStride, Avx.Permute2x128(u3, u7, 0x20));
        Avx.Store(d + 4 * dStride, Avx.Permute2x128(u0, u4, 0x31));
        Avx.Store(d + 5 * dStride, Avx.Permute2x128(u1, u5, 0x31));
        Avx.Store(d + 6 * dStride, Avx.Permute2x128(u2, u6, 0x31));
        Avx.Store(d + 7 * dStride, Avx.Permute2x128(u3, u7, 0x31));
    }
}
#endif
