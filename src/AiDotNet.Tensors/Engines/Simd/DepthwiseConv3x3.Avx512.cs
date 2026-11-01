// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET9_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

// Sixteen-channel blocks for AVX-512 hosts (Zen 4 and later, Intel since Skylake-SP). The eight-channel weight gradient
// loads each of its nine input vectors for every output position (ten loads for nine FMAs: load-bound), because sixteen
// AVX2 registers cannot hold nine accumulators beside a 3x3 window. With thirty-two registers the window slides along the
// row: per output position one new input column (three vectors at stride 1, six at stride 2) and the output gradient.
internal static unsafe partial class DepthwiseConv3x3
{
    /// <summary>Tests only: run the sixteen-channel kernels on a host without AVX-512 (Vector512 is emulated there).</summary>
    internal static bool s_forceAvx512ForTest;

    // Operator opt-out (A/B switch).
    private static readonly bool s_avx512Env = Environment.GetEnvironmentVariable("AIDOTNET_DW_AVX512") != "0";

    private static bool UseAvx512Blocks(int channels, int wd)
        => UseChannelBlocks(channels, wd) && (channels & 15) == 0 && s_avx512Env
            && (Vector512.IsHardwareAccelerated || s_forceAvx512ForTest);

    /// <summary>Sixteen planes [hw] (plane stride hw) into d [hw, 16].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Pack16(float* s, int hw, float* d)
    {
        int q = 0;
        for (; q + 8 <= hw; q += 8)
        {
            Transpose8x8(s + q, hw, d + q * 16, 16);
            Transpose8x8(s + 8L * hw + q, hw, d + q * 16 + 8, 16);
        }
        for (; q < hw; q++)
            for (int l = 0; l < 16; l++) d[q * 16 + l] = s[(long)l * hw + q];
    }

    /// <summary>Sixteen planes [h, wd] into the interior of the zero-bordered tile [(h + 2), (wd + 2), 16], row by row.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PackPadded16(float* s, int h, int wd, float* tile)
    {
        int hw = h * wd, tw = wd + 2;
        for (int i = 0; i < h; i++)
        {
            float* src = s + i * wd, dst = tile + ((i + 1) * tw + 1) * 16;
            int q = 0;
            for (; q + 8 <= wd; q += 8)
            {
                Transpose8x8(src + q, hw, dst + q * 16, 16);
                Transpose8x8(src + 8L * hw + q, hw, dst + q * 16 + 8, 16);
            }
            for (; q < wd; q++)
                for (int l = 0; l < 16; l++) dst[q * 16 + l] = src[(long)l * hw + q];
        }
    }

    /// <summary>dw for channels multiple of sixteen on planes the channel blocks take; see the eight-channel form.</summary>
    private static void BackwardKernelChannelBlocks16(float[] dy, float[] x, float[] dw, int batch, int channels, int h,
        int wd, int stride)
    {
        int oH = (h - 1) / stride + 1, oW = (wd - 1) / stride + 1;
        int blocks = channels / 16;
        CpuParallelSettings.ParallelForOrSerial(0, blocks, (long)batch * channels * oH * oW * 9,
            [MethodImpl(MethodImplOptions.AggressiveOptimization)] (blk) =>
        {
            int tw = wd + 2, tileLen = (h + 2) * tw * 16;
            var tile = Tile(ref t_tile, tileLen);
            var gTile = Tile(ref t_scratch, oH * oW * 16 + 64);
            int c0 = blk * 16;
            Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0;
            fixed (float* px = x, pdy = dy, tp = tile, gp = gTile)
            {
                new Span<float>(tp, tileLen).Clear();
                int rowStep = tw * 16;
                for (int b = 0; b < batch; b++)
                {
                    PackPadded16(px + ((long)b * channels + c0) * h * wd, h, wd, tp);
                    Pack16(pdy + ((long)b * channels + c0) * oH * oW, oH * oW, gp);
                    for (int i = 0; i < oH; i++)
                    {
                        float* r0 = tp + i * stride * rowStep, r1 = r0 + rowStep, r2 = r1 + rowStep;
                        float* g = gp + i * oW * 16;
                        // Window columns c, c + 1 (input columns of output column j at stride s start at s j).
                        var x00 = Vector512.Load(r0); var x10 = Vector512.Load(r1); var x20 = Vector512.Load(r2);
                        if (stride == 1)
                        {
                            var x01 = Vector512.Load(r0 + 16); var x11 = Vector512.Load(r1 + 16); var x21 = Vector512.Load(r2 + 16);
                            for (int j = 0; j < oW; j++)
                            {
                                int c2 = (j + 2) * 16;
                                var x02 = Vector512.Load(r0 + c2); var x12 = Vector512.Load(r1 + c2); var x22 = Vector512.Load(r2 + c2);
                                var gv = Vector512.Load(g + j * 16);
                                a0 = Vector512.FusedMultiplyAdd(gv, x00, a0); a1 = Vector512.FusedMultiplyAdd(gv, x01, a1);
                                a2 = Vector512.FusedMultiplyAdd(gv, x02, a2); a3 = Vector512.FusedMultiplyAdd(gv, x10, a3);
                                a4 = Vector512.FusedMultiplyAdd(gv, x11, a4); a5 = Vector512.FusedMultiplyAdd(gv, x12, a5);
                                a6 = Vector512.FusedMultiplyAdd(gv, x20, a6); a7 = Vector512.FusedMultiplyAdd(gv, x21, a7);
                                a8 = Vector512.FusedMultiplyAdd(gv, x22, a8);
                                x00 = x01; x01 = x02; x10 = x11; x11 = x12; x20 = x21; x21 = x22;
                            }
                        }
                        else
                        {
                            for (int j = 0; j < oW; j++)
                            {
                                int c1 = (2 * j + 1) * 16, c2 = c1 + 16;
                                var x01 = Vector512.Load(r0 + c1); var x11 = Vector512.Load(r1 + c1); var x21 = Vector512.Load(r2 + c1);
                                var x02 = Vector512.Load(r0 + c2); var x12 = Vector512.Load(r1 + c2); var x22 = Vector512.Load(r2 + c2);
                                var gv = Vector512.Load(g + j * 16);
                                a0 = Vector512.FusedMultiplyAdd(gv, x00, a0); a1 = Vector512.FusedMultiplyAdd(gv, x01, a1);
                                a2 = Vector512.FusedMultiplyAdd(gv, x02, a2); a3 = Vector512.FusedMultiplyAdd(gv, x10, a3);
                                a4 = Vector512.FusedMultiplyAdd(gv, x11, a4); a5 = Vector512.FusedMultiplyAdd(gv, x12, a5);
                                a6 = Vector512.FusedMultiplyAdd(gv, x20, a6); a7 = Vector512.FusedMultiplyAdd(gv, x21, a7);
                                a8 = Vector512.FusedMultiplyAdd(gv, x22, a8);
                                x00 = x02; x10 = x12; x20 = x22;
                            }
                        }
                    }
                }
            }
            // Lane l of accumulator k is dw[c0 + l][k].
            float* acc = stackalloc float[144];
            Vector512.Store(a0, acc); Vector512.Store(a1, acc + 16); Vector512.Store(a2, acc + 32);
            Vector512.Store(a3, acc + 48); Vector512.Store(a4, acc + 64); Vector512.Store(a5, acc + 80);
            Vector512.Store(a6, acc + 96); Vector512.Store(a7, acc + 112); Vector512.Store(a8, acc + 128);
            for (int l = 0; l < 16; l++)
                for (int kk = 0; kk < 9; kk++)
                    dw[(c0 + l) * 9 + kk] = acc[kk * 16 + l];
        }, deterministicSafe: true);
    }
}
#endif
