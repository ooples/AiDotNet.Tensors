// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET9_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines.Simd;

// The AVX-512 weight-gradient tile. The AVX2 tile holds twelve 8-lane accumulators (four output channels x three kernel
// columns) and spends three loads and four broadcasts per twelve FMAs. With thirty-two registers one pass holds all
// eight output channels of the block x three columns, sixteen input channels wide: per position three 512-bit loads and
// eight broadcasts feed twenty-four 16-lane FMAs. The input is packed sixteen channels per pixel (PackInputWide) so each
// load is a single vmovups; the eight-wide packing needed a vinsertf64x4 join per load, which competes with the FMAs
// for port 5 on Intel server cores (#1445).
internal static partial class DirectConvAvx2
{
    private const int AccumulatorsPerPass512 = 24;

    /// <summary>Whether <see cref="BackwardKernel"/> runs <see cref="BackwardKernelTile512"/> for this input width.</summary>
    private static bool UseAvx512BackwardKernel(int inChannels) => UseAvx512Forward && inChannels % (2 * Block) == 0;

    /// <summary>
    /// dW for output block <paramref name="ob"/> x input blocks 2 <paramref name="ib2"/> and 2 <paramref name="ib2"/> + 1,
    /// as <see cref="BackwardKernelTile"/> computes it: each output's FMA chain is the same (positions in the same order,
    /// from zero, one rounding per FMA), so the results are bit-identical. <paramref name="packedInput"/> is sixteen-wide
    /// (<see cref="PackInputWide"/>), <paramref name="planeIn"/> one sixteen-channel plane and the offsets in its pixels.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void BackwardKernelTile512(float[] packedInput, float[] packedGrad, int[] offsets, float[] dest,
        int destOffset, bool accumulate, int ob, int ib2, int batchBegin, int batchEnd, int inChannels, int outBlocks,
        int paddedW, int planeIn, int kernelHeight, int kernelWidth, int dilationH, int dilationW, int positions)
    {
        const int Lanes = 2 * Block;
        int ib = 2 * ib2;
        int columnGroups = (kernelWidth + 2) / 3;
        int passes = kernelHeight * columnGroups;
        int parkFloats = passes * AccumulatorsPerPass512 * Lanes;
        float* parked = stackalloc float[parkFloats];
        new Span<float>(parked, parkFloats).Clear();
        int step = dilationW * Lanes;
        fixed (float* px = packedInput)
        fixed (float* pg = packedGrad)
        fixed (int* pOff = offsets)
        fixed (float* pd = dest)
        {
            float* xb = px + (long)ib2 * planeIn;
            for (int b = batchBegin; b < batchEnd; b++)
            {
                float* gImage = pg + (long)(b * outBlocks + ob) * positions * Block;
                int* oImage = pOff + b * positions;
                for (int q0 = 0; q0 < positions; q0 += KernelGradPositionChunk)
                {
                    int q1 = Math.Min(positions, q0 + KernelGradPositionChunk);
                    int pass = 0;
                    for (int y = 0; y < kernelHeight; y++)
                    {
                        for (int x0 = 0; x0 < kernelWidth; x0 += 3, pass++)
                        {
                            int columns = Math.Min(3, kernelWidth - x0);
                            float* park = parked + pass * AccumulatorsPerPass512 * Lanes;
                            float* xTap = xb + (y * dilationH * paddedW + x0 * dilationW) * Lanes;
                            float* gq = gImage + q0 * Block;
                            if (columns == 3) TilePass512Three(park, xTap, gq, oImage, q0, q1, step);
                            else TilePass512(park, xTap, gq, oImage, q0, q1, step, columns);
                        }
                    }
                }
            }

            // dW[ob*8 + j, ib*8 + lane, y, x0 + k] = lane of accumulator (j, k).
            int taps = kernelHeight * kernelWidth;
            float* d = pd + destOffset;
            int passIndex = 0;
            for (int y = 0; y < kernelHeight; y++)
            {
                for (int x0 = 0; x0 < kernelWidth; x0 += 3, passIndex++)
                {
                    int columns = Math.Min(3, kernelWidth - x0);
                    float* park = parked + passIndex * AccumulatorsPerPass512 * Lanes;
                    for (int j = 0; j < Block; j++)
                    {
                        float* row = d + ((long)(ob * Block + j) * inChannels + ib * Block) * taps + y * kernelWidth + x0;
                        for (int k = 0; k < columns; k++)
                        {
                            float* acc = park + (j * 3 + k) * Lanes;
                            float* target = row + k;
                            if (accumulate)
                                for (int lane = 0; lane < Lanes; lane++) target[(long)lane * taps] += acc[lane];
                            else
                                for (int lane = 0; lane < Lanes; lane++) target[(long)lane * taps] = acc[lane];
                        }
                    }
                }
            }
        }
    }

    // TilePass512 for a full group of three kernel columns (every column of a 3x3 kernel): no per-position column tests.
    [MethodImpl(Hot)]
    private static unsafe void TilePass512Three(float* park, float* xTap, float* gq, int* oImage, int q0, int q1, int step)
    {
        var a00 = Vector512.Load(park); var a01 = Vector512.Load(park + 16); var a02 = Vector512.Load(park + 32);
        var a10 = Vector512.Load(park + 48); var a11 = Vector512.Load(park + 64); var a12 = Vector512.Load(park + 80);
        var a20 = Vector512.Load(park + 96); var a21 = Vector512.Load(park + 112); var a22 = Vector512.Load(park + 128);
        var a30 = Vector512.Load(park + 144); var a31 = Vector512.Load(park + 160); var a32 = Vector512.Load(park + 176);
        var a40 = Vector512.Load(park + 192); var a41 = Vector512.Load(park + 208); var a42 = Vector512.Load(park + 224);
        var a50 = Vector512.Load(park + 240); var a51 = Vector512.Load(park + 256); var a52 = Vector512.Load(park + 272);
        var a60 = Vector512.Load(park + 288); var a61 = Vector512.Load(park + 304); var a62 = Vector512.Load(park + 320);
        var a70 = Vector512.Load(park + 336); var a71 = Vector512.Load(park + 352); var a72 = Vector512.Load(park + 368);
        for (int q = q0; q < q1; q++, gq += Block)
        {
            float* xs = xTap + oImage[q];
            var x0v = Vector512.Load(xs);
            var x1v = Vector512.Load(xs + step);
            var x2v = Vector512.Load(xs + 2 * step);
            var g = Vector512.Create(gq[0]);
            a00 = Vector512.FusedMultiplyAdd(x0v, g, a00); a01 = Vector512.FusedMultiplyAdd(x1v, g, a01); a02 = Vector512.FusedMultiplyAdd(x2v, g, a02);
            g = Vector512.Create(gq[1]);
            a10 = Vector512.FusedMultiplyAdd(x0v, g, a10); a11 = Vector512.FusedMultiplyAdd(x1v, g, a11); a12 = Vector512.FusedMultiplyAdd(x2v, g, a12);
            g = Vector512.Create(gq[2]);
            a20 = Vector512.FusedMultiplyAdd(x0v, g, a20); a21 = Vector512.FusedMultiplyAdd(x1v, g, a21); a22 = Vector512.FusedMultiplyAdd(x2v, g, a22);
            g = Vector512.Create(gq[3]);
            a30 = Vector512.FusedMultiplyAdd(x0v, g, a30); a31 = Vector512.FusedMultiplyAdd(x1v, g, a31); a32 = Vector512.FusedMultiplyAdd(x2v, g, a32);
            g = Vector512.Create(gq[4]);
            a40 = Vector512.FusedMultiplyAdd(x0v, g, a40); a41 = Vector512.FusedMultiplyAdd(x1v, g, a41); a42 = Vector512.FusedMultiplyAdd(x2v, g, a42);
            g = Vector512.Create(gq[5]);
            a50 = Vector512.FusedMultiplyAdd(x0v, g, a50); a51 = Vector512.FusedMultiplyAdd(x1v, g, a51); a52 = Vector512.FusedMultiplyAdd(x2v, g, a52);
            g = Vector512.Create(gq[6]);
            a60 = Vector512.FusedMultiplyAdd(x0v, g, a60); a61 = Vector512.FusedMultiplyAdd(x1v, g, a61); a62 = Vector512.FusedMultiplyAdd(x2v, g, a62);
            g = Vector512.Create(gq[7]);
            a70 = Vector512.FusedMultiplyAdd(x0v, g, a70); a71 = Vector512.FusedMultiplyAdd(x1v, g, a71); a72 = Vector512.FusedMultiplyAdd(x2v, g, a72);
        }
        Vector512.Store(a00, park); Vector512.Store(a01, park + 16); Vector512.Store(a02, park + 32);
        Vector512.Store(a10, park + 48); Vector512.Store(a11, park + 64); Vector512.Store(a12, park + 80);
        Vector512.Store(a20, park + 96); Vector512.Store(a21, park + 112); Vector512.Store(a22, park + 128);
        Vector512.Store(a30, park + 144); Vector512.Store(a31, park + 160); Vector512.Store(a32, park + 176);
        Vector512.Store(a40, park + 192); Vector512.Store(a41, park + 208); Vector512.Store(a42, park + 224);
        Vector512.Store(a50, park + 240); Vector512.Store(a51, park + 256); Vector512.Store(a52, park + 272);
        Vector512.Store(a60, park + 288); Vector512.Store(a61, park + 304); Vector512.Store(a62, park + 320);
        Vector512.Store(a70, park + 336); Vector512.Store(a71, park + 352); Vector512.Store(a72, park + 368);
    }

    // One pass over positions [q0, q1): accumulator (j, k) at park + (3 j + k) 16, output channel j, kernel column k.
    // Columns past `columns` multiply zeros into accumulators that are never written out.
    [MethodImpl(Hot)]
    private static unsafe void TilePass512(float* park, float* xTap, float* gq, int* oImage, int q0, int q1, int step,
        int columns)
    {
        var a00 = Vector512.Load(park); var a01 = Vector512.Load(park + 16); var a02 = Vector512.Load(park + 32);
        var a10 = Vector512.Load(park + 48); var a11 = Vector512.Load(park + 64); var a12 = Vector512.Load(park + 80);
        var a20 = Vector512.Load(park + 96); var a21 = Vector512.Load(park + 112); var a22 = Vector512.Load(park + 128);
        var a30 = Vector512.Load(park + 144); var a31 = Vector512.Load(park + 160); var a32 = Vector512.Load(park + 176);
        var a40 = Vector512.Load(park + 192); var a41 = Vector512.Load(park + 208); var a42 = Vector512.Load(park + 224);
        var a50 = Vector512.Load(park + 240); var a51 = Vector512.Load(park + 256); var a52 = Vector512.Load(park + 272);
        var a60 = Vector512.Load(park + 288); var a61 = Vector512.Load(park + 304); var a62 = Vector512.Load(park + 320);
        var a70 = Vector512.Load(park + 336); var a71 = Vector512.Load(park + 352); var a72 = Vector512.Load(park + 368);
        for (int q = q0; q < q1; q++, gq += Block)
        {
            float* xs = xTap + oImage[q];
            var x0v = Vector512.Load(xs);
            var x1v = columns > 1 ? Vector512.Load(xs + step) : Vector512<float>.Zero;
            var x2v = columns > 2 ? Vector512.Load(xs + 2 * step) : Vector512<float>.Zero;
            var g = Vector512.Create(gq[0]);
            a00 = Vector512.FusedMultiplyAdd(x0v, g, a00); a01 = Vector512.FusedMultiplyAdd(x1v, g, a01); a02 = Vector512.FusedMultiplyAdd(x2v, g, a02);
            g = Vector512.Create(gq[1]);
            a10 = Vector512.FusedMultiplyAdd(x0v, g, a10); a11 = Vector512.FusedMultiplyAdd(x1v, g, a11); a12 = Vector512.FusedMultiplyAdd(x2v, g, a12);
            g = Vector512.Create(gq[2]);
            a20 = Vector512.FusedMultiplyAdd(x0v, g, a20); a21 = Vector512.FusedMultiplyAdd(x1v, g, a21); a22 = Vector512.FusedMultiplyAdd(x2v, g, a22);
            g = Vector512.Create(gq[3]);
            a30 = Vector512.FusedMultiplyAdd(x0v, g, a30); a31 = Vector512.FusedMultiplyAdd(x1v, g, a31); a32 = Vector512.FusedMultiplyAdd(x2v, g, a32);
            g = Vector512.Create(gq[4]);
            a40 = Vector512.FusedMultiplyAdd(x0v, g, a40); a41 = Vector512.FusedMultiplyAdd(x1v, g, a41); a42 = Vector512.FusedMultiplyAdd(x2v, g, a42);
            g = Vector512.Create(gq[5]);
            a50 = Vector512.FusedMultiplyAdd(x0v, g, a50); a51 = Vector512.FusedMultiplyAdd(x1v, g, a51); a52 = Vector512.FusedMultiplyAdd(x2v, g, a52);
            g = Vector512.Create(gq[6]);
            a60 = Vector512.FusedMultiplyAdd(x0v, g, a60); a61 = Vector512.FusedMultiplyAdd(x1v, g, a61); a62 = Vector512.FusedMultiplyAdd(x2v, g, a62);
            g = Vector512.Create(gq[7]);
            a70 = Vector512.FusedMultiplyAdd(x0v, g, a70); a71 = Vector512.FusedMultiplyAdd(x1v, g, a71); a72 = Vector512.FusedMultiplyAdd(x2v, g, a72);
        }
        Vector512.Store(a00, park); Vector512.Store(a01, park + 16); Vector512.Store(a02, park + 32);
        Vector512.Store(a10, park + 48); Vector512.Store(a11, park + 64); Vector512.Store(a12, park + 80);
        Vector512.Store(a20, park + 96); Vector512.Store(a21, park + 112); Vector512.Store(a22, park + 128);
        Vector512.Store(a30, park + 144); Vector512.Store(a31, park + 160); Vector512.Store(a32, park + 176);
        Vector512.Store(a40, park + 192); Vector512.Store(a41, park + 208); Vector512.Store(a42, park + 224);
        Vector512.Store(a50, park + 240); Vector512.Store(a51, park + 256); Vector512.Store(a52, park + 272);
        Vector512.Store(a60, park + 288); Vector512.Store(a61, park + 304); Vector512.Store(a62, park + 320);
        Vector512.Store(a70, park + 336); Vector512.Store(a71, park + 352); Vector512.Store(a72, park + 368);
    }
}
#endif
