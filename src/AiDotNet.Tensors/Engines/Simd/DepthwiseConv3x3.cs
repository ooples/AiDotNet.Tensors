// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Numerics;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Float 3x3 depthwise convolution with padding 1 and stride 1 or 2 (MobileNet's and EfficientNet's every depthwise
/// layer): forward, input gradient and weight gradient, NCHW, one filter per channel. Each plane is copied once into a
/// zero-bordered scratch plane (for stride 2 split into its even and odd columns), so every tap is an unconditional
/// vector load along the output row; the nine taps of a vector of outputs accumulate in a register.
/// <para>
/// The input gradient is the same stride-1 product with the kernel turned 180 degrees, over the output gradient (for
/// stride 2 spread onto the input grid with zeros between). The weight gradient keeps nine vector accumulators per
/// channel across the batch and sums their lanes at the end. Forward and input gradient split over planes, the weight
/// gradient over channels, and every output is summed by one task in one order, so results do not depend on the thread
/// count.
/// </para>
/// <para>
/// The generic loops these replace ran the backward serially and scalar: on a 4-vCPU runner MobileNetV2's b16 depthwise
/// layers took 818 ms for forward plus both gradients against PyTorch's 68 ms.
/// </para>
/// </summary>
internal static unsafe partial class DepthwiseConv3x3
{
    /// <summary>A/B and test switch: true sends every shape to the generic loops.</summary>
    internal static bool s_disable;

    [ThreadStatic] private static float[]? t_scratch;

    /// <summary>Whether these kernels cover the convolution.</summary>
    internal static bool Applies(int multiplier, int kH, int kW, int strideH, int strideW, int padH, int padW)
        => !s_disable && Vector.IsHardwareAccelerated && multiplier == 1 && kH == 3 && kW == 3 && padH == 1 && padW == 1
           && strideH == strideW && (strideH == 1 || strideH == 2);

    private static int VW => Vector<float>.Count;

    private static int RoundUp(int n) => (n + VW - 1) / VW * VW;

    private static float[] Scratch(int length)
    {
        var s = t_scratch;
        if (s is null || s.Length < length) t_scratch = s = new float[length];
        return s;
    }

    // Planes per task: enough outputs that the dispatch and the scratch setup are small next to the work (one 7x7
    // plane per task cost more in dispatch than in arithmetic).
    private const int TaskOutputs = 8192;

    private static int GroupSize(int planeOutputs) => Math.Max(1, TaskOutputs / Math.Max(1, planeOutputs));

    /// <summary>y [B, C, oH, oW] = depthwise 3x3 of x [B, C, H, W], w [C, 1, 3, 3].</summary>
    internal static void Forward(float[] x, float[] w, float[] y, int batch, int channels, int h, int wd, int stride)
    {
#if NET5_0_OR_GREATER
        if (stride == 1 ? UseChannelBlocksStride1Product(channels, wd) : UseChannelBlocks(channels, wd))
        {
            ForwardChannelBlocks(x, w, y, batch, channels, h, wd, stride, flip: false);
            return;
        }
#endif
        int oH = (h - 1) / stride + 1, oW = (wd - 1) / stride + 1;
        int planes = batch * channels;
        int group = GroupSize(oH * oW), tasks = (planes + group - 1) / group;
        CpuParallelSettings.ParallelForOrSerial(0, tasks, (long)planes * oH * oW * 9, t =>
        {
            int rowLen = PaddedRowLength(oW, stride);
            int planeLen = (h + 2) * rowLen;
            var scratch = Scratch(planeLen * stride);
            float* k = stackalloc float[9];
            fixed (float* px = x, pw = w, py = y, sp = scratch)
            {
                // Pad never writes the border, so one clear serves every plane of the task.
                new Span<float>(sp, planeLen * stride).Clear();
                for (int p = t * group, end = Math.Min(planes, p + group); p < end; p++)
                {
                    float* wc = pw + (p % channels) * 9;
                    for (int i = 0; i < 9; i++) k[i] = wc[i];
                    Pad(px + (long)p * h * wd, h, wd, stride, rowLen, sp);
                    ConvPadded(sp, rowLen, planeLen, stride, k, py + (long)p * oH * oW, oH, oW);
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>dx [B, C, H, W] from dy [B, C, oH, oW] and w [C, 1, 3, 3]; every element written.</summary>
    internal static void BackwardInput(float[] dy, float[] w, float[] dx, int batch, int channels, int h, int wd, int stride)
    {
#if NET5_0_OR_GREATER
        if (stride == 1 && UseChannelBlocksStride1Product(channels, wd))
        {
            ForwardChannelBlocks(dy, w, dx, batch, channels, h, wd, 1, flip: true);
            return;
        }
        if (stride == 2 && UseChannelBlocks(channels, wd))
        {
            BackwardInputStride2ChannelBlocks(dy, w, dx, batch, channels, h, wd);
            return;
        }
#endif
        int oH = (h - 1) / stride + 1, oW = (wd - 1) / stride + 1;
        int planes = batch * channels;
        int group = GroupSize(h * wd), tasks = (planes + group - 1) / group;
        CpuParallelSettings.ParallelForOrSerial(0, tasks, (long)planes * h * wd * 9, t =>
        {
            // dx[i][j] = sum over (a, b) of D[i - 1 + a][j - 1 + b] * w[2 - a][2 - b], D the output gradient on the input
            // grid (stride 2: at even coordinates, zero between), i.e. the stride-1 product of D, zero-bordered, with the
            // kernel turned 180 degrees.
            int rowLen = PaddedRowLength(wd, 1);
            int planeLen = (h + 2) * rowLen;
            var scratch = Scratch(planeLen);
            float* k = stackalloc float[9];
            fixed (float* pdy = dy, pw = w, pdx = dx, sp = scratch)
            {
                // Every plane writes the same positions, so the zeros between them survive from this one clear.
                new Span<float>(sp, planeLen).Clear();
                for (int p = t * group, end = Math.Min(planes, p + group); p < end; p++)
                {
                    float* wc = pw + (p % channels) * 9;
                    for (int i = 0; i < 9; i++) k[i] = wc[8 - i];
                    float* src = pdy + (long)p * oH * oW;
                    if (stride == 1)
                    {
                        Pad(src, h, wd, 1, rowLen, sp);
                    }
                    else
                    {
                        for (int oh = 0; oh < oH; oh++)
                        {
                            float* dst = sp + (2 * oh + 1) * rowLen + 1;
                            float* row = src + oh * oW;
                            for (int ow = 0; ow < oW; ow++) dst[2 * ow] = row[ow];
                        }
                    }
                    ConvPadded(sp, rowLen, planeLen, 1, k, pdx + (long)p * h * wd, h, wd);
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>Row length of the zero-bordered scratch plane(s) for an output row of oW at the stride: room for the
    /// last vector's three taps.</summary>
    private static int PaddedRowLength(int oW, int stride) => stride == 1 ? RoundUp(oW) + 2 : RoundUp(oW) + 1;

    /// <summary>dw [C, 1, 3, 3] from dy [B, C, oH, oW] and x [B, C, H, W]; summed over the batch per channel.</summary>
    internal static void BackwardKernel(float[] dy, float[] x, float[] dw, int batch, int channels, int h, int wd, int stride)
    {
#if NET5_0_OR_GREATER
        if (UseChannelBlocks(channels, wd))
        {
            BackwardKernelChannelBlocks(dy, x, dw, batch, channels, h, wd, stride);
            return;
        }
#endif
        int oH = (h - 1) / stride + 1, oW = (wd - 1) / stride + 1;
        CpuParallelSettings.ParallelForOrSerial(0, channels, (long)batch * channels * oH * oW * 9, c =>
        {
            int vw = VW;
            int rowLen = PaddedRowLength(oW, stride);
            int planeLen = (h + 2) * rowLen;
            var scratch = Scratch(planeLen * stride + vw);
            Vector<float> a0 = Vector<float>.Zero, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0;
            fixed (float* s = scratch)
            {
                float* g = s + planeLen * stride;   // the zero-padded tail chunk of a dy row
                new Span<float>(s, planeLen * stride).Clear();
                for (int b = 0; b < batch; b++)
                {
                    long p = (long)b * channels + c;
                    fixed (float* px = &x[p * h * wd])
                    fixed (float* pdy = &dy[p * oH * oW])
                    {
                        Pad(px, h, wd, stride, rowLen, s);
                        float* e = s, o = s + planeLen;
                        for (int oh = 0; oh < oH; oh++)
                        {
                            float* dyRow = pdy + oh * oW;
                            for (int ow = 0; ow < oW; ow += vw)
                            {
                                Vector<float> gv;
                                if (ow + vw <= oW) gv = *(Vector<float>*)(dyRow + ow);
                                else
                                {
                                    for (int i = 0; i < vw; i++) g[i] = ow + i < oW ? dyRow[ow + i] : 0f;
                                    gv = *(Vector<float>*)g;
                                }
                                if (stride == 1)
                                {
                                    float* r0 = e + oh * rowLen + ow, r1 = r0 + rowLen, r2 = r1 + rowLen;
                                    a0 += gv * *(Vector<float>*)r0; a1 += gv * *(Vector<float>*)(r0 + 1); a2 += gv * *(Vector<float>*)(r0 + 2);
                                    a3 += gv * *(Vector<float>*)r1; a4 += gv * *(Vector<float>*)(r1 + 1); a5 += gv * *(Vector<float>*)(r1 + 2);
                                    a6 += gv * *(Vector<float>*)r2; a7 += gv * *(Vector<float>*)(r2 + 1); a8 += gv * *(Vector<float>*)(r2 + 2);
                                }
                                else
                                {
                                    int r = 2 * oh * rowLen + ow;
                                    float* e0 = e + r, o0 = o + r, e1 = e0 + rowLen, o1 = o0 + rowLen, e2 = e1 + rowLen, o2 = o1 + rowLen;
                                    a0 += gv * *(Vector<float>*)e0; a1 += gv * *(Vector<float>*)o0; a2 += gv * *(Vector<float>*)(e0 + 1);
                                    a3 += gv * *(Vector<float>*)e1; a4 += gv * *(Vector<float>*)o1; a5 += gv * *(Vector<float>*)(e1 + 1);
                                    a6 += gv * *(Vector<float>*)e2; a7 += gv * *(Vector<float>*)o2; a8 += gv * *(Vector<float>*)(e2 + 1);
                                }
                            }
                        }
                    }
                }
            }
            int k = c * 9;
            dw[k] = Vector.Dot(a0, Vector<float>.One); dw[k + 1] = Vector.Dot(a1, Vector<float>.One);
            dw[k + 2] = Vector.Dot(a2, Vector<float>.One); dw[k + 3] = Vector.Dot(a3, Vector<float>.One);
            dw[k + 4] = Vector.Dot(a4, Vector<float>.One); dw[k + 5] = Vector.Dot(a5, Vector<float>.One);
            dw[k + 6] = Vector.Dot(a6, Vector<float>.One); dw[k + 7] = Vector.Dot(a7, Vector<float>.One);
            dw[k + 8] = Vector.Dot(a8, Vector<float>.One);
        }, deterministicSafe: true);
    }

    /// <summary>
    /// Copies plane src [h, wd] into scratch, whose border the caller has zeroed (this never writes it): stride 1 one plane P[r][c] = src[r - 1][c - 1]; stride 2
    /// its even columns E[r][j] = P[r][2j] then its odd columns O[r][j] = P[r][2j + 1], each (h + 2) x rowLen.
    /// </summary>
    private static void Pad(float* src, int h, int wd, int stride, int rowLen, float* dst)
    {
        int planeLen = (h + 2) * rowLen;
        if (stride == 1)
        {
            for (int r = 0; r < h; r++)
                Buffer.MemoryCopy(src + r * wd, dst + (r + 1) * rowLen + 1, (long)wd * 4, (long)wd * 4);
            return;
        }
        float* odd = dst + planeLen;
        for (int r = 0; r < h; r++)
        {
            float* s = src + r * wd;
            float* ev = dst + (r + 1) * rowLen, od = odd + (r + 1) * rowLen;
            // P column c holds src column c - 1: even P columns 2j take src 2j - 1, odd ones 2j + 1 take src 2j.
            int j = 0;
#if NET5_0_OR_GREATER
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            {
                // 16 source columns per pass: in-lane shuffles pick the even and odd columns of each 128-bit half, and a
                // 64-bit permute puts the halves in order.
                for (; 2 * j + 16 <= wd; j += 8)
                {
                    var a = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 2 * j);
                    var b = System.Runtime.Intrinsics.X86.Avx.LoadVector256(s + 2 * j + 8);
                    var evens = System.Runtime.Intrinsics.X86.Avx.Shuffle(a, b, 0b10_00_10_00);
                    var odds = System.Runtime.Intrinsics.X86.Avx.Shuffle(a, b, 0b11_01_11_01);
                    evens = System.Runtime.Intrinsics.X86.Avx2.Permute4x64(evens.AsDouble(), 0b11_01_10_00).AsSingle();
                    odds = System.Runtime.Intrinsics.X86.Avx2.Permute4x64(odds.AsDouble(), 0b11_01_10_00).AsSingle();
                    System.Runtime.Intrinsics.X86.Avx.Store(od + j, evens);
                    System.Runtime.Intrinsics.X86.Avx.Store(ev + j + 1, odds);
                }
            }
#endif
            for (; 2 * j < wd; j++)
            {
                od[j] = s[2 * j];
                if (2 * j + 1 < wd) ev[j + 1] = s[2 * j + 1];
            }
        }
    }

    /// <summary>One plane's 3x3 product (kernel k) at the given stride from its padded scratch into y [oH, oW].</summary>
    private static void ConvPadded(float* s, int rowLen, int planeLen, int stride, float* k, float* y, int oH, int oW)
    {
        if (stride == 1)
        {
            Stride1FromPadded(s, rowLen, k, y, oH, oW);
            return;
        }
        int vw = VW;
        float* tail = stackalloc float[vw];
        Vector<float> k0 = new(k[0]), k1 = new(k[1]), k2 = new(k[2]), k3 = new(k[3]), k4 = new(k[4]),
            k5 = new(k[5]), k6 = new(k[6]), k7 = new(k[7]), k8 = new(k[8]);
        float* e = s, o = s + planeLen;
        for (int oh = 0; oh < oH; oh++)
        {
            float* yRow = y + oh * oW;
            for (int ow = 0; ow < oW; ow += vw)
            {
                int r = 2 * oh * rowLen + ow;
                float* e0 = e + r, o0 = o + r, e1 = e0 + rowLen, o1 = o0 + rowLen, e2 = e1 + rowLen, o2 = o1 + rowLen;
                var acc = *(Vector<float>*)e0 * k0 + *(Vector<float>*)o0 * k1 + *(Vector<float>*)(e0 + 1) * k2
                    + *(Vector<float>*)e1 * k3 + *(Vector<float>*)o1 * k4 + *(Vector<float>*)(e1 + 1) * k5
                    + *(Vector<float>*)e2 * k6 + *(Vector<float>*)o2 * k7 + *(Vector<float>*)(e2 + 1) * k8;
                Store(acc, yRow, ow, oW, tail);
            }
        }
    }

    /// <summary>Stride-1 3x3 product over a zero-bordered plane P (row length rowLen) into y [oH, oW].</summary>
    private static void Stride1FromPadded(float* p, int rowLen, float* k, float* y, int oH, int oW)
    {
        int vw = VW;
        float* tail = stackalloc float[vw];
        Vector<float> k0 = new(k[0]), k1 = new(k[1]), k2 = new(k[2]), k3 = new(k[3]), k4 = new(k[4]),
            k5 = new(k[5]), k6 = new(k[6]), k7 = new(k[7]), k8 = new(k[8]);
        for (int oh = 0; oh < oH; oh++)
        {
            float* yRow = y + oh * oW;
            float* r0 = p + oh * rowLen, r1 = r0 + rowLen, r2 = r1 + rowLen;
            for (int ow = 0; ow < oW; ow += vw)
            {
                var acc = *(Vector<float>*)(r0 + ow) * k0 + *(Vector<float>*)(r0 + ow + 1) * k1 + *(Vector<float>*)(r0 + ow + 2) * k2
                    + *(Vector<float>*)(r1 + ow) * k3 + *(Vector<float>*)(r1 + ow + 1) * k4 + *(Vector<float>*)(r1 + ow + 2) * k5
                    + *(Vector<float>*)(r2 + ow) * k6 + *(Vector<float>*)(r2 + ow + 1) * k7 + *(Vector<float>*)(r2 + ow + 2) * k8;
                Store(acc, yRow, ow, oW, tail);
            }
        }
    }

    private static void Store(Vector<float> v, float* row, int ow, int oW, float* tail)
    {
        int vw = VW;
        if (ow + vw <= oW)
        {
            *(Vector<float>*)(row + ow) = v;
            return;
        }
        *(Vector<float>*)tail = v;
        for (int i = 0; ow + i < oW; i++) row[ow + i] = tail[i];
    }
}
