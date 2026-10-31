// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// Fused Winograd F(2x2, 3x3) forward convolution for OpenCL, ported from the CUDA kernel
    /// (<c>CUDA/Kernels/CudaConvWinogradFusedKernels.cs</c>) to 256-thread work groups. The unfused route stores the input
    /// transform V[16][C][P] and the products M[16][K][P] in global memory between launches; here each work group
    /// transforms its input patches and filters into local memory, runs the sixteen per-point products from there and
    /// applies the output transform in its epilogue. The arithmetic is the unfused route's transforms, fp32, with the
    /// reduction in channel order.
    /// </summary>
    /// <remarks>
    /// A work group owns 32 output channels x 32 output tiles x all 16 components. Every work item loads one 4x4 input
    /// patch (channel tid &gt;&gt; 5, tile tid &amp; 31) and one filter (channel tid &amp; 7, output channel tid &gt;&gt; 3)
    /// per 8-channel step. For the products, work item (x = tid &gt;&gt; 4, lane = tid &amp; 15) owns component x and an
    /// 8 (output channel) x 8 (tile) sub-block.
    /// </remarks>
    internal static class WinogradFusedKernels
    {
        internal const string Forward = "wino_f23_fused_forward";

        internal const string SplitReduce = "wino_f23_split_reduce";

        // Built with these kernels for their shared conv_epilogue: a direct convolution over few input channels.
        internal const string SmallChannel = "conv2d_smallc";
        internal const string SmallChannel3x3 = "conv2d_smallc3";
        internal const string CenterTapGather = "conv_center_tap_gather";

        /// <summary>Output channels one conv2d_smallc work item computes (SC_OB in the source).</summary>
        internal const int SmallChannelBlock = 16;

        /// <summary>
        /// The most filter taps (input channels x kernel height x kernel width) conv2d_smallc holds in local memory
        /// (SC_MAX_TAPS in the source): its ws[SC_OB * SC_MAX_TAPS] array. The dispatcher admits no larger filter.
        /// </summary>
        internal const int SmallChannelMaxTaps = 196;

        /// <summary>The widest filter row conv2d_smallc loads at once (SC_MAX_KW in the source); the dispatcher admits no wider.</summary>
        internal const int SmallChannelMaxKernelWidth = 7;

        internal static string[] GetKernelNames() => new[] { Forward, SplitReduce, SmallChannel, SmallChannel3x3, CenterTapGather };

        internal const string Source = @"
#define WF_BO 32
#define WF_BP 32
#define WF_CK 8
#define WF_ULD 36
// A convolution's epilogue: epi bit 0 adds bias[channel], bit 1 applies ReLU (NaN passes through, as the relu kernel).
// The same arithmetic as the conv2d_bias_add and relu passes it replaces.
float conv_epilogue(float v, __global const float* bias, int channel, int epi)
{
    if (epi & 1) v += bias[channel];
    if (epi & 2) v = isnan(v) ? v : fmax(0.0f, v);
    return v;
}
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void wino_f23_fused_forward(__global const float* restrict in, __global const float* restrict wt,
    __global float* restrict out, const int N, const int R, const int IH, const int IW, const int O, const int OH,
    const int OW, const int pad, const int tilesH, const int tilesW, __global const float* bias, const int epi,
    const int rChunk)
{
    // Work-group layer z reduces channels [z * rChunk, z * rChunk + rChunk) and writes its partial output plane z
    // (the host then sums the planes and applies the epilogue, passing epi = 0 here); one layer is the whole sum.
    const int rBegin = get_group_id(2) * rChunk, rEnd = min(R, rBegin + rChunk);
    out += (long)get_group_id(2) * N * O * OH * OW;
    // U | V during the reduction; the epilogue reuses the first 16 x 16 x 32 floats.
    __local float sm[16 * WF_CK * WF_ULD + 16 * WF_CK * 32];
    __local float* us = sm;
    __local float* vs = sm + 16 * WF_CK * WF_ULD;
    const int tid = get_local_id(0);
    const int tilesPerImage = tilesH * tilesW;
    const int P = N * tilesPerImage;
    const int p0 = get_group_id(0) * WF_BP, o0 = get_group_id(1) * WF_BO;

    // Patch loader: channel lc, tile lp.
    const int lp = tid & 31, lc = tid >> 5;
    const int gp = p0 + lp;
    uint pm = 0;
    long inOff = 0;
    if (gp < P)
    {
        int n = gp / tilesPerImage, t = gp - n * tilesPerImage;
        int th = t / tilesW, tw = t - th * tilesW;
        int h0 = th * 2 - pad, w0 = tw * 2 - pad;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (h0 + i >= 0 && h0 + i < IH && w0 + j >= 0 && w0 + j < IW) pm |= 1u << (i * 4 + j);
        inOff = (long)n * R * IH * IW + (long)h0 * IW + w0;
    }
    const long chanStride = (long)IH * IW;
    // Filter loader: reduction channel fr, output channel fo.
    const int fr = tid & 7, fo = tid >> 3;

    // Products: component x, an 8 x 8 (output channel, tile) sub-block.
    const int x = tid >> 4, lane = tid & 15;
    const int oo = (lane >> 2) * 8, pp = (lane & 3) * 8;
    float acc[8][8];
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    const int steps = (rEnd - rBegin + WF_CK - 1) / WF_CK;
    // The next step's patch and filter are read into registers before this step's products, so the global loads
    // overlap the arithmetic (one local buffer; the barriers order the stores against the reads).
    float d[16], g[9];
    #define WF_LOAD(R0) {         const int r_ = (R0) + lc;         const uint m_ = r_ < rEnd ? pm : 0u;         __global const float* src_ = in + inOff + (long)r_ * chanStride;         for (int i = 0; i < 4; i++)             for (int j = 0; j < 4; j++) d[i * 4 + j] = ((m_ >> (i * 4 + j)) & 1u) ? src_[i * IW + j] : 0.0f;         const int wr_ = (R0) + fr, wo_ = o0 + fo;         if (wr_ < rEnd && wo_ < O) { __global const float* w_ = wt + ((long)wo_ * R + wr_) * 9; for (int q = 0; q < 9; q++) g[q] = w_[q]; }         else { for (int q = 0; q < 9; q++) g[q] = 0.0f; } }
    if (steps > 0) WF_LOAD(rBegin)
    for (int step = 0; step < steps; step++)
    {
        // V = B^T d B, B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]].
        {
            float t[16];
            for (int j = 0; j < 4; j++)
            {
                t[j] = d[j] - d[8 + j];
                t[4 + j] = d[4 + j] + d[8 + j];
                t[8 + j] = d[8 + j] - d[4 + j];
                t[12 + j] = d[4 + j] - d[12 + j];
            }
            __local float* v = vs + lc * 32 + lp;
            for (int i = 0; i < 4; i++)
            {
                v[(i * 4 + 0) * WF_CK * 32] = t[i * 4] - t[i * 4 + 2];
                v[(i * 4 + 1) * WF_CK * 32] = t[i * 4 + 1] + t[i * 4 + 2];
                v[(i * 4 + 2) * WF_CK * 32] = t[i * 4 + 2] - t[i * 4 + 1];
                v[(i * 4 + 3) * WF_CK * 32] = t[i * 4 + 1] - t[i * 4 + 3];
            }
        }
        // U = G g G^T, G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]].
        {
            float t[12];
            for (int j = 0; j < 3; j++)
            {
                float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
                t[j] = g0;
                t[3 + j] = 0.5f * (g0 + g1 + g2);
                t[6 + j] = 0.5f * (g0 - g1 + g2);
                t[9 + j] = g2;
            }
            __local float* uu = us + fr * WF_ULD + fo;
            for (int i = 0; i < 4; i++)
            {
                float a = t[i * 3], b = t[i * 3 + 1], c = t[i * 3 + 2];
                uu[(i * 4 + 0) * WF_CK * WF_ULD] = a;
                uu[(i * 4 + 1) * WF_CK * WF_ULD] = 0.5f * (a + b + c);
                uu[(i * 4 + 2) * WF_CK * WF_ULD] = 0.5f * (a - b + c);
                uu[(i * 4 + 3) * WF_CK * WF_ULD] = c;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (step + 1 < steps) WF_LOAD(rBegin + (step + 1) * WF_CK)
        __local const float* ub = us + x * WF_CK * WF_ULD + oo;
        __local const float* vb = vs + x * WF_CK * 32 + pp;
        for (int r = 0; r < WF_CK; r++)
        {
            float4 a0 = vload4(0, ub + r * WF_ULD), a1 = vload4(1, ub + r * WF_ULD);
            float4 b0 = vload4(0, vb + r * 32), b1 = vload4(1, vb + r * 32);
            float a[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            float b[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) acc[i][j] = mad(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #undef WF_LOAD

    // Epilogue, 16 output channels at a time through local memory (16 components x 16 channels x 32 tiles), then
    // Y = A^T m A, A^T = [[1,1,1,0],[0,1,-1,-1]], into each (channel, tile) pair's in-bounds 2x2 outputs.
    __local float* ms = sm;
    for (int part = 0; part < 2; part++)
    {
        if ((oo >> 4) == part)
        {
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) ms[(x * 16 + oo - 16 * part + i) * 32 + pp + j] = acc[i][j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int pair = tid; pair < 16 * 32; pair += 256)
        {
            const int ol = pair >> 5, pl = pair & 31;
            const int o = o0 + 16 * part + ol, p = p0 + pl;
            if (o < O && p < P)
            {
                float m[16];
                for (int c = 0; c < 16; c++) m[c] = ms[(c * 16 + ol) * 32 + pl];
                float t0[4], t1[4];
                for (int j = 0; j < 4; j++)
                {
                    t0[j] = m[j] + m[4 + j] + m[8 + j];
                    t1[j] = m[4 + j] - m[8 + j] - m[12 + j];
                }
                int n = p / tilesPerImage, t = p - n * tilesPerImage;
                int th = t / tilesW, tw = t - th * tilesW;
                int h = th * 2, w = tw * 2;
                __global float* dst = out + (((long)n * O + o) * OH + h) * OW + w;
                dst[0] = conv_epilogue(t0[0] + t0[1] + t0[2], bias, o, epi);
                if (w + 1 < OW) dst[1] = conv_epilogue(t0[1] - t0[2] - t0[3], bias, o, epi);
                if (h + 1 < OH)
                {
                    dst[OW] = conv_epilogue(t1[0] + t1[1] + t1[2], bias, o, epi);
                    if (w + 1 < OW) dst[OW + 1] = conv_epilogue(t1[1] - t1[2] - t1[3], bias, o, epi);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
// out[i] = the epilogue of the sum over s of partial[s][i], i < count, in split order; channel (i / hw) % O.
__kernel void wino_f23_split_reduce(__global const float* restrict partial, __global float* restrict out,
    const int count, const int splits, __global const float* bias, const int epi, const int hw, const int O)
{
    int i = get_global_id(0);
    if (i >= count) return;
    float sum = 0.0f;
    for (int s = 0; s < splits; s++) sum += partial[(long)s * count + i];
    out[i] = conv_epilogue(sum, bias, (i / hw) % O, epi);
}
// A convolution over few input channels (a network's stem on 3 RGB channels), one work item per output pixel and 16
// output channels: the pixel's C x KH x KW inputs are read once for all 16, where the direct kernel reads them once per
// output channel. The group's 16 filters (output channel block get_group_id(1)) are staged in local memory first, so
// the 16 x C x KH x KW weight reads per pixel are local broadcasts, not global loads; C x KH x KW is at most 196.
#define SC_MAX_TAPS 196
#define SC_MAX_KW 7
#define SC_OB 16
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void conv2d_smallc(__global const float* restrict in, __global const float* restrict wt, __global float* restrict out,
    const int N, const int C, const int IH, const int IW, const int O, const int OH, const int OW, const int KH,
    const int KW, const int SH, const int SW, const int PH, const int PW, const int DH, const int DW,
    __global const float* bias, const int epi)
{
    __local float ws[SC_OB * SC_MAX_TAPS];
    const int npix = OH * OW;
    const int pix = get_global_id(0);
    const int o0 = get_group_id(1) * SC_OB;
    const int taps = C * KH * KW;
    for (int e = get_local_id(0); e < SC_OB * taps; e += 64)
    {
        const int j = e / taps;
        ws[e] = o0 + j < O ? wt[(long)o0 * taps + e] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (pix >= N * npix) return;
    const int n = pix / npix, q = pix - n * npix;
    const int oh = q / OW, ow = q - oh * OW;
    float acc[SC_OB];
    for (int j = 0; j < SC_OB; j++) acc[j] = 0.0f;
    // A filter row's window (KW <= SC_MAX_KW) is loaded at once, out-of-bounds taps predicated to zero, so the loads
    // are in flight together; behind per-tap bounds branches they ran one at a time (a 7 x 7 stem at batch 128 took
    // 1.1 ms; 0.85 ms this way). Two or four pixels per work item, to reuse each weight read, measured slower still.
    const int ih0 = oh * SH - PH, iw0 = ow * SW - PW;
    for (int c = 0; c < C; c++)
    {
        __global const float* src = in + (long)(n * C + c) * IH * IW;
        for (int kh = 0; kh < KH; kh++)
        {
            const int ih = ih0 + kh * DH;
            const bool rowInside = ih >= 0 && ih < IH;
            float v[SC_MAX_KW];
            #pragma unroll
            for (int kw = 0; kw < SC_MAX_KW; kw++)
            {
                const int iw = iw0 + kw * DW;
                const bool inside = rowInside && kw < KW && iw >= 0 && iw < IW;
                v[kw] = inside ? src[inside ? ih * IW + iw : 0] : 0.0f;
            }
            __local const float* w = ws + (c * KH + kh) * KW;
            #pragma unroll
            for (int kw = 0; kw < SC_MAX_KW; kw++)
            {
                if (kw >= KW) break;
                for (int j = 0; j < SC_OB; j++) acc[j] = mad(v[kw], w[j * taps + kw], acc[j]);
            }
        }
    }
    for (int j = 0; j < SC_OB; j++)
    {
        const int o = o0 + j;
        if (o < O) out[(long)(n * O + o) * npix + q] = conv_epilogue(acc[j], bias, o, epi);
    }
}
// wcT[c][o] = w[o][c][centre]: the GEMM operand of a convolution whose 1 x 1 output reads a 1 x 1 input through its
// filters' centre tap alone (an odd filter padded by half its size).
__kernel void conv_center_tap_gather(__global const float* restrict w, __global float* restrict wcT, const int O,
    const int C, const int taps, const int centre)
{
    const int i = get_global_id(0);
    if (i >= O * C) return;
    const int c = i / O, o = i - c * O;
    wcT[i] = w[((long)o * C + c) * taps + centre];
}
// conv2d_smallc for a 3 x 3 filter, unrolled: a channel's nine window values are loaded together, out-of-bounds taps
// predicated to zero, before its products. In the general form each tap's load sat behind its bounds branches, so the
// loads ran one at a time and the 3-channel stem took about three times its output's write time.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void conv2d_smallc3(__global const float* restrict in, __global const float* restrict wt, __global float* restrict out,
    const int N, const int C, const int IH, const int IW, const int O, const int OH, const int OW, const int SH,
    const int SW, const int PH, const int PW, const int DH, const int DW, __global const float* bias, const int epi)
{
    __local float ws[SC_OB * SC_MAX_TAPS];
    const int npix = OH * OW;
    const int pix = get_global_id(0);
    const int o0 = get_group_id(1) * SC_OB;
    const int taps = C * 9;
    for (int e = get_local_id(0); e < SC_OB * taps; e += 64)
    {
        const int j = e / taps;
        ws[e] = o0 + j < O ? wt[(long)o0 * taps + e] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (pix >= N * npix) return;
    const int n = pix / npix, q = pix - n * npix;
    const int oh = q / OW, ow = q - oh * OW;
    const int ih0 = oh * SH - PH, iw0 = ow * SW - PW;
    float acc[SC_OB];
    for (int j = 0; j < SC_OB; j++) acc[j] = 0.0f;
    for (int c = 0; c < C; c++)
    {
        __global const float* src = in + (long)(n * C + c) * IH * IW;
        float v[9];
        #pragma unroll
        for (int kh = 0; kh < 3; kh++)
        {
            const int ih = ih0 + kh * DH;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++)
            {
                const int iw = iw0 + kw * DW;
                const bool inside = ih >= 0 && ih < IH && iw >= 0 && iw < IW;
                v[kh * 3 + kw] = inside ? src[inside ? ih * IW + iw : 0] : 0.0f;
            }
        }
        __local const float* w = ws + c * 9;
        #pragma unroll
        for (int t = 0; t < 9; t++)
            for (int j = 0; j < SC_OB; j++) acc[j] = mad(v[t], w[j * taps + t], acc[j]);
    }
    for (int j = 0; j < SC_OB; j++)
    {
        const int o = o0 + j;
        if (o < O) out[(long)(n * O + o) * npix + q] = conv_epilogue(acc[j], bias, o, epi);
    }
}
";
    }
}
