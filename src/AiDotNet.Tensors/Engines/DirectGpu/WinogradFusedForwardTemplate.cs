// Copyright (c) CrowdTrainer LLC. All rights reserved.
// #1432: OpenCL's fused Winograd F(2x2, 3x3) forward (WinogradFusedKernels, #1431) in WinogradF23KernelTemplate's
// index-based conventions, emitted as Metal Shading Language and Vulkan GLSL. WGSL keeps its own copy
// (WebGpu/WebGpuWinogradFusedKernels.cs).
using System;
using System.Globalization;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The fused Winograd F(2x2, 3x3) forward convolution: each 256-wide group transforms its input patches and filters into
/// shared memory, runs the sixteen per-point products from there and applies the output transform (with the bias + ReLU
/// epilogue) itself, so the 16-plane V and M never reach global memory.
/// </summary>
/// <remarks>
/// A group owns 32 output channels x 32 output tiles x all 16 components, on a 1-D grid
/// (<c>grp = outputChannelBlock * pBlocks + tileBlock</c>). Per reduction step of <c>ck</c> input channels, work items
/// below <c>ck * 32</c> load one 4x4 patch (channel <c>tid &gt;&gt; 5</c>, tile <c>tid &amp; 31</c>) and one 3x3 filter
/// (channel <c>tid % ck</c>, output channel <c>tid / ck</c>); for the products, work item (component <c>tid &gt;&gt; 4</c>,
/// lane <c>tid &amp; 15</c>) owns an 8 (output channel) x 8 (tile) sub-block. The epilogue stages the products through
/// shared memory in parts of <c>partChannels</c> output channels. OpenCL's instance steps 8 channels with a padded U row
/// (34.8 KB); <see cref="Spec"/> sizes shared memory as max(U + V, epilogue staging), which is 32 KB at ck 8 and
/// partChannels 16 (Metal's threadgroup limit) and 16 KB at ck 4 and partChannels 8 (WebGPU's guaranteed minimum).
/// </remarks>
internal static class WinogradFusedForwardTemplate
{
    /// <summary>The group size its thread roles assume.</summary>
    internal const int GroupSize = 256;

    /// <summary>Output channels and output tiles per group.</summary>
    internal const int Tile = 32;

    internal static string Name(int ck) => "wino_f23_fused_forward_c" + ck.ToString(CultureInfo.InvariantCulture);

    /// <summary>Shared-memory floats of the <paramref name="ck"/>-channel-step instance.</summary>
    internal static int SharedFloats(int ck, int partChannels) => Math.Max(2 * 16 * ck * Tile, 16 * partChannels * Tile);

    /// <summary>The instance stepping <paramref name="ck"/> (4 or 8) input channels, staging <paramref name="partChannels"/>
    /// (8 or 16) output channels per epilogue part.</summary>
    internal static WinogradF23KernelTemplate.Spec Spec(int ck, int partChannels)
    {
        if (ck != 4 && ck != 8) throw new ArgumentOutOfRangeException(nameof(ck));
        if (partChannels != 8 && partChannels != 16) throw new ArgumentOutOfRangeException(nameof(partChannels));
        string body = Body.Replace("CK", ck.ToString(CultureInfo.InvariantCulture))
            .Replace("PARTCH", partChannels.ToString(CultureInfo.InvariantCulture));
        return new WinogradF23KernelTemplate.Spec(Name(ck), new[] { "x", "wt", "y", "bias" },
            new[] { "N", "R", "IH", "IW", "O", "OH", "OW", "pad", "tilesH", "tilesW", "epi", "pBlocks" }, body,
            SharedFloats(ck, partChannels), written: new[] { 2 });
    }

    /// <summary>
    /// The launch of a forward convolution: its groups and the spec's ints, in order. False when the shape is not
    /// admitted (3x3, stride 1, dilation 1, equal padding up to 2) or an index or the grid would overflow 32 bits.
    /// </summary>
    internal static bool TryPlan(in TunedConv2DArgs a, out int groups, out int[] ints)
    {
        groups = 0;
        ints = Array.Empty<int>();
        if (!WinogradF23Pipeline.Admits(a)) return false;
        int tilesH = (a.OutHeight + 1) / 2, tilesW = (a.OutWidth + 1) / 2;
        long p = (long)a.Batch * tilesH * tilesW;
        long pBlocks = (p + Tile - 1) / Tile, oBlocks = ((long)a.OutChannels + Tile - 1) / Tile;
        // Every index in the body is 32-bit (GLSL and WGSL have no portable 64-bit integer).
        if ((long)a.Batch * a.InChannels * a.InHeight * a.InWidth > int.MaxValue
            || (long)a.Batch * a.OutChannels * a.OutHeight * a.OutWidth > int.MaxValue
            || (long)a.OutChannels * a.InChannels * 9 > int.MaxValue
            || pBlocks * oBlocks * GroupSize > int.MaxValue)
            return false;
        groups = (int)(pBlocks * oBlocks);
        ints = new[]
        {
            a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels, a.OutHeight, a.OutWidth, a.PadH, tilesH, tilesW,
            ConvEpilogue.Flags(a), (int)pBlocks,
        };
        return true;
    }

    // U[16][CK][32] at 0 and V[16][CK][32] after it during the reduction; the epilogue reuses the front.
    private const string Body = @"
    int tid = lid;
    int tilesPerImage = tilesH * tilesW;
    int P = N * tilesPerImage;
    int pb = grp % pBlocks;
    int p0 = pb * 32, o0 = (grp / pBlocks) * 32;
    int vBase = 16 * CK * 32;

    // Patch loader: channel lc, tile lp (work items below CK * 32).
    int lp = tid & 31, lc = tid >> 5;
    int gp = p0 + lp;
    uint pmask = 0u;
    int inOff = 0;
    if (gp < P && lc < CK)
    {
        int n = gp / tilesPerImage, t = gp - n * tilesPerImage;
        int th = t / tilesW, tw = t - th * tilesW;
        int h0 = th * 2 - pad, w0 = tw * 2 - pad;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (h0 + i >= 0 && h0 + i < IH && w0 + j >= 0 && w0 + j < IW) pmask |= (1u << (i * 4 + j));
        inOff = n * R * IH * IW + h0 * IW + w0;
    }
    int chanStride = IH * IW;
    // Filter loader: reduction channel fr, output channel fo (work items below CK * 32).
    int fr = tid % CK, fo = tid / CK;

    // Products: component xc, an 8 x 8 (output channel, tile) sub-block.
    int xc = tid >> 4, lane = tid & 15;
    int oo = (lane >> 2) * 8, pp = (lane & 3) * 8;
    float acc[64];
    for (int q = 0; q < 64; q++) acc[q] = 0.0f;

    int steps = (R + CK - 1) / CK;
    for (int step = 0; step < steps; step++)
    {
        int r0 = step * CK;
        if (lc < CK)
        {
            int r = r0 + lc;
            uint m = r < R ? pmask : 0u;
            float d[16];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                {
                    d[i * 4 + j] = 0.0f;
                    if (((m >> (i * 4 + j)) & 1u) != 0u) d[i * 4 + j] = x[inOff + r * chanStride + i * IW + j];
                }
            // V = B^T d B, B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]].
            float t[16];
            for (int j = 0; j < 4; j++)
            {
                t[j] = d[j] - d[8 + j];
                t[4 + j] = d[4 + j] + d[8 + j];
                t[8 + j] = d[8 + j] - d[4 + j];
                t[12 + j] = d[4 + j] - d[12 + j];
            }
            int vs = vBase + lc * 32 + lp;
            for (int i = 0; i < 4; i++)
            {
                sm[vs + (i * 4 + 0) * CK * 32] = t[i * 4] - t[i * 4 + 2];
                sm[vs + (i * 4 + 1) * CK * 32] = t[i * 4 + 1] + t[i * 4 + 2];
                sm[vs + (i * 4 + 2) * CK * 32] = t[i * 4 + 2] - t[i * 4 + 1];
                sm[vs + (i * 4 + 3) * CK * 32] = t[i * 4 + 1] - t[i * 4 + 3];
            }
        }
        if (tid < CK * 32)
        {
            int wr = r0 + fr, wo = o0 + fo;
            float g[9];
            for (int q = 0; q < 9; q++) g[q] = 0.0f;
            if (wr < R && wo < O)
                for (int q = 0; q < 9; q++) g[q] = wt[(wo * R + wr) * 9 + q];
            // U = G g G^T, G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]].
            float t[12];
            for (int j = 0; j < 3; j++)
            {
                float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
                t[j] = g0;
                t[3 + j] = 0.5f * (g0 + g1 + g2);
                t[6 + j] = 0.5f * (g0 - g1 + g2);
                t[9 + j] = g2;
            }
            int us = fr * 32 + fo;
            for (int i = 0; i < 4; i++)
            {
                float a = t[i * 3], b = t[i * 3 + 1], c = t[i * 3 + 2];
                sm[us + (i * 4 + 0) * CK * 32] = a;
                sm[us + (i * 4 + 1) * CK * 32] = 0.5f * (a + b + c);
                sm[us + (i * 4 + 2) * CK * 32] = 0.5f * (a - b + c);
                sm[us + (i * 4 + 3) * CK * 32] = c;
            }
        }
        BARRIER();
        int ub = xc * CK * 32 + oo, vb = vBase + xc * CK * 32 + pp;
        for (int r = 0; r < CK; r++)
        {
            float a[8];
            float b[8];
            for (int i = 0; i < 8; i++) { a[i] = sm[ub + r * 32 + i]; b[i] = sm[vb + r * 32 + i]; }
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) acc[i * 8 + j] = FMA(a[i], b[j], acc[i * 8 + j]);
        }
        BARRIER();
    }

    // Epilogue, PARTCH output channels at a time through shared memory (16 components x PARTCH channels x 32 tiles),
    // then Y = A^T m A, A^T = [[1,1,1,0],[0,1,-1,-1]], into each (channel, tile) pair's in-bounds 2x2 outputs, with
    // the bias + ReLU epilogue (epi bit 0 adds bias[o], bit 1 applies ReLU; v < 0 is false for NaN, which passes).
    for (int part = 0; part < 32 / PARTCH; part++)
    {
        if (oo / PARTCH == part)
        {
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) sm[(xc * PARTCH + oo - PARTCH * part + i) * 32 + pp + j] = acc[i * 8 + j];
        }
        BARRIER();
        for (int pair = tid; pair < PARTCH * 32; pair += 256)
        {
            int ol = pair >> 5, pl = pair & 31;
            int o = o0 + PARTCH * part + ol, p = p0 + pl;
            if (o < O && p < P)
            {
                float mm[16];
                for (int c = 0; c < 16; c++) mm[c] = sm[(c * PARTCH + ol) * 32 + pl];
                float t0[4];
                float t1[4];
                for (int j = 0; j < 4; j++)
                {
                    t0[j] = mm[j] + mm[4 + j] + mm[8 + j];
                    t1[j] = mm[4 + j] - mm[8 + j] - mm[12 + j];
                }
                int n = p / tilesPerImage, t = p - n * tilesPerImage;
                int th = t / tilesW, tw = t - th * tilesW;
                int h = th * 2, w = tw * 2;
                int dst = ((n * O + o) * OH + h) * OW + w;
                float bk = ((epi & 1) != 0) ? bias[o] : 0.0f;
                bool relu = (epi & 2) != 0;
                float v00 = t0[0] + t0[1] + t0[2] + bk;
                if (relu && v00 < 0.0f) v00 = 0.0f;
                y[dst] = v00;
                if (w + 1 < OW)
                {
                    float v01 = t0[1] - t0[2] - t0[3] + bk;
                    if (relu && v01 < 0.0f) v01 = 0.0f;
                    y[dst + 1] = v01;
                }
                if (h + 1 < OH)
                {
                    float v10 = t1[0] + t1[1] + t1[2] + bk;
                    if (relu && v10 < 0.0f) v10 = 0.0f;
                    y[dst + OW] = v10;
                    if (w + 1 < OW)
                    {
                        float v11 = t1[1] - t1[2] - t1[3] + bk;
                        if (relu && v11 < 0.0f) v11 = 0.0f;
                        y[dst + OW + 1] = v11;
                    }
                }
            }
        }
        BARRIER();
    }
";
}
