// Copyright (c) AiDotNet. All rights reserved.
// One definition of the Winograd F(2x2, 3x3) / F(3x3, 2x2) tile transforms, emitted as OpenCL C, HIP C++, Vulkan GLSL
// and Metal Shading Language (#1368, #1369, #1370, #1371). WGSL has no C-like dialect and keeps its own copy
// (WebGpu/WebGpuWinogradKernels.cs).
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>The source language a <see cref="WinogradF23KernelTemplate"/> kernel is emitted in.</summary>
internal enum WinogradKernelDialect
{
    OpenCl,
    Hip,
    Glsl,
    Metal,
}

/// <summary>
/// Which route's kernels a module or a readiness check covers. The fused weight gradient is the only route whose
/// kernels declare work-group shared memory and a fixed 256-wide group, so it is the one a device compiler is most
/// likely to reject; keeping it in its own group means such a rejection drops only the fused candidate, and the
/// forward, input-gradient and unfused weight-gradient routes still run.
/// </summary>
internal enum WinogradKernelGroup
{
    /// <summary>Every kernel in the template.</summary>
    All,

    /// <summary>The forward, input-gradient and unfused weight-gradient kernels.</summary>
    Base,

    /// <summary>The fused weight gradient's two accumulation kernels and its cross-split finish.</summary>
    Fused,
}

/// <summary>
/// The tile transforms of <see cref="WinogradF23Pipeline"/>, written once and emitted per dialect. The arithmetic is
/// CUDA's (<c>CUDA/Kernels/CudaConvWinogradKernels.cs</c>), operation for operation, so every backend rounds the same.
/// </summary>
/// <remarks>
/// THE BODIES ARE INDEX-BASED, NOT POINTER-BASED, because GLSL has no pointers: a Vulkan buffer is a global array named
/// in its binding. Each body therefore names its buffers and its integer parameters directly, and each dialect's wrapper
/// declares them under those names -- kernel parameters in OpenCL and HIP, bindings and push constants in GLSL. Indices
/// are 32-bit everywhere: GLSL has no portable 64-bit integer, and the pipeline refuses any plane that does not fit one.
/// Identifiers avoid GLSL's reserved words (<c>input</c>, <c>output</c>, <c>filter</c>).
/// </remarks>
internal static class WinogradF23KernelTemplate
{
    internal const string Filter = "wino_f23_filter";
    internal const string FilterFlip = "wino_f23_filter_flip";
    internal const string Input = "wino_f23_input";
    internal const string Output = "wino_f23_output";
    internal const string GradOutputTiles = "wino_f32w_dy";
    internal const string InputTilesTileMajor = "wino_f32w_input";
    internal const string FilterGradient = "wino_f32w_output";
    internal const string FusedFilterGradient = "wino_f32w_fused";
    internal const string FusedFilterGradientDirect = "wino_f32w_fused_direct";
    internal const string FusedFilterGradientFinish = "wino_f32w_fused_finish";

    /// <summary>The fused weight gradient's block: 32 output x 32 input channels per group of 256.</summary>
    internal const int FusedTile = 32;

    /// <summary>The fused weight gradient's tiles per shared-memory chunk; splits are a multiple of it.</summary>
    internal const int FusedChunk = 4;

    /// <summary>The fused weight gradient's work-group size, fixed: its thread roles assume exactly 256.</summary>
    internal const int FusedGroupSize = 256;

    /// <summary>
    /// One kernel: its buffers in binding order (the second written, every other one read), its int parameters, and
    /// the floats of work-group shared memory it declares (zero for the elementwise kernels). A shared-memory kernel
    /// sees <c>lid</c> (its index in the 256-wide group) and <c>grp</c> (the group's index in the 1-D grid), and writes
    /// <c>BARRIER();</c> and <c>FMA(a, b, c)</c>, which each dialect spells its own way.
    /// </summary>
    internal sealed class Spec
    {
        internal Spec(string name, string[] buffers, string[] ints, string body, int sharedFloats = 0, int[]? written = null)
        {
            Name = name; Buffers = buffers; Ints = ints; Body = body; SharedFloats = sharedFloats;
            Written = written ?? new[] { 1 };
        }

        internal string Name { get; }
        internal string[] Buffers { get; }
        internal string[] Ints { get; }
        internal string Body { get; }
        internal int SharedFloats { get; }

        /// <summary>The buffers the kernel writes, by binding index; every other one is read only.</summary>
        internal int[] Written { get; }
    }

    // G g G^T for the 3x3 filter already in g[9] (row-major), stored to the sixteen planes of u at idx.
    private const string FilterTail = @"
    float t[12];
    for (int j = 0; j < 3; j++)
    {
        float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
        t[j] = g0;
        t[3 + j] = 0.5f * (g0 + g1 + g2);
        t[6 + j] = 0.5f * (g0 - g1 + g2);
        t[9 + j] = g2;
    }
    int plane = K * C;
    for (int i = 0; i < 4; i++)
    {
        float a = t[i * 3], b = t[i * 3 + 1], e = t[i * 3 + 2];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = 0.5f * (a + b + e);
        u[(i * 4 + 2) * plane + idx] = 0.5f * (a - b + e);
        u[(i * 4 + 3) * plane + idx] = e;
    }
";

    // B^T d B for the 4x4 patch of tile p of channel c (zero outside the image), stored to the sixteen planes of v at idx.
    private const string InputTail = @"
    int n = p / (tilesH * tilesW), tt = p - n * (tilesH * tilesW);
    int th = tt / tilesW, tw = tt - th * tilesW;
    int h0 = th * 2 - pad, w0 = tw * 2 - pad;
    int src = (n * C + c) * H * W;
    float d[16];
    for (int i = 0; i < 4; i++)
    {
        int hh = h0 + i;
        for (int j = 0; j < 4; j++)
        {
            int ww = w0 + j;
            d[i * 4 + j] = (hh >= 0 && hh < H && ww >= 0 && ww < W) ? x[src + hh * W + ww] : 0.0f;
        }
    }
    float s[16];
    for (int j = 0; j < 4; j++)
    {
        s[j] = d[j] - d[8 + j];
        s[4 + j] = d[4 + j] + d[8 + j];
        s[8 + j] = d[8 + j] - d[4 + j];
        s[12 + j] = d[4 + j] - d[12 + j];
    }
    int plane = C * P;
    for (int i = 0; i < 4; i++)
    {
        v[(i * 4 + 0) * plane + idx] = s[i * 4] - s[i * 4 + 2];
        v[(i * 4 + 1) * plane + idx] = s[i * 4 + 1] + s[i * 4 + 2];
        v[(i * 4 + 2) * plane + idx] = s[i * 4 + 2] - s[i * 4 + 1];
        v[(i * 4 + 3) * plane + idx] = s[i * 4 + 1] - s[i * 4 + 3];
    }
";

    // The fused weight gradient (CUDA's wino_f32w_fused, CUDA/Kernels/CudaConvWinogradKernels.cs, restated for 16 KB of
    // shared memory, WebGPU's guaranteed minimum). Group grp owns input channels c0 .. c0 + 31, output channels
    // k0 .. k0 + 31 and the tiles [bz * tilesPerSplit, ...) (grp = (bz * gy + by) * gx + bx). Per chunk of 4 tiles,
    // threads 0..127 transform dY (U = A g A^T) and threads 128..255 transform x (V = B^T d B) into sm (U at 0, V at
    // 2048, each [16 x][4 tiles][32 channels] with the channel column XOR-swizzled by 8 * tile, so a warp's 8 rows x 4
    // tiles of stores hit 32 banks and each thread's 4-channel quad stays contiguous). Thread (tx, tk, tc) then
    // accumulates M[x][k][c] += U[x][p][k] V[x][p][c] for components 4tx .. 4tx + 3, channels 4tk .., 4tc ..: 64
    // accumulators, one fma chain per (x, k, c) in ascending tile order, as CUDA's.
    private const string FusedMain = @"
    int tid = lid;
    int gx = (C + 31) / 32, gy = (K + 31) / 32;
    int bz = grp / (gx * gy);
    int rem = grp - bz * (gx * gy);
    int by = rem / gx, bx = rem - by * gx;
    int c0 = bx * 32, k0 = by * 32;
    int TT = tilesH * tilesW, P = N * TT;
    int pBegin = bz * tilesPerSplit, pEnd = min(P, pBegin + tilesPerSplit);
    int tc = tid & 7, tk = (tid >> 3) & 7, tx = tid >> 6;
    int tj = tid & 3, tr = (tid >> 2) & 31, role = tid >> 7;
    int col = tj * 32 + (tr ^ (tj * 8));
    float acc[64];
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;
    for (int pc = pBegin; pc < pEnd; pc += 4)
    {
        int p = pc + tj;
        bool pin = p < pEnd;
        int n = 0, th = 0, tw = 0;
        if (pin) { n = p / TT; int tt = p - n * TT; th = tt / tilesW; tw = tt - th * tilesW; }
        if (role == 0)
        {
            int k = k0 + tr;
            float g00 = 0.0f, g01 = 0.0f, g10 = 0.0f, g11 = 0.0f;
            if (pin && k < K)
            {
                int h = th * 2, w2 = tw * 2;
                int src = (n * K + k) * outH * outW + h * outW + w2;
                g00 = dy[src];
                if (w2 + 1 < outW) g01 = dy[src + 1];
                if (h + 1 < outH)
                {
                    g10 = dy[src + outW];
                    if (w2 + 1 < outW) g11 = dy[src + outW + 1];
                }
            }
            float ra[4];
            float rb[4];
            ra[0] = g00; ra[1] = g00 + g10; ra[2] = g00 - g10; ra[3] = -g10;
            rb[0] = g01; rb[1] = g01 + g11; rb[2] = g01 - g11; rb[3] = -g11;
            for (int i = 0; i < 4; i++)
            {
                sm[(i * 4 + 0) * 128 + col] = ra[i];
                sm[(i * 4 + 1) * 128 + col] = ra[i] + rb[i];
                sm[(i * 4 + 2) * 128 + col] = ra[i] - rb[i];
                sm[(i * 4 + 3) * 128 + col] = -rb[i];
            }
        }
        else
        {
            int c = c0 + tr;
            bool cin = pin && c < C;
            int h0 = th * 2 - pad, w0 = tw * 2 - pad;
            int src = (n * C + (cin ? c : 0)) * H * W;
            float d[16];
            for (int i = 0; i < 4; i++)
            {
                int hh = h0 + i;
                for (int j = 0; j < 4; j++)
                {
                    int ww = w0 + j;
                    d[i * 4 + j] = 0.0f;
                    if (cin && hh >= 0 && hh < H && ww >= 0 && ww < W) d[i * 4 + j] = x[src + hh * W + ww];
                }
            }
            float s[16];
            for (int j = 0; j < 4; j++)
            {
                s[j] = d[j] - d[8 + j];
                s[4 + j] = d[4 + j] + d[8 + j];
                s[8 + j] = d[8 + j] - d[4 + j];
                s[12 + j] = d[4 + j] - d[12 + j];
            }
            for (int i = 0; i < 4; i++)
            {
                sm[2048 + (i * 4 + 0) * 128 + col] = s[i * 4] - s[i * 4 + 2];
                sm[2048 + (i * 4 + 1) * 128 + col] = s[i * 4 + 1] + s[i * 4 + 2];
                sm[2048 + (i * 4 + 2) * 128 + col] = s[i * 4 + 2] - s[i * 4 + 1];
                sm[2048 + (i * 4 + 3) * 128 + col] = s[i * 4 + 1] - s[i * 4 + 3];
            }
        }
        BARRIER();
        for (int j = 0; j < 4; j++)
        {
            for (int xi = 0; xi < 4; xi++)
            {
                int row = (tx * 4 + xi) * 128 + j * 32;
                int uo = row + ((tk * 4) ^ (j * 8)), vo = 2048 + row + ((tc * 4) ^ (j * 8));
                float u0 = sm[uo], u1 = sm[uo + 1], u2 = sm[uo + 2], u3 = sm[uo + 3];
                float v0 = sm[vo], v1 = sm[vo + 1], v2 = sm[vo + 2], v3 = sm[vo + 3];
                int a0 = xi * 16;
                acc[a0 + 0] = FMA(u0, v0, acc[a0 + 0]); acc[a0 + 1] = FMA(u0, v1, acc[a0 + 1]);
                acc[a0 + 2] = FMA(u0, v2, acc[a0 + 2]); acc[a0 + 3] = FMA(u0, v3, acc[a0 + 3]);
                acc[a0 + 4] = FMA(u1, v0, acc[a0 + 4]); acc[a0 + 5] = FMA(u1, v1, acc[a0 + 5]);
                acc[a0 + 6] = FMA(u1, v2, acc[a0 + 6]); acc[a0 + 7] = FMA(u1, v3, acc[a0 + 7]);
                acc[a0 + 8] = FMA(u2, v0, acc[a0 + 8]); acc[a0 + 9] = FMA(u2, v1, acc[a0 + 9]);
                acc[a0 + 10] = FMA(u2, v2, acc[a0 + 10]); acc[a0 + 11] = FMA(u2, v3, acc[a0 + 11]);
                acc[a0 + 12] = FMA(u3, v0, acc[a0 + 12]); acc[a0 + 13] = FMA(u3, v1, acc[a0 + 13]);
                acc[a0 + 14] = FMA(u3, v2, acc[a0 + 14]); acc[a0 + 15] = FMA(u3, v3, acc[a0 + 15]);
            }
        }
        BARRIER();
    }
";

    // The epilogue, in eighths of the group's k rows (eighth e is the rows its tk == e threads hold): they gather each
    // (k, c)'s 16 components into Ms = sm[0 ..][16 x][4 k][33] (row stride 33: conflict-free), and threads 0..127 take
    // one (k, c) each and apply G^T M G (3x3).
    private const string FusedEpilogueHead = @"
    for (int e = 0; e < 8; e++)
    {
        if (tk == e)
        {
            for (int xi = 0; xi < 4; xi++)
                for (int a = 0; a < 4; a++)
                    for (int b = 0; b < 4; b++)
                        sm[(tx * 4 + xi) * 132 + a * 33 + tc * 4 + b] = acc[xi * 16 + a * 4 + b];
        }
        BARRIER();
        int kl = tid >> 5, cl = tid & 31;
        float r9[9];
        if (tid < 128)
        {
            float m[16];
            for (int q = 0; q < 16; q++) m[q] = sm[q * 132 + kl * 33 + cl];
            float t[12];
            for (int j = 0; j < 4; j++)
            {
                float m0 = m[j], m1 = m[4 + j], m2 = m[8 + j], m3 = m[12 + j];
                t[j] = m0 + 0.5f * (m1 + m2);
                t[4 + j] = 0.5f * (m1 - m2);
                t[8 + j] = 0.5f * (m1 + m2) + m3;
            }
            for (int i = 0; i < 3; i++)
            {
                float a = t[i * 4], b = t[i * 4 + 1], cc = t[i * 4 + 2], dd = t[i * 4 + 3];
                r9[i * 3 + 0] = a + 0.5f * (b + cc);
                r9[i * 3 + 1] = 0.5f * (b - cc);
                r9[i * 3 + 2] = 0.5f * (b + cc) + dd;
            }
        }
";

    // Unsplit: stage the 4 rows' dW spans (dW[k][c0 .. c0 + 31][3][3], contiguous) in sm[2112 ..] and copy them out.
    private const string FusedEpilogueDirect = FusedEpilogueHead + @"
        if (tid < 128)
            for (int q = 0; q < 9; q++) sm[2112 + kl * 288 + cl * 9 + q] = r9[q];
        BARRIER();
        int span = min(32, C - c0) * 9;
        for (int i = tid; i < 4 * 288; i += 256)
        {
            int rowk = i / 288, r = i - rowk * 288;
            int k = k0 + e * 4 + rowk;
            if (k < K && r < span) res[(k * C + c0) * 9 + r] = sm[2112 + i];
        }
        BARRIER();
    }
";

    // Split: component-major partials res[bz][q][k][c] (q = 3i + j), summed by wino_f32w_fused_finish.
    private const string FusedEpilogueSplit = FusedEpilogueHead + @"
        int k = k0 + e * 4 + kl, c = c0 + cl;
        if (tid < 128 && k < K && c < C)
        {
            int plane = K * C;
            int dst = bz * 9 * plane + k * C + c;
            for (int q = 0; q < 9; q++) res[dst + q * plane] = r9[q];
        }
        BARRIER();
    }
";

    private static readonly string[] FusedInts =
        { "N", "C", "H", "W", "K", "outH", "outW", "pad", "tilesH", "tilesW", "tilesPerSplit" };

    internal static readonly Spec[] Kernels =
    {
        // Weight gradient, one launch per split of the tiles: x, the written dW (or partials), dY. Groups of 256, a 1-D
        // grid of ceil(C/32) * ceil(K/32) * splits groups.
        new Spec(FusedFilterGradientDirect, new[] { "x", "res", "dy" }, FusedInts, FusedMain + FusedEpilogueDirect, 4096),
        new Spec(FusedFilterGradient, new[] { "x", "res", "dy" }, FusedInts, FusedMain + FusedEpilogueSplit, 4096),

        // dW[k][c][q] = sum over the splits of partials[z][q][k][c], in split order; idx = (k*C + c)*9 + q.
        new Spec(FusedFilterGradientFinish, new[] { "partials", "dw" }, new[] { "KC", "splits" }, @"
    int idx = gid;
    if (idx >= KC * 9) return;
    int kc = idx / 9, q = idx - kc * 9;
    float sum = 0.0f;
    for (int z = 0; z < splits; z++) sum += partials[(z * 9 + q) * KC + kc];
    dw[idx] = sum;
"),

        // U[x][k][c] = (G g G^T)[x], G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]]; idx = k*C + c.
        new Spec(Filter, new[] { "w", "u" }, new[] { "K", "C" }, @"
    int idx = gid;
    if (idx >= K * C) return;
    float g[9];
    for (int q = 0; q < 9; q++) g[q] = w[idx * 9 + q];
" + FilterTail),

        // Input gradient: U'[x][c][k] of w[k][c] rotated 180 degrees; idx = c*K + k.
        new Spec(FilterFlip, new[] { "w", "u" }, new[] { "K", "C" }, @"
    int idx = gid;
    if (idx >= K * C) return;
    int c = idx / K, k = idx - c * K;
    int src = (k * C + c) * 9;
    float g[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) g[i * 3 + j] = w[src + (2 - i) * 3 + (2 - j)];
" + FilterTail),

        // V[x][c][p], B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]; idx = c*P + p.
        new Spec(Input, new[] { "x", "v" }, new[] { "N", "C", "H", "W", "pad", "tilesH", "tilesW" }, @"
    int P = N * tilesH * tilesW;
    int idx = gid;
    if (idx >= C * P) return;
    int c = idx / P, p = idx - c * P;
" + InputTail),

        // Weight gradient: the same transform stored tile-major, V[x][p][c]; idx = p*C + c.
        new Spec(InputTilesTileMajor, new[] { "x", "v" }, new[] { "N", "C", "H", "W", "pad", "tilesH", "tilesW" }, @"
    int P = N * tilesH * tilesW;
    int idx = gid;
    if (idx >= C * P) return;
    int p = idx / C, c = idx - p * C;
" + InputTail),

        // Y = A^T m A, A^T = [[1,1,1,0],[0,1,-1,-1]]; writes the in-bounds 2x2 outputs of tile p of channel k. The
        // convolution's epilogue: epi bit 0 adds bias[k], bit 1 applies ReLU (v < 0 is false for NaN, which passes).
        new Spec(Output, new[] { "mm", "y", "bias" }, new[] { "N", "K", "outH", "outW", "tilesH", "tilesW", "epi" }, @"
    int P = N * tilesH * tilesW;
    int idx = gid;
    if (idx >= K * P) return;
    int k = idx / P, p = idx - k * P;
    int n = p / (tilesH * tilesW), tt = p - n * (tilesH * tilesW);
    int th = tt / tilesW, tw = tt - th * tilesW;
    int plane = K * P;
    float m[16];
    for (int q = 0; q < 16; q++) m[q] = mm[q * plane + idx];
    float s[8];
    for (int j = 0; j < 4; j++)
    {
        s[j] = m[j] + m[4 + j] + m[8 + j];
        s[4 + j] = m[4 + j] - m[8 + j] - m[12 + j];
    }
    int dst = (n * K + k) * outH * outW;
    int h = th * 2, w2 = tw * 2;
    float bk = ((epi & 1) != 0) ? bias[k] : 0.0f;
    bool relu = (epi & 2) != 0;
    for (int i = 0; i < 2; i++)
    {
        if (h + i >= outH) break;
        float o0 = s[i * 4] + s[i * 4 + 1] + s[i * 4 + 2] + bk;
        if (relu && o0 < 0.0f) o0 = 0.0f;
        y[dst + (h + i) * outW + w2] = o0;
        if (w2 + 1 < outW)
        {
            float o1 = s[i * 4 + 1] - s[i * 4 + 2] - s[i * 4 + 3] + bk;
            if (relu && o1 < 0.0f) o1 = 0.0f;
            y[dst + (h + i) * outW + w2 + 1] = o1;
        }
    }
"),

        // Weight gradient, F(3x3, 2x2): U[x][k][p] = (A g A^T)[x] of dY tile p, A = [[1,0],[1,1],[1,-1],[0,-1]].
        new Spec(GradOutputTiles, new[] { "dy", "u" }, new[] { "N", "K", "outH", "outW", "tilesH", "tilesW" }, @"
    int P = N * tilesH * tilesW;
    int idx = gid;
    if (idx >= K * P) return;
    int k = idx / P, p = idx - k * P;
    int n = p / (tilesH * tilesW), tt = p - n * (tilesH * tilesW);
    int th = tt / tilesW, tw = tt - th * tilesW;
    int src = (n * K + k) * outH * outW;
    int h = th * 2, w2 = tw * 2;
    float g00 = dy[src + h * outW + w2];
    float g01 = (w2 + 1 < outW) ? dy[src + h * outW + w2 + 1] : 0.0f;
    float g10 = (h + 1 < outH) ? dy[src + (h + 1) * outW + w2] : 0.0f;
    float g11 = (h + 1 < outH && w2 + 1 < outW) ? dy[src + (h + 1) * outW + w2 + 1] : 0.0f;
    // t = A g (4x2): rows g0, g0+g1, g0-g1, -g1 over the two columns.
    float r[8];
    r[0] = g00; r[1] = g01;
    r[2] = g00 + g10; r[3] = g01 + g11;
    r[4] = g00 - g10; r[5] = g01 - g11;
    r[6] = -g10; r[7] = -g11;
    int plane = K * P;
    for (int i = 0; i < 4; i++)
    {
        float a = r[i * 2], b = r[i * 2 + 1];
        u[(i * 4 + 0) * plane + idx] = a;
        u[(i * 4 + 1) * plane + idx] = a + b;
        u[(i * 4 + 2) * plane + idx] = a - b;
        u[(i * 4 + 3) * plane + idx] = -b;
    }
"),

        // dW[k][c] = G^T M G (3x3), G^T = [[1,1/2,1/2,0],[0,1/2,-1/2,0],[0,1/2,1/2,1]], from M[x][split][k][c] with the
        // split-K partials summed in order; idx = k*C + c.
        new Spec(FilterGradient, new[] { "mm", "dw" }, new[] { "K", "C", "splits" }, @"
    int idx = gid;
    if (idx >= K * C) return;
    int plane = K * C;
    float m[16];
    for (int q = 0; q < 16; q++)
    {
        float acc = 0.0f;
        for (int sp = 0; sp < splits; sp++) acc += mm[(q * splits + sp) * plane + idx];
        m[q] = acc;
    }
    float t[12];
    for (int j = 0; j < 4; j++)
    {
        float m0 = m[j], m1 = m[4 + j], m2 = m[8 + j], m3 = m[12 + j];
        t[j] = m0 + 0.5f * (m1 + m2);
        t[4 + j] = 0.5f * (m1 - m2);
        t[8 + j] = 0.5f * (m1 + m2) + m3;
    }
    int dst = idx * 9;
    for (int i = 0; i < 3; i++)
    {
        float a = t[i * 4], b = t[i * 4 + 1], e = t[i * 4 + 2], f = t[i * 4 + 3];
        dw[dst + i * 3 + 0] = a + 0.5f * (b + e);
        dw[dst + i * 3 + 1] = 0.5f * (b - e);
        dw[dst + i * 3 + 2] = 0.5f * (b + e) + f;
    }
"),
    };

    /// <summary>The fused weight gradient's kernels, in <see cref="Kernels"/> order.</summary>
    internal static readonly string[] FusedKernels =
        { FusedFilterGradientDirect, FusedFilterGradient, FusedFilterGradientFinish };

    /// <summary>Whether <paramref name="name"/> belongs to the fused weight gradient's route.</summary>
    internal static bool IsFusedKernel(string name)
    {
        foreach (var fused in FusedKernels)
            if (fused == name) return true;
        return false;
    }

    private static bool InGroup(Spec spec, WinogradKernelGroup group) => group switch
    {
        WinogradKernelGroup.Base => !IsFusedKernel(spec.Name),
        WinogradKernelGroup.Fused => IsFusedKernel(spec.Name),
        _ => true,
    };

    internal static string[] GetKernelNames(WinogradKernelGroup group = WinogradKernelGroup.All)
    {
        var names = new List<string>(Kernels.Length);
        foreach (var spec in Kernels)
            if (InGroup(spec, group)) names.Add(spec.Name);
        return names.ToArray();
    }

    internal static Spec Get(string name)
    {
        foreach (var s in Kernels)
            if (s.Name == name) return s;
        throw new ArgumentException($"No Winograd kernel named '{name}'.", nameof(name));
    }

    /// <summary>
    /// Every kernel in one OpenCL C, HIP C++ or Metal Shading Language module. Metal's buffers sit at indices 0 and 1 and
    /// its int parameters at 2.., one <c>constant int&amp;</c> each, which is how MetalBackend's 1-D conv dispatcher binds
    /// them (buffers, then each int by setBytes).
    /// </summary>
    internal static string GetModuleSource(WinogradKernelDialect dialect,
        WinogradKernelGroup group = WinogradKernelGroup.All)
    {
        var specs = new List<Spec>();
        foreach (var spec in Kernels)
            if (InGroup(spec, group)) specs.Add(spec);
        return EmitModule(specs, dialect);
    }

    /// <summary>
    /// <paramref name="specs"/> in one OpenCL C, HIP C++ or Metal Shading Language module (any kernel family written in
    /// this template's body conventions; see <see cref="Spec"/>).
    /// </summary>
    internal static string EmitModule(IEnumerable<Spec> specs, WinogradKernelDialect dialect)
    {
        if (dialect == WinogradKernelDialect.Glsl)
            throw new ArgumentException("GLSL has one kernel per shader; use GetGlslSource.", nameof(dialect));
        var sb = new StringBuilder();
        if (dialect == WinogradKernelDialect.Metal) sb.Append("#include <metal_stdlib>\nusing namespace metal;\n");
        foreach (var spec in specs)
        {
            sb.Append('\n');
            sb.Append(dialect switch
            {
                WinogradKernelDialect.OpenCl => "__kernel void ",
                WinogradKernelDialect.Metal => "kernel void ",
                _ => "extern \"C\" __global__ void ",
            });
            sb.Append(spec.Name).Append('(');
            var parameters = new List<string>();
            for (int i = 0; i < spec.Buffers.Length; i++)
            {
                bool read = Array.IndexOf(spec.Written, i) < 0;
                parameters.Add(dialect switch
                {
                    WinogradKernelDialect.OpenCl => (read ? "__global const float* restrict " : "__global float* restrict ") + spec.Buffers[i],
                    WinogradKernelDialect.Metal => (read ? "device const float* " : "device float* ") + spec.Buffers[i] + " [[buffer(" + i + ")]]",
                    _ => (read ? "const float* __restrict__ " : "float* __restrict__ ") + spec.Buffers[i],
                });
            }
            for (int i = 0; i < spec.Ints.Length; i++)
                parameters.Add(dialect switch
                {
                    WinogradKernelDialect.OpenCl => "const int " + spec.Ints[i],
                    WinogradKernelDialect.Metal => "constant int& " + spec.Ints[i] + " [[buffer(" + (spec.Buffers.Length + i) + ")]]",
                    _ => "int " + spec.Ints[i],
                });
            if (dialect == WinogradKernelDialect.Metal)
            {
                parameters.Add("uint gidU [[thread_position_in_grid]]");
                if (spec.SharedFloats > 0)
                {
                    parameters.Add("uint lidU [[thread_index_in_threadgroup]]");
                    parameters.Add("uint grpU [[threadgroup_position_in_grid]]");
                }
            }
            sb.Append(string.Join(", ", parameters)).Append(")\n{\n");
            sb.Append(dialect switch
            {
                WinogradKernelDialect.OpenCl => "    int gid = (int)get_global_id(0);\n",
                WinogradKernelDialect.Metal => "    int gid = (int)gidU;\n",
                _ => "    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
            });
            if (spec.SharedFloats > 0)
                sb.Append(dialect switch
                {
                    WinogradKernelDialect.OpenCl => $"    __local float sm[{spec.SharedFloats}];\n"
                        + "    int lid = (int)get_local_id(0), grp = (int)get_group_id(0);\n",
                    WinogradKernelDialect.Metal => $"    threadgroup float sm[{spec.SharedFloats}];\n"
                        + "    int lid = (int)lidU, grp = (int)grpU;\n",
                    _ => $"    __shared__ float sm[{spec.SharedFloats}];\n"
                        + "    int lid = (int)threadIdx.x, grp = (int)blockIdx.x;\n",
                });
            sb.Append(Spell(spec.Body, dialect)).Append("}\n");
        }
        return sb.ToString();
    }

    // The bodies' dialect-neutral spellings: the work-group barrier, fused multiply-add, reciprocal square root, float
    // maximum, and an int parameter's bits read as a float (float parameters ride in the int list).
    private static string Spell(string body, WinogradKernelDialect dialect)
        => body.Replace("BARRIER();", dialect switch
        {
            WinogradKernelDialect.OpenCl => "barrier(CLK_LOCAL_MEM_FENCE);",
            WinogradKernelDialect.Metal => "threadgroup_barrier(mem_flags::mem_threadgroup);",
            WinogradKernelDialect.Glsl => "memoryBarrierShared(); barrier();",
            _ => "__syncthreads();",
        }).Replace("FMA(", dialect == WinogradKernelDialect.Hip ? "fmaf(" : "fma(")
        .Replace("RSQRT(", dialect switch
        {
            WinogradKernelDialect.Glsl => "inversesqrt(",
            WinogradKernelDialect.Hip => "rsqrtf(",
            _ => "rsqrt(",
        })
        .Replace("FMAX(", dialect switch
        {
            WinogradKernelDialect.Glsl => "max(",
            WinogradKernelDialect.Hip => "fmaxf(",
            _ => "fmax(",
        })
        .Replace("PRECISE ", dialect == WinogradKernelDialect.Glsl ? "precise " : "")
        .Replace("IBITS(", dialect switch
        {
            WinogradKernelDialect.OpenCl => "as_float(",
            WinogradKernelDialect.Glsl => "intBitsToFloat(",
            WinogradKernelDialect.Metal => "as_type<float>(",
            _ => "__int_as_float(",
        });

    /// <summary>
    /// One kernel as a Vulkan compute shader: the buffers at bindings 0 and 1 and the int parameters as push constants,
    /// in <see cref="Spec.Ints"/> order, over a 1-D grid of 256-wide groups.
    /// </summary>
    internal static string GetGlslSource(string name) => EmitGlsl(Get(name));

    /// <summary>One kernel of any family in this template's conventions as a Vulkan compute shader (see
    /// <see cref="GetGlslSource"/>).</summary>
    internal static string EmitGlsl(Spec spec)
    {
        var sb = new StringBuilder("#version 450\nlayout(local_size_x = 256) in;\n");
        for (int i = 0; i < spec.Buffers.Length; i++)
            sb.Append("layout(set=0, binding=").Append(i).Append(") buffer B").Append(i).Append(" { float ")
                .Append(spec.Buffers[i]).Append("[]; };\n");
        sb.Append("layout(push_constant) uniform PC { ");
        foreach (var p in spec.Ints) sb.Append("int ").Append(p).Append("; ");
        sb.Append("};\n");
        if (spec.SharedFloats > 0) sb.Append("shared float sm[").Append(spec.SharedFloats).Append("];\n");
        sb.Append("void main()\n{\n    int gid = int(gl_GlobalInvocationID.x);\n");
        if (spec.SharedFloats > 0)
            sb.Append("    int lid = int(gl_LocalInvocationID.x), grp = int(gl_WorkGroupID.x);\n");
        sb.Append(Spell(spec.Body, WinogradKernelDialect.Glsl)).Append("}\n");
        return sb.ToString();
    }
}