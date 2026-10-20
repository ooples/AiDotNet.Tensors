// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// Winograd F(4x4, 3x3) tile transforms for the OpenCL forward convolution (Lavin and Gray 2016, section 4.2). Each
    /// 4x4 output tile reads a 6x6 input patch, so the convolution becomes 36 per-point GEMMs with 4x fewer multiplies
    /// than a direct convolution (F(2x2, 3x3): 16 points, 2.25x fewer). Layouts match
    /// <see cref="WinogradF23KernelTemplate"/>: U[36][K][C],
    /// V[36][C][P], M[36][K][P] with P = N * tilesH * tilesW, so the GEMMs are the backend's concatenated-batch layout.
    /// </summary>
    internal static class WinogradF43Kernels
    {
        internal const string Filter = "wino_f43_filter";
        internal const string Input = "wino_f43_input";
        internal const string Output = "wino_f43_output";

        internal static string[] GetKernelNames() => new[] { Filter, Input, Output };

        internal const string Source = @"
// U[x][k][c] = (G g G^T)[x], G = [[1/4,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],[1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]].
__kernel void wino_f43_filter(__global const float* restrict w, __global float* restrict u, const int K, const int C)
{
    int idx = (int)get_global_id(0);
    if (idx >= K * C) return;
    float g[9];
    for (int q = 0; q < 9; q++) g[q] = w[idx * 9 + q];
    float t[18];
    for (int j = 0; j < 3; j++)
    {
        float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
        t[j] = 0.25f * g0;
        t[3 + j] = -(g0 + g1 + g2) / 6.0f;
        t[6 + j] = -(g0 - g1 + g2) / 6.0f;
        t[9 + j] = g0 / 24.0f + g1 / 12.0f + g2 / 6.0f;
        t[12 + j] = g0 / 24.0f - g1 / 12.0f + g2 / 6.0f;
        t[15 + j] = g2;
    }
    int plane = K * C;
    for (int i = 0; i < 6; i++)
    {
        float a = t[i * 3], b = t[i * 3 + 1], e = t[i * 3 + 2];
        u[(i * 6 + 0) * plane + idx] = 0.25f * a;
        u[(i * 6 + 1) * plane + idx] = -(a + b + e) / 6.0f;
        u[(i * 6 + 2) * plane + idx] = -(a - b + e) / 6.0f;
        u[(i * 6 + 3) * plane + idx] = a / 24.0f + b / 12.0f + e / 6.0f;
        u[(i * 6 + 4) * plane + idx] = a / 24.0f - b / 12.0f + e / 6.0f;
        u[(i * 6 + 5) * plane + idx] = e;
    }
}

// V[x][c][p] = (B^T d B)[x] for the 6x6 patch of tile p of channel c (zero outside the image); idx = c*P + p.
// B^T = [[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],[0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]].
__kernel void wino_f43_input(__global const float* restrict x, __global float* restrict v, const int N, const int C,
    const int H, const int W, const int pad, const int tilesH, const int tilesW)
{
    int P = N * tilesH * tilesW;
    int idx = (int)get_global_id(0);
    if (idx >= C * P) return;
    int c = idx / P, p = idx - c * P;
    int n = p / (tilesH * tilesW), tt = p - n * (tilesH * tilesW);
    int th = tt / tilesW, tw = tt - th * tilesW;
    int h0 = th * 4 - pad, w0 = tw * 4 - pad;
    int src = (n * C + c) * H * W;
    float d[36];
    for (int i = 0; i < 6; i++)
    {
        int hh = h0 + i;
        for (int j = 0; j < 6; j++)
        {
            int ww = w0 + j;
            d[i * 6 + j] = (hh >= 0 && hh < H && ww >= 0 && ww < W) ? x[src + hh * W + ww] : 0.0f;
        }
    }
    float s[36];
    for (int j = 0; j < 6; j++)
    {
        float z0 = d[j], z1 = d[6 + j], z2 = d[12 + j], z3 = d[18 + j], z4 = d[24 + j], z5 = d[30 + j];
        s[j] = 4.0f * z0 - 5.0f * z2 + z4;
        s[6 + j] = -4.0f * z1 - 4.0f * z2 + z3 + z4;
        s[12 + j] = 4.0f * z1 - 4.0f * z2 - z3 + z4;
        s[18 + j] = -2.0f * z1 - z2 + 2.0f * z3 + z4;
        s[24 + j] = 2.0f * z1 - z2 - 2.0f * z3 + z4;
        s[30 + j] = 4.0f * z1 - 5.0f * z3 + z5;
    }
    int plane = C * P;
    for (int i = 0; i < 6; i++)
    {
        float z0 = s[i * 6], z1 = s[i * 6 + 1], z2 = s[i * 6 + 2], z3 = s[i * 6 + 3], z4 = s[i * 6 + 4], z5 = s[i * 6 + 5];
        v[(i * 6 + 0) * plane + idx] = 4.0f * z0 - 5.0f * z2 + z4;
        v[(i * 6 + 1) * plane + idx] = -4.0f * z1 - 4.0f * z2 + z3 + z4;
        v[(i * 6 + 2) * plane + idx] = 4.0f * z1 - 4.0f * z2 - z3 + z4;
        v[(i * 6 + 3) * plane + idx] = -2.0f * z1 - z2 + 2.0f * z3 + z4;
        v[(i * 6 + 4) * plane + idx] = 2.0f * z1 - z2 - 2.0f * z3 + z4;
        v[(i * 6 + 5) * plane + idx] = 4.0f * z1 - 5.0f * z3 + z5;
    }
}

// Y = A^T m A, A^T = [[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]]; writes the in-bounds 4x4 outputs
// of tile p of channel k; idx = k*P + p.
// A convolution's epilogue: epi bit 0 adds bias[channel], bit 1 applies ReLU (NaN passes through, as the relu kernel).
// The same arithmetic as the conv2d_bias_add and relu passes it replaces.
float conv_epilogue(float v, __global const float* bias, int channel, int epi)
{
    if (epi & 1) v += bias[channel];
    if (epi & 2) v = isnan(v) ? v : fmax(0.0f, v);
    return v;
}
__kernel void wino_f43_output(__global const float* restrict mm, __global float* restrict y, const int N, const int K,
    const int outH, const int outW, const int tilesH, const int tilesW, __global const float* bias, const int epi)
{
    int P = N * tilesH * tilesW;
    int idx = (int)get_global_id(0);
    if (idx >= K * P) return;
    int k = idx / P, p = idx - k * P;
    int n = p / (tilesH * tilesW), tt = p - n * (tilesH * tilesW);
    int th = tt / tilesW, tw = tt - th * tilesW;
    int plane = K * P;
    float m[36];
    for (int q = 0; q < 36; q++) m[q] = mm[q * plane + idx];
    float s[24];
    for (int j = 0; j < 6; j++)
    {
        float z0 = m[j], z1 = m[6 + j], z2 = m[12 + j], z3 = m[18 + j], z4 = m[24 + j], z5 = m[30 + j];
        s[j] = z0 + z1 + z2 + z3 + z4;
        s[6 + j] = z1 - z2 + 2.0f * z3 - 2.0f * z4;
        s[12 + j] = z1 + z2 + 4.0f * z3 + 4.0f * z4;
        s[18 + j] = z1 - z2 + 8.0f * z3 - 8.0f * z4 + z5;
    }
    int dst = (n * K + k) * outH * outW;
    int h = th * 4, w4 = tw * 4;
    for (int i = 0; i < 4; i++)
    {
        if (h + i >= outH) break;
        float z0 = s[i * 6], z1 = s[i * 6 + 1], z2 = s[i * 6 + 2], z3 = s[i * 6 + 3], z4 = s[i * 6 + 4], z5 = s[i * 6 + 5];
        float o[4];
        o[0] = z0 + z1 + z2 + z3 + z4;
        o[1] = z1 - z2 + 2.0f * z3 - 2.0f * z4;
        o[2] = z1 + z2 + 4.0f * z3 + 4.0f * z4;
        o[3] = z1 - z2 + 8.0f * z3 - 8.0f * z4 + z5;
        int row = dst + (h + i) * outW;
        for (int j = 0; j < 4; j++)
            if (w4 + j < outW) y[row + w4 + j] = conv_epilogue(o[j], bias, k, epi);
    }
}
";
    }
}
