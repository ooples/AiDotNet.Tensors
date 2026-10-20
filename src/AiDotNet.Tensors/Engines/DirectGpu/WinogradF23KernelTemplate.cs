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

    /// <summary>One kernel: its buffers in binding order (the second written, every other one read) and its int
    /// parameters.</summary>
    internal sealed class Spec
    {
        internal Spec(string name, string[] buffers, string[] ints, string body)
        {
            Name = name; Buffers = buffers; Ints = ints; Body = body;
        }

        internal string Name { get; }
        internal string[] Buffers { get; }
        internal string[] Ints { get; }
        internal string Body { get; }
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

    internal static readonly Spec[] Kernels =
    {
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

    internal static string[] GetKernelNames()
    {
        var names = new string[Kernels.Length];
        for (int i = 0; i < Kernels.Length; i++) names[i] = Kernels[i].Name;
        return names;
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
    internal static string GetModuleSource(WinogradKernelDialect dialect)
    {
        if (dialect == WinogradKernelDialect.Glsl)
            throw new ArgumentException("GLSL has one kernel per shader; use GetGlslSource.", nameof(dialect));
        var sb = new StringBuilder();
        if (dialect == WinogradKernelDialect.Metal) sb.Append("#include <metal_stdlib>\nusing namespace metal;\n");
        foreach (var spec in Kernels)
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
                bool read = i != 1;
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
            if (dialect == WinogradKernelDialect.Metal) parameters.Add("uint gidU [[thread_position_in_grid]]");
            sb.Append(string.Join(", ", parameters)).Append(")\n{\n");
            sb.Append(dialect switch
            {
                WinogradKernelDialect.OpenCl => "    int gid = (int)get_global_id(0);\n",
                WinogradKernelDialect.Metal => "    int gid = (int)gidU;\n",
                _ => "    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n",
            });
            sb.Append(spec.Body).Append("}\n");
        }
        return sb.ToString();
    }

    /// <summary>
    /// One kernel as a Vulkan compute shader: the buffers at bindings 0 and 1 and the int parameters as push constants,
    /// in <see cref="Spec.Ints"/> order, over a 1-D grid of 256-wide groups.
    /// </summary>
    internal static string GetGlslSource(string name)
    {
        var spec = Get(name);
        var sb = new StringBuilder("#version 450\nlayout(local_size_x = 256) in;\n");
        for (int i = 0; i < spec.Buffers.Length; i++)
            sb.Append("layout(set=0, binding=").Append(i).Append(") buffer B").Append(i).Append(" { float ")
                .Append(spec.Buffers[i]).Append("[]; };\n");
        sb.Append("layout(push_constant) uniform PC { ");
        foreach (var p in spec.Ints) sb.Append("int ").Append(p).Append("; ");
        sb.Append("};\nvoid main()\n{\n    int gid = int(gl_GlobalInvocationID.x);\n").Append(spec.Body).Append("}\n");
        return sb.ToString();
    }
}