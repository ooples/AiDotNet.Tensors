// Copyright (c) CrowdTrainer LLC. All rights reserved.
// One definition of the tuned-kernel registry's generated row / column variants, emitted as OpenCL C and as Metal
// Shading Language (#1114, #1116).
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>The source language a <see cref="TunedRowKernelTemplate"/> instance is emitted in.</summary>
    internal enum TunedKernelDialect
    {
        OpenCl,
        Metal,
    }

    /// <summary>
    /// The row-wise and column-wise variants the tuned-kernel registry measures against each backend's established
    /// kernels, written once and emitted per dialect. The established kernels give each row a whole work group (or one
    /// thread) however short the row is; these give a row <c>L</c> lanes (L = 8, 16 or 32, so a 256-item group serves
    /// 256 / L rows) or give a 32-column tile <c>R</c> row lanes.
    /// </summary>
    /// <remarks>
    /// The bodies are dialect-neutral: address spaces and the group barrier are macros, and every work-item index is a
    /// parameter, so OpenCL and Metal run the same arithmetic in the same order. That is what lets the OpenCL tests on a
    /// real device stand for the Metal instance on hosts without one. Each lane group reduces through group memory with
    /// a fixed halving tree (lane i adds lane i + o); every work item reaches every barrier (inactive tail rows carry
    /// neutral values). No atomics, so every variant is deterministic.
    /// </remarks>
    internal static class TunedRowKernelTemplate
    {
        internal const int GroupSize = 256;

        /// <summary>Lane counts each row-kernel family is instantiated for.</summary>
        internal static readonly int[] RowLanes = { 8, 16, 32 };

        /// <summary>Row-lane counts the column-reduction family is instantiated for.</summary>
        internal static readonly int[] ColumnRowLanes = { 8, 32 };

        internal static string SoftmaxName(int lanes) => "tk_softmax_l" + lanes.ToString(CultureInfo.InvariantCulture);
        internal static string SoftmaxBackwardName(int lanes) => "tk_softmax_backward_l" + lanes.ToString(CultureInfo.InvariantCulture);
        internal static string LayerNormName(int lanes) => "tk_layernorm_forward_l" + lanes.ToString(CultureInfo.InvariantCulture);
        internal static string LayerNormBackwardName(int lanes) => "tk_layernorm_backward_l" + lanes.ToString(CultureInfo.InvariantCulture);
        internal static string LayerNormGradParamsName(int rowLanes) => "tk_layernorm_grad_params_r" + rowLanes.ToString(CultureInfo.InvariantCulture);
        internal static string RmsNormGradGammaName(int rowLanes) => "tk_rmsnorm_grad_gamma_r" + rowLanes.ToString(CultureInfo.InvariantCulture);

        /// <summary>The im2col variant writing [P, B * L] for the single-GEMM convolution candidate.</summary>
        internal const string Im2ColName = "tk_im2col_pbl";

        internal static string[] GetKernelNames()
        {
            var names = new List<string> { Im2ColName };
            foreach (int l in RowLanes)
            {
                names.Add(SoftmaxName(l));
                names.Add(SoftmaxBackwardName(l));
                names.Add(LayerNormName(l));
                names.Add(LayerNormBackwardName(l));
            }
            foreach (int r in ColumnRowLanes)
            {
                names.Add(LayerNormGradParamsName(r));
                names.Add(RmsNormGradGammaName(r));
            }
            return names.ToArray();
        }

        internal static string GetSource(TunedKernelDialect dialect)
        {
            var sb = new StringBuilder();
            sb.Append(dialect == TunedKernelDialect.Metal ? MetalPrelude : OpenClPrelude);
            sb.Append(Bodies);
            foreach (int l in RowLanes) sb.Append(RowKernels(dialect, l));
            foreach (int r in ColumnRowLanes) sb.Append(ColumnKernels(dialect, r));
            sb.Append(Im2ColKernel(dialect));
            return sb.ToString();
        }

        private const string OpenClPrelude = @"
#define TK_GLOBAL __global
#define TK_LOCAL __local
#define TK_BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
";

        private const string MetalPrelude = @"
#include <metal_stdlib>
using namespace metal;
#define TK_GLOBAL device
#define TK_LOCAL threadgroup
#define TK_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
";

        private const string Bodies = @"
// Sum / max over the L consecutive lanes of a row group, through group scratch[GroupSize]; every lane gets the result.
float tk_sum(float v, TK_LOCAL float* scratch, int L, int lid)
{
    scratch[lid] = v;
    TK_BARRIER();
    for (int o = L / 2; o > 0; o >>= 1)
    {
        if ((lid % L) < o) scratch[lid] += scratch[lid + o];
        TK_BARRIER();
    }
    float r = scratch[lid - lid % L];
    TK_BARRIER();
    return r;
}

float tk_max(float v, TK_LOCAL float* scratch, int L, int lid)
{
    scratch[lid] = v;
    TK_BARRIER();
    for (int o = L / 2; o > 0; o >>= 1)
    {
        if ((lid % L) < o) scratch[lid] = fmax(scratch[lid], scratch[lid + o]);
        TK_BARRIER();
    }
    float r = scratch[lid - lid % L];
    TK_BARRIER();
    return r;
}

// Row softmax: max, exp written to output, sum, scale by 1 / sum (or 1 when the sum is not positive).
void tk_softmax_body(TK_GLOBAL const float* input, TK_GLOBAL float* output, int rows, int n, int L, TK_LOCAL float* scratch,
    int t, int lid)
{
    int row = t / L, lane = t % L;
    bool active = row < rows;
    TK_GLOBAL const float* x = input + (long)(active ? row : 0) * n;
    TK_GLOBAL float* y = output + (long)(active ? row : 0) * n;
    float m = -INFINITY;
    if (active) for (int f = lane; f < n; f += L) m = fmax(m, x[f]);
    m = tk_max(m, scratch, L, lid);
    float s = 0.0f;
    if (active) for (int f = lane; f < n; f += L) { float e = exp(x[f] - m); y[f] = e; s += e; }
    s = tk_sum(s, scratch, L, lid);
    float inv = (s > 0.0f) ? (1.0f / s) : 1.0f;
    if (active) for (int f = lane; f < n; f += L) y[f] *= inv;
}

void tk_softmax_backward_body(TK_GLOBAL const float* gradOutput, TK_GLOBAL const float* output, TK_GLOBAL float* gradInput,
    int rows, int n, int L, TK_LOCAL float* scratch, int t, int lid)
{
    int row = t / L, lane = t % L;
    bool active = row < rows;
    long base = (long)(active ? row : 0) * n;
    float dot = 0.0f;
    if (active) for (int f = lane; f < n; f += L) dot += gradOutput[base + f] * output[base + f];
    dot = tk_sum(dot, scratch, L, lid);
    if (active) for (int f = lane; f < n; f += L) gradInput[base + f] = output[base + f] * (gradOutput[base + f] - dot);
}

void tk_layernorm_forward_body(TK_GLOBAL const float* input, TK_GLOBAL float* output, TK_GLOBAL const float* gamma,
    TK_GLOBAL const float* beta, TK_GLOBAL float* saveMean, TK_GLOBAL float* saveInvVar, int rows, int n, float epsilon, int L,
    TK_LOCAL float* scratch, int t, int lid)
{
    int row = t / L, lane = t % L;
    bool active = row < rows;
    long base = (long)(active ? row : 0) * n;
    float s = 0.0f;
    if (active) for (int i = lane; i < n; i += L) s += input[base + i];
    float mean = tk_sum(s, scratch, L, lid) / (float)n;
    float v = 0.0f;
    if (active) for (int i = lane; i < n; i += L) { float d = input[base + i] - mean; v += d * d; }
    float invVar = rsqrt(tk_sum(v, scratch, L, lid) / (float)n + epsilon);
    if (!active) return;
    if (lane == 0) { saveMean[row] = mean; saveInvVar[row] = invVar; }
    for (int i = lane; i < n; i += L)
    {
        float normalized = (input[base + i] - mean) * invVar;
        output[base + i] = gamma[i] * normalized + beta[i];
    }
}

void tk_layernorm_backward_body(TK_GLOBAL const float* gradOutput, TK_GLOBAL const float* input, TK_GLOBAL const float* gamma,
    TK_GLOBAL const float* saveMean, TK_GLOBAL const float* saveInvVar, TK_GLOBAL float* gradInput, int rows, int n, int L,
    TK_LOCAL float* scratch, int t, int lid)
{
    int row = t / L, lane = t % L;
    bool active = row < rows;
    int r = active ? row : 0;
    long base = (long)r * n;
    float mean = saveMean[r], invVar = saveInvVar[r];
    float sumDy = 0.0f, sumDyXmu = 0.0f;
    if (active) for (int i = lane; i < n; i += L)
    {
        float dy = gradOutput[base + i] * gamma[i];
        sumDy += dy;
        sumDyXmu += dy * (input[base + i] - mean);
    }
    sumDy = tk_sum(sumDy, scratch, L, lid);
    sumDyXmu = tk_sum(sumDyXmu, scratch, L, lid);
    if (!active) return;
    for (int i = lane; i < n; i += L)
    {
        float xmu = input[base + i] - mean;
        float dxhat = gradOutput[base + i] * gamma[i];
        gradInput[base + i] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)n);
    }
}

// Column reduction over a 32-column tile with R row lanes (work group 32 x R): gradGamma (and gradBeta), or RMSNorm's
// gradGamma. Group tiles padded to 33 columns. cx / ry: the item's position in its group; gx: the group's column tile.
void tk_column_grad_body(TK_GLOBAL const float* gradOutput, TK_GLOBAL const float* input, TK_GLOBAL const float* stat0,
    TK_GLOBAL const float* stat1, TK_GLOBAL float* out0, TK_GLOBAL float* out1, int rows, int n, int R, bool rms,
    TK_LOCAL float* s0, TK_LOCAL float* s1, int cx, int ry, int gx)
{
    int col = gx * 32 + cx;
    float a0 = 0.0f, a1 = 0.0f;
    if (col < n)
    {
        for (int b = ry; b < rows; b += R)
        {
            long idx = (long)b * n + col;
            float g = gradOutput[idx];
            if (rms) a0 += g * input[idx] * (1.0f / stat0[b]);
            else { float normalized = (input[idx] - stat0[b]) * stat1[b]; a0 += g * normalized; a1 += g; }
        }
    }
    s0[ry * 33 + cx] = a0; s1[ry * 33 + cx] = a1;
    TK_BARRIER();
    for (int h = R / 2; h > 0; h >>= 1)
    {
        if (ry < h) { s0[ry * 33 + cx] += s0[(ry + h) * 33 + cx]; s1[ry * 33 + cx] += s1[(ry + h) * 33 + cx]; }
        TK_BARRIER();
    }
    if (ry == 0 && col < n)
    {
        out0[col] = s0[cx];
        if (!rms) out1[col] = s1[cx];
    }
}

// im2col to [P, B * L] (P = Cin*kH*kW, L = outH*outW) for the registry's im2col + GEMM convolution candidate: the batch
// sits inside each row, so ONE GEMM covers every batch item. One work item per (batch, output pixel).
void tk_im2col_body(TK_GLOBAL const float* input, TK_GLOBAL float* col, int batch, int channels, int height, int width,
    int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int outH, int outW,
    int idx)
{
    int l = outH * outW;
    if (idx >= batch * l) return;
    int b = idx / l, s = idx % l, oh = s / outW, ow = s % outW;
    for (int c = 0; c < channels; c++)
        for (int kh = 0; kh < kernelH; kh++)
            for (int kw = 0; kw < kernelW; kw++)
            {
                int ih = oh * strideH - padH + kh * dilationH, iw = ow * strideW - padW + kw * dilationW;
                float v = (ih >= 0 && ih < height && iw >= 0 && iw < width) ? input[((b * channels + c) * height + ih) * width + iw] : 0.0f;
                col[(long)((c * kernelH + kh) * kernelW + kw) * (batch * l) + b * l + s] = v;
            }
}
";

        private static string RowKernels(TunedKernelDialect dialect, int l)
        {
            string ls = l.ToString(CultureInfo.InvariantCulture);
            if (dialect == TunedKernelDialect.OpenCl)
            {
                const string Ids = "(int)get_global_id(0), (int)get_local_id(0)";
                string attr = $"__kernel __attribute__((reqd_work_group_size({GroupSize}, 1, 1))) void";
                return $@"
{attr} {SoftmaxName(l)}(__global const float* input, __global float* output, int rows, int n)
{{ __local float scratch[{GroupSize}]; tk_softmax_body(input, output, rows, n, {ls}, scratch, {Ids}); }}
{attr} {SoftmaxBackwardName(l)}(__global const float* gradOutput, __global const float* output, __global float* gradInput, int rows, int n)
{{ __local float scratch[{GroupSize}]; tk_softmax_backward_body(gradOutput, output, gradInput, rows, n, {ls}, scratch, {Ids}); }}
{attr} {LayerNormName(l)}(__global const float* input, __global float* output, __global const float* gamma, __global const float* beta, __global float* saveMean, __global float* saveInvVar, int rows, int n, float epsilon)
{{ __local float scratch[{GroupSize}]; tk_layernorm_forward_body(input, output, gamma, beta, saveMean, saveInvVar, rows, n, epsilon, {ls}, scratch, {Ids}); }}
{attr} {LayerNormBackwardName(l)}(__global const float* gradOutput, __global const float* input, __global const float* gamma, __global const float* saveMean, __global const float* saveInvVar, __global float* gradInput, int rows, int n)
{{ __local float scratch[{GroupSize}]; tk_layernorm_backward_body(gradOutput, input, gamma, saveMean, saveInvVar, gradInput, rows, n, {ls}, scratch, {Ids}); }}
";
            }

            // Metal: buffers then scalars in argument order; the host dispatches threadgroups of GroupSize threads.
            const string MetalIds = "uint t [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]]";
            const string Pass = "(int)t, (int)lid";
            return $@"
kernel void {SoftmaxName(l)}(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant int& rows [[buffer(2)]], constant int& n [[buffer(3)]], {MetalIds})
{{ threadgroup float scratch[{GroupSize}]; tk_softmax_body(input, output, rows, n, {ls}, scratch, {Pass}); }}
kernel void {SoftmaxBackwardName(l)}(device const float* gradOutput [[buffer(0)]], device const float* output [[buffer(1)]],
    device float* gradInput [[buffer(2)]], constant int& rows [[buffer(3)]], constant int& n [[buffer(4)]], {MetalIds})
{{ threadgroup float scratch[{GroupSize}]; tk_softmax_backward_body(gradOutput, output, gradInput, rows, n, {ls}, scratch, {Pass}); }}
kernel void {LayerNormName(l)}(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]], device const float* beta [[buffer(3)]], device float* saveMean [[buffer(4)]],
    device float* saveInvVar [[buffer(5)]], constant int& rows [[buffer(6)]], constant int& n [[buffer(7)]],
    constant float& epsilon [[buffer(8)]], {MetalIds})
{{ threadgroup float scratch[{GroupSize}]; tk_layernorm_forward_body(input, output, gamma, beta, saveMean, saveInvVar, rows, n, epsilon, {ls}, scratch, {Pass}); }}
kernel void {LayerNormBackwardName(l)}(device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]],
    device const float* gamma [[buffer(2)]], device const float* saveMean [[buffer(3)]], device const float* saveInvVar [[buffer(4)]],
    device float* gradInput [[buffer(5)]], constant int& rows [[buffer(6)]], constant int& n [[buffer(7)]], {MetalIds})
{{ threadgroup float scratch[{GroupSize}]; tk_layernorm_backward_body(gradOutput, input, gamma, saveMean, saveInvVar, gradInput, rows, n, {ls}, scratch, {Pass}); }}
";
        }

        private static string ColumnKernels(TunedKernelDialect dialect, int r)
        {
            string rs = r.ToString(CultureInfo.InvariantCulture);
            if (dialect == TunedKernelDialect.OpenCl)
            {
                const string Ids = "(int)get_local_id(0), (int)get_local_id(1), (int)get_group_id(0)";
                string attr = $"__kernel __attribute__((reqd_work_group_size(32, {rs}, 1))) void";
                return $@"
{attr} {LayerNormGradParamsName(r)}(__global const float* gradOutput, __global const float* input, __global const float* saveMean, __global const float* saveInvVar, __global float* gradGamma, __global float* gradBeta, int rows, int n)
{{ __local float s0[{rs} * 33]; __local float s1[{rs} * 33]; tk_column_grad_body(gradOutput, input, saveMean, saveInvVar, gradGamma, gradBeta, rows, n, {rs}, false, s0, s1, {Ids}); }}
{attr} {RmsNormGradGammaName(r)}(__global const float* gradOutput, __global const float* input, __global const float* saveRms, __global float* gradGamma, int rows, int n)
{{ __local float s0[{rs} * 33]; __local float s1[{rs} * 33]; tk_column_grad_body(gradOutput, input, saveRms, saveRms, gradGamma, gradGamma, rows, n, {rs}, true, s0, s1, {Ids}); }}
";
            }

            // Metal: threadgroups of (32, R, 1), one per 32-column tile.
            const string MetalIds = "uint2 lid [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]";
            const string Pass = "(int)lid.x, (int)lid.y, (int)group.x";
            return $@"
kernel void {LayerNormGradParamsName(r)}(device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]],
    device const float* saveMean [[buffer(2)]], device const float* saveInvVar [[buffer(3)]], device float* gradGamma [[buffer(4)]],
    device float* gradBeta [[buffer(5)]], constant int& rows [[buffer(6)]], constant int& n [[buffer(7)]], {MetalIds})
{{ threadgroup float s0[{rs} * 33]; threadgroup float s1[{rs} * 33]; tk_column_grad_body(gradOutput, input, saveMean, saveInvVar, gradGamma, gradBeta, rows, n, {rs}, false, s0, s1, {Pass}); }}
kernel void {RmsNormGradGammaName(r)}(device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]],
    device const float* saveRms [[buffer(2)]], device float* gradGamma [[buffer(3)]], constant int& rows [[buffer(4)]],
    constant int& n [[buffer(5)]], {MetalIds})
{{ threadgroup float s0[{rs} * 33]; threadgroup float s1[{rs} * 33]; tk_column_grad_body(gradOutput, input, saveRms, saveRms, gradGamma, gradGamma, rows, n, {rs}, true, s0, s1, {Pass}); }}
";
        }

        private static string Im2ColKernel(TunedKernelDialect dialect)
        {
            const string Geometry = "batch, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, outH, outW";
            if (dialect == TunedKernelDialect.OpenCl)
                return $@"
__kernel void {Im2ColName}(__global const float* input, __global float* col, int batch, int channels, int height, int width,
    int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int outH, int outW)
{{ tk_im2col_body(input, col, {Geometry}, (int)get_global_id(0)); }}
";

            var sb = new StringBuilder();
            sb.Append($"\nkernel void {Im2ColName}(device const float* input [[buffer(0)]], device float* col [[buffer(1)]]");
            string[] scalars = Geometry.Split(new[] { ", " }, System.StringSplitOptions.None);
            for (int i = 0; i < scalars.Length; i++)
                sb.Append($",\n    constant int& {scalars[i]} [[buffer({(i + 2).ToString(CultureInfo.InvariantCulture)})]]");
            sb.Append($",\n    uint idx [[thread_position_in_grid]])\n{{ tk_im2col_body(input, col, {Geometry}, (int)idx); }}\n");
            return sb.ToString();
        }
    }
}
