// Copyright (c) AiDotNet. All rights reserved.
// Generated row / column kernel variants for OpenCL, served through the tuned-kernel registry (#1114).
#if !NET462
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// OpenCL instances of the registry's row-wise and column-wise variants (CUDA: CudaTunedRowKernels), generated from
    /// one definition per op over a lane count. The established kernels give each row a whole work group however short
    /// the row is; these give a row <c>L</c> lanes (L = 8, 16 or 32, so a 256-item group serves 256 / L rows) or give a
    /// 32-column tile <c>R</c> row lanes, and the registry measures which wins per shape class.
    /// </summary>
    /// <remarks>
    /// OpenCL 1.2 has no portable sub-group shuffle, so each lane group reduces through local memory with the same
    /// halving tree as CUDA's shuffle-down reduction (lane i adds lane i + o); every work item reaches every barrier
    /// (inactive tail rows carry neutral values). The reductions run in a fixed order with no atomics, so every variant
    /// is deterministic.
    /// </remarks>
    internal static class OpenClTunedRowKernels
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

        public static string GetSource()
        {
            var sb = new StringBuilder();
            sb.Append(@"
// Sum / max over the L consecutive lanes of a row group, through local scratch[GroupSize]; every lane gets the result.
float tk_sum(float v, __local float* scratch, int L)
{
    int lid = (int)get_local_id(0);
    scratch[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = L / 2; o > 0; o >>= 1)
    {
        if ((lid % L) < o) scratch[lid] += scratch[lid + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = scratch[lid - lid % L];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

float tk_max(float v, __local float* scratch, int L)
{
    int lid = (int)get_local_id(0);
    scratch[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = L / 2; o > 0; o >>= 1)
    {
        if ((lid % L) < o) scratch[lid] = fmax(scratch[lid], scratch[lid + o]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = scratch[lid - lid % L];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

// Row softmax: max, exp written to output, sum, scale by 1 / sum (or 1 when the sum is not positive).
void tk_softmax_body(__global const float* input, __global float* output, int rows, int n, int L, __local float* scratch)
{
    int t = (int)get_global_id(0);
    int row = t / L, lane = t % L;
    bool active = row < rows;
    __global const float* x = input + (long)(active ? row : 0) * n;
    __global float* y = output + (long)(active ? row : 0) * n;
    float m = -INFINITY;
    if (active) for (int f = lane; f < n; f += L) m = fmax(m, x[f]);
    m = tk_max(m, scratch, L);
    float s = 0.0f;
    if (active) for (int f = lane; f < n; f += L) { float e = exp(x[f] - m); y[f] = e; s += e; }
    s = tk_sum(s, scratch, L);
    float inv = (s > 0.0f) ? (1.0f / s) : 1.0f;
    if (active) for (int f = lane; f < n; f += L) y[f] *= inv;
}

void tk_softmax_backward_body(__global const float* gradOutput, __global const float* output, __global float* gradInput,
    int rows, int n, int L, __local float* scratch)
{
    int t = (int)get_global_id(0);
    int row = t / L, lane = t % L;
    bool active = row < rows;
    long base = (long)(active ? row : 0) * n;
    float dot = 0.0f;
    if (active) for (int f = lane; f < n; f += L) dot += gradOutput[base + f] * output[base + f];
    dot = tk_sum(dot, scratch, L);
    if (active) for (int f = lane; f < n; f += L) gradInput[base + f] = output[base + f] * (gradOutput[base + f] - dot);
}

void tk_layernorm_forward_body(__global const float* input, __global float* output, __global const float* gamma,
    __global const float* beta, __global float* saveMean, __global float* saveInvVar, int rows, int n, float epsilon, int L,
    __local float* scratch)
{
    int t = (int)get_global_id(0);
    int row = t / L, lane = t % L;
    bool active = row < rows;
    long base = (long)(active ? row : 0) * n;
    float s = 0.0f;
    if (active) for (int i = lane; i < n; i += L) s += input[base + i];
    float mean = tk_sum(s, scratch, L) / (float)n;
    float v = 0.0f;
    if (active) for (int i = lane; i < n; i += L) { float d = input[base + i] - mean; v += d * d; }
    float invVar = rsqrt(tk_sum(v, scratch, L) / (float)n + epsilon);
    if (!active) return;
    if (lane == 0) { saveMean[row] = mean; saveInvVar[row] = invVar; }
    for (int i = lane; i < n; i += L)
    {
        float normalized = (input[base + i] - mean) * invVar;
        output[base + i] = gamma[i] * normalized + beta[i];
    }
}

void tk_layernorm_backward_body(__global const float* gradOutput, __global const float* input, __global const float* gamma,
    __global const float* saveMean, __global const float* saveInvVar, __global float* gradInput, int rows, int n, int L,
    __local float* scratch)
{
    int t = (int)get_global_id(0);
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
    sumDy = tk_sum(sumDy, scratch, L);
    sumDyXmu = tk_sum(sumDyXmu, scratch, L);
    if (!active) return;
    for (int i = lane; i < n; i += L)
    {
        float xmu = input[base + i] - mean;
        float dxhat = gradOutput[base + i] * gamma[i];
        gradInput[base + i] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)n);
    }
}

// Column reduction over a 32-column tile with R row lanes (work group 32 x R): gradGamma (and gradBeta), or RMSNorm's
// gradGamma. Local tiles padded to 33 columns.
void tk_column_grad_body(__global const float* gradOutput, __global const float* input, __global const float* stat0,
    __global const float* stat1, __global float* out0, __global float* out1, int rows, int n, int R, bool rms,
    __local float* s0, __local float* s1)
{
    int cx = (int)get_local_id(0), ry = (int)get_local_id(1);
    int col = (int)get_group_id(0) * 32 + cx;
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
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = R / 2; h > 0; h >>= 1)
    {
        if (ry < h) { s0[ry * 33 + cx] += s0[(ry + h) * 33 + cx]; s1[ry * 33 + cx] += s1[(ry + h) * 33 + cx]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (ry == 0 && col < n)
    {
        out0[col] = s0[cx];
        if (!rms) out1[col] = s1[cx];
    }
}
");
            foreach (int l in RowLanes)
            {
                string ls = l.ToString(CultureInfo.InvariantCulture);
                sb.Append($@"
__kernel __attribute__((reqd_work_group_size({GroupSize}, 1, 1))) void {SoftmaxName(l)}(__global const float* input, __global float* output, int rows, int n)
{{ __local float scratch[{GroupSize}]; tk_softmax_body(input, output, rows, n, {ls}, scratch); }}
__kernel __attribute__((reqd_work_group_size({GroupSize}, 1, 1))) void {SoftmaxBackwardName(l)}(__global const float* gradOutput, __global const float* output, __global float* gradInput, int rows, int n)
{{ __local float scratch[{GroupSize}]; tk_softmax_backward_body(gradOutput, output, gradInput, rows, n, {ls}, scratch); }}
__kernel __attribute__((reqd_work_group_size({GroupSize}, 1, 1))) void {LayerNormName(l)}(__global const float* input, __global float* output, __global const float* gamma, __global const float* beta, __global float* saveMean, __global float* saveInvVar, int rows, int n, float epsilon)
{{ __local float scratch[{GroupSize}]; tk_layernorm_forward_body(input, output, gamma, beta, saveMean, saveInvVar, rows, n, epsilon, {ls}, scratch); }}
__kernel __attribute__((reqd_work_group_size({GroupSize}, 1, 1))) void {LayerNormBackwardName(l)}(__global const float* gradOutput, __global const float* input, __global const float* gamma, __global const float* saveMean, __global const float* saveInvVar, __global float* gradInput, int rows, int n)
{{ __local float scratch[{GroupSize}]; tk_layernorm_backward_body(gradOutput, input, gamma, saveMean, saveInvVar, gradInput, rows, n, {ls}, scratch); }}
");
            }
            foreach (int r in ColumnRowLanes)
            {
                string rs = r.ToString(CultureInfo.InvariantCulture);
                sb.Append($@"
__kernel __attribute__((reqd_work_group_size(32, {rs}, 1))) void {LayerNormGradParamsName(r)}(__global const float* gradOutput, __global const float* input, __global const float* saveMean, __global const float* saveInvVar, __global float* gradGamma, __global float* gradBeta, int rows, int n)
{{ __local float s0[{rs} * 33]; __local float s1[{rs} * 33]; tk_column_grad_body(gradOutput, input, saveMean, saveInvVar, gradGamma, gradBeta, rows, n, {rs}, false, s0, s1); }}
__kernel __attribute__((reqd_work_group_size(32, {rs}, 1))) void {RmsNormGradGammaName(r)}(__global const float* gradOutput, __global const float* input, __global const float* saveRms, __global float* gradGamma, int rows, int n)
{{ __local float s0[{rs} * 33]; __local float s1[{rs} * 33]; tk_column_grad_body(gradOutput, input, saveRms, saveRms, gradGamma, gradGamma, rows, n, {rs}, true, s0, s1); }}
");
            }
            return sb.ToString();
        }

        public static string[] GetKernelNames()
        {
            var names = new List<string>();
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
    }
}
#endif