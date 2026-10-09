// Copyright (c) AiDotNet. All rights reserved.
// Generated row/column kernel variants served through the tuned-kernel registry.

using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// Generates the registry's row-wise and column-wise CUDA kernel variants from ONE definition per op,
    /// instantiated over a lane-count parameter. The established kernels (<c>softmax</c>, <c>softmax_backward</c>,
    /// <c>layernorm_forward/backward/grad_params</c>, <c>rmsnorm_grad_gamma</c>) map work to threads in one fixed way
    /// whatever the row length: a 256-thread block per row (so a 32-wide attention row leaves 224 threads idle), a
    /// single thread per row (softmax backward, strided uncoalesced loads), or a single thread per COLUMN looping over
    /// every row (the affine-parameter gradients: 64 threads total for a d=64 LayerNorm). These variants instead
    /// assign <c>L</c> lanes to each row (L = 8, 16 or 32; several rows per warp) or a 32-column tile with
    /// <c>R</c> row lanes to each block, and the registry measures which variant wins for each shape class.
    /// </summary>
    /// <remarks>
    /// <para>Every variant is deterministic: reductions run in a fixed order with no atomics.</para>
    /// <para>Bit-compatibility: the per-lane partial sums are formed in the same order as the reference kernels'
    /// per-thread partials and then combined by the same <c>__shfl_down_sync</c> tree, so for rows no longer than the
    /// lane count (softmax) or twice the lane count when L = 32 (LayerNorm, whose reference folds lane+32 first) the
    /// outputs are bit-identical to the reference; longer rows associate differently and agree to rounding.</para>
    /// <para>Inactive tail rows keep executing the shuffles (with neutral values) so the full-warp masks are valid.</para>
    /// <para>CUDA only for now. The other backends run their established kernels for these ops, unchanged; registry
    /// dispatch and generated variants for them are tracked per backend: HIP #1113, OpenCL #1114, Vulkan #1115,
    /// Metal #1116, WebGPU #1117.</para>
    /// </remarks>
    internal static class CudaTunedRowKernels
    {
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
#include <math.h>

template <int L> __device__ __forceinline__ float tk_sum(float v) {
    #pragma unroll
    for (int o = L / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o, L);
    return __shfl_sync(0xffffffffu, v, 0, L);
}
template <int L> __device__ __forceinline__ float tk_max(float v) {
    #pragma unroll
    for (int o = L / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o, L));
    return __shfl_sync(0xffffffffu, v, 0, L);
}

// Row softmax, L lanes per row. Matches softmax: max, exp written to output, sum, scale by 1/sum (or 1 when the
// sum is not positive).
template <int L> __device__ __forceinline__ void tk_softmax_body(
    const float* __restrict__ input, float* __restrict__ output, int rows, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int row = t / L, lane = t % L;
    bool active = row < rows;
    const float* x = input + (size_t)(active ? row : 0) * n;
    float* y = output + (size_t)(active ? row : 0) * n;
    float m = -INFINITY;
    if (active) for (int f = lane; f < n; f += L) m = fmaxf(m, x[f]);
    m = tk_max<L>(m);
    float s = 0.0f;
    if (active) for (int f = lane; f < n; f += L) { float e = expf(x[f] - m); y[f] = e; s += e; }
    s = tk_sum<L>(s);
    float inv = (s > 0.0f) ? (1.0f / s) : 1.0f;
    if (active) for (int f = lane; f < n; f += L) y[f] *= inv;
}

// Row softmax backward: dx = y * (dy - sum(dy * y)).
template <int L> __device__ __forceinline__ void tk_softmax_backward_body(
    const float* __restrict__ gradOutput, const float* __restrict__ output, float* __restrict__ gradInput,
    int rows, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int row = t / L, lane = t % L;
    bool active = row < rows;
    size_t base = (size_t)(active ? row : 0) * n;
    float dot = 0.0f;
    if (active) for (int f = lane; f < n; f += L) dot += gradOutput[base + f] * output[base + f];
    dot = tk_sum<L>(dot);
    if (active) for (int f = lane; f < n; f += L) gradInput[base + f] = output[base + f] * (gradOutput[base + f] - dot);
}

// LayerNorm forward, L lanes per row: two-pass mean / variance like layernorm_forward.
template <int L> __device__ __forceinline__ void tk_layernorm_forward_body(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ saveMean, float* __restrict__ saveInvVar, int rows, int n, float epsilon)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int row = t / L, lane = t % L;
    bool active = row < rows;
    size_t base = (size_t)(active ? row : 0) * n;
    float s = 0.0f;
    if (active) for (int i = lane; i < n; i += L) s += input[base + i];
    float mean = tk_sum<L>(s) / (float)n;
    float v = 0.0f;
    if (active) for (int i = lane; i < n; i += L) { float d = input[base + i] - mean; v += d * d; }
    float invVar = rsqrtf(tk_sum<L>(v) / (float)n + epsilon);
    if (!active) return;
    if (lane == 0) { saveMean[row] = mean; saveInvVar[row] = invVar; }
    for (int i = lane; i < n; i += L) {
        float normalized = (input[base + i] - mean) * invVar;
        output[base + i] = gamma[i] * normalized + beta[i];
    }
}

// LayerNorm backward (input gradient only), L lanes per row, same formula as layernorm_backward.
template <int L> __device__ __forceinline__ void tk_layernorm_backward_body(
    const float* __restrict__ gradOutput, const float* __restrict__ input, const float* __restrict__ gamma,
    const float* __restrict__ saveMean, const float* __restrict__ saveInvVar, float* __restrict__ gradInput,
    int rows, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int row = t / L, lane = t % L;
    bool active = row < rows;
    int r = active ? row : 0;
    size_t base = (size_t)r * n;
    float mean = saveMean[r], invVar = saveInvVar[r];
    float sumDy = 0.0f, sumDyXmu = 0.0f;
    if (active) for (int i = lane; i < n; i += L) {
        float dy = gradOutput[base + i] * gamma[i];
        sumDy += dy;
        sumDyXmu += dy * (input[base + i] - mean);
    }
    sumDy = tk_sum<L>(sumDy);
    sumDyXmu = tk_sum<L>(sumDyXmu);
    if (!active) return;
    for (int i = lane; i < n; i += L) {
        float xmu = input[base + i] - mean;
        float dxhat = gradOutput[base + i] * gamma[i];
        gradInput[base + i] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)n);
    }
}

// Column reduction over rows for affine-parameter gradients. Block = 32 columns x R row lanes; each row lane walks
// rows lane, lane+R, ... in order, then the R partials are folded in a fixed tree. No atomics.
template <int R, bool RMS> __device__ __forceinline__ void tk_column_grad_body(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    const float* __restrict__ stat0, const float* __restrict__ stat1,
    float* __restrict__ out0, float* __restrict__ out1, int rows, int n)
{
    __shared__ float s0[R][33];
    __shared__ float s1[R][33];
    int cx = threadIdx.x, ry = threadIdx.y;
    int col = blockIdx.x * 32 + cx;
    float a0 = 0.0f, a1 = 0.0f;
    if (col < n) {
        for (int b = ry; b < rows; b += R) {
            size_t idx = (size_t)b * n + col;
            float g = gradOutput[idx];
            if (RMS) {
                a0 += g * input[idx] * (1.0f / stat0[b]);
            } else {
                float normalized = (input[idx] - stat0[b]) * stat1[b];
                a0 += g * normalized;
                a1 += g;
            }
        }
    }
    s0[ry][cx] = a0; s1[ry][cx] = a1;
    __syncthreads();
    #pragma unroll
    for (int h = R / 2; h > 0; h >>= 1) {
        if (ry < h) { s0[ry][cx] += s0[ry + h][cx]; s1[ry][cx] += s1[ry + h][cx]; }
        __syncthreads();
    }
    if (ry == 0 && col < n) {
        out0[col] = s0[0][cx];
        if (!RMS) out1[col] = s1[0][cx];
    }
}
");
            foreach (int l in RowLanes)
            {
                string ls = l.ToString(CultureInfo.InvariantCulture);
                sb.Append($@"
extern ""C"" __global__ __launch_bounds__(256) void {SoftmaxName(l)}(const float* __restrict__ input, float* __restrict__ output, int rows, int n)
{{ tk_softmax_body<{ls}>(input, output, rows, n); }}
extern ""C"" __global__ __launch_bounds__(256) void {SoftmaxBackwardName(l)}(const float* __restrict__ gradOutput, const float* __restrict__ output, float* __restrict__ gradInput, int rows, int n)
{{ tk_softmax_backward_body<{ls}>(gradOutput, output, gradInput, rows, n); }}
extern ""C"" __global__ __launch_bounds__(256) void {LayerNormName(l)}(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ saveMean, float* __restrict__ saveInvVar, int rows, int n, float epsilon)
{{ tk_layernorm_forward_body<{ls}>(input, output, gamma, beta, saveMean, saveInvVar, rows, n, epsilon); }}
extern ""C"" __global__ __launch_bounds__(256) void {LayerNormBackwardName(l)}(const float* __restrict__ gradOutput, const float* __restrict__ input, const float* __restrict__ gamma, const float* __restrict__ saveMean, const float* __restrict__ saveInvVar, float* __restrict__ gradInput, int rows, int n)
{{ tk_layernorm_backward_body<{ls}>(gradOutput, input, gamma, saveMean, saveInvVar, gradInput, rows, n); }}
");
            }
            foreach (int r in ColumnRowLanes)
            {
                string rs = r.ToString(CultureInfo.InvariantCulture);
                sb.Append($@"
extern ""C"" __global__ __launch_bounds__({32 * r}) void {LayerNormGradParamsName(r)}(const float* __restrict__ gradOutput, const float* __restrict__ input, const float* __restrict__ saveMean, const float* __restrict__ saveInvVar, float* __restrict__ gradGamma, float* __restrict__ gradBeta, int rows, int n)
{{ tk_column_grad_body<{rs}, false>(gradOutput, input, saveMean, saveInvVar, gradGamma, gradBeta, rows, n); }}
extern ""C"" __global__ __launch_bounds__({32 * r}) void {RmsNormGradGammaName(r)}(const float* __restrict__ gradOutput, const float* __restrict__ input, const float* __restrict__ saveRms, float* __restrict__ gradGamma, int rows, int n)
{{ tk_column_grad_body<{rs}, true>(gradOutput, input, saveRms, saveRms, gradGamma, gradGamma, rows, n); }}
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
