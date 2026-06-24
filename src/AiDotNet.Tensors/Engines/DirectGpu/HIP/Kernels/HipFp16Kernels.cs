// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for FP16 (half-precision) conversion operations.
// HIP is source-compatible with CUDA for FP16 intrinsics.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for FP16 (half-precision) conversion operations.
/// These kernels convert between FP32 (float) and FP16 (half) precision.
/// </summary>
public static class HipFp16Kernels
{
    public static string GetSource()
    {
        // HIP provides FP16 support via hip_fp16.h which is automatically available in hiprtc
        // The __half type and conversion functions are available when targeting AMD GPUs with FP16 support
        return @"
#include <hip/hip_fp16.h>

// ============================================================================
// FP16 CONVERSION KERNELS
// ============================================================================

// Convert FP32 (float) array to FP16 (half) array
// input: float array of size 'size'
// output: half array of size 'size' (stored as unsigned short for compatibility)
extern ""C"" __global__ __launch_bounds__(256) void convert_fp32_to_fp16(
    const float* input, unsigned short* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h = __float2half(input[idx]);
    output[idx] = *reinterpret_cast<unsigned short*>(&h);
}

// Convert FP16 (half) array to FP32 (float) array
// input: half array of size 'size' (stored as unsigned short for compatibility)
// output: float array of size 'size'
extern ""C"" __global__ __launch_bounds__(256) void convert_fp16_to_fp32(
    const unsigned short* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h = *reinterpret_cast<const __half*>(&input[idx]);
    output[idx] = __half2float(h);
}

// Convert FP32 to FP16 with rounding mode control
// roundMode: 0 = round to nearest even (default), 1 = round toward zero, 2 = round down, 3 = round up
extern ""C"" __global__ __launch_bounds__(256) void convert_fp32_to_fp16_rounding(
    const float* input, unsigned short* output, int size, int roundMode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h;
    switch (roundMode) {
        case 1: h = __float2half_rz(input[idx]); break;  // Round toward zero
        case 2: h = __float2half_rd(input[idx]); break;  // Round down
        case 3: h = __float2half_ru(input[idx]); break;  // Round up
        default: h = __float2half_rn(input[idx]); break; // Round to nearest even
    }
    output[idx] = *reinterpret_cast<unsigned short*>(&h);
}

// Vectorized FP32 to FP16 conversion (processes 2 elements per thread for better performance)
extern ""C"" __global__ __launch_bounds__(256) void convert_fp32_to_fp16_vec2(
    const float2* input, unsigned int* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPairs = size / 2;
    if (idx >= numPairs) return;

    float2 f2 = input[idx];
    __half2 h2 = __floats2half2_rn(f2.x, f2.y);
    output[idx] = *reinterpret_cast<unsigned int*>(&h2);
}

// Vectorized FP16 to FP32 conversion (processes 2 elements per thread for better performance)
extern ""C"" __global__ __launch_bounds__(256) void convert_fp16_to_fp32_vec2(
    const unsigned int* input, float2* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPairs = size / 2;
    if (idx >= numPairs) return;

    __half2 h2 = *reinterpret_cast<const __half2*>(&input[idx]);
    float2 f2 = __half22float2(h2);
    output[idx] = f2;
}

// ============================================================================
// FP16 CONV im2col (#1650/#638): fused im2col + FP32->FP16 in TRANSPOSED [K,N] layout (K=channels*kernelH*kernelW,
// N=batch*outH*outW) so the conv becomes a plain NN GEMM out[outC,N] = weights[outC,K] . col[K,N] via
// GemmFp16In32fOut (hipBLAS, FP16 multiply / FP32 accumulate on the MFMA matrix units) — the industry conv path.
// ONE THREAD PER col ELEMENT (N*K threads) → coalesced col[t]=col[k*N+n] writes (adjacent threads → adjacent n →
// adjacent addresses) + full occupancy (a thread-per-patch version launched only N threads → starved).
// ============================================================================
extern ""C"" __global__ __launch_bounds__(256) void im2col_kn_fp16hw(
    const float* __restrict__ input, unsigned short* __restrict__ output,
    int batch, int channels, int height, int width,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW, int outH, int outW)
{
    int N = batch * outH * outW;
    int K = channels * kernelH * kernelW;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)N * K) return;
    int n = (int)(t % N);     // output position (fastest — coalesced)
    int k = (int)(t / N);     // unrolled (c,kh,kw) row
    int b = n / (outH * outW);
    int rem = n % (outH * outW);
    int oh = rem / outW;
    int ow = rem % outW;
    int kw = k % kernelW;
    int kh = (k / kernelW) % kernelH;
    int c  = k / (kernelW * kernelH);
    int ih = oh * strideH - padH + kh * dilationH;
    int iw = ow * strideW - padW + kw * dilationW;
    float val = (ih >= 0 && ih < height && iw >= 0 && iw < width)
        ? input[((b * channels + c) * height + ih) * width + iw] : 0.0f;
    __half h = __float2half(val);
    output[t] = *reinterpret_cast<unsigned short*>(&h);
}

// ============================================================================
// FP16 ELEMENT-WISE ARITHMETIC (packed __half2 for 2x throughput)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void fp16_add(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 va = *reinterpret_cast<const __half2*>(&a[pairIdx]);
        __half2 vb = *reinterpret_cast<const __half2*>(&b[pairIdx]);
        *reinterpret_cast<__half2*>(&out[pairIdx]) = __hadd2(va, vb);
    } else if (pairIdx < size) {
        __half ha = *reinterpret_cast<const __half*>(&a[pairIdx]);
        __half hb = *reinterpret_cast<const __half*>(&b[pairIdx]);
        *reinterpret_cast<__half*>(&out[pairIdx]) = __hadd(ha, hb);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_sub(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 va = *reinterpret_cast<const __half2*>(&a[pairIdx]);
        __half2 vb = *reinterpret_cast<const __half2*>(&b[pairIdx]);
        *reinterpret_cast<__half2*>(&out[pairIdx]) = __hsub2(va, vb);
    } else if (pairIdx < size) {
        __half ha = *reinterpret_cast<const __half*>(&a[pairIdx]);
        __half hb = *reinterpret_cast<const __half*>(&b[pairIdx]);
        *reinterpret_cast<__half*>(&out[pairIdx]) = __hsub(ha, hb);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_mul(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 va = *reinterpret_cast<const __half2*>(&a[pairIdx]);
        __half2 vb = *reinterpret_cast<const __half2*>(&b[pairIdx]);
        *reinterpret_cast<__half2*>(&out[pairIdx]) = __hmul2(va, vb);
    } else if (pairIdx < size) {
        __half ha = *reinterpret_cast<const __half*>(&a[pairIdx]);
        __half hb = *reinterpret_cast<const __half*>(&b[pairIdx]);
        *reinterpret_cast<__half*>(&out[pairIdx]) = __hmul(ha, hb);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_fma(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    const unsigned short* __restrict__ c, unsigned short* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 va = *reinterpret_cast<const __half2*>(&a[pairIdx]);
        __half2 vb = *reinterpret_cast<const __half2*>(&b[pairIdx]);
        __half2 vc = *reinterpret_cast<const __half2*>(&c[pairIdx]);
        *reinterpret_cast<__half2*>(&out[pairIdx]) = __hfma2(va, vb, vc);
    } else if (pairIdx < size) {
        __half ha = *reinterpret_cast<const __half*>(&a[pairIdx]);
        __half hb = *reinterpret_cast<const __half*>(&b[pairIdx]);
        __half hc = *reinterpret_cast<const __half*>(&c[pairIdx]);
        *reinterpret_cast<__half*>(&out[pairIdx]) = __hfma(ha, hb, hc);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_scale(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output,
    unsigned short scaleVal, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    __half s = *reinterpret_cast<const __half*>(&scaleVal);
    __half2 s2 = __half2half2(s);
    if (pairIdx + 1 < size) {
        __half2 v = *reinterpret_cast<const __half2*>(&input[pairIdx]);
        *reinterpret_cast<__half2*>(&output[pairIdx]) = __hmul2(v, s2);
    } else if (pairIdx < size) {
        __half v = *reinterpret_cast<const __half*>(&input[pairIdx]);
        *reinterpret_cast<__half*>(&output[pairIdx]) = __hmul(v, s);
    }
}

// ============================================================================
// FP16 ACTIVATION KERNELS
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void fp16_relu(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    __half2 zero2 = __float2half2_rn(0.0f);
    if (pairIdx + 1 < size) {
        __half2 v = *reinterpret_cast<const __half2*>(&input[pairIdx]);
        *reinterpret_cast<__half2*>(&output[pairIdx]) = __hmax2(v, zero2);
    } else if (pairIdx < size) {
        __half v = *reinterpret_cast<const __half*>(&input[pairIdx]);
        __half zero = __float2half(0.0f);
        *reinterpret_cast<__half*>(&output[pairIdx]) = __hmax(v, zero);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_sigmoid(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 v = *reinterpret_cast<const __half2*>(&input[pairIdx]);
        float2 fv = __half22float2(v);
        fv.x = 1.0f / (1.0f + expf(-fv.x));
        fv.y = 1.0f / (1.0f + expf(-fv.y));
        *reinterpret_cast<__half2*>(&output[pairIdx]) = __floats2half2_rn(fv.x, fv.y);
    } else if (pairIdx < size) {
        __half v = *reinterpret_cast<const __half*>(&input[pairIdx]);
        float fv = 1.0f / (1.0f + expf(-__half2float(v)));
        *reinterpret_cast<__half*>(&output[pairIdx]) = __float2half(fv);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_tanh(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 v = *reinterpret_cast<const __half2*>(&input[pairIdx]);
        float2 fv = __half22float2(v);
        fv.x = tanhf(fv.x);
        fv.y = tanhf(fv.y);
        *reinterpret_cast<__half2*>(&output[pairIdx]) = __floats2half2_rn(fv.x, fv.y);
    } else if (pairIdx < size) {
        __half v = *reinterpret_cast<const __half*>(&input[pairIdx]);
        *reinterpret_cast<__half*>(&output[pairIdx]) = __float2half(tanhf(__half2float(v)));
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_gelu(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 v = *reinterpret_cast<const __half2*>(&input[pairIdx]);
        float2 fv = __half22float2(v);
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            float x = (i == 0) ? fv.x : fv.y;
            float x3 = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x3);
            float result = 0.5f * x * (1.0f + tanhf(inner));
            if (i == 0) fv.x = result; else fv.y = result;
        }
        *reinterpret_cast<__half2*>(&output[pairIdx]) = __floats2half2_rn(fv.x, fv.y);
    } else if (pairIdx < size) {
        float x = __half2float(*reinterpret_cast<const __half*>(&input[pairIdx]));
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        *reinterpret_cast<__half*>(&output[pairIdx]) = __float2half(0.5f * x * (1.0f + tanhf(inner)));
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_swish(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 v = *reinterpret_cast<const __half2*>(&input[pairIdx]);
        float2 fv = __half22float2(v);
        fv.x = fv.x / (1.0f + expf(-fv.x));
        fv.y = fv.y / (1.0f + expf(-fv.y));
        *reinterpret_cast<__half2*>(&output[pairIdx]) = __floats2half2_rn(fv.x, fv.y);
    } else if (pairIdx < size) {
        float x = __half2float(*reinterpret_cast<const __half*>(&input[pairIdx]));
        *reinterpret_cast<__half*>(&output[pairIdx]) = __float2half(x / (1.0f + expf(-x)));
    }
}

// ============================================================================
// FP16 REDUCTION (warp shuffle + shared memory, FP32 accumulation)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void fp16_reduce_sum(
    const unsigned short* __restrict__ input, float* __restrict__ output, int size)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    for (unsigned int i = idx; i < (unsigned int)size; i += blockDim.x * gridDim.x) {
        __half h = *reinterpret_cast<const __half*>(&input[i]);
        val += __half2float(h);
    }

    unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);

    unsigned int lane = tid & 31;
    unsigned int warpId = tid >> 5;
    if (lane == 0) scratch[warpId] = val;
    __syncthreads();

    unsigned int numWarps = (blockDim.x + 31) >> 5;
    val = (tid < numWarps) ? scratch[tid] : 0.0f;
    unsigned int warp_mask = (numWarps >= 32) ? 0xFFFFFFFF : ((1u << numWarps) - 1);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(warp_mask, val, offset);
    // NON-DETERMINISTIC (issue #382); see fp16_reduce_sum_deterministic.
    if (tid == 0) atomicAdd(&output[0], val);
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_reduce_sum_deterministic(
    const unsigned short* __restrict__ input, float* __restrict__ output, int size)
{
    extern __shared__ float scratch_d[];
    unsigned int tid = threadIdx.x;
    float val = 0.0f;
    for (unsigned int i = tid; i < (unsigned int)size; i += blockDim.x) {
        __half h = *reinterpret_cast<const __half*>(&input[i]);
        val += __half2float(h);
    }
    unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    unsigned int lane = tid & 31;
    unsigned int warpId = tid >> 5;
    if (lane == 0) scratch_d[warpId] = val;
    __syncthreads();
    unsigned int numWarps = (blockDim.x + 31) >> 5;
    val = (tid < numWarps) ? scratch_d[tid] : 0.0f;
    unsigned int warp_mask = (numWarps >= 32) ? 0xFFFFFFFF : ((1u << numWarps) - 1);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(warp_mask, val, offset);
    if (tid == 0) *output = val;
}

// ============================================================================
// FP16 SOFTMAX (mixed precision: FP16 I/O, FP32 accumulation)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void fp16_softmax(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output,
    int rows, int cols)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned short* rowIn = input + row * cols;
    unsigned short* rowOut = output + row * cols;

    float maxVal = -1e30f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __half2float(*reinterpret_cast<const __half*>(&rowIn[c]));
        maxVal = fmaxf(maxVal, v);
    }
    unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        maxVal = fmaxf(maxVal, __shfl_down_sync(mask, maxVal, offset));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = maxVal;
    __syncthreads();
    {
        unsigned int nw = (blockDim.x + 31) >> 5;
        maxVal = (threadIdx.x < nw) ? smem[threadIdx.x] : -1e30f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            maxVal = fmaxf(maxVal, __shfl_down_sync(0xFFFFFFFF, maxVal, offset));
        if (threadIdx.x == 0) smem[0] = maxVal;
    }
    __syncthreads();
    maxVal = smem[0];

    float sumVal = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __half2float(*reinterpret_cast<const __half*>(&rowIn[c]));
        sumVal += expf(v - maxVal);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sumVal += __shfl_down_sync(mask, sumVal, offset);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = sumVal;
    __syncthreads();
    {
        unsigned int nw2 = (blockDim.x + 31) >> 5;
        sumVal = (threadIdx.x < nw2) ? smem[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sumVal += __shfl_down_sync(0xFFFFFFFF, sumVal, offset);
        if (threadIdx.x == 0) smem[0] = sumVal;
    }
    __syncthreads();
    float invSum = 1.0f / smem[0];

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __half2float(*reinterpret_cast<const __half*>(&rowIn[c]));
        *reinterpret_cast<__half*>(&rowOut[c]) = __float2half(expf(v - maxVal) * invSum);
    }
}

// ============================================================================
// FP16 LAYERNORM (FP16 I/O + gamma/beta, FP32 accumulation). ONE BLOCK PER ROW.
// Full-block shared-memory reduction (no warp-size assumption, correct on wave32/wave64).
// Population variance (/cols), eps inside rsqrt — matches the engine's LayerNorm convention.
// Writes per-row FP32 mean + variance for the backward; meanOut/varOut are optional + independent.
// ============================================================================
extern ""C"" __global__ void fp16_layernorm(
    const unsigned short* __restrict__ input, const unsigned short* __restrict__ gamma,
    const unsigned short* __restrict__ beta, unsigned short* __restrict__ output,
    float* __restrict__ meanOut, float* __restrict__ varOut, int rows, int cols, float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* in = input + (long long)row * cols;
    unsigned short* out = output + (long long)row * cols;
    extern __shared__ float sdata[];
    int tid = threadIdx.x; int bs = blockDim.x;
    float s = 0.0f;
    for (int i = tid; i < cols; i += bs) s += __half2float(*reinterpret_cast<const __half*>(&in[i]));
    sdata[tid] = s; __syncthreads();
    for (int st = bs >> 1; st > 0; st >>= 1) { if (tid < st) sdata[tid] += sdata[tid + st]; __syncthreads(); }
    float mean = sdata[0] / (float)cols; __syncthreads();
    float vv = 0.0f;
    for (int i = tid; i < cols; i += bs) { float d = __half2float(*reinterpret_cast<const __half*>(&in[i])) - mean; vv += d * d; }
    sdata[tid] = vv; __syncthreads();
    for (int st = bs >> 1; st > 0; st >>= 1) { if (tid < st) sdata[tid] += sdata[tid + st]; __syncthreads(); }
    float var = sdata[0] / (float)cols; __syncthreads();
    float invstd = rsqrtf(var + eps);
    if (tid == 0) { if (meanOut) meanOut[row] = mean; if (varOut) varOut[row] = var; }
    for (int i = tid; i < cols; i += bs) {
        float norm = (__half2float(*reinterpret_cast<const __half*>(&in[i])) - mean) * invstd;
        float g = __half2float(*reinterpret_cast<const __half*>(&gamma[i]));
        float b = __half2float(*reinterpret_cast<const __half*>(&beta[i]));
        *reinterpret_cast<__half*>(&out[i]) = __float2half(norm * g + b);
    }
}
";
    }

    /// <summary>
    /// Gets the list of kernel names for compilation.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "im2col_kn_fp16hw",
            "convert_fp32_to_fp16",
            "convert_fp16_to_fp32",
            "convert_fp32_to_fp16_rounding",
            "convert_fp32_to_fp16_vec2",
            "convert_fp16_to_fp32_vec2",
            "fp16_add",
            "fp16_sub",
            "fp16_mul",
            "fp16_fma",
            "fp16_scale",
            "fp16_relu",
            "fp16_sigmoid",
            "fp16_tanh",
            "fp16_gelu",
            "fp16_swish",
            "fp16_reduce_sum",
            "fp16_reduce_sum_deterministic",
            "fp16_softmax",
            "fp16_layernorm"
        };
    }
}
