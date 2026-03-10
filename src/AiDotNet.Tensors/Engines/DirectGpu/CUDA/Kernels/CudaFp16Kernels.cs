namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for FP16 (half-precision) conversion and compute operations.
/// Includes conversion between FP32/FP16, element-wise arithmetic, and activations
/// that operate directly on half-precision data using __half2 packed arithmetic.
/// </summary>
public static class CudaFp16Kernels
{
    public static string GetSource()
    {
        return @"
#include <cuda_fp16.h>

// ============================================================================
// FP16 CONVERSION KERNELS
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void convert_fp32_to_fp16(
    const float* input, unsigned short* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h = __float2half(input[idx]);
    output[idx] = *reinterpret_cast<unsigned short*>(&h);
}

extern ""C"" __global__ __launch_bounds__(256) void convert_fp16_to_fp32(
    const unsigned short* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h = *reinterpret_cast<const __half*>(&input[idx]);
    output[idx] = __half2float(h);
}

extern ""C"" __global__ __launch_bounds__(256) void convert_fp32_to_fp16_rounding(
    const float* input, unsigned short* output, int size, int roundMode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h;
    switch (roundMode) {
        case 1: h = __float2half_rz(input[idx]); break;
        case 2: h = __float2half_rd(input[idx]); break;
        case 3: h = __float2half_ru(input[idx]); break;
        default: h = __float2half_rn(input[idx]); break;
    }
    output[idx] = *reinterpret_cast<unsigned short*>(&h);
}

extern ""C"" __global__ __launch_bounds__(256) void convert_fp32_to_fp16_vec2(
    const float2* input, uint* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPairs = size / 2;
    if (idx >= numPairs) return;

    float2 f2 = input[idx];
    __half2 h2 = __floats2half2_rn(f2.x, f2.y);
    output[idx] = *reinterpret_cast<uint*>(&h2);
}

extern ""C"" __global__ __launch_bounds__(256) void convert_fp16_to_fp32_vec2(
    const uint* input, float2* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPairs = size / 2;
    if (idx >= numPairs) return;

    __half2 h2 = *reinterpret_cast<const __half2*>(&input[idx]);
    float2 f2 = __half22float2(h2);
    output[idx] = f2;
}

// ============================================================================
// FP16 ELEMENT-WISE ARITHMETIC (packed __half2 for 2x throughput)
// ============================================================================

// All arithmetic kernels process 2 elements per thread using __half2 packed ops.
// Input/output stored as unsigned short (16-bit). Grid covers size/2 pairs.

extern ""C"" __global__ __launch_bounds__(256) void fp16_add(
    const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 va = *reinterpret_cast<const __half2*>(&a[pairIdx]);
        __half2 vb = *reinterpret_cast<const __half2*>(&b[pairIdx]);
        __half2 vc = __hadd2(va, vb);
        *reinterpret_cast<__half2*>(&out[pairIdx]) = vc;
    } else if (pairIdx < size) {
        __half ha = *reinterpret_cast<const __half*>(&a[pairIdx]);
        __half hb = *reinterpret_cast<const __half*>(&b[pairIdx]);
        __half hc = __hadd(ha, hb);
        *reinterpret_cast<__half*>(&out[pairIdx]) = hc;
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
        __half2 vc = __hsub2(va, vb);
        *reinterpret_cast<__half2*>(&out[pairIdx]) = vc;
    } else if (pairIdx < size) {
        __half ha = *reinterpret_cast<const __half*>(&a[pairIdx]);
        __half hb = *reinterpret_cast<const __half*>(&b[pairIdx]);
        __half hc = __hsub(ha, hb);
        *reinterpret_cast<__half*>(&out[pairIdx]) = hc;
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
        __half2 vc = __hmul2(va, vb);
        *reinterpret_cast<__half2*>(&out[pairIdx]) = vc;
    } else if (pairIdx < size) {
        __half ha = *reinterpret_cast<const __half*>(&a[pairIdx]);
        __half hb = *reinterpret_cast<const __half*>(&b[pairIdx]);
        __half hc = __hmul(ha, hb);
        *reinterpret_cast<__half*>(&out[pairIdx]) = hc;
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
        __half2 vr = __hfma2(va, vb, vc);
        *reinterpret_cast<__half2*>(&out[pairIdx]) = vr;
    } else if (pairIdx < size) {
        __half ha = *reinterpret_cast<const __half*>(&a[pairIdx]);
        __half hb = *reinterpret_cast<const __half*>(&b[pairIdx]);
        __half hc = *reinterpret_cast<const __half*>(&c[pairIdx]);
        __half hr = __hfma(ha, hb, hc);
        *reinterpret_cast<__half*>(&out[pairIdx]) = hr;
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
// FP16 ACTIVATION KERNELS (packed __half2)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void fp16_relu(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    __half2 zero2 = __float2half2_rn(0.0f);
    if (pairIdx + 1 < size) {
        __half2 v = *reinterpret_cast<const __half2*>(&input[pairIdx]);
        // max(v, 0) using comparison + blend
        __half2 mask = __hge2(v, zero2);
        __half2 result;
        // Use FP16 multiply by mask (1.0 or 0.0)
        result = __hmul2(v, mask);
        *reinterpret_cast<__half2*>(&output[pairIdx]) = result;
    } else if (pairIdx < size) {
        __half v = *reinterpret_cast<const __half*>(&input[pairIdx]);
        __half zero = __float2half(0.0f);
        __half result = __hgt(v, zero) ? v : zero;
        *reinterpret_cast<__half*>(&output[pairIdx]) = result;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void fp16_sigmoid(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairIdx = idx * 2;
    if (pairIdx + 1 < size) {
        __half2 v = *reinterpret_cast<const __half2*>(&input[pairIdx]);
        // Compute sigmoid in FP32 for accuracy, store as FP16
        float2 fv = __half22float2(v);
        fv.x = 1.0f / (1.0f + expf(-fv.x));
        fv.y = 1.0f / (1.0f + expf(-fv.y));
        *reinterpret_cast<__half2*>(&output[pairIdx]) = __floats2half2_rn(fv.x, fv.y);
    } else if (pairIdx < size) {
        __half v = *reinterpret_cast<const __half*>(&input[pairIdx]);
        float fv = __half2float(v);
        fv = 1.0f / (1.0f + expf(-fv));
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
        float fv = tanhf(__half2float(v));
        *reinterpret_cast<__half*>(&output[pairIdx]) = __float2half(fv);
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
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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
        __half v = *reinterpret_cast<const __half*>(&input[pairIdx]);
        float x = __half2float(v);
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float result = 0.5f * x * (1.0f + tanhf(inner));
        *reinterpret_cast<__half*>(&output[pairIdx]) = __float2half(result);
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
        __half v = *reinterpret_cast<const __half*>(&input[pairIdx]);
        float x = __half2float(v);
        *reinterpret_cast<__half*>(&output[pairIdx]) = __float2half(x / (1.0f + expf(-x)));
    }
}

// ============================================================================
// FP16 REDUCTION KERNELS (warp shuffle + shared memory)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void fp16_reduce_sum(
    const unsigned short* __restrict__ input, float* __restrict__ output, int size)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Accumulate in FP32 for precision
    float val = 0.0f;
    for (unsigned int i = idx; i < (unsigned int)size; i += blockDim.x * gridDim.x) {
        __half h = *reinterpret_cast<const __half*>(&input[i]);
        val += __half2float(h);
    }

    // Warp shuffle reduction
    unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);

    unsigned int lane = tid & 31;
    unsigned int warpId = tid >> 5;
    if (lane == 0) scratch[warpId] = val;
    __syncthreads();

    unsigned int numWarps = (blockDim.x + 31) >> 5;
    if (tid < numWarps) {
        val = scratch[tid];
        unsigned int warp_mask = (numWarps >= 32) ? 0xFFFFFFFF : ((1u << numWarps) - 1);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(warp_mask, val, offset);
        if (tid == 0) atomicAdd(&output[0], val);
    }
}

// ============================================================================
// MIXED-PRECISION: FP16 input, FP32 accumulate, FP16 output
// Useful for attention softmax where precision matters for the reduction
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

    // Pass 1: find max (in FP32)
    float maxVal = -1e30f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __half2float(*reinterpret_cast<const __half*>(&rowIn[c]));
        maxVal = fmaxf(maxVal, v);
    }
    // Warp reduce max
    unsigned int mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        maxVal = fmaxf(maxVal, __shfl_down_sync(mask, maxVal, offset));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = maxVal;
    __syncthreads();
    if (threadIdx.x < ((blockDim.x + 31) >> 5)) {
        maxVal = smem[threadIdx.x];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            maxVal = fmaxf(maxVal, __shfl_down_sync(mask, maxVal, offset));
        if (threadIdx.x == 0) smem[0] = maxVal;
    }
    __syncthreads();
    maxVal = smem[0];

    // Pass 2: exp(x - max) and sum (in FP32)
    float sumVal = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __half2float(*reinterpret_cast<const __half*>(&rowIn[c]));
        sumVal += expf(v - maxVal);
    }
    // Warp reduce sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sumVal += __shfl_down_sync(mask, sumVal, offset);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = sumVal;
    __syncthreads();
    if (threadIdx.x < ((blockDim.x + 31) >> 5)) {
        sumVal = smem[threadIdx.x];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sumVal += __shfl_down_sync(mask, sumVal, offset);
        if (threadIdx.x == 0) smem[0] = sumVal;
    }
    __syncthreads();
    float invSum = 1.0f / smem[0];

    // Pass 3: normalize and convert back to FP16
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __half2float(*reinterpret_cast<const __half*>(&rowIn[c]));
        float result = expf(v - maxVal) * invSum;
        *reinterpret_cast<__half*>(&rowOut[c]) = __float2half(result);
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
            "fp16_softmax"
        };
    }
}
