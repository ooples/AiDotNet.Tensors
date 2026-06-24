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

// nvrtc has no <sys/types.h>, so `uint` (used by the vec2 conversion kernels below) is undefined under nvrtc
// even with the toolkit headers present — this is why the whole module historically failed to compile. Provide
// the typedef (identical re-typedef is legal C++ if a CUDA header also defines it).
typedef unsigned int uint;

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
        // NON-DETERMINISTIC (issue #382); see fp16_reduce_sum_deterministic.
        if (tid == 0) atomicAdd(&output[0], val);
    }
}

// fp16_reduce_sum — bit-deterministic variant (issue #382).
// Single-block strided reduction; no inter-block atomic.
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
    if (tid < numWarps) {
        val = scratch_d[tid];
        unsigned int warp_mask = (numWarps >= 32) ? 0xFFFFFFFF : ((1u << numWarps) - 1);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(warp_mask, val, offset);
        if (tid == 0) *output = val;
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
    {
        unsigned int nw = (blockDim.x + 31) >> 5;
        sumVal = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sumVal += __shfl_down_sync(0xFFFFFFFF, sumVal, offset);
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

// FUSED im2col + hardware FP32->FP16, TRANSPOSED [K,N] layout (K=channels*kH*kW, N=batch*outH*outW). This is
// the industry FP16-conv path: feed col[K,N] (FP16) + weights[outC,K] (FP16) to a Tensor-Core GEMM
// (GemmFp16: out[outC,N] = weights @ col, FP16 multiply / FP32 accumulate). Writing FP16 directly here (hw
// __float2half) avoids a separate cast pass + halves col bandwidth. #1650/#638 replay-floor lever.
extern ""C"" __global__ __launch_bounds__(256) void im2col_kn_fp16hw(
    const float* __restrict__ input, unsigned short* __restrict__ output,
    int batch, int channels, int height, int width,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW, int outH, int outW)
{
    // ONE THREAD PER col ELEMENT (N*K threads) — fills the GPU and writes col[t]=col[k*N+n] fully coalesced
    // (adjacent threads → adjacent n → adjacent addresses). The previous thread-per-patch version launched only
    // N threads (~4 blocks → ~6% occupancy → ~107us for a ~6us-of-bandwidth job); this is the fix.
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

// #1650/#638 (#671): DIRECT FP16 conv for the tiny-spatial deep convs the im2col+GEMM path skips (small N where
// im2col + launch overhead exceeds the Tensor-Core GEMM win). FP32 input rounded to FP16 per access (mirrors
// im2col_kn_fp16hw), FP16 weights, FP32 multiply-accumulate (matches GemmFp16In32fOut numerics), FP32 output.
// One thread per output (ow, oh) with a 3D grid z = b*outChannels. Capture-safe (single kernel, no device alloc).
extern ""C"" __global__ void conv2d_direct_fp16hw(
    const float* __restrict__ input, const unsigned short* __restrict__ weightsHalf, float* __restrict__ output,
    int batch, int inChannels, int inHeight, int inWidth,
    int outChannels, int outHeight, int outWidth,
    int kernelH, int kernelW, int strideH, int strideW,
    int padH, int padW, int dilationH, int dilationW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z;                 // b * outChannels + oc
    if (ow >= outWidth || oh >= outHeight) return;
    int oc = bz % outChannels;
    int b  = bz / outChannels;
    float acc = 0.0f;
    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH - padH + kh * dilationH;
            if (ih < 0 || ih >= inHeight) continue;
            for (int kw = 0; kw < kernelW; kw++) {
                int iw = ow * strideW - padW + kw * dilationW;
                if (iw < 0 || iw >= inWidth) continue;
                float inV = input[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                int wIdx = ((oc * inChannels + ic) * kernelH + kh) * kernelW + kw;
                __half wH = *reinterpret_cast<const __half*>(&weightsHalf[wIdx]);
                // FP16 operands, FP32 multiply-accumulate (Tensor-Core semantics): round input to FP16, mul in FP32.
                acc += __half2float(__float2half(inV)) * __half2float(wH);
            }
        }
    }
    output[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = acc;
}

// FP16-IN variant of im2col_kn_fp16hw: input is already FP16 (unsigned short), output FP16 col[K,N] in the SAME
// TRANSPOSED [K,N] layout (N=batch*outH*outW, K=channels*kH*kW). Each tap is loaded via __half2float (oob → 0);
// avoids a separate FP16→FP32 cast pass when the activation feeding the conv is already half-resident
// (#1650/#638 #3 — FP16-activation diffusion inference). ONE THREAD PER col element (N*K threads), fully coalesced.
extern ""C"" __global__ __launch_bounds__(256) void im2col_kn_fp16_from_fp16(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output,
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
    float val;
    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
        int idx = ((b * channels + c) * height + ih) * width + iw;
        val = __half2float(*reinterpret_cast<const __half*>(&input[idx]));
    } else {
        val = 0.0f;
    }
    __half h = __float2half(val);
    output[t] = *reinterpret_cast<unsigned short*>(&h); // store the half BITS (NOT a value-convert __half→ushort)
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
            "conv2d_direct_fp16hw",
            "im2col_kn_fp16_from_fp16",
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
            "fp16_softmax"
        };
    }
}
