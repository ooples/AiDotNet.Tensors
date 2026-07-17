// Copyright (c) AiDotNet. All rights reserved.
// Weight-only fused dequant-GEMM CUDA kernels (P0: quantized LLM-serving decode hot path).

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// nvRTC CUDA kernels for weight-only quantized matmul: C[M,N] = act[M,K] · dequant(W[K,N]).
    /// Contract matches the CPU oracle FusedDequantMatmulKernels (symmetric scales, per-tensor when
    /// scaleCount==1 else per-group over the flattened K*N buffer; int4 = 2 signed nibbles/byte,
    /// low nibble even; fp8 = OCP E4M3 decode matching Float8E4M3.ToFloat). Naive one-thread-per-output
    /// baseline (correctness-first); tensor-core (WMMA) fusion is the follow-up perf work.
    /// </summary>
    internal static class CudaQuantGemmKernels
    {
        public static string[] GetKernelNames() => new[] { "dequant_gemm_int8", "dequant_gemm_int4", "dequant_gemm_fp8_e4m3" };

        public static string GetSource() => @"
extern ""C"" __global__ void dequant_gemm_int8(
    const float* act, const signed char* w, const float* scales, float* outbuf,
    int M, int K, int N, int groupSize, int scaleCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int i = idx / N, j = idx % N;
    float acc = 0.0f;
    if (scaleCount == 1) {
        float s = scales[0];
        for (int k = 0; k < K; ++k) acc += act[i*K+k] * (float)w[k*N+j];
        acc *= s;
    } else {
        for (int k = 0; k < K; ++k) { int flat = k*N+j; acc += act[i*K+k] * (float)w[flat] * scales[flat/groupSize]; }
    }
    outbuf[idx] = acc;
}

extern ""C"" __global__ void dequant_gemm_int4(
    const float* act, const unsigned char* w, const float* scales, float* outbuf,
    int M, int K, int N, int groupSize, int scaleCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int i = idx / N, j = idx % N;
    float acc = 0.0f;
    if (scaleCount == 1) {
        float s = scales[0];
        for (int k = 0; k < K; ++k) { int flat=k*N+j; unsigned char b=w[flat>>1]; int nib=(flat&1)?((b>>4)&0xF):(b&0xF); int val=(nib&0x7)-(nib&0x8); acc += act[i*K+k]*(float)val; }
        acc *= s;
    } else {
        for (int k = 0; k < K; ++k) { int flat=k*N+j; unsigned char b=w[flat>>1]; int nib=(flat&1)?((b>>4)&0xF):(b&0xF); int val=(nib&0x7)-(nib&0x8); acc += act[i*K+k]*(float)val*scales[flat/groupSize]; }
    }
    outbuf[idx] = acc;
}

__device__ __forceinline__ float decode_e4m3(unsigned char raw){
    unsigned int r = raw;
    if ((r & 0x7F) == 0x7F) return __uint_as_float(0x7FC00000u);
    if ((r & 0x7F) == 0) return 0.0f;
    unsigned int sign=(r&0x80)>>7, exp4=(r&0x78)>>3, m3=(r&0x07);
    int exp32=(int)exp4 - 7 + 127;
    unsigned int bits=(sign<<31)|((unsigned int)(exp32 & 0xFF)<<23)|(m3<<20);
    return __uint_as_float(bits);
}

extern ""C"" __global__ void dequant_gemm_fp8_e4m3(
    const float* act, const unsigned char* w, const float* scales, float* outbuf,
    int M, int K, int N, int groupSize, int scaleCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int i = idx / N, j = idx % N;
    float acc = 0.0f;
    if (scaleCount == 1) {
        float s = scales[0];
        for (int k = 0; k < K; ++k) acc += act[i*K+k] * decode_e4m3(w[k*N+j]);
        acc *= s;
    } else {
        for (int k = 0; k < K; ++k) { int flat=k*N+j; acc += act[i*K+k] * decode_e4m3(w[flat]) * scales[flat/groupSize]; }
    }
    outbuf[idx] = acc;
}
";
    }
}
