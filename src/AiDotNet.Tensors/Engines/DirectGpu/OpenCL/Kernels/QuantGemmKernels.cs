// Copyright (c) AiDotNet. All rights reserved.
// Weight-only fused dequant-GEMM kernels (P0: LLM-serving quantized inference hot path).
// Works on ALL .NET versions including .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for weight-only quantized matmul: <c>C[M,N] = activations[M,K] · dequant(W[K,N])</c>,
    /// where W is int8-quantized (symmetric). This is the decode hot path for quantized LLM serving —
    /// the fused kernel folds dequantization into the MAC loop so the int8 payload is read once and no
    /// full-precision weight tensor is materialized.
    /// </summary>
    /// <remarks>
    /// Contract MUST match the CPU oracle (Engines/Simd/FusedDequantMatmulKernels.Q8MatMul):
    /// weights are row-major with flat index <c>k*N + j</c>; scales are symmetric (no zero-points) and
    /// either per-tensor (scaleCount == 1) or per-group over the FLATTENED buffer
    /// (scale index = flat / groupSize). This is a correct, naive one-thread-per-output baseline;
    /// tiling / tensor-core (WMMA-equivalent) and int4/fp8 variants are follow-up optimizations.
    /// </remarks>
    internal static class QuantGemmKernels
    {
        public static string[] GetKernelNames() => new[] { "dequant_gemm_int8", "dequant_gemm_int4", "dequant_gemm_fp8_e4m3" };

        public static string GetSource()
        {
            return @"
// C[M,N] = act[M,K] . dequant(w_int8[K,N]); symmetric scales; row-major flat = k*N + j.
__kernel void dequant_gemm_int8(
    __global const float* act,     // [M*K]
    __global const char*  w,       // [K*N] int8 weights
    __global const float* scales,  // [scaleCount]
    __global float*       outbuf,  // [M*N]
    const int M, const int K, const int N,
    const int groupSize, const int scaleCount)
{
    const int i = get_global_id(0); // row in [0, M)
    const int j = get_global_id(1); // col in [0, N)
    if (i >= M || j >= N) return;

    const int actRow = i * K;
    float acc = 0.0f;

    if (scaleCount == 1) {
        // Per-tensor: single scale folded in once at the end.
        const float s = scales[0];
        for (int k = 0; k < K; ++k) {
            acc += act[actRow + k] * (float)(w[k * N + j]);
        }
        acc *= s;
    } else {
        // Per-group: scale changes as flat = k*N + j crosses a group boundary.
        for (int k = 0; k < K; ++k) {
            const int flat = k * N + j;
            const float s = scales[flat / groupSize];
            acc += act[actRow + k] * (float)(w[flat]) * s;
        }
    }

    outbuf[i * N + j] = acc;
}

// int4 variant: weights are 2 signed nibbles per byte (low nibble = even element,
// high nibble = odd element; two's-complement, matching PackedInt4 / llama.cpp Q4_0).
// C[M,N] = act[M,K] . dequant(int4 W[K,N]); symmetric scales; flat = k*N + j.
__kernel void dequant_gemm_int4(
    __global const float* act,      // [M*K]
    __global const uchar* wpacked,  // [ceil(K*N/2)] two int4 per byte
    __global const float* scales,   // [scaleCount]
    __global float*       outbuf,   // [M*N]
    const int M, const int K, const int N,
    const int groupSize, const int scaleCount)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i >= M || j >= N) return;

    const int actRow = i * K;
    float acc = 0.0f;

    if (scaleCount == 1) {
        const float s = scales[0];
        for (int k = 0; k < K; ++k) {
            const int flat = k * N + j;
            const uchar b = wpacked[flat >> 1];
            const int nib = (flat & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
            const int val = (nib & 0x07) - (nib & 0x08); // sign-extend 4-bit two's-complement
            acc += act[actRow + k] * (float)val;
        }
        acc *= s;
    } else {
        for (int k = 0; k < K; ++k) {
            const int flat = k * N + j;
            const uchar b = wpacked[flat >> 1];
            const int nib = (flat & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
            const int val = (nib & 0x07) - (nib & 0x08);
            const float s = scales[flat / groupSize];
            acc += act[actRow + k] * (float)val * s;
        }
    }

    outbuf[i * N + j] = acc;
}

// Decode one OCP FP8 E4M3 byte to float, bit-for-bit matching Float8E4M3.ToFloat()
// (1 sign, 4 exp bias-7, 3 mantissa; 0x7F/0xFF = NaN; no Inf).
inline float decode_e4m3(uchar raw) {
    if ((raw & 0x7F) == 0x7F) return NAN;
    if ((raw & 0x7F) == 0)    return (raw & 0x80) ? -0.0f : 0.0f;
    uint sign  = (uint)((raw & 0x80) >> 7);
    uint exp4  = (uint)((raw & 0x78) >> 3);
    uint m3    = (uint)(raw & 0x07);
    int  exp32 = (int)exp4 - 7 + 127;
    uint bits  = (sign << 31) | (((uint)(exp32 & 0xFF)) << 23) | (m3 << 20);
    return as_float(bits);
}

// fp8 (E4M3) weight-only variant: C[M,N] = act[M,K] . (scale * decode_e4m3(W[K,N])).
// Symmetric scales; per-tensor (scaleCount==1) or per-group over flat k*N.
__kernel void dequant_gemm_fp8_e4m3(
    __global const float* act,     // [M*K]
    __global const uchar* w,       // [K*N] fp8 e4m3 bytes
    __global const float* scales,  // [scaleCount]
    __global float*       outbuf,  // [M*N]
    const int M, const int K, const int N,
    const int groupSize, const int scaleCount)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i >= M || j >= N) return;

    const int actRow = i * K;
    float acc = 0.0f;

    if (scaleCount == 1) {
        const float s = scales[0];
        for (int k = 0; k < K; ++k)
            acc += act[actRow + k] * decode_e4m3(w[k * N + j]);
        acc *= s;
    } else {
        for (int k = 0; k < K; ++k) {
            const int flat = k * N + j;
            const float s = scales[flat / groupSize];
            acc += act[actRow + k] * decode_e4m3(w[flat]) * s;
        }
    }

    outbuf[i * N + j] = acc;
}
";
        }
    }
}
