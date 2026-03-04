// Copyright (c) AiDotNet. All rights reserved.
// Tensor Core WMMA kernels for Volta+ GPUs (sm_70+).
// Uses nvcuda::wmma for 16x16x16 FP16 matrix multiply with FP32 accumulation.
// FP32 inputs are converted to FP16 in shared memory before tensor core execution.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA Tensor Core GEMM kernels using WMMA (Warp Matrix Multiply-Accumulate).
    /// Requires compute capability >= 7.0 (Volta, Turing, Ampere, Hopper).
    /// Falls back gracefully to Phase 3 tiled GEMM on older hardware.
    /// </summary>
    internal static class CudaWmmaKernels
    {
        /// <summary>
        /// Gets WMMA kernel sources. Requires sm_70+ for compilation.
        /// Uses 64x64 CTA tile with 4 warps, each computing 16x16 via tensor cores.
        /// FP32 → FP16 conversion happens in shared memory before WMMA operations.
        /// </summary>
        public static string GetSource()
        {
            return @"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// ===========================================================================
// WMMA PARAMETERS
// CTA tile: 64x64, split into 4x4 = 16 warp tiles of 16x16 each
// Thread block: 256 threads = 8 warps (only 16 warps needed for 4x4)
// Actually: 4 warps in 2x2 arrangement, each warp handles one 16x16 tile
// Block: 128 threads = 4 warps
// ===========================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WMMA_TILE_M 32
#define WMMA_TILE_N 32
#define WMMA_BLOCK_THREADS 128

// ===========================================================================
// SHARED MEMORY LAYOUT
// As: [WMMA_TILE_M][WMMA_K] in half = 32 * 16 * 2 = 1024 bytes
// Bs: [WMMA_K][WMMA_TILE_N] in half = 16 * 32 * 2 = 1024 bytes
// Cs: [WMMA_TILE_M][WMMA_TILE_N] in float = 32 * 32 * 4 = 4096 bytes
// Total: ~6 KB shared memory per block (very low, allows high occupancy)
// ===========================================================================

// Macro for the common WMMA GEMM body (load, compute, store)
// EPILOGUE_CODE is applied to each output element 'val' at (gRow, gCol)
#define WMMA_GEMM_BODY(EPILOGUE_CODE) \
    int warpId = threadIdx.x / 32; \
    int warpRow = warpId / 2; \
    int warpCol = warpId % 2; \
    int tileRowStart = blockIdx.y * WMMA_TILE_M; \
    int tileColStart = blockIdx.x * WMMA_TILE_N; \
    \
    __shared__ half As[WMMA_TILE_M][WMMA_K]; \
    __shared__ half Bs[WMMA_K][WMMA_TILE_N]; \
    __shared__ float Cs[WMMA_TILE_M][WMMA_TILE_N]; \
    \
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc; \
    wmma::fill_fragment(acc, 0.0f); \
    \
    for (int kTile = 0; kTile < K; kTile += WMMA_K) { \
        /* Cooperative load A[WMMA_TILE_M x WMMA_K] -> FP16 shared memory */ \
        for (int i = threadIdx.x; i < WMMA_TILE_M * WMMA_K; i += WMMA_BLOCK_THREADS) { \
            int r = i / WMMA_K; \
            int c = i % WMMA_K; \
            int gRow = tileRowStart + r; \
            int gCol = kTile + c; \
            As[r][c] = (gRow < M && gCol < K) ? __float2half(A[gRow * K + gCol]) : __float2half(0.0f); \
        } \
        /* Cooperative load B[WMMA_K x WMMA_TILE_N] -> FP16 shared memory */ \
        for (int i = threadIdx.x; i < WMMA_K * WMMA_TILE_N; i += WMMA_BLOCK_THREADS) { \
            int r = i / WMMA_TILE_N; \
            int c = i % WMMA_TILE_N; \
            int gRow = kTile + r; \
            int gCol = tileColStart + c; \
            Bs[r][c] = (gRow < K && gCol < N) ? __float2half(B[gRow * N + gCol]) : __float2half(0.0f); \
        } \
        __syncthreads(); \
        \
        /* WMMA matrix multiply: each warp computes its 16x16 fragment */ \
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag; \
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag; \
        wmma::load_matrix_sync(a_frag, &As[warpRow * WMMA_M][0], WMMA_K); \
        wmma::load_matrix_sync(b_frag, &Bs[0][warpCol * WMMA_N], WMMA_TILE_N); \
        wmma::mma_sync(acc, a_frag, b_frag, acc); \
        \
        __syncthreads(); \
    } \
    \
    /* Store WMMA result to shared memory */ \
    wmma::store_matrix_sync(&Cs[warpRow * WMMA_M][warpCol * WMMA_N], acc, WMMA_TILE_N, wmma::mem_row_major); \
    __syncthreads(); \
    \
    /* Apply epilogue and write to global memory */ \
    for (int i = threadIdx.x; i < WMMA_TILE_M * WMMA_TILE_N; i += WMMA_BLOCK_THREADS) { \
        int r = i / WMMA_TILE_N; \
        int c = i % WMMA_TILE_N; \
        int gRow = tileRowStart + r; \
        int gCol = tileColStart + c; \
        if (gRow < M && gCol < N) { \
            float val = Cs[r][c] + bias[gCol]; \
            EPILOGUE_CODE \
            C[gRow * N + gCol] = val; \
        } \
    }

// ===========================================================================
// FUSED WMMA GEMM + BIAS + ACTIVATION KERNELS
// Each kernel uses tensor cores for the GEMM and applies activation inline.
// ===========================================================================

extern ""C"" __global__ void gemm_bias_relu_wmma(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    WMMA_GEMM_BODY(val = fmaxf(0.0f, val);)
}

extern ""C"" __global__ void gemm_bias_gelu_wmma(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    WMMA_GEMM_BODY(
        float x3 = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * x3);
        val = 0.5f * val * (1.0f + tanhf(inner));
    )
}

extern ""C"" __global__ void gemm_bias_sigmoid_wmma(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    WMMA_GEMM_BODY(val = 1.0f / (1.0f + expf(-val));)
}

extern ""C"" __global__ void gemm_bias_tanh_wmma(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    WMMA_GEMM_BODY(val = tanhf(val);)
}

extern ""C"" __global__ void gemm_bias_wmma(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    WMMA_GEMM_BODY(/* no activation */;)
}

extern ""C"" __global__ void gemm_bias_swish_wmma(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    WMMA_GEMM_BODY(
        float sigmoid = 1.0f / (1.0f + expf(-val));
        val = val * sigmoid;
    )
}

extern ""C"" __global__ void gemm_bias_leaky_relu_wmma(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K, float alpha)
{
    // Cannot use the macro directly due to extra parameter; inline the body
    int warpId = threadIdx.x / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;
    int tileRowStart = blockIdx.y * WMMA_TILE_M;
    int tileColStart = blockIdx.x * WMMA_TILE_N;

    __shared__ half As[WMMA_TILE_M][WMMA_K];
    __shared__ half Bs[WMMA_K][WMMA_TILE_N];
    __shared__ float Cs[WMMA_TILE_M][WMMA_TILE_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int kTile = 0; kTile < K; kTile += WMMA_K) {
        for (int i = threadIdx.x; i < WMMA_TILE_M * WMMA_K; i += WMMA_BLOCK_THREADS) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gRow = tileRowStart + r;
            int gCol = kTile + c;
            As[r][c] = (gRow < M && gCol < K) ? __float2half(A[gRow * K + gCol]) : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < WMMA_K * WMMA_TILE_N; i += WMMA_BLOCK_THREADS) {
            int r = i / WMMA_TILE_N;
            int c = i % WMMA_TILE_N;
            int gRow = kTile + r;
            int gCol = tileColStart + c;
            Bs[r][c] = (gRow < K && gCol < N) ? __float2half(B[gRow * N + gCol]) : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, &As[warpRow * WMMA_M][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &Bs[0][warpCol * WMMA_N], WMMA_TILE_N);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    wmma::store_matrix_sync(&Cs[warpRow * WMMA_M][warpCol * WMMA_N], acc, WMMA_TILE_N, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < WMMA_TILE_M * WMMA_TILE_N; i += WMMA_BLOCK_THREADS) {
        int r = i / WMMA_TILE_N;
        int c = i % WMMA_TILE_N;
        int gRow = tileRowStart + r;
        int gCol = tileColStart + c;
        if (gRow < M && gCol < N) {
            float val = Cs[r][c] + bias[gCol];
            val = val >= 0.0f ? val : alpha * val;
            C[gRow * N + gCol] = val;
        }
    }
}
";
        }

        /// <summary>
        /// Gets WMMA kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "gemm_bias_relu_wmma",
                "gemm_bias_gelu_wmma",
                "gemm_bias_sigmoid_wmma",
                "gemm_bias_tanh_wmma",
                "gemm_bias_wmma",
                "gemm_bias_swish_wmma",
                "gemm_bias_leaky_relu_wmma"
            };
        }
    }
}
