// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

internal static class CudaFusedAdvancedKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "fused_lora_forward",
        "fused_ddim_step",
        "fused_sparse_linear"
    };

    public static string GetSource() => @"
#include <math.h>

__device__ __forceinline__ float fused_activate(float x, int activation)
{
    switch (activation)
    {
        case 1: return x < 0.0f ? 0.0f : x;
        case 2: return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        case 3: return 1.0f / (1.0f + expf(-x));
        case 4: return tanhf(x);
        case 5: return x > 0.0f ? x : 0.01f * x;
        case 6: return x / (1.0f + expf(-x));
        default: return x;
    }
}

// Issue #301 fused LoRA forward: output = base_output + scaling · (input · A · B)
//
// Two-stage decomposition (the only way to hit the issue's ≥5× target vs the
// decomposed path):
//   Stage 1 (per row of input): proj[r] = Σ_i input[b,i] · A[i,r]   — O(in × rank)
//   Stage 2 (per (row, col) of output): delta = Σ_r proj[r] · B[r,j]
//                                      output[b,j] = base[b,j] + scaling · delta — O(rank)
// Total per batch row: O(rank · (in + out))
// Total work: O(batch · rank · (in + out))
//
// The decomposed-into-fused-loop variant (compute proj inside the (b,j) loop)
// pays O(batch · in · rank · out) — out× more — and is strictly slower than
// the unfused path it claims to replace.
//
// Launch contract:
//   grid.x = batch_size,  block.x = min(output_features, max_threads_per_block)
//   shared mem = rank * sizeof(float) bytes (dynamic)
extern ""C"" __global__ void fused_lora_forward(
    const float* __restrict__ input,
    const float* __restrict__ base_output,
    const float* __restrict__ lora_a,
    const float* __restrict__ lora_b,
    float* __restrict__ output,
    int batch_size,
    int input_features,
    int rank,
    int output_features,
    float scaling)
{
    extern __shared__ float proj[];   // [rank]

    int b = blockIdx.x;
    if (b >= batch_size) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* in_row = input + b * input_features;

    // Stage 1: cooperatively compute proj[r] for r in [0, rank).
    // Strided over threads in the block. Each thread runs a full input_features
    // dot product with column r of A.
    for (int r = tid; r < rank; r += block_size)
    {
        float acc = 0.0f;
        for (int i = 0; i < input_features; i++)
            acc += in_row[i] * lora_a[i * rank + r];
        proj[r] = acc;
    }

    __syncthreads();

    // Stage 2: each thread emits one or more output columns, reusing proj[].
    int row_base = b * output_features;
    for (int j = tid; j < output_features; j += block_size)
    {
        float delta = 0.0f;
        for (int r = 0; r < rank; r++)
            delta += proj[r] * lora_b[r * output_features + j];
        output[row_base + j] = base_output[row_base + j] + scaling * delta;
    }
}

extern ""C"" __global__ void fused_ddim_step(
    const float* __restrict__ x_t,
    const float* __restrict__ epsilon_theta,
    float* __restrict__ output,
    int size,
    float alpha_bar_t,
    float alpha_bar_t_minus_1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float sqrt_at = sqrtf(alpha_bar_t);
    float sqrt_atm1 = sqrtf(alpha_bar_t_minus_1);
    float c_xt = sqrt_atm1 / sqrt_at;
    float c_eps = sqrtf(1.0f - alpha_bar_t_minus_1) - sqrtf(1.0f - alpha_bar_t) * sqrt_atm1 / sqrt_at;
    output[idx] = c_xt * x_t[idx] + c_eps * epsilon_theta[idx];
}

extern ""C"" __global__ void fused_sparse_linear(
    const float* __restrict__ input,
    const int* __restrict__ packed_csr,
    const float* __restrict__ sparse_values,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_features,
    int output_features,
    int nnz,
    int has_bias,
    int activation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_features;
    if (idx >= total) return;

    int b = idx / output_features;
    int j = idx - b * output_features;
    const int* row_offsets = packed_csr;
    const int* col_indices = packed_csr + output_features + 1;
    int start = row_offsets[j];
    int end = row_offsets[j + 1];
    if (start < 0 || end < start || end > nnz)
    {
        output[idx] = 0.0f;
        return;
    }

    float sum = has_bias != 0 ? bias[j] : 0.0f;
    for (int p = start; p < end; p++)
    {
        int col = col_indices[p];
        if ((unsigned int)col < (unsigned int)input_features)
            sum += input[b * input_features + col] * sparse_values[p];
    }
    output[idx] = fused_activate(sum, activation);
}
";
}
