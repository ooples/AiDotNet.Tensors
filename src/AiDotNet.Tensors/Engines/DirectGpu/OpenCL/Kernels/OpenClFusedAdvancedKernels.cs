// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class OpenClFusedAdvancedKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "fused_lora_forward",
        "fused_ddim_step",
        "fused_sparse_linear"
    };

    public static string GetSource() => @"
inline float fused_activate(float x, int activation)
{
    switch (activation)
    {
        case 1: return x < 0.0f ? 0.0f : x;
        case 2: return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        case 3: return 1.0f / (1.0f + exp(-x));
        case 4: return tanh(x);
        case 5: return x > 0.0f ? x : 0.01f * x;
        case 6: return x / (1.0f + exp(-x));
        default: return x;
    }
}

// See CudaFusedAdvancedKernels.cs for the full design rationale.
// Two-stage shared-memory variant — O(batch · rank · (in + out)) instead of
// the broken O(batch · in · rank · out) inner-recompute variant.
//
// Launch contract:
//   global_size = batch_size * local_size
//   local_size  = min(output_features, max work-group size)
//   __local proj[rank] passed via clSetKernelArg(... NULL ...) for dynamic
//   shared mem (in OpenCL: __local float* proj as a kernel arg, then
//   clSetKernelArg(kernel, idx, rank * sizeof(float), NULL)).
__kernel void fused_lora_forward(
    __global const float* input,
    __global const float* base_output,
    __global const float* lora_a,
    __global const float* lora_b,
    __global float* output,
    int batch_size,
    int input_features,
    int rank,
    int output_features,
    float scaling,
    __local float* proj)
{
    int b = get_group_id(0);
    if (b >= batch_size) return;

    int tid = get_local_id(0);
    int block_size = get_local_size(0);

    __global const float* in_row = input + b * input_features;

    // Stage 1: cooperatively compute proj[r] = sum_i in_row[i] * lora_a[i, r].
    for (int r = tid; r < rank; r += block_size)
    {
        float acc = 0.0f;
        for (int i = 0; i < input_features; i++)
            acc += in_row[i] * lora_a[i * rank + r];
        proj[r] = acc;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Stage 2: each thread emits one or more output columns, reusing proj.
    int row_base = b * output_features;
    for (int j = tid; j < output_features; j += block_size)
    {
        float delta = 0.0f;
        for (int r = 0; r < rank; r++)
            delta += proj[r] * lora_b[r * output_features + j];
        output[row_base + j] = base_output[row_base + j] + scaling * delta;
    }
}

__kernel void fused_ddim_step(
    __global const float* x_t,
    __global const float* epsilon_theta,
    __global float* output,
    int size,
    float alpha_bar_t,
    float alpha_bar_t_minus_1)
{
    int idx = get_global_id(0);
    if (idx >= size) return;

    float sqrt_at = sqrt(alpha_bar_t);
    float sqrt_atm1 = sqrt(alpha_bar_t_minus_1);
    float c_xt = sqrt_atm1 / sqrt_at;
    float c_eps = sqrt(1.0f - alpha_bar_t_minus_1) - sqrt(1.0f - alpha_bar_t) * sqrt_atm1 / sqrt_at;
    output[idx] = c_xt * x_t[idx] + c_eps * epsilon_theta[idx];
}

__kernel void fused_sparse_linear(
    __global const float* input,
    __global const int* packed_csr,
    __global const float* sparse_values,
    __global const float* bias,
    __global float* output,
    int batch_size,
    int input_features,
    int output_features,
    int nnz,
    int has_bias,
    int activation)
{
    int idx = get_global_id(0);
    int total = batch_size * output_features;
    if (idx >= total) return;

    int b = idx / output_features;
    int j = idx - b * output_features;
    __global const int* row_offsets = packed_csr;
    __global const int* col_indices = packed_csr + output_features + 1;
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
        if ((uint)col < (uint)input_features)
            sum += input[b * input_features + col] * sparse_values[p];
    }
    output[idx] = fused_activate(sum, activation);
}
";
}
