// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class Issue301FusedKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "issue301_fused_lora_forward",
        "issue301_fused_ddim_step",
        "issue301_fused_sparse_linear"
    };

    public static string GetSource() => @"
inline float issue301_activate(float x, int activation)
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

__kernel void issue301_fused_lora_forward(
    __global const float* input,
    __global const float* base_output,
    __global const float* lora_a,
    __global const float* lora_b,
    __global float* output,
    int batch_size,
    int input_features,
    int rank,
    int output_features,
    float scaling)
{
    int idx = get_global_id(0);
    int total = batch_size * output_features;
    if (idx >= total) return;

    int b = idx / output_features;
    int j = idx - b * output_features;
    float delta = 0.0f;
    for (int r = 0; r < rank; r++)
    {
        float acc = 0.0f;
        for (int i = 0; i < input_features; i++)
            acc += input[b * input_features + i] * lora_a[i * rank + r];
        delta += acc * lora_b[r * output_features + j];
    }
    output[idx] = base_output[idx] + scaling * delta;
}

__kernel void issue301_fused_ddim_step(
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

__kernel void issue301_fused_sparse_linear(
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
    output[idx] = issue301_activate(sum, activation);
}
";
}
