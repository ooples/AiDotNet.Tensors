// Copyright (c) AiDotNet. All rights reserved.
// OpenCL fused linear (MatMul + Bias + Activation) kernels.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class FusedLinearKernels
{
    public static string GetSource()
    {
        return @"
// Activation helpers
inline float act_relu(float x) { return max(x, 0.0f); }
inline float act_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float act_tanh_fn(float x) { return tanh(x); }
inline float act_gelu(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}
inline float act_swish(float x) { return x * act_sigmoid(x); }

// Backward helpers
inline float bw_relu(float grad, float preact) { return preact > 0.0f ? grad : 0.0f; }
inline float bw_sigmoid(float grad, float output) { return grad * output * (1.0f - output); }
inline float bw_tanh_fn(float grad, float output) { return grad * (1.0f - output * output); }
inline float bw_gelu(float grad, float preact) {
    float t = tanh(0.7978845608f * (preact + 0.044715f * preact * preact * preact));
    float dt = 0.7978845608f * (1.0f + 0.134145f * preact * preact) * (1.0f - t * t);
    return grad * (0.5f * (1.0f + t) + 0.5f * preact * dt);
}
inline float bw_swish(float grad, float preact) {
    float sig = act_sigmoid(preact);
    return grad * (sig + preact * sig * (1.0f - sig));
}

// Forward kernels: output[b,j] = activation(sum_k(input[b,k] * weight[k,j]) + bias[j])
__kernel void fused_linear_relu(
    __global const float* input, __global const float* weight, __global const float* bias,
    __global float* output, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * outFeatures) return;
    int b = idx / outFeatures, j = idx % outFeatures;
    float sum = bias[j];
    for (int k = 0; k < inFeatures; k++)
        sum += input[b * inFeatures + k] * weight[k * outFeatures + j];
    output[idx] = act_relu(sum);
}

__kernel void fused_linear_sigmoid(
    __global const float* input, __global const float* weight, __global const float* bias,
    __global float* output, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * outFeatures) return;
    int b = idx / outFeatures, j = idx % outFeatures;
    float sum = bias[j];
    for (int k = 0; k < inFeatures; k++)
        sum += input[b * inFeatures + k] * weight[k * outFeatures + j];
    output[idx] = act_sigmoid(sum);
}

__kernel void fused_linear_tanh(
    __global const float* input, __global const float* weight, __global const float* bias,
    __global float* output, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * outFeatures) return;
    int b = idx / outFeatures, j = idx % outFeatures;
    float sum = bias[j];
    for (int k = 0; k < inFeatures; k++)
        sum += input[b * inFeatures + k] * weight[k * outFeatures + j];
    output[idx] = act_tanh_fn(sum);
}

__kernel void fused_linear_gelu(
    __global const float* input, __global const float* weight, __global const float* bias,
    __global float* output, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * outFeatures) return;
    int b = idx / outFeatures, j = idx % outFeatures;
    float sum = bias[j];
    for (int k = 0; k < inFeatures; k++)
        sum += input[b * inFeatures + k] * weight[k * outFeatures + j];
    output[idx] = act_gelu(sum);
}

__kernel void fused_linear_swish(
    __global const float* input, __global const float* weight, __global const float* bias,
    __global float* output, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * outFeatures) return;
    int b = idx / outFeatures, j = idx % outFeatures;
    float sum = bias[j];
    for (int k = 0; k < inFeatures; k++)
        sum += input[b * inFeatures + k] * weight[k * outFeatures + j];
    output[idx] = act_swish(sum);
}

// Backward: grad_input kernel
__kernel void fused_linear_relu_backward_grad_input(
    __global const float* gradOutput, __global const float* weight, __global const float* saved,
    __global float* gradInput, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * inFeatures) return;
    int b = idx / inFeatures, i = idx % inFeatures;
    float sum = 0.0f;
    for (int j = 0; j < outFeatures; j++) {
        float masked = bw_relu(gradOutput[b * outFeatures + j], saved[b * outFeatures + j]);
        sum += masked * weight[i * outFeatures + j];
    }
    gradInput[idx] = sum;
}

__kernel void fused_linear_sigmoid_backward_grad_input(
    __global const float* gradOutput, __global const float* weight, __global const float* saved,
    __global float* gradInput, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * inFeatures) return;
    int b = idx / inFeatures, i = idx % inFeatures;
    float sum = 0.0f;
    for (int j = 0; j < outFeatures; j++) {
        float masked = bw_sigmoid(gradOutput[b * outFeatures + j], saved[b * outFeatures + j]);
        sum += masked * weight[i * outFeatures + j];
    }
    gradInput[idx] = sum;
}

__kernel void fused_linear_tanh_backward_grad_input(
    __global const float* gradOutput, __global const float* weight, __global const float* saved,
    __global float* gradInput, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * inFeatures) return;
    int b = idx / inFeatures, i = idx % inFeatures;
    float sum = 0.0f;
    for (int j = 0; j < outFeatures; j++) {
        float masked = bw_tanh_fn(gradOutput[b * outFeatures + j], saved[b * outFeatures + j]);
        sum += masked * weight[i * outFeatures + j];
    }
    gradInput[idx] = sum;
}

__kernel void fused_linear_gelu_backward_grad_input(
    __global const float* gradOutput, __global const float* weight, __global const float* saved,
    __global float* gradInput, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * inFeatures) return;
    int b = idx / inFeatures, i = idx % inFeatures;
    float sum = 0.0f;
    for (int j = 0; j < outFeatures; j++) {
        float masked = bw_gelu(gradOutput[b * outFeatures + j], saved[b * outFeatures + j]);
        sum += masked * weight[i * outFeatures + j];
    }
    gradInput[idx] = sum;
}

__kernel void fused_linear_swish_backward_grad_input(
    __global const float* gradOutput, __global const float* weight, __global const float* saved,
    __global float* gradInput, int batchSize, int inFeatures, int outFeatures)
{
    int idx = get_global_id(0);
    if (idx >= batchSize * inFeatures) return;
    int b = idx / inFeatures, i = idx % inFeatures;
    float sum = 0.0f;
    for (int j = 0; j < outFeatures; j++) {
        float masked = bw_swish(gradOutput[b * outFeatures + j], saved[b * outFeatures + j]);
        sum += masked * weight[i * outFeatures + j];
    }
    gradInput[idx] = sum;
}

// Weight gradient
__kernel void fused_linear_weight_grad(
    __global const float* gradOutput, __global const float* input, __global const float* saved,
    __global float* gradWeight, int batchSize, int inFeatures, int outFeatures, int activationType)
{
    int idx = get_global_id(0);
    if (idx >= inFeatures * outFeatures) return;
    int i = idx / outFeatures, j = idx % outFeatures;
    float sum = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        float go = gradOutput[b * outFeatures + j];
        float s = saved[b * outFeatures + j];
        float masked;
        if (activationType == 0) masked = bw_relu(go, s);
        else if (activationType == 1) masked = bw_sigmoid(go, s);
        else if (activationType == 2) masked = bw_tanh_fn(go, s);
        else if (activationType == 3) masked = bw_gelu(go, s);
        else if (activationType == 4) masked = bw_swish(go, s);
        else masked = go;
        sum += input[b * inFeatures + i] * masked;
    }
    gradWeight[idx] = sum;
}

// Bias gradient
__kernel void fused_linear_bias_grad(
    __global const float* gradOutput, __global const float* saved,
    __global float* gradBias, int batchSize, int outFeatures, int activationType)
{
    int j = get_global_id(0);
    if (j >= outFeatures) return;
    float sum = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        float go = gradOutput[b * outFeatures + j];
        float s = saved[b * outFeatures + j];
        float masked;
        if (activationType == 0) masked = bw_relu(go, s);
        else if (activationType == 1) masked = bw_sigmoid(go, s);
        else if (activationType == 2) masked = bw_tanh_fn(go, s);
        else if (activationType == 3) masked = bw_gelu(go, s);
        else if (activationType == 4) masked = bw_swish(go, s);
        else masked = go;
        sum += masked;
    }
    gradBias[j] = sum;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "fused_linear_relu", "fused_linear_sigmoid", "fused_linear_tanh",
            "fused_linear_gelu", "fused_linear_swish",
            "fused_linear_relu_backward_grad_input", "fused_linear_sigmoid_backward_grad_input",
            "fused_linear_tanh_backward_grad_input", "fused_linear_gelu_backward_grad_input",
            "fused_linear_swish_backward_grad_input",
            "fused_linear_weight_grad", "fused_linear_bias_grad",
        };
    }
}
