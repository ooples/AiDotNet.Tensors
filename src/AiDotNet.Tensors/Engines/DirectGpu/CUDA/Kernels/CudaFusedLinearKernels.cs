// Copyright (c) AiDotNet. All rights reserved.
// CUDA fused linear (MatMul + Bias + Activation) kernels for forward and backward passes.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for fused linear operations: output = activation(input @ weight + bias).
/// Single kernel invocation instead of separate MatMul + Add + Activation calls.
/// </summary>
internal static class CudaFusedLinearKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

// ===========================================================================
// Activation helpers (device functions)
// ===========================================================================

__device__ __forceinline__ float activate_relu(float x) { return x > 0.0f ? x : 0.0f; }
__device__ __forceinline__ float activate_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float activate_tanh_act(float x) { return tanhf(x); }
__device__ __forceinline__ float activate_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float activate_swish(float x) { return x * activate_sigmoid(x); }

__device__ __forceinline__ float backward_relu(float grad, float preact) { return preact > 0.0f ? grad : 0.0f; }
__device__ __forceinline__ float backward_sigmoid(float grad, float output) {
    return grad * output * (1.0f - output);
}
__device__ __forceinline__ float backward_tanh_act(float grad, float output) {
    return grad * (1.0f - output * output);
}
__device__ __forceinline__ float backward_gelu(float grad, float preact) {
    float t = tanhf(0.7978845608f * (preact + 0.044715f * preact * preact * preact));
    float dt = 0.7978845608f * (1.0f + 0.134145f * preact * preact) * (1.0f - t * t);
    return grad * (0.5f * (1.0f + t) + 0.5f * preact * dt);
}
__device__ __forceinline__ float backward_swish(float grad, float preact) {
    float sig = activate_sigmoid(preact);
    return grad * (sig + preact * sig * (1.0f - sig));
}

// ===========================================================================
// Forward: output[b,j] = activation(sum_k(input[b,k] * weight[k,j]) + bias[j])
// ===========================================================================

#define FUSED_LINEAR_FORWARD(name, activate_fn) \
extern ""C"" __global__ void name( \
    const float* __restrict__ input, \
    const float* __restrict__ weight, \
    const float* __restrict__ bias, \
    float* __restrict__ output, \
    int batchSize, int inFeatures, int outFeatures) \
{ \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    int total = batchSize * outFeatures; \
    if (idx >= total) return; \
    int b = idx / outFeatures; \
    int j = idx % outFeatures; \
    float sum = __ldg(&bias[j]); \
    for (int k = 0; k < inFeatures; k++) { \
        sum += __ldg(&input[b * inFeatures + k]) * __ldg(&weight[k * outFeatures + j]); \
    } \
    output[idx] = activate_fn(sum); \
}

FUSED_LINEAR_FORWARD(fused_linear_relu, activate_relu)
FUSED_LINEAR_FORWARD(fused_linear_sigmoid, activate_sigmoid)
FUSED_LINEAR_FORWARD(fused_linear_tanh, activate_tanh_act)
FUSED_LINEAR_FORWARD(fused_linear_gelu, activate_gelu)
FUSED_LINEAR_FORWARD(fused_linear_swish, activate_swish)

// ===========================================================================
// Backward: computes gradInput, gradWeight, gradBias from gradOutput
// activation_backward depends on the activation type
// ===========================================================================

#define FUSED_LINEAR_BACKWARD_GRADIN(name, backward_fn, uses_preact) \
extern ""C"" __global__ void name##_grad_input( \
    const float* __restrict__ gradOutput, \
    const float* __restrict__ weight, \
    const float* __restrict__ saved, \
    float* __restrict__ gradInput, \
    int batchSize, int inFeatures, int outFeatures) \
{ \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    int total = batchSize * inFeatures; \
    if (idx >= total) return; \
    int b = idx / inFeatures; \
    int i = idx % inFeatures; \
    float sum = 0.0f; \
    for (int j = 0; j < outFeatures; j++) { \
        float go = __ldg(&gradOutput[b * outFeatures + j]); \
        float s = __ldg(&saved[b * outFeatures + j]); \
        float masked = backward_fn(go, s); \
        sum += masked * __ldg(&weight[i * outFeatures + j]); \
    } \
    gradInput[idx] = sum; \
}

FUSED_LINEAR_BACKWARD_GRADIN(fused_linear_relu_backward, backward_relu, 1)
FUSED_LINEAR_BACKWARD_GRADIN(fused_linear_sigmoid_backward, backward_sigmoid, 0)
FUSED_LINEAR_BACKWARD_GRADIN(fused_linear_tanh_backward, backward_tanh_act, 0)
FUSED_LINEAR_BACKWARD_GRADIN(fused_linear_gelu_backward, backward_gelu, 1)
FUSED_LINEAR_BACKWARD_GRADIN(fused_linear_swish_backward, backward_swish, 1)

// Weight gradient: gradWeight[i,j] = sum_b(input[b,i] * masked_grad[b,j])
// This is equivalent to input^T @ masked_gradient
extern ""C"" __global__ void fused_linear_weight_grad(
    const float* __restrict__ gradOutput,
    const float* __restrict__ input,
    const float* __restrict__ saved,
    float* __restrict__ gradWeight,
    int batchSize, int inFeatures, int outFeatures,
    int activationType)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = inFeatures * outFeatures;
    if (idx >= total) return;
    int i = idx / outFeatures;
    int j = idx % outFeatures;
    float sum = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        float go = __ldg(&gradOutput[b * outFeatures + j]);
        float s = __ldg(&saved[b * outFeatures + j]);
        float masked;
        switch (activationType) {
            case 0: masked = backward_relu(go, s); break;
            case 1: masked = backward_sigmoid(go, s); break;
            case 2: masked = backward_tanh_act(go, s); break;
            case 3: masked = backward_gelu(go, s); break;
            case 4: masked = backward_swish(go, s); break;
            default: masked = go; break;
        }
        sum += __ldg(&input[b * inFeatures + i]) * masked;
    }
    gradWeight[idx] = sum;
}

// Bias gradient: sum of activation-masked gradOutput along batch dimension
extern ""C"" __global__ void fused_linear_bias_grad(
    const float* __restrict__ gradOutput,
    const float* __restrict__ saved,
    float* __restrict__ gradBias,
    int batchSize, int outFeatures,
    int activationType)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= outFeatures) return;
    float sum = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        float go = __ldg(&gradOutput[b * outFeatures + j]);
        float s = __ldg(&saved[b * outFeatures + j]);
        float masked;
        switch (activationType) {
            case 0: masked = backward_relu(go, s); break;
            case 1: masked = backward_sigmoid(go, s); break;
            case 2: masked = backward_tanh_act(go, s); break;
            case 3: masked = backward_gelu(go, s); break;
            case 4: masked = backward_swish(go, s); break;
            default: masked = go; break;
        }
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
            "fused_linear_relu",
            "fused_linear_sigmoid",
            "fused_linear_tanh",
            "fused_linear_gelu",
            "fused_linear_swish",
            "fused_linear_relu_backward_grad_input",
            "fused_linear_sigmoid_backward_grad_input",
            "fused_linear_tanh_backward_grad_input",
            "fused_linear_gelu_backward_grad_input",
            "fused_linear_swish_backward_grad_input",
            "fused_linear_weight_grad",
            "fused_linear_bias_grad",
        };
    }
}
