// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    internal static class MetalIssue301FusedKernels
    {
        public static string[] GetKernelNames() => new[]
        {
            "issue301_fused_lora_forward",
            "issue301_fused_ddim_step",
            "issue301_fused_sparse_linear",
        };

        public const string Source = @"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

static inline float issue301_apply_activation(float x, uint activation)
{
    if (activation == 1u) return max(x, 0.0f);
    if (activation == 2u)
    {
        float k = sqrt(2.0f / 3.14159265358979323846f);
        return 0.5f * x * (1.0f + tanh(k * (x + 0.044715f * x * x * x)));
    }
    if (activation == 3u) return 1.0f / (1.0f + exp(-x));
    if (activation == 4u) return tanh(x);
    if (activation == 5u) return x > 0.0f ? x : 0.01f * x;
    if (activation == 6u)
    {
        float sigmoid = 1.0f / (1.0f + exp(-x));
        return x * sigmoid;
    }
    return x;
}

kernel void issue301_fused_lora_forward(
    device const float* input [[buffer(0)]],
    device const float* baseOutput [[buffer(1)]],
    device const float* loraA [[buffer(2)]],
    device const float* loraB [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& batchSize [[buffer(5)]],
    constant uint& inputFeatures [[buffer(6)]],
    constant uint& rank [[buffer(7)]],
    constant uint& outputFeatures [[buffer(8)]],
    constant float& scaling [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batchSize * outputFeatures;
    if (gid >= total) return;

    uint b = gid / outputFeatures;
    uint o = gid - b * outputFeatures;
    float delta = 0.0f;

    for (uint r = 0; r < rank; ++r)
    {
        float hidden = 0.0f;
        for (uint i = 0; i < inputFeatures; ++i)
        {
            hidden += input[b * inputFeatures + i] * loraA[i * rank + r];
        }
        delta += hidden * loraB[r * outputFeatures + o];
    }

    output[gid] = baseOutput[gid] + scaling * delta;
}

kernel void issue301_fused_ddim_step(
    device const float* xT [[buffer(0)]],
    device const float* epsilonTheta [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& alphaBarT [[buffer(4)]],
    constant float& alphaBarTMinus1 [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    float eps = epsilonTheta[gid];
    float x0Pred = (xT[gid] - sqrt(max(0.0f, 1.0f - alphaBarT)) * eps) / sqrt(alphaBarT);
    output[gid] = sqrt(alphaBarTMinus1) * x0Pred + sqrt(max(0.0f, 1.0f - alphaBarTMinus1)) * eps;
}

kernel void issue301_fused_sparse_linear(
    device const float* input [[buffer(0)]],
    device const float* packedCsr [[buffer(1)]],
    device const float* sparseValues [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& batchSize [[buffer(5)]],
    constant uint& inputFeatures [[buffer(6)]],
    constant uint& outputFeatures [[buffer(7)]],
    constant uint& nnz [[buffer(8)]],
    constant uint& hasBias [[buffer(9)]],
    constant uint& activation [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = batchSize * outputFeatures;
    if (gid >= total) return;

    uint b = gid / outputFeatures;
    uint o = gid - b * outputFeatures;
    int rowStart = as_type<int>(packedCsr[o]);
    int rowEnd = as_type<int>(packedCsr[o + 1u]);
    uint colBase = outputFeatures + 1u;
    float sum = hasBias != 0u ? bias[o] : 0.0f;

    for (int idx = rowStart; idx < rowEnd; ++idx)
    {
        int col = as_type<int>(packedCsr[colBase + uint(idx)]);
        sum += input[b * inputFeatures + uint(col)] * sparseValues[uint(idx)];
    }

    output[gid] = issue301_apply_activation(sum, activation);
}
";
    }
}
