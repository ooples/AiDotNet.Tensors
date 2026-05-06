// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanIssue301FusedKernels
{
    public const string LoRAForward = @"
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input { float inputData[]; };
layout(std430, binding = 1) readonly buffer BaseOutput { float baseOutputData[]; };
layout(std430, binding = 2) readonly buffer LoraA { float loraAData[]; };
layout(std430, binding = 3) readonly buffer LoraB { float loraBData[]; };
layout(std430, binding = 4) writeonly buffer Output { float outputData[]; };

layout(push_constant) uniform Params
{
    uint batchSize;
    uint inputFeatures;
    uint rank;
    uint outputFeatures;
    uint scalingBits;
} p;

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    uint total = p.batchSize * p.outputFeatures;
    if (gid >= total) return;

    uint b = gid / p.outputFeatures;
    uint o = gid - b * p.outputFeatures;
    float delta = 0.0;

    for (uint r = 0; r < p.rank; ++r)
    {
        float hidden = 0.0;
        for (uint i = 0; i < p.inputFeatures; ++i)
        {
            hidden += inputData[b * p.inputFeatures + i] * loraAData[i * p.rank + r];
        }
        delta += hidden * loraBData[r * p.outputFeatures + o];
    }

    outputData[gid] = baseOutputData[gid] + uintBitsToFloat(p.scalingBits) * delta;
}
";

    public const string DDIMStep = @"
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer XT { float xTData[]; };
layout(std430, binding = 1) readonly buffer Epsilon { float epsilonData[]; };
layout(std430, binding = 2) writeonly buffer Output { float outputData[]; };

layout(push_constant) uniform Params
{
    uint size;
    uint alphaBarTBits;
    uint alphaBarTMinus1Bits;
} p;

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= p.size) return;

    float alphaBarT = uintBitsToFloat(p.alphaBarTBits);
    float alphaBarTMinus1 = uintBitsToFloat(p.alphaBarTMinus1Bits);
    float eps = epsilonData[gid];
    float x0Pred = (xTData[gid] - sqrt(max(0.0, 1.0 - alphaBarT)) * eps) / sqrt(alphaBarT);
    outputData[gid] = sqrt(alphaBarTMinus1) * x0Pred + sqrt(max(0.0, 1.0 - alphaBarTMinus1)) * eps;
}
";

    public const string SparseLinear = @"
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input { float inputData[]; };
layout(std430, binding = 1) readonly buffer PackedCsr { float packedCsrData[]; };
layout(std430, binding = 2) readonly buffer Values { float sparseValuesData[]; };
layout(std430, binding = 3) readonly buffer Bias { float biasData[]; };
layout(std430, binding = 4) writeonly buffer Output { float outputData[]; };

layout(push_constant) uniform Params
{
    uint batchSize;
    uint inputFeatures;
    uint outputFeatures;
    uint nnz;
    uint hasBias;
    uint activation;
} p;

float issue301_apply_activation(float x)
{
    if (p.activation == 1u) return max(x, 0.0);
    if (p.activation == 2u)
    {
        float k = sqrt(2.0 / 3.14159265358979323846);
        return 0.5 * x * (1.0 + tanh(k * (x + 0.044715 * x * x * x)));
    }
    if (p.activation == 3u) return 1.0 / (1.0 + exp(-x));
    if (p.activation == 4u) return tanh(x);
    if (p.activation == 5u) return x > 0.0 ? x : 0.01 * x;
    if (p.activation == 6u)
    {
        float sigmoid = 1.0 / (1.0 + exp(-x));
        return x * sigmoid;
    }
    return x;
}

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    uint total = p.batchSize * p.outputFeatures;
    if (gid >= total) return;

    uint b = gid / p.outputFeatures;
    uint o = gid - b * p.outputFeatures;
    int rowStart = floatBitsToInt(packedCsrData[o]);
    int rowEnd = floatBitsToInt(packedCsrData[o + 1u]);
    uint colBase = p.outputFeatures + 1u;
    float sum = p.hasBias != 0u ? biasData[o] : 0.0;

    for (int idx = rowStart; idx < rowEnd; ++idx)
    {
        int col = floatBitsToInt(packedCsrData[colBase + uint(idx)]);
        sum += inputData[b * p.inputFeatures + uint(col)] * sparseValuesData[uint(idx)];
    }

    outputData[gid] = issue301_apply_activation(sum);
}
";
}
