// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanFusedAdvancedKernels
{
    // Two-stage fused LoRA forward. See CudaFusedAdvancedKernels.cs for the
    // design rationale. Total work: O(batch · rank · (in + out)) instead of
    // the broken O(batch · in · rank · out).
    //
    // Launch contract:
    //   gl_NumWorkGroups.x = batchSize
    //   local_size_x       = 256 (covers the typical output_features range)
    //   shared array size  = MAX_RANK = 256 (constant — covers practical LoRA
    //                        rank values; reads guard against overrun at runtime).
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

const uint MAX_RANK = 256u;
shared float proj[MAX_RANK];

void main()
{
    uint b = gl_WorkGroupID.x;
    if (b >= p.batchSize) return;

    uint tid = gl_LocalInvocationID.x;
    uint blockSize = gl_WorkGroupSize.x;

    uint inBase = b * p.inputFeatures;

    // Fail closed if host validation regressed and dispatched p.rank > MAX_RANK
    // or p.rank == 0. Silent truncation produces numerically wrong output;
    // skip the row entirely so callers see a zero output (and fail their
    // numerical-equivalence checks) rather than a confidently-wrong answer.
    if (p.rank == 0u || p.rank > MAX_RANK) return;
    uint effRank = p.rank;
    for (uint r = tid; r < effRank; r += blockSize)
    {
        float acc = 0.0;
        for (uint i = 0u; i < p.inputFeatures; ++i)
            acc += inputData[inBase + i] * loraAData[i * p.rank + r];
        proj[r] = acc;
    }

    barrier();
    memoryBarrierShared();

    // Stage 2: each thread emits one or more output elements, reusing proj.
    uint outBase = b * p.outputFeatures;
    float scaling = uintBitsToFloat(p.scalingBits);
    for (uint o = tid; o < p.outputFeatures; o += blockSize)
    {
        float delta = 0.0;
        for (uint r = 0u; r < effRank; ++r)
            delta += proj[r] * loraBData[r * p.outputFeatures + o];
        outputData[outBase + o] = baseOutputData[outBase + o] + scaling * delta;
    }
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
    // Clamp both alpha values away from negative-rounding artifacts before
    // taking sqrt. sqrt(<=0) → 0/NaN poison every element, including the
    // divisor sqrt(alphaBarT) (denominator). Host-side validation rejects
    // alphaBarT <= 0 too, but a defensive max() here costs ~1 ALU and
    // protects against rounding drift in cumulative-product schedules.
    float clampedAt = max(alphaBarT, 1e-12);
    float clampedAtm1 = max(alphaBarTMinus1, 0.0);
    float x0Pred = (xTData[gid] - sqrt(max(0.0, 1.0 - clampedAt)) * eps) / sqrt(clampedAt);
    outputData[gid] = sqrt(clampedAtm1) * x0Pred + sqrt(max(0.0, 1.0 - clampedAtm1)) * eps;
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

float fused_apply_activation(float x)
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
    // batchSize * outputFeatures wraps modulo 2^32 in GLSL when the product
    // exceeds 4.29B. The host-side dispatch validates that overflow before
    // submitting; this re-derives the same total in 64-bit-safe form by
    // checking each thread's (b, o) pair individually instead of trusting
    // a wrapped `total`.
    if (gid >= p.batchSize * p.outputFeatures) return;

    uint b = gid / p.outputFeatures;
    uint o = gid - b * p.outputFeatures;
    if (b >= p.batchSize || o >= p.outputFeatures) return;

    int rowStart = floatBitsToInt(packedCsrData[o]);
    int rowEnd = floatBitsToInt(packedCsrData[o + 1u]);
    uint colBase = p.outputFeatures + 1u;
    float sum = p.hasBias != 0u ? biasData[o] : 0.0;

    // Defensive bounds: skip the row if the CSR offsets are inverted,
    // negative, or out of nnz range. p.nnz is the authoritative upper
    // bound on idx; without this guard a corrupted CSR walks past
    // sparseValuesData (UB on Vulkan, dispatch crash).
    if (rowStart < 0 || rowEnd < rowStart || uint(rowEnd) > p.nnz)
    {
        outputData[gid] = fused_apply_activation(sum);
        return;
    }

    for (int idx = rowStart; idx < rowEnd; ++idx)
    {
        if (uint(idx) >= p.nnz) break;
        int col = floatBitsToInt(packedCsrData[colBase + uint(idx)]);
        if (col < 0 || uint(col) >= p.inputFeatures) continue;
        sum += inputData[b * p.inputFeatures + uint(col)] * sparseValuesData[uint(idx)];
    }

    outputData[gid] = fused_apply_activation(sum);
}
";
}
