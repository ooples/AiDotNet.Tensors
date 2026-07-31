namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for capsule network operations.
/// HIP is API-compatible with CUDA — kernel source is identical.
/// </summary>
internal static class HipCapsuleKernels
{
    public static string GetSource()
    {
        // HIP kernel source is identical to CUDA — same __global__ syntax
        return AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaCapsuleKernels.GetSource() + @"

extern ""C"" __global__ __launch_bounds__(256) void squash(
    const float* input, float* output, int numCapsules, int capsuleDim, float epsilon)
{
    int capsule = blockIdx.x * blockDim.x + threadIdx.x;
    if (capsule >= numCapsules) return;

    int offset = capsule * capsuleDim;
    float squaredNorm = 0.0f;
    for (int d = 0; d < capsuleDim; d++) {
        float value = input[offset + d];
        squaredNorm += value * value;
    }
    float denominator = (1.0f + squaredNorm) * (sqrtf(squaredNorm) + epsilon);
    float scale = denominator != 0.0f ? squaredNorm / denominator : 0.0f;
    for (int d = 0; d < capsuleDim; d++)
        output[offset + d] = scale * input[offset + d];
}

extern ""C"" __global__ __launch_bounds__(256) void squash_backward(
    const float* gradOutput, const float* input, float* gradInput,
    int numCapsules, int capsuleDim, float epsilon)
{
    int capsule = blockIdx.x * blockDim.x + threadIdx.x;
    if (capsule >= numCapsules) return;

    int offset = capsule * capsuleDim;
    float squaredNorm = 0.0f;
    float dot = 0.0f;
    for (int d = 0; d < capsuleDim; d++) {
        float value = input[offset + d];
        squaredNorm += value * value;
        dot += value * gradOutput[offset + d];
    }
    float norm = sqrtf(squaredNorm);
    float normPlusEpsilon = norm + epsilon;
    float denominator = (1.0f + squaredNorm) * normPlusEpsilon;
    float scale = denominator != 0.0f ? squaredNorm / denominator : 0.0f;
    float coefficient = denominator != 0.0f
        ? (norm + 2.0f * epsilon - squaredNorm * norm) / (denominator * denominator)
        : 0.0f;
    for (int d = 0; d < capsuleDim; d++)
        gradInput[offset + d] = scale * gradOutput[offset + d]
            + coefficient * input[offset + d] * dot;
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            .. AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaCapsuleKernels.GetKernelNames(),
            "squash",
            "squash_backward"
        ];
    }
}
