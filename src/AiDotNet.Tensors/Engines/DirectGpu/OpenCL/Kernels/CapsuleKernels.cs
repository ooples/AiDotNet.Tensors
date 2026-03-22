namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for capsule network operations.
/// </summary>
internal static class CapsuleKernels
{
    public static string GetSource()
    {
        return @"
__kernel void capsule_predictions(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
{
    int idx = get_global_id(0);
    int total = batchSize * inputCapsules * outputCapsules * outputDim;
    if (idx >= total) return;

    int d = idx % outputDim;
    int temp = idx / outputDim;
    int c = temp % outputCapsules;
    temp = temp / outputCapsules;
    int i = temp % inputCapsules;
    int b = temp / inputCapsules;

    float sum = 0.0f;
    for (int k = 0; k < inputDim; k++) {
        int inputIdx = b * inputCapsules * inputDim + i * inputDim + k;
        int weightIdx = i * outputCapsules * inputDim * outputDim + c * inputDim * outputDim + k * outputDim + d;
        sum += input[inputIdx] * weights[weightIdx];
    }
    output[idx] = sum;
}

__kernel void capsule_transform(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
{
    int idx = get_global_id(0);
    int total = batchSize * inputCapsules * numCapsules * capsuleDim;
    if (idx >= total) return;

    int d = idx % capsuleDim;
    int temp = idx / capsuleDim;
    int j = temp % numCapsules;
    temp = temp / numCapsules;
    int i = temp % inputCapsules;
    int b = temp / inputCapsules;

    float sum = 0.0f;
    for (int k = 0; k < inputDim; k++) {
        int inputIdx = b * inputCapsules * inputDim + i * inputDim + k;
        int weightIdx = i * inputDim * numCapsules * capsuleDim + k * numCapsules * capsuleDim + j * capsuleDim + d;
        sum += input[inputIdx] * weights[weightIdx];
    }
    output[idx] = sum;
}

__kernel void capsule_weighted_sum(
    __global const float* coupling,
    __global const float* predictions,
    __global float* output,
    int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
{
    int idx = get_global_id(0);
    int total = batchSize * outputCapsules * capsuleDim;
    if (idx >= total) return;

    int d = idx % capsuleDim;
    int temp = idx / capsuleDim;
    int c = temp % outputCapsules;
    int b = temp / outputCapsules;

    float sum = 0.0f;
    for (int i = 0; i < inputCapsules; i++) {
        int couplingIdx = b * inputCapsules * outputCapsules + i * outputCapsules + c;
        int predIdx = b * inputCapsules * outputCapsules * capsuleDim + i * outputCapsules * capsuleDim + c * capsuleDim + d;
        sum += coupling[couplingIdx] * predictions[predIdx];
    }
    output[idx] = sum;
}

__kernel void capsule_agreement(
    __global const float* predictions,
    __global const float* output_caps,
    __global float* agreement,
    int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
{
    int idx = get_global_id(0);
    int total = batchSize * inputCapsules * outputCapsules;
    if (idx >= total) return;

    int c = idx % outputCapsules;
    int temp = idx / outputCapsules;
    int i = temp % inputCapsules;
    int b = temp / inputCapsules;

    float sum = 0.0f;
    for (int d = 0; d < capsuleDim; d++) {
        int predIdx = b * inputCapsules * outputCapsules * capsuleDim + i * outputCapsules * capsuleDim + c * capsuleDim + d;
        int outIdx = b * outputCapsules * capsuleDim + c * capsuleDim + d;
        sum += predictions[predIdx] * output_caps[outIdx];
    }
    agreement[idx] = sum;
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "capsule_predictions",
            "capsule_transform",
            "capsule_weighted_sum",
            "capsule_agreement"
        ];
    }
}
