// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP categorical sampling — its own HIPRTC module, deliberately.
/// </summary>
/// <remarks>
/// <para>
/// ISOLATED FOR TWO REASONS, both learned from the OpenCL side.
/// </para>
/// <para>
/// A module is compiled as one hiprtcCompileProgram call, so appending this kernel to the shared
/// neural-net source made every kernel in that module hostage to it: one rejection and
/// _neuralNetModule, generate_random_uniform and the rest all fail to load. Its own module fails
/// alone and leaves the engine on its CPU reference.
/// </para>
/// <para>
/// And the shared module compiles with fast math, which permits the compiler to reassociate
/// floating-point arithmetic. This kernel exists to reproduce the CPU's inverse-CDF walk EXACTLY,
/// in ordered double accumulation; reassociating the running sum moves the bucket boundary and
/// selects a different category, which the exact-parity test reads as a hard mismatch. It is
/// therefore compiled with fast math off.
/// </para>
/// </remarks>
internal static class HipCategoricalKernels
{
    public static string[] GetKernelNames() => new[] { "categorical_sample" };

    public static string GetSource() => @"
extern ""C"" __global__ __launch_bounds__(256) void categorical_sample(
    const float* probabilities, float* oneHot, int rows, int classes, unsigned long long seed)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    unsigned int seed32 = (unsigned int)seed ^ (unsigned int)(seed >> 32);
    unsigned int state = (unsigned int)row * 747796405u + seed32 + 2891336453u;
    unsigned int word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    unsigned int sample = (word >> 22) ^ word;
    float uniform = (float)(sample >> 8) * (1.0f / 16777216.0f);

    int offset = row * classes;

    double sum = 0.0;
    for (int c = 0; c < classes; c++) sum += (double)probabilities[offset + c];

    double target = (double)uniform * sum;
    double cumulative = 0.0;
    int selected = classes - 1;
    for (int c = 0; c < classes; c++)
    {
        cumulative += (double)probabilities[offset + c];
        if (target < cumulative) { selected = c; break; }
    }

    for (int c = 0; c < classes; c++) oneHot[offset + c] = 0.0f;
    oneHot[offset + selected] = 1.0f;
}
";
}
