// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL categorical sampling — the device counterpart of <c>CpuEngine.TensorCategoricalSample</c>.
/// </summary>
/// <remarks>
/// <para>
/// This exists so the op stops silently running on the CPU after the caller selected the GPU engine.
/// <c>ICategoricalSamplingBackend</c> and <c>ISeededGumbelSoftmaxBackend</c> were implemented only by
/// CUDA and Vulkan, so on OpenCL the engine had no device route and fell back — measured as 0 kernel
/// launches against a required 1 in the GPU residency probe.
/// </para>
/// <para>
/// BIT-EXACTNESS IS THE WHOLE POINT, because the parity test compares CPU and GPU one-hot outputs
/// exactly and a single differing category is a hard mismatch. Three things therefore mirror the CPU
/// implementation rather than merely resembling it:
/// </para>
/// <list type="number">
/// <item>The uniform draw is <c>StatelessRandom.Uniform01(seed, row)</c> — the same PCG hash, the
/// same three constants, the same 24-bit mantissa scaling. Keying on the ROW index (not draw order)
/// is what lets a work-item compute its own uniform without replaying the rows before it.</item>
/// <item>The sum and the cumulative walk accumulate in <c>double</c>, as the CPU does. In float the
/// running total drifts, and a target sitting near a bucket edge selects the neighbouring category —
/// correct-looking output, exact-parity failure. Devices without fp64 do not advertise this kernel
/// and fall back to the CPU, which is the same convention the geometry kernels use.</item>
/// <item>The comparison is <c>target &lt; cumulative</c> with the same category ordering and the same
/// <c>classes - 1</c> default when floating-point drift leaves the target just past the final
/// boundary.</item>
/// </list>
/// </remarks>
internal static class CategoricalKernels
{
    /// <summary>Kernel names to register when the program compiles.</summary>
    public static string[] GetKernelNames() => new[] { "categorical_sample" };

    public static string GetSource() => @"
// double accumulation is required for exact CPU parity (see the C# remarks); a device without
// fp64 fails to compile this program and the engine keeps using the managed reference.
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

// StatelessRandom.Uniform01(seed, index) — identical constants to the managed helper and to
// GenerateRandomUniform in RandomKernels.
inline float stateless_uniform01(uint seed32, uint index)
{
    uint state = index * 747796405u + seed32 + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    uint sample = (word >> 22) ^ word;
    return (float)(sample >> 8) * (1.0f / 16777216.0f);
}

__kernel void categorical_sample(
    __global const float* probabilities,
    __global float* oneHot,
    const int rows,
    const int classes,
    const ulong seed)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    // The engine passes (ulong)(uint)seed, so the fold is a no-op there; it is kept so an
    // unfolded 64-bit seed degrades the same way GenerateRandomUniform does.
    uint seed32 = (uint)seed ^ (uint)(seed >> 32);
    float uniform = stateless_uniform01(seed32, (uint)row);

    int offset = row * classes;

    double sum = 0.0;
    for (int c = 0; c < classes; c++)
    {
        sum += (double)probabilities[offset + c];
    }

    double target = (double)uniform * sum;
    double cumulative = 0.0;
    int selected = classes - 1;
    for (int c = 0; c < classes; c++)
    {
        cumulative += (double)probabilities[offset + c];
        if (target < cumulative)
        {
            selected = c;
            break;
        }
    }

    for (int c = 0; c < classes; c++)
    {
        oneHot[offset + c] = 0.0f;
    }

    oneHot[offset + selected] = 1.0f;
}
";
}
