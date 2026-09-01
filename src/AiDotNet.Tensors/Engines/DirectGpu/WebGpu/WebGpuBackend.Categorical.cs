// Copyright (c) AiDotNet. All rights reserved.
// WebGPU launcher for categorical sampling. Dispatches the WGSL twin of the OpenCL
// categorical_sample kernel through the same Dispatch2BufferAsync path the other ops use.
#if NET8_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Gives WebGPU a device route for <c>TensorCategoricalSample</c>.
/// </summary>
/// <remarks>
/// <para>
/// Without this the op fell back to the CPU after the caller had explicitly selected the WebGPU
/// engine — correct output, but measured by the residency probe as 0 kernel launches against a
/// required 1, which is a portability bug rather than an optimisation gap.
/// </para>
/// <para>
/// The kernel differs from the other backends' in ONE deliberate respect: WGSL has no f64, so the
/// probability sum and the cumulative walk are carried in a compensated two-float expansion. See
/// <see cref="WebGpuCategoricalKernels"/> for why that reproduces the managed reference's category
/// selection.
/// </para>
/// <para>
/// NOT EXECUTED IN VERIFICATION: this needs a WebGPU adapter and none was available on the host this
/// was written on, so the route is unverified against the CPU oracle.
/// </para>
/// </remarks>
public sealed partial class WebGpuBackend : ICategoricalSamplingBackend
{
    /// <inheritdoc/>
    public bool CanCategoricalSample(int rows, int classes)
        => rows > 0
           && classes > 0
           // One invocation per row; a row wider than this is still correct but the serial walk over
           // classes stops being the right shape for a GPU and the CPU reference is faster.
           && (long)rows * classes <= int.MaxValue;

    /// <inheritdoc/>
    public bool TryCategoricalSample(
        IGpuBuffer probabilities,
        IGpuBuffer oneHot,
        int rows,
        int classes,
        ulong seed)
    {
        if (!CanCategoricalSample(rows, classes)) return false;
        if (probabilities is not WebGpuBuffer probabilityBuffer) return false;
        if (oneHot is not WebGpuBuffer oneHotBuffer) return false;

        // Both buffers are indexed as row * classes + c, so both must actually hold that many
        // elements. CanCategoricalSample only checks the DIMENSIONS; an undersized buffer would be
        // written past its end on the device, which corrupts whatever is next in device memory and
        // is reported, if at all, by some unrelated later operation.
        long addressed = (long)rows * classes;
        GpuKernelDiagnostics.ValidateCapacity(
            "categorical_sample", nameof(probabilities), probabilityBuffer.Size, addressed);
        GpuKernelDiagnostics.ValidateCapacity(
            "categorical_sample", nameof(oneHot), oneHotBuffer.Size, addressed);

        if (probabilityBuffer.Size < addressed || oneHotBuffer.Size < addressed)
        {
            // Even with deep checks off, refuse rather than launch an out-of-range write.
            return false;
        }

        // The uniform block is four u32s; the dispatch helper takes floats, so the bits are carried
        // across verbatim rather than value-converted. Reading them as f32 would round the seed.
        var uniforms = new[]
        {
            BitConverter.Int32BitsToSingle(rows),
            BitConverter.Int32BitsToSingle(classes),
            BitConverter.Int32BitsToSingle(unchecked((int)(uint)(seed & 0xFFFFFFFF))),
            BitConverter.Int32BitsToSingle(unchecked((int)(uint)(seed >> 32))),
        };

        Dispatch2BufferAsync(
            "Categorical",
            WebGpuCategoricalKernels.CategoricalSample,
            "categorical_sample",
            probabilities,
            oneHot,
            uniforms,
            rows).GetAwaiter().GetResult();

        return true;
    }
}
#endif
