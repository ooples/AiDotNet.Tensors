// Copyright (c) AiDotNet. All rights reserved.
// Metal launcher shim for categorical sampling. Mirrors MetalBackend.Ann.cs — pipeline resolved via
// the library handle compiled at init, dispatched with a 1-D thread grid of one thread per row.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Gives Metal a device route for <c>TensorCategoricalSample</c>.
/// </summary>
/// <remarks>
/// <para>
/// Without this the op fell back to the CPU after the caller had explicitly selected the Metal
/// engine — correct output, but measured by the residency probe as 0 kernel launches against a
/// required 1, which is a portability bug rather than an optimisation gap.
/// </para>
/// <para>
/// The kernel differs from the other backends' in ONE deliberate respect: MSL has no <c>double</c>,
/// so the probability sum and the cumulative walk are carried in a compensated two-float expansion.
/// See <see cref="MetalCategoricalKernels"/> for why that reproduces the managed reference's
/// category selection and what it depends on.
/// </para>
/// <para>
/// NOT EXECUTED IN VERIFICATION: Metal runs only on macOS and this was written on Windows, so this
/// route is verified to compile and register but its output has not been observed against the CPU
/// oracle.
/// </para>
/// </remarks>
public sealed partial class MetalBackend : ICategoricalSamplingBackend
{
    private const string CategoricalLibName = "Categorical";

    /// <inheritdoc/>
    public bool CanCategoricalSample(int rows, int classes)
        // Availability is decided by whether the library COMPILED, which is also the fp-capability
        // answer: a device or compiler that rejects the kernel leaves this zero and the engine keeps
        // using the managed reference.
        => _categoricalLibrary != IntPtr.Zero
           && rows > 0
           && classes > 0
           // One thread per row; a row wider than this is still correct but the serial walk over
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
        if (probabilities is not MetalGpuBuffer probabilityBuffer) return false;
        if (oneHot is not MetalGpuBuffer oneHotBuffer) return false;

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

        ThrowIfDisposed();

        var pipeline = GetPipeline(CategoricalLibName, _categoricalLibrary, "categorical_sample");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(rows);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(probabilityBuffer, 0);
        encoder.SetBuffer(oneHotBuffer, 1);
        encoder.SetBytes(rows, 2);
        encoder.SetBytes(classes, 3);
        // Split as the kernel folds it: seed32 = lo ^ hi, matching the managed helper and every
        // other backend's kernel.
        encoder.SetBytes((uint)(seed & 0xFFFFFFFF), 4);
        encoder.SetBytes((uint)(seed >> 32), 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);

        return true;
    }
}
