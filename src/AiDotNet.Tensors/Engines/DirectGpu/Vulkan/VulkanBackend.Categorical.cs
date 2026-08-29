// Copyright (c) AiDotNet. All rights reserved.
// Vulkan launcher for categorical sampling. Dispatches the GLSL twin of the OpenCL
// categorical_sample kernel through the same GlslUnaryOp path the other softmax-family ops use.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Gives Vulkan a device route for <c>TensorCategoricalSample</c>.
/// </summary>
/// <remarks>
/// <para>
/// Without this the op fell back to the CPU after the caller had explicitly selected the Vulkan
/// engine — correct output, but measured by the residency probe as 0 kernel launches against a
/// required 1, which is a portability bug rather than an optimisation gap.
/// </para>
/// <para>
/// AVAILABILITY IS DECIDED BY WHETHER THE SHADER RAN, not by parsing device feature strings. The
/// kernel accumulates in <c>double</c> for exact CPU parity, which requires the device to support
/// <c>shaderFloat64</c>; a device that lacks it fails pipeline creation, and the first failure
/// latches this route off so the engine keeps using the managed reference instead of retrying a
/// compile that cannot succeed. Sampling at lower precision would be worse than falling back: the
/// parity test compares one-hot outputs exactly, and a target near a bucket edge would select the
/// neighbouring category.
/// </para>
/// <para>
/// NOT EXECUTED IN VERIFICATION: this host has the Vulkan loader but no usable Vulkan compute path
/// (every Vulkan test skips with "Vulkan not available on this system"), so this route is verified
/// to compile but its output has NOT been observed against the CPU oracle. Unlike the Metal and
/// WebGpu ports it is at least algorithmically identical to the verified OpenCL kernel, double
/// accumulation included, rather than a precision-adapted rewrite.
/// </para>
/// </remarks>
public sealed partial class VulkanBackend : ICategoricalSamplingBackend
{
    /// <summary>0 = not yet attempted, 1 = dispatched successfully, 2 = unsupported on this device.</summary>
    private int _categoricalState;

    /// <inheritdoc/>
    public bool CanCategoricalSample(int rows, int classes)
        => _categoricalState != 2
           && rows > 0
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
        if (probabilities is null || oneHot is null) return false;

        // Both buffers are indexed as row * classes + c, so both must actually hold that many
        // elements. CanCategoricalSample only checks the DIMENSIONS; an undersized buffer would be
        // written past its end on the device, which corrupts whatever is next in device memory and
        // is reported, if at all, by some unrelated later operation.
        long addressed = (long)rows * classes;
        GpuKernelDiagnostics.ValidateCapacity(
            "categorical_sample", nameof(probabilities), probabilities.Size, addressed);
        GpuKernelDiagnostics.ValidateCapacity(
            "categorical_sample", nameof(oneHot), oneHot.Size, addressed);

        if (probabilities.Size < addressed || oneHot.Size < addressed)
        {
            // Even with deep checks off, refuse rather than launch an out-of-range write.
            return false;
        }

        try
        {
            GlslUnaryOp(
                VulkanGlslKernels.CategoricalSampleGlsl,
                probabilities,
                oneHot,
                rows,
                new uint[] { (uint)rows, (uint)classes, (uint)(seed & 0xFFFFFFFF), (uint)(seed >> 32) },
                4 * sizeof(uint));
        }
        catch (InvalidOperationException ex)
        {
            // NARROW BY INTENT. GlslUnaryOp reports an unavailable pipeline as exactly this, which is
            // what a device without shaderFloat64 — or a host without libshaderc — produces for this
            // kernel. That is a capability answer, so it is latched: retrying a compile that cannot
            // succeed on every sample is pure cost. EVERY OTHER EXCEPTION PROPAGATES. A transient
            // dispatch failure is not evidence about the device, and permanently marking the backend
            // unsupported because of one is how a recoverable error becomes a silent downgrade.
            //
            // Returning false rather than throwing is deliberate and is NOT a host fallback:
            // VulkanBackend implements IGpuBatchExecution, so TensorCategoricalSample routes to the
            // Gumbel-max identity ON THE DEVICE. Throwing would remove a working device route
            // instead of protecting one. (The engine does throw NotSupportedException if Try* returns
            // false after Can* has already claimed support, so a false claim is never silent.)
            _categoricalState = 2;
            System.Diagnostics.Debug.WriteLine(
                $"[VulkanBackend] Categorical sampling unavailable on this device ({ex.Message}). "
                + "Routing TensorCategoricalSample to the on-device Gumbel-max path.");
            return false;
        }

        _categoricalState = 1;
        return true;
    }
}
