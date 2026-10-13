// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    /// <summary>The largest rank the single-dispatch LoRA kernel holds in work-group shared memory.</summary>
    internal const int LoRASharedMemoryRank = 256;

    public void FusedLoRAForward(
        IGpuBuffer input,
        IGpuBuffer baseOutput,
        IGpuBuffer loraA,
        IGpuBuffer loraB,
        IGpuBuffer output,
        int batchSize,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        // The two-stage LoRA kernel uses one workgroup per batch row. The
        // GlslQuintOp wrapper computes workgroup count via
        // ceil(dispatchSize / 256). Pass batchSize * 256 to land on exactly
        // batchSize workgroups.
        if (batchSize <= 0 || rank <= 0) return;
        if (rank > LoRASharedMemoryRank)
        {
            // The single-dispatch kernel keeps a row's projection in work-group shared memory, which holds 256
            // entries. A larger rank runs as two dispatches with the projection in a device buffer (#1230).
            if (inputFeatures <= 0 || outputFeatures <= 0) return;
            using var projection = AllocateBuffer(checked(batchSize * rank));
            GlslNaryOp(VulkanFusedAdvancedKernels.LoRAProject, new[] { input, loraA, projection }, batchSize * rank,
                new[] { (uint)batchSize, (uint)inputFeatures, (uint)rank });
            GlslNaryOp(VulkanFusedAdvancedKernels.LoRAApply, new[] { baseOutput, projection, loraB, output },
                checked(batchSize * outputFeatures),
                new[] { (uint)batchSize, (uint)rank, (uint)outputFeatures, FloatBits(scaling) });
            return;
        }

        var pushConstants = new[]
        {
            (uint)batchSize,
            (uint)inputFeatures,
            (uint)rank,
            (uint)outputFeatures,
            FloatBits(scaling)
        };

        GlslQuintOp(
            VulkanFusedAdvancedKernels.LoRAForward,
            input, baseOutput, loraA, loraB, output,
            batchSize * VulkanKernels.WorkgroupSize,
            pushConstants,
            5 * sizeof(uint));
    }

    public void FusedDDIMStep(
        IGpuBuffer xT,
        IGpuBuffer epsilonTheta,
        IGpuBuffer output,
        int size,
        float alphaBarT,
        float alphaBarTMinus1)
    {
        if (size <= 0) return;
        // The Vulkan kernel divides by sqrt(alphaBarT). Reject ᾱ <= 0 here
        // (defence-in-depth on top of the kernel's max() clamp).
        if (!(alphaBarT > 0f && alphaBarT <= 1f))
            throw new ArgumentOutOfRangeException(nameof(alphaBarT),
                $"alphaBarT must be in (0, 1]; got {alphaBarT}.");
        if (!(alphaBarTMinus1 >= 0f && alphaBarTMinus1 <= 1f))
            throw new ArgumentOutOfRangeException(nameof(alphaBarTMinus1),
                $"alphaBarTMinus1 must be in [0, 1]; got {alphaBarTMinus1}.");

        var pushConstants = new[]
        {
            (uint)size,
            FloatBits(alphaBarT),
            FloatBits(alphaBarTMinus1)
        };

        GlslBinaryOp(
            VulkanFusedAdvancedKernels.DDIMStep,
            xT, epsilonTheta, output,
            size,
            pushConstants,
            3 * sizeof(uint));
    }

    public void FusedSparseLinear(
        IGpuBuffer input,
        IGpuBuffer packedCsr,
        IGpuBuffer sparseValues,
        IGpuBuffer bias,
        IGpuBuffer output,
        int batchSize,
        int inputFeatures,
        int outputFeatures,
        int nnz,
        int hasBias,
        int activation)
    {
        if (batchSize <= 0 || outputFeatures <= 0) return;
        // Two compounding overflow risks to validate against:
        //   1) GLSL `uint total = p.batchSize * p.outputFeatures` wraps modulo
        //      2^32 if the product exceeds ~4.29B (defeats the in-shader
        //      bounds guard).
        //   2) The GlslQuintOp wrapper takes `int dispatchSize`, so the C#
        //      side itself overflows on cast for products > int.MaxValue
        //      (~2.15B), even though they'd still fit in uint32.
        // The tighter bound (int.MaxValue) is the operative one — anything
        // smaller is also < uint.MaxValue, so we only need one check.
        long totalLong = (long)batchSize * outputFeatures;
        if (totalLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                $"batchSize ({batchSize}) * outputFeatures ({outputFeatures}) = {totalLong} " +
                "exceeds int.MaxValue (Vulkan dispatch wrapper is int-bounded); split the dispatch.");
        if (nnz < 0)
            throw new ArgumentOutOfRangeException(nameof(nnz), $"nnz must be >= 0; got {nnz}.");

        var pushConstants = new[]
        {
            (uint)batchSize,
            (uint)inputFeatures,
            (uint)outputFeatures,
            (uint)nnz,
            (uint)hasBias,
            (uint)activation
        };

        GlslQuintOp(
            VulkanFusedAdvancedKernels.SparseLinear,
            input, packedCsr, sparseValues, bias, output,
            (int)totalLong,
            pushConstants,
            6 * sizeof(uint));
    }
}
