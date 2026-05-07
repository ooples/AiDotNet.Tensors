// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
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
        if ((uint)rank > 256u)
            throw new NotSupportedException(
                $"Vulkan LoRA fused kernel currently caps rank at 256 (got {rank}). " +
                "Increase MAX_RANK in VulkanIssue301FusedKernels.LoRAForward and " +
                "redeploy the SPIR-V if higher ranks are required.");

        var pushConstants = new[]
        {
            (uint)batchSize,
            (uint)inputFeatures,
            (uint)rank,
            (uint)outputFeatures,
            FloatBits(scaling)
        };

        GlslQuintOp(
            VulkanIssue301FusedKernels.LoRAForward,
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
            VulkanIssue301FusedKernels.DDIMStep,
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
        // GLSL `uint total = p.batchSize * p.outputFeatures` wraps modulo 2^32
        // when the product exceeds ~4.29B. Validate on the C# side before
        // submitting so the kernel's bounds guard isn't silently bypassed.
        long totalLong = (long)batchSize * outputFeatures;
        if (totalLong > uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                $"batchSize ({batchSize}) * outputFeatures ({outputFeatures}) = {totalLong} " +
                "overflows uint32 in the GLSL kernel; split the dispatch.");
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
            VulkanIssue301FusedKernels.SparseLinear,
            input, packedCsr, sparseValues, bias, output,
            (int)totalLong,
            pushConstants,
            6 * sizeof(uint));
    }
}
