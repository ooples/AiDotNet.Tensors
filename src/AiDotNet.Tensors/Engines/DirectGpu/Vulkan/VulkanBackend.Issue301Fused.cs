// Copyright (c) AiDotNet. All rights reserved.

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
            batchSize * outputFeatures,
            pushConstants,
            6 * sizeof(uint));
    }
}
