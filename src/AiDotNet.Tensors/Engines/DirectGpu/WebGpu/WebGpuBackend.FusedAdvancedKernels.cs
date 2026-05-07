// Copyright (c) AiDotNet. All rights reserved.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
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
        if (batchSize <= 0 || rank <= 0 || outputFeatures <= 0) return;
        if (rank > 256)
            throw new System.NotSupportedException(
                $"WebGPU LoRA fused kernel currently caps rank at 256 (got {rank}). " +
                "Increase MAX_RANK in WebGpuFusedAdvancedKernels.LoRAForward and the " +
                "workgroup-private array size if higher ranks are required.");

        // The two-stage WGSL kernel uses one workgroup per batch row with a
        // workgroup_size of 256. Dispatch5BufferAsync's CalculateWorkgroups1D
        // computes ceil(workSize / 256), so passing batchSize * 256 yields
        // exactly batchSize workgroups.
        int workSize = batchSize * 256;
        Dispatch5BufferAsync(
            "FusedLoRA",
            WebGpuFusedAdvancedKernels.LoRAForward,
            "main",
            input, baseOutput, loraA, loraB, output,
            FusedLoRAUniform(batchSize, inputFeatures, rank, outputFeatures, scaling),
            workSize).GetAwaiter().GetResult();
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
        Dispatch3BufferAsync(
            "FusedDDIM",
            WebGpuFusedAdvancedKernels.DDIMStep,
            "main",
            xT, epsilonTheta, output,
            new[] { System.BitConverter.Int32BitsToSingle(size), alphaBarT, alphaBarTMinus1, 0f },
            size).GetAwaiter().GetResult();
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
        int total = batchSize * outputFeatures;
        if (total <= 0) return;
        Dispatch5BufferAsync(
            "FusedSparseLinear",
            WebGpuFusedAdvancedKernels.SparseLinear,
            "main",
            input, packedCsr, sparseValues, bias, output,
            FusedKernelUniformInts(batchSize, inputFeatures, outputFeatures, nnz, hasBias, activation),
            total).GetAwaiter().GetResult();
    }

    private static float[] FusedLoRAUniform(int batchSize, int inputFeatures, int rank, int outputFeatures, float scaling) =>
        new[]
        {
            System.BitConverter.Int32BitsToSingle(batchSize),
            System.BitConverter.Int32BitsToSingle(inputFeatures),
            System.BitConverter.Int32BitsToSingle(rank),
            System.BitConverter.Int32BitsToSingle(outputFeatures),
            scaling,
            0f,
            0f,
            0f
        };

    private static float[] FusedKernelUniformInts(params int[] values)
    {
        int padded = ((values.Length + 3) / 4) * 4;
        var uniforms = new float[padded];
        for (int i = 0; i < values.Length; i++)
            uniforms[i] = System.BitConverter.Int32BitsToSingle(values[i]);
        return uniforms;
    }
}
#endif
