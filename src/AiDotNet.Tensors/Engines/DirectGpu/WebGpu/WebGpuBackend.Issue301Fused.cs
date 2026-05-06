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
        int total = batchSize * outputFeatures;
        if (total <= 0) return;
        Dispatch5BufferAsync(
            "Issue301FusedLoRA",
            WebGpuIssue301FusedKernels.LoRAForward,
            "main",
            input, baseOutput, loraA, loraB, output,
            Issue301UniformLoRA(batchSize, inputFeatures, rank, outputFeatures, scaling),
            total).GetAwaiter().GetResult();
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
            "Issue301FusedDDIM",
            WebGpuIssue301FusedKernels.DDIMStep,
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
            "Issue301FusedSparseLinear",
            WebGpuIssue301FusedKernels.SparseLinear,
            "main",
            input, packedCsr, sparseValues, bias, output,
            Issue301UniformInts(batchSize, inputFeatures, outputFeatures, nnz, hasBias, activation),
            total).GetAwaiter().GetResult();
    }

    private static float[] Issue301UniformLoRA(int batchSize, int inputFeatures, int rank, int outputFeatures, float scaling) =>
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

    private static float[] Issue301UniformInts(params int[] values)
    {
        int padded = ((values.Length + 3) / 4) * 4;
        var uniforms = new float[padded];
        for (int i = 0; i < values.Length; i++)
            uniforms[i] = System.BitConverter.Int32BitsToSingle(values[i]);
        return uniforms;
    }
}
#endif
