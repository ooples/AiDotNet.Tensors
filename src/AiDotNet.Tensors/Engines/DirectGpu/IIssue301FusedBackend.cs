// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Backend extension point for issue #301 fused kernels. Kept out of
/// IDirectGpuBackend so older/custom backends keep compiling and simply fall
/// back to the CPU fused helpers until they implement these kernels.
/// </summary>
internal interface IIssue301FusedBackend
{
    void FusedLoRAForward(
        IGpuBuffer input,
        IGpuBuffer baseOutput,
        IGpuBuffer loraA,
        IGpuBuffer loraB,
        IGpuBuffer output,
        int batchSize,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling);

    void FusedDDIMStep(
        IGpuBuffer xT,
        IGpuBuffer epsilonTheta,
        IGpuBuffer output,
        int size,
        float alphaBarT,
        float alphaBarTMinus1);

    void FusedSparseLinear(
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
        int activation);
}
