// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Backend extension point for advanced fused kernels: LoRA forward
/// (parameter-efficient adapter), DDIM step (diffusion sampler update),
/// and fused sparse-linear forward.
///
/// <para>Sits between <see cref="DirectGpuTensorEngine"/> (the engine) and
/// the underlying native backend (CudaBackend, HipBackend, OpenClBackend,
/// MetalBackend, VulkanBackend, WebGpuBackend) — NOT at the engine layer.
/// The fused ops themselves are exposed on
/// <see cref="DirectGpuTensorEngine"/> with CPU fallback, so engine-layer
/// callers don't need to know which native backends light up here.</para>
///
/// <para>Same probe-interface pattern <see cref="IAsyncGpuBackend"/> uses
/// for async-execution capability. Kept out of
/// <see cref="IDirectGpuBackend"/> so older / custom backends keep
/// compiling and fall back to the CPU fused helpers until they implement
/// these kernels.</para>
/// </summary>
internal interface IFusedAdvancedKernels
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
