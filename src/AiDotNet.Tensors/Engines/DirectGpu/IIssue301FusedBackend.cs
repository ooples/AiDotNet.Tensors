// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Backend extension point for issue #301 fused kernels — sits between
/// <see cref="DirectGpuTensorEngine"/> (the engine) and the underlying
/// native backend (CudaBackend, HipBackend, OpenClBackend, MetalBackend,
/// VulkanBackend, WebGpuBackend), NOT at the engine layer.
///
/// <para>This is the same probe-interface pattern <see cref="IAsyncGpuBackend"/>
/// already uses for the async-execution capability: an optional contract a
/// native backend may implement, which the engine probes via type-check
/// before dispatching. The fused ops themselves are exposed on
/// <see cref="DirectGpuTensorEngine"/> (with CPU fallback), so the engine-
/// layer surface is unaffected by which native backends light up here.</para>
///
/// <para>Kept out of <see cref="IDirectGpuBackend"/> so older / custom
/// backends keep compiling and simply fall back to the CPU fused helpers
/// until they implement these kernels.</para>
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
