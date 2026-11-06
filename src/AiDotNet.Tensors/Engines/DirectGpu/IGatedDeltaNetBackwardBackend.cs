// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Internal capability interface implemented by GPU backends that provide a fused device-resident
/// BPTT backward pass for <see cref="IDirectGpuBackend.GatedDeltaNetScanForward"/>.
/// Keeping this internal avoids breaking external <see cref="IDirectGpuBackend"/> implementers
/// while allowing <see cref="DirectGpuTensorEngine"/> to keep GatedDeltaNet training 100% on-device.
/// </summary>
internal interface IGatedDeltaNetBackwardBackend
{
    void GatedDeltaNetScanBackward(
        IGpuBuffer dOut,
        IGpuBuffer q,
        IGpuBuffer k,
        IGpuBuffer v,
        IGpuBuffer alpha,
        IGpuBuffer beta,
        IGpuBuffer dQ,
        IGpuBuffer dK,
        IGpuBuffer dV,
        IGpuBuffer dAlpha,
        IGpuBuffer dBeta,
        int batch,
        int seqLen,
        int modelDim,
        int numHeads,
        int headDim);
}

internal static class GatedDeltaNetBackwardResolver
{
    public static bool TryResolve(IDirectGpuBackend backend, out IGatedDeltaNetBackwardBackend resolved)
    {
        while (backend is Gpu.DelegatingGpuBackend wrapper)
            backend = wrapper.InnerBackend;
        if (backend is IGatedDeltaNetBackwardBackend gdn)
        {
            resolved = gdn;
            return true;
        }
        resolved = null!;
        return false;
    }
}
