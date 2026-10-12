namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability for a backend that can generate a versioned random mask
/// inside the bias-add/dropout consumer instead of materializing a private mask
/// in a separate launch.
/// </summary>
[BackendSpecificRoute("CUDA direct-PTX Philox mask fused into bias-add and dropout; every backend runs dropout and bias-add natively",
    nameof(IDirectGpuBackend.Dropout), nameof(IDirectGpuBackend.BiasAdd))]
internal interface IPhiloxBiasDropoutBackend
{
    bool TryFusedBiasPhiloxDropout(
        IGpuBuffer input,
        IGpuBuffer output,
        IGpuBuffer bias,
        IGpuBuffer mask,
        int rows,
        int cols,
        float dropoutRate,
        ulong seed);
}
