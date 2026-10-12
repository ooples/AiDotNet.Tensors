namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional resident capability for fusing Philox slope generation with the
/// training RReLU consumer while retaining the public saved-noise tensor.
/// </summary>
[BackendSpecificRoute("CUDA direct-PTX Philox slope generation fused into training RReLU; every backend draws the slopes and applies RReLU natively",
    nameof(IDirectGpuBackend.GenerateRandomUniform), nameof(IDirectGpuBackend.RRelu))]
internal interface IPhiloxRReluBackend
{
    bool TryFusedPhiloxRRelu(
        IGpuBuffer input,
        IGpuBuffer noise,
        IGpuBuffer output,
        int elementCount,
        float lower,
        float upper,
        ulong seed);
}
