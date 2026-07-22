namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional resident capability for fusing Philox slope generation with the
/// training RReLU consumer while retaining the public saved-noise tensor.
/// </summary>
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
