namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability for a backend that can generate a versioned random mask
/// inside the bias-add/dropout consumer instead of materializing a private mask
/// in a separate launch.
/// </summary>
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
