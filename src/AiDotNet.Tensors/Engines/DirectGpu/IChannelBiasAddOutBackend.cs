namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: a per-channel bias add written out of place, <c>output[b, c, s] = input[b, c, s] + bias[c]</c>,
/// in one pass. Every backend forms it natively as a copy of the input followed by the in-place
/// <see cref="IDirectGpuBackend.Conv2DBiasAdd"/>.
/// </summary>
[BackendSpecificRoute("CUDA out-of-place channel bias add; every backend copies and adds in place natively",
    nameof(IDirectGpuBackend.Copy), nameof(IDirectGpuBackend.Conv2DBiasAdd))]
internal interface IChannelBiasAddOutBackend
{
    /// <summary><c>output = input + bias</c> per channel of [batch, channels, spatialSize], every element of
    /// <paramref name="output"/> written. False -- nothing launched -- when unavailable.</summary>
    bool TryChannelBiasAddOut(IGpuBuffer input, IGpuBuffer bias, IGpuBuffer output, int batch, int channels, int spatialSize);
}