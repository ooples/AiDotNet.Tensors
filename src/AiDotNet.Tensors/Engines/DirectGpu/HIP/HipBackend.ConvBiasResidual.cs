namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IConvBiasResidualBackend
{
    bool IConvBiasResidualBackend.TryConvBiasResidual(in ConvBiasResidualArgs args)
        => ConvBiasResidualComposition.TryExecute(this, args);
}

