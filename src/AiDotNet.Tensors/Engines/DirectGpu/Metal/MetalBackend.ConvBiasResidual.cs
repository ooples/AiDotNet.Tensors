namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IConvBiasResidualBackend
{
    bool IConvBiasResidualBackend.TryConvBiasResidual(in ConvBiasResidualArgs args)
        => ConvBiasResidualComposition.TryExecute(this, args);
}

