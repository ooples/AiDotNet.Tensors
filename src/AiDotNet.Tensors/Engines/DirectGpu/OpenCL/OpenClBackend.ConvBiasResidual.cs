namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IConvBiasResidualBackend
{
    bool IConvBiasResidualBackend.TryConvBiasResidual(in ConvBiasResidualArgs args)
        => ConvBiasResidualComposition.TryExecute(this, args);
}

