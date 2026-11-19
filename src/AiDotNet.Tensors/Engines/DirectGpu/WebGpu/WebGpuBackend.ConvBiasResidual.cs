#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IConvBiasResidualBackend
{
    bool IConvBiasResidualBackend.TryConvBiasResidual(in ConvBiasResidualArgs args)
        => ConvBiasResidualComposition.TryExecute(this, args);
}
#endif

