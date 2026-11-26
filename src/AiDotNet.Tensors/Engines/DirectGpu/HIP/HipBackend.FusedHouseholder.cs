// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IFusedHouseholderChain
{
    void IFusedHouseholderChain.HouseholderChainForward(
        IGpuBuffer input, IGpuBuffer reflectors, IGpuBuffer output,
        int rows, int numReflectors, int dim, bool transpose, float normFloor)
        => PortableHouseholderChain.Forward(this, input, reflectors, output, rows, numReflectors, dim, transpose, normFloor);

    void IFusedHouseholderChain.HouseholderChainBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer reflectors, IGpuBuffer gradInput, IGpuBuffer gradReflectors,
        int rows, int numReflectors, int dim, bool transpose, float normFloor)
        => PortableHouseholderChain.Backward(this, gradOutput, input, reflectors, gradInput, gradReflectors, rows, numReflectors, dim, transpose, normFloor);
}
