// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Matrix-free product of <c>K</c> Householder reflections <c>H_0 H_1 ... H_{K-1}</c> (or reverse order when
/// <c>transpose</c> is true) over row-major <c>input [rows, dim]</c> and unconstrained <c>reflectors [numReflectors, dim]</c>,
/// using overflow-safe row normalization with squared-norm floor <c>normFloor</c>. CUDA executes single-launch
/// shared-SRAM warp-shuffle forward and 2-launch deterministic shared-SRAM backward kernels; all other backends (and
/// oversized shapes) execute the portable device kernels in <see cref="PortableHouseholderChain"/>.
/// </summary>
internal interface IFusedHouseholderChain
{
    void HouseholderChainForward(
        IGpuBuffer input,
        IGpuBuffer reflectors,
        IGpuBuffer output,
        int rows,
        int numReflectors,
        int dim,
        bool transpose,
        float normFloor);

    void HouseholderChainBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer reflectors,
        IGpuBuffer gradInput,
        IGpuBuffer gradReflectors,
        int rows,
        int numReflectors,
        int dim,
        bool transpose,
        float normFloor);
}
