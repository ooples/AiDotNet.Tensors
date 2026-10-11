namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional internal surface for exact vision specializations: CUDA's
/// disabled-by-default direct-PTX kernels, and Vulkan's fused Meshgrid2D (#923).
/// Other backends continue through established routes. Meshgrid2D specialization
/// parity is tracked for HIP (#920), Metal (#921), OpenCL (#922), and WebGPU (#924);
/// each keeps the GPU-resident reshape-plus-broadcast route until its native
/// specialization clears the oracle.
/// </summary>
[BackendSpecificRoute("CUDA direct-PTX specialization of the fused meshgrid; every backend runs the operation natively",
    nameof(IMeshgridBackend.Meshgrid2DPair))]
internal interface IDirectPtxVisionBackend
{
    bool CanDirectPtxMeshgrid2D(int n0, int n1, bool xy);

    bool TryDirectPtxMeshgrid2DPair(
        IGpuBuffer source0,
        IGpuBuffer source1,
        IGpuBuffer output0,
        IGpuBuffer output1,
        int n0,
        int n1,
        bool xy);
}
