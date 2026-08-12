namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional internal surface for exact, disabled-by-default direct-PTX vision
/// specializations. Non-NVIDIA backends continue through established routes.
/// Meshgrid2D specialization parity is tracked for HIP (#920), Metal (#921),
/// OpenCL (#922), Vulkan (#923), and WebGPU (#924); each keeps the GPU-resident
/// reshape-plus-broadcast route until its native specialization clears the oracle.
/// </summary>
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
