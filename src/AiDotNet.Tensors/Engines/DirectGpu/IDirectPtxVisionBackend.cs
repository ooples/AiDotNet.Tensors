namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional internal surface for exact, disabled-by-default direct-PTX vision
/// specializations. Non-NVIDIA backends continue through established routes.
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
