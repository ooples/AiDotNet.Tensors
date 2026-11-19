namespace AiDotNet.Tensors.Engines;

internal interface IInferencePrivateGradientWorkspaceOwner
{
    bool PrivateGradientWorkspaceDemandPending { get; }
    void ReturnPrivateGradientWorkspaceForInference(DirectGpuTensorEngine engine, DirectGpu.IDirectGpuBackend backend);
}
