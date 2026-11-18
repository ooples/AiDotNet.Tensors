namespace AiDotNet.Tensors.Engines;

internal interface IInferencePrivateGradientWorkspaceOwner
{
    void ReturnPrivateGradientWorkspaceForInference(DirectGpuTensorEngine engine, DirectGpu.IDirectGpuBackend backend);
}
