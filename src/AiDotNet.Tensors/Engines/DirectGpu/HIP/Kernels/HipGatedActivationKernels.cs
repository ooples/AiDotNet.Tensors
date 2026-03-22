namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP GatedActivation kernels — shared source with CUDA (HIP is CUDA-compatible).
/// </summary>
public static class HipGatedActivationKernels
{
    public static string GetSource() => AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaGatedActivationKernels.GetSource();
    public static string[] GetKernelNames() => AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaGatedActivationKernels.GetKernelNames();
}
