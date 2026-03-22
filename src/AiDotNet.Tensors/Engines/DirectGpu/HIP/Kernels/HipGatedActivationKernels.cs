namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP GatedActivation kernels — shared source with CUDA (HIP is CUDA-compatible).
/// </summary>
public static class HipGatedActivationKernels
{
    public static string GetSource() => CUDA.Kernels.CudaGatedActivationKernels.GetSource();
    public static string[] GetKernelNames() => CUDA.Kernels.CudaGatedActivationKernels.GetKernelNames();
}
