namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP LossForward kernels — shared source with CUDA (HIP is CUDA-compatible).
/// </summary>
public static class HipLossForwardKernels
{
    public static string GetSource() => CUDA.Kernels.CudaLossForwardKernels.GetSource();
    public static string[] GetKernelNames() => CUDA.Kernels.CudaLossForwardKernels.GetKernelNames();
}
